"""
tools/idempotency.py
--------------------
B5: Idempotency keys on every side-effecting tool.

Email idempotency (reservation-based, per-tenant):
  key = "{user_id}:" + sha1(sorted_recipients + subject + body_first_100_chars)
  Stored in SQLite (local) tool_invocations table, or Redis (cloud) under
  "turtle:idem:{key}".

The key is scoped per user_id so two different tenants sending an identical
email within the same window do NOT collide (the previous key was global,
which meant tenant B's identical send inside tenant A's 60s window would
silently no-op).

Dedup is RESERVATION-based, not check-then-act: `is_duplicate_invocation`
atomically claims the key (SET ... NX EX 60 in Redis / INSERT on a PRIMARY
KEY in SQLite) BEFORE the caller sends anything. This closes a race where
two concurrent identical sends both saw "not yet recorded" and both fired,
because nothing was written until after the SMTP call returned.

Both backends are FAIL CLOSED: if the reservation store itself cannot be
reached, `is_duplicate_invocation` raises `IdempotencyReservationError` and
the caller MUST refuse the send. This is a deliberate behaviour flip from
the prior fail-open posture (Redis/SQLite errors used to be swallowed and
treated as "proceed, this is new"). A duplicate email is unrecoverable; a
send delayed by a transient Redis/DB blip is not.

Usage::

    from tools.idempotency import (
        build_email_idempotency_key,
        is_duplicate_invocation,
        record_invocation,
        IdempotencyReservationError,
    )

    key = build_email_idempotency_key(user_id, recipients, subject, body)
    try:
        cached = is_duplicate_invocation(key)
    except IdempotencyReservationError:
        return "The duplicate-send safety check is unavailable; refusing to send."
    if cached is not None:
        return cached          # duplicate: cached result, or "still sending" message
    result = send_email_now(...)
    record_invocation(key, result)
    return result
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_IDEMPOTENCY_WINDOW_S = 60          # Re-attempts within this window are no-ops
_DB_PATH: Path | None = None        # Resolved lazily from env / default

# Sentinel result value written by the reservation itself, before the SMTP
# call has returned. Distinguishing "pending" from a completed result lets a
# concurrent duplicate be told "still in flight" instead of getting a stale
# cached result or silently re-sending.
_PENDING_SENTINEL = "__pending__"

_PENDING_MESSAGE = (
    "An identical email is already being sent (it started moments ago). "
    "Please wait a few seconds before trying again to avoid sending it twice."
)


class IdempotencyReservationError(RuntimeError):
    """Raised when the idempotency reservation store could not be reached.

    Callers MUST treat this as fail-closed: refuse the send rather than risk
    a duplicate, and say plainly that nothing was sent (so the model does not
    tell the user their mail went out).
    """


def _get_db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        # Anchor to settings.data_dir (repo-root default, honors TURTLE_DATA_DIR).
        # The old bare Path("data") was CWD-relative: launching the server from
        # another directory silently created a fresh tool_invocations.db beside
        # that CWD instead of under <repo>/data — the exact hazard core/config.py
        # (data_dir field + _anchor_data_dir) fixed for everything else. Reusing
        # settings.data_dir keeps idempotency state co-located with the rest of
        # the data volume. Safe from import cycles: core.config is a leaf module
        # (stdlib + pydantic only) and tools/ already imports it (calendar_tool).
        from core.config import settings
        base = settings.data_dir
        base.mkdir(parents=True, exist_ok=True)
        # Log the resolved location once: deployments that previously launched
        # from a non-repo CWD had a stray CWD-relative DB; the log makes the
        # anchor change visible instead of silently "losing" old entries.
        print(f"LOG: idempotency DB at {base / 'tool_invocations.db'}")
        _DB_PATH = base / "tool_invocations.db"
    return _DB_PATH


# ---------------------------------------------------------------------------
# DB setup (creates table once on first call)
# ---------------------------------------------------------------------------

_DB_INITIALIZED = False


def _ensure_db() -> sqlite3.Connection:
    global _DB_INITIALIZED
    conn = sqlite3.connect(str(_get_db_path()), timeout=5)
    if not _DB_INITIALIZED:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_invocations (
                idempotency_key TEXT PRIMARY KEY,
                result          TEXT NOT NULL,
                created_at_s    REAL NOT NULL
            )
        """)
        # Prune old rows on startup (older than 1 hour keeps the table lean)
        conn.execute(
            "DELETE FROM tool_invocations WHERE created_at_s < ?",
            (time.time() - 3600,),
        )
        conn.commit()
        _DB_INITIALIZED = True
    return conn


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------

def build_email_idempotency_key(
    user_id: str,
    recipients: list[str],
    subject: str,
    body: str,
    *,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
) -> str:
    """Build a stable, per-tenant idempotency key for an email send.

    `user_id` scopes the key so two different tenants sending byte-identical
    emails within the same window never collide (the previous key was
    global — a cross-tenant dedup bug). The created_at_s / TTL window in
    is_duplicate_invocation already bounds the dedup horizon; a wall-clock
    bucket in the key made dedup fail exactly when retries straddled a
    minute boundary, so none is included here either.

    Returns "{user_id}:email:{sha1}". The "email:" discriminator (added
    alongside the calendar draft/confirm flow) keeps an email key and a
    calendar key for the same user from colliding as raw strings — the
    "turtle:idem:" storage prefix added by the Redis backend does not by
    itself guarantee that, since it is common to every tool's key. Callers
    pass this straight to is_duplicate_invocation/record_invocation.

    NOTE: this changes the key's raw string versus the pre-existing
    "{user_id}:{sha1}" shape (the sha1 digest itself is unchanged — same
    canonical string, same hash). Any reservation held under the old shape
    at the moment this ships stops being reachable by its old key; that is
    at most a 60s dedup window (_IDEMPOTENCY_WINDOW_S) per in-flight send,
    not a correctness bug — a concurrent duplicate send in that exact
    window would no longer be caught, but no email is lost or double-sent
    outside of that pre-existing race.
    """
    sorted_recipients = sorted(r.lower().strip() for r in recipients)
    sorted_cc = sorted(r.lower().strip() for r in (cc or []))
    sorted_bcc = sorted(r.lower().strip() for r in (bcc or []))
    body_prefix = (body or "")[:100]
    canonical = (
        f"to:{','.join(sorted_recipients)}"
        f"|cc:{','.join(sorted_cc)}"
        f"|bcc:{','.join(sorted_bcc)}"
        f"|sub:{(subject or '').strip()}"
        f"|body:{body_prefix}"
    )
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
    safe_user = (user_id or "").strip() or "anonymous"
    return f"{safe_user}:email:{digest}"


def build_calendar_idempotency_key(
    user_id: str,
    title: str,
    start_iso: str,
    end_iso: str,
    attendee_emails: list[str] | None = None,
) -> str:
    """Build a stable, per-tenant idempotency key for a calendar_confirm.

    Mirrors build_email_idempotency_key's shape: "{user_id}:cal:{sha1}",
    hashing (title, start, end, attendees) per the ledger's chosen key
    shape (`turtle:idem:{uid}:cal:{sha1(title,start,end,attendees)}` once
    the Redis backend's "turtle:idem:" prefix is added).
    """
    sorted_attendees = sorted(e.lower().strip() for e in (attendee_emails or []))
    canonical = (
        f"title:{(title or '').strip()}"
        f"|start:{(start_iso or '').strip()}"
        f"|end:{(end_iso or '').strip()}"
        f"|attendees:{','.join(sorted_attendees)}"
    )
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
    safe_user = (user_id or "").strip() or "anonymous"
    return f"{safe_user}:cal:{digest}"


# ---------------------------------------------------------------------------
# Idempotency reservation + finalize
# ---------------------------------------------------------------------------

def is_duplicate_invocation(idempotency_key: str) -> Optional[str]:
    """Atomically reserve `idempotency_key` for a new send, or report a duplicate.

    Returns None if the reservation was acquired: the caller must proceed to
    send now, then call record_invocation to finalize it.

    Returns a non-None string if this is a duplicate:
      - if another send for the same key is still mid-flight (reserved but
        not yet finalized), a "please wait" message is returned;
      - if a prior send already completed within the window, that cached
        result string is returned.

    Raises IdempotencyReservationError if the reservation store itself could
    not be reached. Callers MUST treat this as fail-closed and refuse the
    send — see the module docstring for why this is a deliberate flip from
    the previous fail-open behaviour.

    In cloud mode (TURTLE_DEPLOY=cloud), delegates to the Redis-backed
    implementation instead: the SQLite file below does not survive a
    serverless cold start, silently disabling dedup on every fresh
    invocation. See core/storage/cloud/redis_backends.py.
    """
    from core.config import settings

    if settings.is_cloud:
        from core.storage.cloud.redis_backends import (
            IdempotencyReservationError as _RedisUnavailable,
            redis_is_duplicate_invocation,
        )

        try:
            return redis_is_duplicate_invocation(idempotency_key)
        except _RedisUnavailable as exc:
            raise IdempotencyReservationError(str(exc)) from exc

    try:
        conn = _ensure_db()
        cutoff = time.time() - _IDEMPOTENCY_WINDOW_S
        # Prune this key if its prior reservation/result has aged out, so a
        # fresh attempt after the window can re-claim it (mirrors Redis's
        # own EX 60 expiry).
        conn.execute(
            "DELETE FROM tool_invocations WHERE idempotency_key = ? AND created_at_s < ?",
            (idempotency_key, cutoff),
        )
        try:
            conn.execute(
                "INSERT INTO tool_invocations (idempotency_key, result, created_at_s) VALUES (?, ?, ?)",
                (idempotency_key, _PENDING_SENTINEL, time.time()),
            )
            conn.commit()
            conn.close()
            return None
        except sqlite3.IntegrityError:
            # Row already exists within the window: someone else holds the
            # reservation (pending) or already finished (final result).
            row = conn.execute(
                "SELECT result FROM tool_invocations WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            conn.close()
            if row is None:
                # Raced with a concurrent finalize/delete; safe to treat as new.
                return None
            existing = str(row[0])
            return _PENDING_MESSAGE if existing == _PENDING_SENTINEL else existing
    except Exception as exc:
        print(f"LOG: Idempotency reservation failed ({exc}) — refusing send (fail closed)")
        raise IdempotencyReservationError(str(exc)) from exc


def record_invocation(idempotency_key: str, result: str, *, success: bool | None = None) -> None:
    """Finalize a reservation previously taken by is_duplicate_invocation.

    On success, overwrite the reservation with the final result so
    subsequent duplicates within the window get the cached result. On
    failure, DELETE the reservation so the user's retry is not blocked by a
    failed send.

    `success` is generalised (originally this only ever sniffed email's
    "Email sent successfully" prefix, which is meaningless for a calendar
    result string). Pass it explicitly for any non-email caller — e.g.
    calendar_confirm knows success from the tool's ToolResult.status, not
    from string-sniffing its rendered text. When omitted, falls back to the
    original email-specific sniff so the existing email call site (which
    does not pass `success`) keeps its exact prior behaviour.

    In cloud mode, delegates to the Redis-backed implementation — see
    is_duplicate_invocation's docstring for why.
    """
    if success is None:
        success = str(result).startswith("Email sent successfully")

    from core.config import settings

    if settings.is_cloud:
        from core.storage.cloud.redis_backends import redis_record_invocation

        redis_record_invocation(idempotency_key, result, success=success)
        return
    try:
        conn = _ensure_db()
        if success:
            conn.execute(
                "INSERT OR REPLACE INTO tool_invocations (idempotency_key, result, created_at_s) VALUES (?, ?, ?)",
                (idempotency_key, result, time.time()),
            )
        else:
            conn.execute(
                "DELETE FROM tool_invocations WHERE idempotency_key = ?",
                (idempotency_key,),
            )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"LOG: Idempotency finalize failed ({exc}) — reservation may linger until its TTL")


async def send_with_reservation(
    idempotency_key: str,
    send_coro_factory: Callable[[], Awaitable[str]],
    *,
    is_success: Callable[[str], bool] | None = None,
) -> str:
    """Run a reserved send, GUARANTEEING the reservation is finalized or
    released however it exits.

    Call this only after is_duplicate_invocation(idempotency_key) has
    already returned None (i.e. the reservation is held). `send_coro_factory`
    is called with no arguments and awaited to actually perform the send;
    it is a factory (not a bare coroutine) so this function can be reused
    safely without "coroutine was never awaited" surprises.

    `is_success`, when given, is called with the returned result string to
    decide whether record_invocation caches it (True) or releases the
    reservation (False) — the email call site omits it and keeps relying on
    record_invocation's own "Email sent successfully" sniff; a calendar
    caller (whose result string is not that sentinel) should pass one, e.g.
    a closure over the ToolResult.status captured before stringifying it.

    Without this wrapper, a send that RAISES instead of returning (a
    cancelled task from a client disconnect mid-SMTP, thread-pool
    exhaustion, an unhandled bug) leaves the reservation stuck at its
    "pending" sentinel for the rest of the 60s window: the caller is told
    the send failed, and then blocked from retrying immediately by the very
    safety mechanism meant to prevent duplicates. That is a regression
    relative to the pre-reservation behaviour (a failed send used to leave
    no trace at all), so this function must not let it happen.

    - Normal return: finalizes with that result via record_invocation
      (success strings are cached as the duplicate-return value; anything
      else deletes the reservation so a retry is not blocked).
    - ANY exception during the send — this deliberately catches
      BaseException, not Exception, so asyncio.CancelledError (which since
      Python 3.8 does NOT subclass Exception) is included — means the send
      never produced a result. The reservation is released (via the same
      delete-on-non-success path record_invocation already has) and the
      original exception is then re-raised UNCHANGED: cancellation is never
      swallowed here, since that would corrupt task-shutdown semantics.
    - If the send itself succeeded but finalizing that success then raises
      (e.g. a bug in record_invocation), the reservation is deliberately
      left AS-IS rather than released: we cannot tell from here whether the
      underlying send actually went out, and releasing it would let a
      concurrent/retried request send a genuine duplicate. The raise
      propagates to the caller, which sees "no result" for this call even
      though the underlying send may have succeeded — the reservation still
      protects against a double-send in that window.
    """
    try:
        result = await send_coro_factory()
    except BaseException:
        try:
            record_invocation(
                idempotency_key,
                "Failed: the operation did not complete",
                success=False,
            )
        except Exception:
            pass
        raise
    record_invocation(
        idempotency_key, result, success=(is_success(result) if is_success else None)
    )
    return result
