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
from typing import Optional


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

    Returns "{user_id}:{sha1}". Callers pass this straight to
    is_duplicate_invocation/record_invocation, which add the "turtle:idem:"
    storage prefix.
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
    return f"{safe_user}:{digest}"


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


def record_invocation(idempotency_key: str, result: str) -> None:
    """Finalize a reservation previously taken by is_duplicate_invocation.

    On success (result starts with "Email sent successfully"), overwrite the
    reservation with the final result so subsequent duplicates within the
    window get the cached result. On failure, DELETE the reservation so the
    user's retry is not blocked by a failed send.

    In cloud mode, delegates to the Redis-backed implementation — see
    is_duplicate_invocation's docstring for why.
    """
    from core.config import settings

    if settings.is_cloud:
        from core.storage.cloud.redis_backends import redis_record_invocation

        redis_record_invocation(idempotency_key, result)
        return
    try:
        conn = _ensure_db()
        if str(result).startswith("Email sent successfully"):
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
