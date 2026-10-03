"""
apps/cron_tick_routes.py
--------------------------
The /internal-prefixed routes that back Turtle's own trusted automation —
GitHub Actions and self-invoking follow-up requests, never end users or
third-party webhooks.

WP 1.B (ledger 1a.3 / S-7.3): these two routes used to share one bearer
secret (CRON_SHARED_SECRET) and one non-constant-time compare. They now use
two DIFFERENT secrets, each scoped to its own endpoint (core.internal_auth.
check_bearer, constant-time):
  - /internal/cron-tick authenticates GitHub Actions with CRON_TICK_SECRET
    only — no payload-by-reference needed, this route reads no body at all.
  - /internal/embed-personal-memory authenticates Turtle's own self-invoke
    (core/worker.py) with INTERNAL_JOB_SECRET only, ADDITIONALLY verified via
    an HMAC signature envelope (core.internal_auth.verify_request) over the
    request's timestamp, nonce and raw body. Its body no longer carries
    user_id directly — only an opaque job id pointing at a payload the
    caller stashed in Redis (core.internal_auth.store_job_payload); this
    endpoint reads-and-deletes it, so the identity fields are never trusted
    from the wire and a captured request can't be replayed.

POST /internal/cron-tick — Vercel migration Phase 2 — cloud replacement for
the in-process APScheduler (core/routine_scheduler.py). Serverless has no
persistent process to hold a live scheduler in, so instead a periodic
external trigger (a GitHub Actions `on: schedule` workflow,
.github/workflows/cron-tick.yml, nominally every 5 minutes but in practice
delivered far less often and hours apart) hits this endpoint. Because ticks
arrive irregularly, the endpoint does not ask "is anything due right now?":
it remembers when it last ran (core/storage/cloud/cron_state_store.py) and
enumerates EVERY occurrence in (last_tick_at, now]. Occurrences up to the
late-fire limit (default 1 hour, env TURTLE_ROUTINE_LATE_FIRE_LIMIT_S) fire;
older ones are recorded as `missed` and the user gets one line on their next
connect. Each occurrence is claimed in Postgres before firing and flipped to
`fired` only after the journal write and delivery succeed; claims stuck in
`claimed` are re-fired by later ticks (at-least-once, with a deterministic
journal event id so the repeat is a no-op). Only meaningful in cloud mode
(TURTLE_DEPLOY=cloud) — local dev keeps using the always-on
RoutineScheduler; this endpoint returns 503 there too, since running it
against local SQLite/APScheduler's own state would double-fire every
routine.

POST /internal/embed-personal-memory — the target of
core/worker.py::dispatch_embed_personal_memory_job's cloud-mode self-invoke.
core/personal_memory_store.py::write_topic() needs to run the
embed_personal_memory job (a live Cohere call + a pgvector upsert) without
itself becoming an async method just to await it — and a bare detached
asyncio.create_task carries the same unconfirmed-survival-past-response risk
already fixed for Discord's deferred interaction processing. So the detached
task's job shrinks to "reliably kick off this endpoint as an independent
request" (see dispatch_embed_personal_memory_job's docstring for the full
"why"), and this endpoint runs the actual job to completion with its own
normal request timeout.

Auth: each route's own secret (see module docstring), constant-time compared
via core.internal_auth.check_bearer. Modeled on apps/admin_routes.py's
admin-token pattern: an unset secret returns 503 (a misconfigured cloud
deploy fails loud rather than leaving the endpoint open), a missing/wrong
token returns 401.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request

from core.config import settings
from core.internal_auth import (
    SignatureError,
    check_bearer,
    require_secret,
    take_job_payload,
    verify_request,
)
from core.storage.cloud import CloudBackendUnavailable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["cron"])

# Occurrences older than this when a tick finally sees them are recorded as
# `missed` instead of fired (ledger 5.7: a news routine delivered five hours
# late is noise; telling the user is honest). Default 1 hour = the confirmed
# decision. The real cause of lateness is GitHub dropping most scheduled runs
# (measured ~2% delivery), not this limit -- raise it via the env var if late
# delivery is preferred to a "missed" line.
DEFAULT_LATE_FIRE_LIMIT_S = 3600
_LATE_FIRE_LIMIT_ENV = "TURTLE_ROUTINE_LATE_FIRE_LIMIT_S"

# A claim still in `claimed` after this long is presumed dead (the tick that
# claimed it crashed or its fire failed) and is re-fired by the next tick.
STUCK_CLAIM_MIN_AGE_S = 600

# Never enumerate further back than this, however old last_tick_at is: bounds
# the work and the number of `missed` rows after a long outage.
MAX_LOOKBACK = timedelta(hours=24)

# routine_last_fired rows older than this are pruned every tick (ledger 5.10).
CLAIM_RETENTION = timedelta(days=7)


def _late_fire_limit_s() -> int:
    """Late-fire limit in seconds. Read from the environment at call time
    (not in core.config.Settings) so it needs no config change; an unset,
    non-integer or non-positive value falls back to the 1-hour default."""
    raw = os.environ.get(_LATE_FIRE_LIMIT_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_LATE_FIRE_LIMIT_S
    try:
        value = int(raw.strip())
    except ValueError:
        value = 0
    if value <= 0:
        logger.warning(
            "%s=%r is not a positive integer; using %ds",
            _LATE_FIRE_LIMIT_ENV, raw, DEFAULT_LATE_FIRE_LIMIT_S,
        )
        return DEFAULT_LATE_FIRE_LIMIT_S
    return value


def _check_cron_auth(authorization: str | None) -> None:
    """CRON_TICK_SECRET only — the one secret GitHub Actions holds."""
    try:
        expected = require_secret(settings.cron_tick_secret, "CRON_TICK_SECRET")
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cron-tick endpoint is disabled ({exc} not set).",
        ) from exc
    if not check_bearer(expected, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized.")


async def _check_job_auth(request: Request, authorization: str | None) -> bytes:
    """INTERNAL_JOB_SECRET only, bearer + HMAC signature envelope. Returns
    the raw request body bytes (already read for the signature check) so the
    caller doesn't re-read the stream.
    """
    try:
        expected = require_secret(settings.internal_job_secret, "INTERNAL_JOB_SECRET")
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Endpoint is disabled ({exc} not set).",
        ) from exc
    if not check_bearer(expected, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized.")

    body = await request.body()
    try:
        await verify_request(
            expected,
            request.headers.get("X-Turtle-Timestamp"),
            request.headers.get("X-Turtle-Nonce"),
            request.headers.get("X-Turtle-Signature"),
            body,
        )
    except SignatureError as exc:
        raise HTTPException(status_code=401, detail=f"Unauthorized: {exc}") from exc
    return body


def _missed_notice_frame(missed: list[Any]) -> dict[str, Any]:
    """ONE user-facing frame summarising every occurrence a tick recorded as
    missed for a user (kept to one frame so missed notices cannot crowd real
    fire frames out of the 5-frame per-user outbox).

    fired_at is derived from the occurrence instants, not the clock, so the
    frame identity (routine_key, fired_at) is stable.
    """
    from core.routine_scheduler import _humanize_routine_key

    latest = max(missed, key=lambda d: d.fire_utc)
    if len(missed) == 1:
        value = latest.event.value
        name = value.get("routine") or _humanize_routine_key(latest.routine_key)
        at = value.get("time") or latest.fire_bucket.split("T")[-1]
        message = f"I missed your {at} routine ({name})."
    else:
        names = sorted(
            {
                d.event.value.get("routine") or _humanize_routine_key(d.routine_key)
                for d in missed
            }
        )
        shown = ", ".join(names[:3]) + (f" (+{len(names) - 3} more)" if len(names) > 3 else "")
        message = f"I missed {len(missed)} routine runs ({shown})."
    return {
        "type": "routine",
        "code": "routine_missed",
        "message": message,
        "routine_key": latest.routine_key,
        "fired_at": latest.fire_utc.isoformat(),
        "missed_count": len(missed),
    }


def _run_tick(now_utc: datetime) -> dict[str, Any]:
    """The tick, under the global tick lock (cron_state FOR UPDATE).

    If another tick holds the lock this returns immediately with
    ``skipped: True`` -- two overlapping ticks must not enumerate the same
    window. Otherwise it runs _tick_body over (last_tick_at, now] and, only
    when every user's routines could be read, advances last_tick_at. An
    exception from the body rolls the lock transaction back (last_tick_at
    unchanged) and propagates, so the run fails visibly and the window is
    retried by the next tick; claims make the retry safe.
    """
    from core.storage.cloud.cron_state_store import locked_tick_state

    limit_s = _late_fire_limit_s()
    with locked_tick_state() as state:
        if state is None:
            logger.info("cron-tick: another tick holds the lock; skipping")
            return {
                "correlation_id": uuid.uuid4().hex,
                "skipped": True,
                "users_checked": 0,
                "routines_fired": 0,
                "error_count": 0,
                "ticked_at": now_utc.isoformat(),
            }
        result, complete = _tick_body(now_utc, state.last_tick_at, limit_s)
        if complete:
            state.advance(now_utc)
        else:
            logger.warning(
                "cron-tick[%s]: window not advanced (some users could not be read);"
                " it will be re-enumerated by the next tick", result["correlation_id"],
            )
        result["window_advanced"] = complete
        return result


def _tick_body(
    now_utc: datetime, last_tick_at: datetime | None, limit_s: int
) -> tuple[dict[str, Any], bool]:
    """The actual scan -- synchronous (every call it makes is itself sync:
    JournalStore in cloud mode, the claim store, _fire_routine_strict), run
    via asyncio.to_thread from the route so it never blocks the event loop.
    Returns (result, complete). complete is False when an occurrence could
    have been skipped without a record (a user's routines could not be read,
    or a claim could not be written), so the caller must not advance
    last_tick_at past it.

    Window: (last_tick_at, now]. Bootstrap, when last_tick_at is None (the
    very first tick ever, or a wiped cron_state): enumerate only the last
    `limit_s` seconds. There is no history to reconcile, and a longer window
    would fire or announce "missed" for occurrences from before the
    scheduler existed; one late-fire limit is exactly the span in which an
    occurrence could still legitimately fire. last_tick_at older than
    MAX_LOOKBACK is clamped to it.

    WP 1.B / S-7.3 output hygiene: the RETURNED dict (which becomes this
    endpoint's HTTP response, and is what .github/workflows/cron-tick.yml
    echoes into the Actions log) carries counts and a correlation id only --
    never a user id. Per-error detail (which DOES include user_id / routine
    key, useful for debugging) goes to the server's own logger instead,
    tagged with the same correlation id so a specific tick's errors can be
    found across the two.
    """
    from core.routine_cron_tick import compute_due_routines, fire_instant_for_bucket
    from core.routine_scheduler import (
        _delivery_hook,
        _fire_routine_strict,
        get_active_routines_for_user,
    )
    from core.storage.cloud import routine_last_fired_store as claims
    from core.storage.cloud.journal_store import list_user_ids_pg

    correlation_id = uuid.uuid4().hex
    counts = {
        "users_checked": 0,
        "routines_fired": 0,
        "routines_missed": 0,
        "routines_refired": 0,
        "claims_failed": 0,
        "claims_pruned": 0,
        "error_count": 0,
    }
    complete = True

    lookback = max(MAX_LOOKBACK, timedelta(seconds=limit_s))
    if last_tick_at is None:
        after_utc = now_utc - timedelta(seconds=limit_s)
    else:
        after_utc = max(last_tick_at, now_utc - lookback)
    window_truncated = last_tick_at is not None and last_tick_at < now_utc - lookback

    def fire_claimed(
        user_id: str, routine_key: str, value: dict[str, Any], bucket: str, fire_utc: datetime
    ) -> bool:
        """Fire an already-claimed occurrence and confirm it. True only when
        the fire really happened; a failure leaves the claim in 'claimed' for
        a later tick to re-fire."""
        try:
            _fire_routine_strict(
                user_id, routine_key, dict(value),
                fire_bucket=bucket, fired_at=fire_utc.isoformat(),
            )
        except Exception as exc:
            counts["error_count"] += 1
            logger.warning(
                "cron-tick[%s]: fire failed for routine %s: %s",
                correlation_id, routine_key, exc,
            )
            return False
        counts["routines_fired"] += 1
        try:
            claims.mark_fired(user_id, routine_key, bucket)
        except Exception as exc:
            # The fire happened; the claim stays 'claimed' and a later tick
            # re-fires it harmlessly (journal no-op, same frame identity).
            counts["error_count"] += 1
            logger.warning(
                "cron-tick[%s]: mark_fired failed for routine %s: %s",
                correlation_id, routine_key, exc,
            )
        return True

    for user_id in list_user_ids_pg():
        counts["users_checked"] += 1
        try:
            routines = get_active_routines_for_user(user_id)
        except Exception as exc:
            counts["error_count"] += 1
            complete = False
            logger.warning(
                "cron-tick[%s]: journal read failed for a user: %s", correlation_id, exc
            )
            continue

        # Only applied, non-rejected routines are live -- a retracted or
        # still-pending-confirmation routine must never fire (mirrors
        # RoutineScheduler.register_for_user's own filter).
        active = {k: e for k, e in routines.items() if e.applied and not e.rejected}
        if not active:
            continue

        due = compute_due_routines(active, after_utc=after_utc, until_utc=now_utc)
        on_time = []
        missed_claimed = []
        for occ in due:
            late_s = (now_utc - occ.fire_utc).total_seconds()
            try:
                if late_s > limit_s:
                    if claims.try_claim_fire(
                        user_id, occ.routine_key, occ.fire_bucket, status="missed"
                    ):
                        missed_claimed.append(occ)
                else:
                    on_time.append(occ)
            except Exception as exc:
                counts["error_count"] += 1
                complete = False
                logger.warning(
                    "cron-tick[%s]: could not record missed routine %s: %s",
                    correlation_id, occ.routine_key, exc,
                )

        # Missed notice FIRST so this tick's real fire frames are the newest
        # in the user's capped outbox (most-recent-kept).
        if missed_claimed:
            counts["routines_missed"] += len(missed_claimed)
            try:
                _delivery_hook(user_id, _missed_notice_frame(missed_claimed))
            except Exception as exc:
                counts["error_count"] += 1
                logger.warning(
                    "cron-tick[%s]: missed-routine notice failed: %s", correlation_id, exc
                )

        for occ in on_time:
            try:
                if not claims.try_claim_fire(user_id, occ.routine_key, occ.fire_bucket):
                    continue  # already claimed by an earlier tick
            except Exception as exc:
                counts["error_count"] += 1
                complete = False
                logger.warning(
                    "cron-tick[%s]: claim failed for routine %s: %s",
                    correlation_id, occ.routine_key, exc,
                )
                continue
            fire_claimed(user_id, occ.routine_key, occ.event.value, occ.fire_bucket, occ.fire_utc)

    # At-least-once: re-fire claims stuck in 'claimed' (the claiming tick died
    # or its fire failed), then give up on ones too old to be worth firing.
    try:
        stuck = claims.list_stuck_claims(STUCK_CLAIM_MIN_AGE_S, limit_s)
    except Exception as exc:
        counts["error_count"] += 1
        stuck = []
        logger.warning("cron-tick[%s]: stuck-claim scan failed: %s", correlation_id, exc)
    by_user: dict[str, list[tuple[str, str]]] = {}
    for user_id, routine_key, bucket in stuck:
        by_user.setdefault(user_id, []).append((routine_key, bucket))
    for user_id, items in by_user.items():
        try:
            active = {
                k: e for k, e in get_active_routines_for_user(user_id).items()
                if e.applied and not e.rejected
            }
        except Exception as exc:
            counts["error_count"] += 1
            logger.warning(
                "cron-tick[%s]: journal read failed re-firing a user's claims: %s",
                correlation_id, exc,
            )
            continue
        for routine_key, bucket in items:
            event = active.get(routine_key)
            fire_utc = fire_instant_for_bucket(event.value, bucket) if event else None
            if event is None or fire_utc is None:
                # Routine retracted/rejected or unusable: nothing to re-fire.
                try:
                    claims.mark_failed(user_id, routine_key, bucket)
                    counts["claims_failed"] += 1
                except Exception as exc:
                    counts["error_count"] += 1
                    logger.warning("cron-tick[%s]: mark_failed failed: %s", correlation_id, exc)
                continue
            if fire_claimed(user_id, routine_key, event.value, bucket, fire_utc):
                counts["routines_refired"] += 1
    try:
        counts["claims_failed"] += claims.fail_stale_claims(limit_s)
        counts["claims_pruned"] = claims.prune_older_than(
            (now_utc - CLAIM_RETENTION).isoformat()
        )
    except Exception as exc:
        counts["error_count"] += 1
        logger.warning("cron-tick[%s]: claim housekeeping failed: %s", correlation_id, exc)

    result = {
        "correlation_id": correlation_id,
        **counts,
        "window_truncated": window_truncated,
        "ticked_at": now_utc.isoformat(),
    }
    return result, complete


@router.post("/cron-tick")
async def cron_tick(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_cron_auth(authorization)
    if not settings.is_cloud:
        raise HTTPException(
            status_code=503,
            detail="Cron-tick is a cloud-mode endpoint; local dev uses RoutineScheduler.",
        )
    now_utc = datetime.now(UTC)
    result = await asyncio.to_thread(_run_tick, now_utc)
    if result["error_count"]:
        logger.warning(
            "cron-tick[%s] completed with %d error(s)",
            result["correlation_id"], result["error_count"],
        )
    return result


@router.post("/embed-personal-memory")
async def embed_personal_memory_endpoint(
    request: Request, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """Runs core.background_tasks.embed_personal_memory to completion as its
    own independent request — see this module's docstring and
    core/worker.py::dispatch_embed_personal_memory_job for the full "why".

    WP 1.B / S-7.3: the wire body carries only an opaque ``job_id`` — NOT
    user_id/topic_name/lines directly. core/worker.py's self-invoke stores
    the real payload in Redis first (core.internal_auth.store_job_payload);
    this endpoint reads-and-deletes it (take_job_payload) so the identity
    field (user_id) is never trusted from the request, and a captured
    request can't be replayed to re-trigger the same embed even if it clears
    the outer signature check twice.
    """
    body = await _check_job_auth(request, authorization)

    try:
        envelope = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    job_id = envelope.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id is required")

    try:
        payload = await take_job_payload(job_id)
    except CloudBackendUnavailable as exc:
        # Redis went from reachable (the signature check above needs it too)
        # to unreachable between there and here — surface as a clean 503,
        # never an unhandled 500 from a raw driver exception.
        raise HTTPException(status_code=503, detail=f"Job store unavailable: {exc}") from exc
    if payload is None:
        raise HTTPException(status_code=401, detail="Unknown or expired job id")

    user_id = payload.get("user_id")
    topic_name = payload.get("topic_name")
    lines = payload.get("lines")
    if not user_id or not topic_name or not isinstance(lines, list):
        raise HTTPException(
            status_code=400, detail="stored job payload is missing required fields"
        )

    from core.background_tasks import embed_personal_memory

    try:
        await embed_personal_memory(user_id=user_id, topic_name=topic_name, lines=lines)
    except Exception as exc:
        logger.error("embed-personal-memory failed user=%s topic=%s: %s", user_id, topic_name, exc)
        raise HTTPException(status_code=500, detail="Embedding failed")

    return {"ok": True}
