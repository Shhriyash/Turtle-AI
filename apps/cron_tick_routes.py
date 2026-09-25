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
.github/workflows/cron-tick.yml, every 5 minutes) hits this endpoint, which
asks "which routines are due right now?" and fires them — a stateless tick
rather than an always-running evaluator. Only meaningful in cloud mode
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
import uuid
from datetime import UTC, datetime
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["cron"])

# GitHub Actions' `on: schedule` cron granularity in this repo's workflow is
# 5 minutes — see .github/workflows/cron-tick.yml. A routine due at HH:05 is
# still due if the tick actually lands at HH:07 (GH Actions doesn't guarantee
# to-the-minute execution), so the due-window width must match this value.
TICK_INTERVAL_MINUTES = 5


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


def _run_tick(now_utc: datetime) -> dict[str, Any]:
    """The actual scan — synchronous (every call it makes is itself sync:
    JournalStore in cloud mode, try_claim_fire, _fire_routine), run via
    asyncio.to_thread from the route so it never blocks the event loop.

    WP 1.B / S-7.3 output hygiene: the RETURNED dict (which becomes this
    endpoint's HTTP response, and is what .github/workflows/cron-tick.yml
    echoes into the Actions log) carries counts and a correlation id only —
    never a user id. Per-error detail (which DOES include user_id / routine
    key, useful for debugging) goes to the server's own logger instead,
    tagged with the same correlation id so a specific tick's errors can be
    found across the two.
    """
    from core.routine_cron_tick import compute_due_routines
    from core.routine_scheduler import _fire_routine, get_active_routines_for_user
    from core.storage.cloud.journal_store import list_user_ids_pg
    from core.storage.cloud.routine_last_fired_store import try_claim_fire

    correlation_id = uuid.uuid4().hex
    users_checked = 0
    routines_fired = 0
    error_count = 0

    for user_id in list_user_ids_pg():
        users_checked += 1
        try:
            routines = get_active_routines_for_user(user_id)
        except Exception as exc:
            error_count += 1
            logger.warning(
                "cron-tick[%s]: journal read failed for a user: %s", correlation_id, exc
            )
            continue

        # Only applied, non-rejected routines are live — a retracted or
        # still-pending-confirmation routine must never fire (mirrors
        # RoutineScheduler.register_for_user's own filter).
        active = {k: e for k, e in routines.items() if e.applied and not e.rejected}
        if not active:
            continue

        due = compute_due_routines(
            active, now_utc=now_utc, tick_interval_minutes=TICK_INTERVAL_MINUTES
        )
        for routine_key, event, fire_bucket in due:
            try:
                if not try_claim_fire(user_id, routine_key, fire_bucket):
                    continue  # already fired this scheduled occurrence
                _fire_routine(user_id, routine_key, dict(event.value))
                routines_fired += 1
            except Exception as exc:
                error_count += 1
                logger.warning(
                    "cron-tick[%s]: fire failed for routine %s: %s",
                    correlation_id, routine_key, exc,
                )

    return {
        "correlation_id": correlation_id,
        "users_checked": users_checked,
        "routines_fired": routines_fired,
        "error_count": error_count,
        "ticked_at": now_utc.isoformat(),
    }


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

    payload = await take_job_payload(job_id)
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
