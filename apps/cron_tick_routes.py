"""
apps/cron_tick_routes.py
--------------------------
Vercel migration Phase 2 — cloud replacement for the in-process APScheduler
(core/routine_scheduler.py). Serverless has no persistent process to hold a
live scheduler in, so instead a periodic external trigger (a GitHub Actions
`on: schedule` workflow, .github/workflows/cron-tick.yml, every 5 minutes)
hits this endpoint, which asks "which routines are due right now?" and fires
them — a stateless tick rather than an always-running evaluator.

Endpoint:
    POST /internal/cron-tick

Auth: bearer token, must equal settings.cron_shared_secret. Modeled on
apps/admin_routes.py's admin-token pattern: an unset secret returns 503 (a
misconfigured cloud deploy fails loud rather than leaving the endpoint open),
a missing/wrong token returns 401.

Only meaningful in cloud mode (TURTLE_DEPLOY=cloud) — local dev keeps using
the always-on RoutineScheduler; this endpoint returns 503 there too, since
running it against local SQLite/APScheduler's own state would double-fire
every routine.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Header, HTTPException

from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["cron"])

# GitHub Actions' `on: schedule` cron granularity in this repo's workflow is
# 5 minutes — see .github/workflows/cron-tick.yml. A routine due at HH:05 is
# still due if the tick actually lands at HH:07 (GH Actions doesn't guarantee
# to-the-minute execution), so the due-window width must match this value.
TICK_INTERVAL_MINUTES = 5


def _check_auth(authorization: str | None) -> None:
    expected = (
        settings.cron_shared_secret.get_secret_value()
        if settings.cron_shared_secret is not None
        else None
    )
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Cron-tick endpoint is disabled (CRON_SHARED_SECRET not set).",
        )
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[len("bearer "):].strip()
    if not token or token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized.")


def _run_tick(now_utc: datetime) -> dict[str, Any]:
    """The actual scan — synchronous (every call it makes is itself sync:
    JournalStore in cloud mode, try_claim_fire, _fire_routine), run via
    asyncio.to_thread from the route so it never blocks the event loop.
    """
    from core.routine_cron_tick import compute_due_routines
    from core.routine_scheduler import _fire_routine, get_active_routines_for_user
    from core.storage.cloud.journal_store import list_user_ids_pg
    from core.storage.cloud.routine_last_fired_store import try_claim_fire

    users_checked = 0
    routines_fired = 0
    errors: list[str] = []

    for user_id in list_user_ids_pg():
        users_checked += 1
        try:
            routines = get_active_routines_for_user(user_id)
        except Exception as exc:
            errors.append(f"{user_id}: journal read failed: {exc}")
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
                errors.append(f"{user_id}/{routine_key}: fire failed: {exc}")

    return {
        "users_checked": users_checked,
        "routines_fired": routines_fired,
        "errors": errors,
        "ticked_at": now_utc.isoformat(),
    }


@router.post("/cron-tick")
async def cron_tick(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_auth(authorization)
    if not settings.is_cloud:
        raise HTTPException(
            status_code=503,
            detail="Cron-tick is a cloud-mode endpoint; local dev uses RoutineScheduler.",
        )
    now_utc = datetime.now(UTC)
    result = await asyncio.to_thread(_run_tick, now_utc)
    if result["errors"]:
        logger.warning("cron-tick completed with errors: %s", result["errors"])
    return result
