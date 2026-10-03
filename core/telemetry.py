"""
core/telemetry.py
-----------------
Phase 7 — minimal funnel telemetry.

Wraps logfire so callers don't have to guard for missing instrumentation.
Each event is also printed to stdout when logfire is unavailable, so local
dev still has a paper trail.

Funnel events the plan calls out:
    onboarding_start
    onboarding_complete
    first_message_sent
    memory_first_confirmed
"""
from __future__ import annotations

from typing import Any

try:
    import logfire as _logfire  # type: ignore
except Exception:  # pragma: no cover
    _logfire = None


def emit(event: str, **fields: Any) -> None:
    """Record a single funnel event. Never raises."""
    payload = {"event": event, **fields}
    if _logfire is not None:
        try:
            _logfire.info("turtle.funnel." + event, **payload)
            return
        except Exception:
            pass
    try:
        print(f"LOG: telemetry {payload}")
    except Exception:
        pass


def emit_once(user_id: str, event: str, **fields: Any) -> bool:
    """Emit ``event`` for ``user_id`` exactly once across process restarts.

    WP2.C (ledger 2.6): local mode keeps the sentinel-file dedup (below) --
    it's a single long-lived process with a real, persistent filesystem, so a
    plain marker file is the simplest thing that works and needs no new
    dependency. Cloud mode is serverless with a filesystem that does not
    persist across cold starts, so a sentinel file there re-emits every cold
    start; it's backed instead by
    core/storage/cloud/telemetry_claim_store.py's try_claim_once, an atomic
    ``INSERT ... ON CONFLICT DO NOTHING`` claim against Postgres (same
    pattern as routine_last_fired_store.try_claim_fire) -- the check and the
    claim happen as one statement, so two concurrent callers for the same
    (user_id, event) can only ever have one winner.

    Returns True if the event was emitted, False if it had already fired.
    """
    if not user_id:
        return False

    from core.config import settings  # avoid circular import at module load

    if settings.is_cloud:
        try:
            from core.storage.cloud.telemetry_claim_store import try_claim_once

            if not try_claim_once(user_id, event):
                return False
        except Exception:
            # Claim store unavailable (DATABASE_URL unset, connection
            # failure, table not yet reachable, ...): same posture as the
            # local sentinel-file fallback below -- emit anyway rather than
            # silently dropping the funnel event. For analytics-grade data a
            # rare duplicate emission is cheaper than a rare gap, and this
            # mirrors the behaviour this function already had before WP2.C.
            pass
    else:
        try:
            from core.paths import personal_memory_dir  # avoid circular import at module load

            marker_dir = personal_memory_dir(user_id) / ".telemetry"
            marker_dir.mkdir(parents=True, exist_ok=True)
            marker = marker_dir / event
            # Exclusive create ("x"): raises FileExistsError if the marker is
            # already there, so the check-and-claim is one atomic filesystem
            # operation instead of the previous exists()-then-write_text()
            # check-then-act, which two concurrent callers could both pass.
            try:
                marker.open("x", encoding="utf-8").close()
            except FileExistsError:
                return False
        except Exception:
            # If we can't touch the sentinel, fall back to emit-every-time.
            pass

    emit(event, user_id=user_id, **fields)
    return True
