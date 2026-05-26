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

    Idempotency is backed by a sentinel file under the user's memory dir:
    ``data/memory/personal/{user_id}/.telemetry/{event}``. Returns True if
    the event was emitted, False if it had already fired.
    """
    if not user_id:
        return False
    try:
        from core.paths import personal_memory_dir  # avoid circular import at module load

        marker_dir = personal_memory_dir(user_id) / ".telemetry"
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker = marker_dir / event
        if marker.exists():
            return False
        marker.write_text("", encoding="utf-8")
    except Exception:
        # If we can't write the sentinel, fall back to emit-every-time.
        pass
    emit(event, user_id=user_id, **fields)
    return True
