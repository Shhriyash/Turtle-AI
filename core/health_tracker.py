"""
core/health_tracker.py
----------------------
Phase 1 / A2: Minimal process-local circuit breaker for model agents.

Tracks per-model cooldown timestamps so the fallback cascade can skip
recently-failed models instead of burning every key in the pool on a
provider outage. Cooldowns are deterministic per failure class:

  - transient (5xx, 429, 402)         -> 60s
  - deterministic provider bug (400)  -> 300s
  - success                            -> clears entry

State is in-process only (a dict). No persistence, no cross-worker sync.
That's enough for single-instance Turtle; the multi-instance story is
out of scope for Phase 1.
"""
from __future__ import annotations

import time
from threading import Lock
from typing import Any


_COOLDOWN_TRANSIENT_S = 60.0
_COOLDOWN_DETERMINISTIC_S = 300.0

_cooldown_until: dict[str, float] = {}
_lock = Lock()


def _model_id(agent_or_model: Any) -> str:
    """Best-effort stable identifier for an agent/model.

    pydantic-ai Agent exposes `.model`; pydantic-ai Model objects expose
    `.model_name` and a provider. We combine them into a string so two
    different keys for the same provider/model collapse to the same id
    (the cooldown applies to the *capacity bucket*, not the key).
    """
    model = getattr(agent_or_model, "model", agent_or_model)
    name = getattr(model, "model_name", None) or getattr(model, "name", None)
    cls = model.__class__.__name__
    return f"{cls}:{name}" if name else cls


def is_cooling(agent_or_model: Any) -> bool:
    mid = _model_id(agent_or_model)
    with _lock:
        until = _cooldown_until.get(mid)
        if until is None:
            return False
        if time.monotonic() >= until:
            _cooldown_until.pop(mid, None)
            return False
        return True


def mark_failure(agent_or_model: Any, exc: Exception) -> None:
    """Mark a cooldown for the given agent based on the failure class."""
    seconds = _cooldown_seconds(exc)
    if seconds <= 0:
        return
    mid = _model_id(agent_or_model)
    with _lock:
        _cooldown_until[mid] = time.monotonic() + seconds
    print(f"LOG: health_tracker cooling {mid} for {seconds:.0f}s ({exc.__class__.__name__})")


def mark_success(agent_or_model: Any) -> None:
    mid = _model_id(agent_or_model)
    with _lock:
        _cooldown_until.pop(mid, None)


def _cooldown_seconds(exc: Exception) -> float:
    try:
        from pydantic_ai.exceptions import ModelHTTPError
    except Exception:
        ModelHTTPError = None  # type: ignore

    if ModelHTTPError is not None and isinstance(exc, ModelHTTPError):
        status = getattr(exc, "status_code", None)
        if status in (429, 402) or (isinstance(status, int) and status >= 500):
            return _COOLDOWN_TRANSIENT_S
        if status == 400:
            message = str(exc).lower()
            if "harmony" in message or "render tokens" in message or "tools should have a name" in message:
                return _COOLDOWN_DETERMINISTIC_S
            return 0.0
        return 0.0

    msg = str(exc).lower()
    if any(tok in msg for tok in ("connection", "timeout", "service unavailable", "reset", "eof")):
        return _COOLDOWN_TRANSIENT_S
    return 0.0


def snapshot() -> dict[str, float]:
    """Return current cooldown map (model_id -> seconds remaining). Debug only."""
    now = time.monotonic()
    with _lock:
        return {mid: max(0.0, until - now) for mid, until in _cooldown_until.items()}
