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


def _bucket_id(agent_or_model: Any) -> str:
    """Best-effort stable family identifier for an agent/model.

    pydantic-ai Agent exposes `.model`; pydantic-ai Model objects expose
    `.model_name` and a provider. We combine them into a string so equivalent
    provider/model rungs share a FAMILY bucket, used only for deterministic
    provider bugs that affect every key.
    """
    model = getattr(agent_or_model, "model", agent_or_model)
    name = getattr(model, "model_name", None) or getattr(model, "name", None)
    cls = model.__class__.__name__
    return f"{cls}:{name}" if name else cls


def _rung_id(agent_or_model: Any) -> str:
    """Per-rung identity for quota-scoped errors.

    The rung is the model object identity, which maps to one API key in the
    pools; one key's 429 must not bench its siblings.
    """
    model = getattr(agent_or_model, "model", agent_or_model)
    return f"{_bucket_id(agent_or_model)}#{id(model):x}"


def _cooling_key_active(key: str, now: float) -> bool:
    until = _cooldown_until.get(key)
    if until is None:
        return False
    if now >= until:
        _cooldown_until.pop(key, None)
        return False
    return True


def is_cooling(agent_or_model: Any) -> bool:
    rid = _rung_id(agent_or_model)
    bid = _bucket_id(agent_or_model)
    with _lock:
        now = time.monotonic()
        return _cooling_key_active(rid, now) or _cooling_key_active(bid, now)


def mark_failure(agent_or_model: Any, exc: Exception) -> None:
    """Mark a cooldown for the given agent based on the failure class."""
    seconds, scope = _cooldown_seconds(exc)
    if seconds <= 0:
        return
    mid = _bucket_id(agent_or_model) if scope == "bucket" else _rung_id(agent_or_model)
    with _lock:
        _cooldown_until[mid] = time.monotonic() + seconds
    print(f"LOG: health_tracker cooling {mid} for {seconds:.0f}s ({exc.__class__.__name__})")


def mark_success(agent_or_model: Any) -> None:
    rid = _rung_id(agent_or_model)
    bid = _bucket_id(agent_or_model)
    with _lock:
        _cooldown_until.pop(rid, None)
        _cooldown_until.pop(bid, None)


def _cooldown_seconds(exc: Exception) -> tuple[float, str]:
    """Return (seconds, scope): scope "rung" for quota-scoped transient errors
    (one key's limit must not bench sibling keys) and "bucket" for
    deterministic provider bugs that affect the whole model family."""
    try:
        from pydantic_ai.exceptions import ModelHTTPError
    except Exception:
        ModelHTTPError = None  # type: ignore

    if ModelHTTPError is not None and isinstance(exc, ModelHTTPError):
        status = getattr(exc, "status_code", None)
        # 413 = Groq TPM "request too large" (per-minute budget). Back off
        # briefly so the cascade prefers a higher-limit provider for ~1 min.
        if status in (413, 429, 402) or (isinstance(status, int) and status >= 500):
            return _COOLDOWN_TRANSIENT_S, "rung"
        if status == 400:
            message = str(exc).lower()
            body = getattr(exc, "body", None)
            if body is not None:
                message = f"{message} {str(body).lower()}"
            if "harmony" in message or "render tokens" in message or "tools should have a name" in message:
                return _COOLDOWN_DETERMINISTIC_S, "bucket"
            return 0.0, "rung"
        return 0.0, "rung"

    msg = str(exc).lower()
    if any(tok in msg for tok in ("connection", "timeout", "service unavailable", "reset", "eof")):
        return _COOLDOWN_TRANSIENT_S, "rung"
    return 0.0, "rung"


def snapshot() -> dict[str, float]:
    """Return current cooldown map (model_id -> seconds remaining). Debug only."""
    now = time.monotonic()
    with _lock:
        return {mid: max(0.0, until - now) for mid, until in _cooldown_until.items()}
