"""
core/health_tracker.py
----------------------
Phase 1 / A2: Minimal process-local circuit breaker for model agents.

Tracks per-model cooldown timestamps so the fallback cascade can skip
recently-failed models instead of burning every key in the pool on a
provider outage. Cooldowns are deterministic per failure class:

  - transient, per-key (5xx, 429, 413)      -> 60s (rung scope)
  - deterministic, provider-wide            -> 300s (bucket scope):
      * 402 credits exhausted (account state, all keys share it)
      * 400 tool-format rejection (gpt-oss harmony, Gemini tool-turn ordering)
  - success                                  -> clears entry

Local state is an in-process dict. In CLOUD mode the BUCKET-scope cooldowns
(402 credits exhausted, deterministic 400s -- the 300s ones) are additionally
mirrored to Redis as ``SET turtle:cooldown:{bucket_id} <deadline> EX <seconds>`` so a
cold serverless instance does not re-burn a provider a warm one already knows
is dead. ``time.monotonic()`` is NOT comparable across processes and is never
stored. The value is the absolute WALL-CLOCK deadline (``time.time()``, which is
comparable across processes); a reader converts it to a local monotonic expiry
so its view lasts as long as the real cooldown (it must outlive the first
rung's LLM call, because the in-loop is_cooling re-check runs after it). The
``EX`` TTL is the backstop: a key with a bad/skewed deadline still self-destructs.
Clock skew between instances is real but small (NTP-synced hosts, ms to low
seconds) and is bounded by that TTL; it makes the deadline approximate, not exact.

  - Each cascade calls refresh_shared()/refresh_shared_sync() ONCE, which does
    a single MGET over the distinct bucket ids of its agents and caches the
    answer for a few seconds; is_cooling() stays synchronous and local.
  - RUNG-scope (60s, per-key) cooldowns are deliberately NOT mirrored: the id
    embeds ``id(model)``, which is per-process, so a sibling instance could
    never match it, and a 60s transient costs a cold instance at most one
    failed call per key.
  - FAIL OPEN: any Redis error/timeout is swallowed and the cascade proceeds
    on local state (same posture as apps/channels/discord.py's interaction
    dedup: a cooldown hint is never worth taking the agent down for). After a
    failure Redis is not retried for _REDIS_BACKOFF_S.
  - Local mode has no Redis: every shared helper is a no-op.
"""
from __future__ import annotations

import asyncio
import time
from threading import Lock
from typing import Any


_COOLDOWN_TRANSIENT_S = 60.0
_COOLDOWN_DETERMINISTIC_S = 300.0

_cooldown_until: dict[str, float] = {}
_lock = Lock()

_SHARED_PREFIX = "turtle:cooldown:"
_SHARED_VIEW_S = 10.0       # fallback view when a key's remaining time is unknown (legacy "1" value)
_MIN_PLAUSIBLE_EPOCH = 1_000_000_000.0  # below this a value is not a wall-clock deadline
_REDIS_TIMEOUT_S = 1.5      # hard ceiling per Redis round trip (client has 1s socket timeout)
_REDIS_BACKOFF_S = 30.0     # skip Redis this long after a failure

# Local view of what Redis said: bucket id -> monotonic expiry of THIS VIEW
# (not of the cooldown; the Redis TTL owns that).
_shared_until: dict[str, float] = {}
_redis_down_until = 0.0
_pending: set[asyncio.Task] = set()


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


def _cooling_key_active(key: str, now: float, table: dict[str, float] | None = None) -> bool:
    table = _cooldown_until if table is None else table
    until = table.get(key)
    if until is None:
        return False
    if now >= until:
        table.pop(key, None)
        return False
    return True


def _shared_enabled() -> bool:
    """True only in cloud mode and outside the post-failure back-off window."""
    try:
        from core.config import settings

        if not settings.is_cloud:
            return False
    except Exception:
        return False
    return time.monotonic() >= _redis_down_until


def _note_redis_failure(what: str, exc: BaseException) -> None:
    global _redis_down_until
    _redis_down_until = time.monotonic() + _REDIS_BACKOFF_S
    print(f"LOG: health_tracker shared cooldown {what} unavailable, failing OPEN: "
          f"{exc.__class__.__name__}: {exc}")


def _view_seconds(val: Any) -> float:
    """Seconds this process should treat a present shared key as cooling.

    The value is an absolute wall-clock deadline. A key written by an older
    deploy holds the literal "1" (and anything unparseable/implausible is
    treated the same): cooling, remaining time unknown -> short fallback window
    rather than crashing or ignoring a key that demonstrably exists.
    """
    try:
        deadline = float(val)
    except (TypeError, ValueError):
        return _SHARED_VIEW_S
    if deadline != deadline or deadline < _MIN_PLAUSIBLE_EPOCH:  # NaN or legacy "1"
        return _SHARED_VIEW_S
    return max(0.0, deadline - time.time())


def _apply_mget(bids: list[str], values: list[Any]) -> None:
    now = time.monotonic()
    with _lock:
        for bid, val in zip(bids, values):
            if val is None:
                _shared_until.pop(bid, None)
            else:
                _shared_until[bid] = now + _view_seconds(val)


def _distinct_bucket_ids(agents: list[Any]) -> list[str]:
    return list(dict.fromkeys(_bucket_id(a) for a in agents))


async def refresh_shared(agents: list[Any]) -> None:
    """ONE MGET for the whole cascade (async runners). Never raises."""
    if not _shared_enabled():
        return
    bids = _distinct_bucket_ids(agents)
    if not bids:
        return
    try:
        from core.storage import cloud

        client = await asyncio.wait_for(cloud.get_redis_client(), _REDIS_TIMEOUT_S)
        values = await asyncio.wait_for(
            client.mget([_SHARED_PREFIX + b for b in bids]), _REDIS_TIMEOUT_S
        )
        _apply_mget(bids, list(values))
    except Exception as exc:  # CancelledError is BaseException: deliberately not caught
        _note_redis_failure("read", exc)


def refresh_shared_sync(agents: list[Any]) -> None:
    """Sync twin of refresh_shared for run_agent_sync_with_fallbacks (offline
    CLI only -- no event loop to freeze). Never raises."""
    if not _shared_enabled():
        return
    bids = _distinct_bucket_ids(agents)
    if not bids:
        return
    try:
        from core.storage import cloud

        values = cloud.get_redis_sync_client().mget([_SHARED_PREFIX + b for b in bids])
        _apply_mget(bids, list(values))
    except Exception as exc:
        _note_redis_failure("read", exc)


def _dispatch_write(async_op: Any, sync_op: Any) -> None:
    """mark_failure/mark_success are sync and called on the event loop. Never
    block it on Redis: schedule the async write as a task when a loop is
    running, else (offline CLI) use the sync client."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        sync_op()
        return
    task = loop.create_task(async_op())
    _pending.add(task)
    task.add_done_callback(_pending.discard)


def _mirror_set(bid: str, seconds: float) -> None:
    key = _SHARED_PREFIX + bid
    ex = max(1, int(seconds))
    deadline = repr(time.time() + seconds)  # wall clock; see module docstring

    async def _a() -> None:
        try:
            from core.storage import cloud

            client = await asyncio.wait_for(cloud.get_redis_client(), _REDIS_TIMEOUT_S)
            await asyncio.wait_for(client.set(key, deadline, ex=ex), _REDIS_TIMEOUT_S)
        except Exception as exc:
            _note_redis_failure("write", exc)

    def _s() -> None:
        try:
            from core.storage import cloud

            cloud.get_redis_sync_client().set(key, deadline, ex=ex)
        except Exception as exc:
            _note_redis_failure("write", exc)

    _dispatch_write(_a, _s)


def _mirror_delete(bid: str) -> None:
    key = _SHARED_PREFIX + bid

    async def _a() -> None:
        try:
            from core.storage import cloud

            client = await asyncio.wait_for(cloud.get_redis_client(), _REDIS_TIMEOUT_S)
            await asyncio.wait_for(client.delete(key), _REDIS_TIMEOUT_S)
        except Exception as exc:
            _note_redis_failure("write", exc)

    def _s() -> None:
        try:
            from core.storage import cloud

            cloud.get_redis_sync_client().delete(key)
        except Exception as exc:
            _note_redis_failure("write", exc)

    _dispatch_write(_a, _s)


def is_cooling(agent_or_model: Any) -> bool:
    rid = _rung_id(agent_or_model)
    bid = _bucket_id(agent_or_model)
    with _lock:
        now = time.monotonic()
        return (
            _cooling_key_active(rid, now)
            or _cooling_key_active(bid, now)
            or _cooling_key_active(bid, now, _shared_until)
        )


def mark_failure(agent_or_model: Any, exc: Exception) -> None:
    """Mark a cooldown for the given agent based on the failure class."""
    seconds, scope = _cooldown_seconds(exc)
    if seconds <= 0:
        return
    mid = _bucket_id(agent_or_model) if scope == "bucket" else _rung_id(agent_or_model)
    with _lock:
        _cooldown_until[mid] = time.monotonic() + seconds
    if scope == "bucket" and _shared_enabled():
        _mirror_set(mid, seconds)
    print(f"LOG: health_tracker cooling {mid} for {seconds:.0f}s ({exc.__class__.__name__})")


def mark_success(agent_or_model: Any) -> None:
    rid = _rung_id(agent_or_model)
    bid = _bucket_id(agent_or_model)
    with _lock:
        _cooldown_until.pop(rid, None)
        was_bucket_cooling = (
            _cooldown_until.pop(bid, None) is not None
            or _shared_until.pop(bid, None) is not None
        )
    # Only touch Redis when this success actually clears a known bucket
    # cooldown -- never a Redis round trip on the healthy hot path.
    if was_bucket_cooling and _shared_enabled():
        _mirror_delete(bid)


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
        # 413 = Groq TPM "request too large" (per-minute budget); 429 = rate
        # limit. Both are per-key and transient — back off briefly, per rung, so
        # the cascade prefers a higher-limit sibling/provider for ~1 min.
        if status in (413, 429) or (isinstance(status, int) and status >= 500):
            return _COOLDOWN_TRANSIENT_S, "rung"
        # 402 = payment required / credits exhausted. This is an ACCOUNT-level
        # state shared by every key of that provider and it will NOT clear in
        # 60s (it needs the operator to add credits). Cooling one rung for 60s
        # left the sibling keys 402ing on every turn, so each turn wasted a full
        # round-trip per key. Cool the whole family for the long window instead
        # so the cascade skips a known-broke provider until it's plausibly
        # topped up. (Observed live: OpenRouter 402 on all 3 keys every turn.)
        if status == 402:
            return _COOLDOWN_DETERMINISTIC_S, "bucket"
        if status == 400:
            message = str(exc).lower()
            body = getattr(exc, "body", None)
            if body is not None:
                message = f"{message} {str(body).lower()}"
            # Deterministic provider/tool-format rejections that recur identically
            # on the same model family. Includes gpt-oss "harmony" render errors
            # AND Gemini's strict tool-turn ordering ("function response turn
            # comes immediately after a function call turn" / INVALID_ARGUMENT),
            # which otherwise 400s on every tool-using turn and was never cooled —
            # so the cascade re-tried the dead Gemini rungs each turn (observed
            # live: ~14-36s TTFR while grinding through them to reach Groq).
            deterministic_400 = (
                "harmony" in message
                or "render tokens" in message
                or "tools should have a name" in message
                or "function response" in message
                or "immediately after a function call" in message
                or ("invalid_argument" in message and "function" in message)
            )
            if deterministic_400:
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
