"""
core/storage/cloud/redis_backends.py
--------------------------------------
Cloud (TURTLE_DEPLOY=cloud) Redis-backed replacements for three in-process
primitives that do not survive serverless (each is a per-process dict/deque
or a local SQLite file — gone on the next cold start, and even on today's
single-VM deploy already documented as "swap for Redis" / "in-memory,
best-effort"):

- RedisWebSocketRateLimiter -> core/guardrails.py::WebSocketRateLimiter
- RedisChannelGateBuffer    -> core/channel_gate.py::ChannelGateBuffer
- Redis idempotency helpers -> tools/idempotency.py's SQLite-backed functions

All three match their local counterparts' call sites EXACTLY (same method
names/signatures, same exception types raised), and all three are
SYNCHRONOUS: every existing call site invokes them directly with no `await`
(apps/turtle_server.py:2253/2259/3002/3056/3769), so like PgChunkVectorStore
these are built on the sync Redis client (core.storage.cloud.get_redis_sync_client),
not the async one.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from core.channel_gate import DEFAULT_TTL_SECONDS, parse_gate_answer
from core.guardrails import WebSocketRateLimitExceeded
from core.storage.cloud import CloudBackendUnavailable, get_redis_sync_client

# Ledger 3.6(a)/(b): the errors that must fail OPEN here (unlike idempotency's
# deliberate fail-CLOSED above -- see this module's docstring intro / the
# governing-principle note in the WP brief). A live Redis outage raises
# redis-py's own driver errors (redis.exceptions.RedisError and everything
# under it, e.g. ConnectionError/TimeoutError) -- NOT CloudBackendUnavailable,
# which get_redis_sync_client only raises for an unset REDIS_URL. Both are
# caught here: an unset URL is a config problem, a live outage is a runtime
# blip, but in neither case should refusing to enforce a soft limit cost a
# user their entire WebSocket connection.
try:
    import redis as _redis_module  # local import: optional dep, mirrors get_redis_sync_client

    _REDIS_FAIL_OPEN_ERRORS: tuple[type[BaseException], ...] = (
        _redis_module.RedisError,
        CloudBackendUnavailable,
    )
except Exception:  # pragma: no cover - redis package itself unavailable
    _REDIS_FAIL_OPEN_ERRORS = (CloudBackendUnavailable,)

# Ledger 3.6(a): in-process counter, incremented every time the rate limiter
# degrades (fails open) because Redis could not be reached. Same "no metrics
# module yet" caveat as routine_outbox_store.get_outbox_failure_counts --
# this does not reach a dashboard until Phase 4.
_RATE_LIMITER_METRICS = {"rate_limiter_degraded": 0}


def get_rate_limiter_degraded_count() -> int:
    """Snapshot of this process's rate_limiter_degraded counter."""
    return _RATE_LIMITER_METRICS["rate_limiter_degraded"]


# Ledger 3.6(b): same counter shape for the channel gate's fail-open path.
_CHANNEL_GATE_METRICS = {"channel_gate_degraded": 0}


def get_channel_gate_degraded_count() -> int:
    """Snapshot of this process's channel_gate_degraded counter."""
    return _CHANNEL_GATE_METRICS["channel_gate_degraded"]


# --- Rate limiter -----------------------------------------------------------

class RedisWebSocketRateLimiter:
    """Drop-in for WebSocketRateLimiter.check_and_record, sliding-window via a
    Redis sorted set per user (score = event timestamp, member = a unique
    per-event token so two events in the same millisecond don't collide and
    silently undercount). Pruned to the day window on every call; the hour
    count is a ZCOUNT over the same set, mirroring the local implementation's
    "filter to day, then count within hour" logic exactly.
    """

    def __init__(self, *, per_hour: int | None = None, per_day: int | None = None) -> None:
        from core.config import settings

        self.per_hour = int(per_hour if per_hour is not None else settings.ws_messages_per_hour)
        self.per_day = int(per_day if per_day is not None else settings.ws_messages_per_day)

    def check_and_record(self, user_id: str) -> None:
        if not user_id:
            return
        try:
            client = get_redis_sync_client()
            key = f"turtle:ws_rate:{user_id}"
            now = time.time()
            day_cutoff = now - 86400
            hour_cutoff = now - 3600

            client.zremrangebyscore(key, "-inf", day_cutoff)

            if self.per_day > 0:
                day_count = client.zcard(key)
                if day_count >= self.per_day:
                    raise WebSocketRateLimitExceeded(user_id, "day", self.per_day)
            if self.per_hour > 0:
                hour_count = client.zcount(key, hour_cutoff, "+inf")
                if hour_count >= self.per_hour:
                    raise WebSocketRateLimitExceeded(user_id, "hour", self.per_hour)

            # Unique member per event: two messages in the same wall-clock instant
            # must both count, not collide into one sorted-set entry.
            member = f"{now}:{id(object())}"
            client.zadd(key, {member: now})
            # Bound the key's own lifetime so an abandoned user's entry doesn't
            # linger in Redis forever once past the day window.
            client.expire(key, 90000)  # a little over 24h
        except WebSocketRateLimitExceeded:
            # A real, enforced limit -- never swallow this one.
            raise
        except _REDIS_FAIL_OPEN_ERRORS as exc:
            # Ledger 3.6(a): previously had NO error handling at all, so a
            # live Redis outage raised redis-py's own ConnectionError/
            # TimeoutError uncaught -- both call sites only catch
            # WebSocketRateLimitExceeded, so the exception reached the outer
            # handler and killed the entire WebSocket connection over an
            # unenforced (but recoverable) rate limit. Fail OPEN instead:
            # a rate limit is a bounded-abuse guard, not an irreversible
            # side effect (contrast tools/idempotency.py's deliberate
            # fail-CLOSED above, which guards a real duplicate-send).
            _RATE_LIMITER_METRICS["rate_limiter_degraded"] += 1
            print(f"LOG: rate limiter degraded (Redis unreachable): {exc}")


# --- Channel gate buffer -----------------------------------------------------

class RedisChannelGateBuffer:
    """Drop-in for ChannelGateBuffer: note_prompt/try_consume_answer/
    has_outstanding/clear, backed by a Redis string per (user_id, channel)
    with a native TTL (SETEX) instead of the local class's manual expiry
    bookkeeping. No max_entries cap needed — Redis's own TTL bounds every
    key's lifetime, so there is nothing to evict.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _redis_key(key: tuple[str, str]) -> str:
        user_id, channel = key
        return f"turtle:gate:{user_id}:{channel}"

    @staticmethod
    def _degrade(exc: BaseException, action: str) -> None:
        # Ledger 3.6(b): same fail-open posture as the rate limiter -- a
        # channel-gate prompt going unanswered because Redis is unreachable
        # is recoverable (the user just doesn't get the yes/no shortcut this
        # turn); refusing the whole channel turn over it is strictly worse.
        _CHANNEL_GATE_METRICS["channel_gate_degraded"] += 1
        print(f"LOG: channel gate degraded (Redis unreachable) during {action}: {exc}")

    def note_prompt(self, key: tuple[str, str], event_ids: tuple[str, ...], **_kwargs) -> None:
        if not event_ids:
            return
        try:
            client = get_redis_sync_client()
            client.set(self._redis_key(key), json.dumps(list(event_ids)), ex=self._ttl_seconds)
        except _REDIS_FAIL_OPEN_ERRORS as exc:
            # Best-effort: the prompt still reaches the user in chat text even
            # if we fail to remember it was asked, it just can't be answered
            # via the yes/no shortcut this turn.
            self._degrade(exc, "note_prompt")

    def try_consume_answer(
        self, key: tuple[str, str], text: str, **_kwargs
    ) -> tuple[bool, tuple[str, ...]] | None:
        try:
            client = get_redis_sync_client()
            redis_key = self._redis_key(key)
            raw = client.get(redis_key)
        except _REDIS_FAIL_OPEN_ERRORS as exc:
            # Fail open to "no pending prompt" -- the caller falls through to
            # treating this message as ordinary chat text instead of an
            # unanswerable gate reply.
            self._degrade(exc, "try_consume_answer")
            return None
        if raw is None:
            return None
        verdict = parse_gate_answer(text)
        if verdict is None:
            # Not a yes/no reply — leave the prompt outstanding (its Redis TTL
            # still governs expiry), matching the local class's behavior.
            return None
        try:
            client.delete(redis_key)
        except _REDIS_FAIL_OPEN_ERRORS as exc:
            self._degrade(exc, "try_consume_answer.delete")
        try:
            event_ids = tuple(json.loads(raw))
        except Exception:
            event_ids = ()
        return verdict, event_ids

    def has_outstanding(self, key: tuple[str, str], **_kwargs) -> bool:
        try:
            client = get_redis_sync_client()
            return bool(client.exists(self._redis_key(key)))
        except _REDIS_FAIL_OPEN_ERRORS as exc:
            self._degrade(exc, "has_outstanding")
            return False

    def clear(self, key: tuple[str, str]) -> None:
        try:
            client = get_redis_sync_client()
            client.delete(self._redis_key(key))
        except _REDIS_FAIL_OPEN_ERRORS as exc:
            self._degrade(exc, "clear")


# --- Tool idempotency ---------------------------------------------------------

_IDEMPOTENCY_WINDOW_S = 60  # Matches tools/idempotency.py's local window.
_SUCCESS_PREFIX = "Email sent successfully"

# Sentinel written by the reservation itself, before the SMTP call has
# returned. Keep this message in sync with tools/idempotency.py's local-mode
# equivalent — both are surfaced verbatim to the model/user.
_PENDING_SENTINEL = "__pending__"
_PENDING_MESSAGE = (
    "An identical email is already being sent (it started moments ago). "
    "Please wait a few seconds before trying again to avoid sending it twice."
)


class IdempotencyReservationError(RuntimeError):
    """Raised when Redis could not be reached to take/verify a reservation.

    Fail-closed: callers MUST refuse the send. This is a deliberate flip
    from the prior behaviour, which caught the exception and treated the
    invocation as new (fail-open) — meaning a Redis blip used to risk a
    duplicate send; now it costs the user a refused send instead.
    """


def redis_is_duplicate_invocation(idempotency_key: str) -> Optional[str]:
    """Reservation-based dedup check, drop-in for tools.idempotency.is_duplicate_invocation.

    Atomically claims `idempotency_key` via SET ... NX EX 60 with a pending
    sentinel BEFORE the caller sends anything (closing the race where two
    concurrent identical sends both saw "not yet recorded" and both fired).
    Returns None when the reservation is acquired, a "still sending" message
    when another send for this key is mid-flight, or the cached completed
    result when a prior send already finished within the window.

    Raises IdempotencyReservationError if Redis is unreachable.
    """
    key = f"turtle:idem:{idempotency_key}"
    try:
        client = get_redis_sync_client()
        acquired = client.set(key, _PENDING_SENTINEL, nx=True, ex=_IDEMPOTENCY_WINDOW_S)
        if acquired:
            return None
        raw = client.get(key)
        if raw is None:
            # Raced with a concurrent finalize/expiry between the failed SET
            # and this GET: the key is gone, so retry the reservation once
            # rather than either sending blind or refusing a legitimate new
            # send (mirrors the local SQLite path's equivalent race).
            acquired = client.set(key, _PENDING_SENTINEL, nx=True, ex=_IDEMPOTENCY_WINDOW_S)
            return None if acquired else _PENDING_MESSAGE
        if raw == _PENDING_SENTINEL:
            return _PENDING_MESSAGE
        return raw
    except Exception as exc:
        print(f"LOG: Redis idempotency reservation failed ({exc}) — refusing send (fail closed)")
        raise IdempotencyReservationError(str(exc)) from exc


def redis_record_invocation(idempotency_key: str, result: str, *, success: bool | None = None) -> None:
    """Finalize a reservation taken by redis_is_duplicate_invocation: overwrite
    with the completed result on success, or DELETE it on failure so the
    user's retry is not blocked by a failed send. Drop-in for
    tools.idempotency.record_invocation.

    `success` is generalised the same way as tools.idempotency.record_invocation:
    pass it explicitly for non-email callers (e.g. calendar_confirm), whose
    result string never starts with _SUCCESS_PREFIX ("Email sent
    successfully") and would otherwise always be treated as a failure —
    which would silently never cache a successful calendar confirm, making
    the reservation useless for it. When omitted, falls back to the
    original email-only string sniff so the existing email call site keeps
    its exact prior behaviour.
    """
    if success is None:
        success = str(result).startswith(_SUCCESS_PREFIX)
    key = f"turtle:idem:{idempotency_key}"
    try:
        client = get_redis_sync_client()
        if success:
            client.set(key, result, ex=_IDEMPOTENCY_WINDOW_S)
        else:
            client.delete(key)
    except Exception as exc:
        print(f"LOG: Redis idempotency finalize failed ({exc}) — reservation may linger until its TTL")
