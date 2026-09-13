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
from core.storage.cloud import get_redis_sync_client


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

    def note_prompt(self, key: tuple[str, str], event_ids: tuple[str, ...], **_kwargs) -> None:
        if not event_ids:
            return
        client = get_redis_sync_client()
        client.set(self._redis_key(key), json.dumps(list(event_ids)), ex=self._ttl_seconds)

    def try_consume_answer(
        self, key: tuple[str, str], text: str, **_kwargs
    ) -> tuple[bool, tuple[str, ...]] | None:
        client = get_redis_sync_client()
        redis_key = self._redis_key(key)
        raw = client.get(redis_key)
        if raw is None:
            return None
        verdict = parse_gate_answer(text)
        if verdict is None:
            # Not a yes/no reply — leave the prompt outstanding (its Redis TTL
            # still governs expiry), matching the local class's behavior.
            return None
        client.delete(redis_key)
        try:
            event_ids = tuple(json.loads(raw))
        except Exception:
            event_ids = ()
        return verdict, event_ids

    def has_outstanding(self, key: tuple[str, str], **_kwargs) -> bool:
        client = get_redis_sync_client()
        return bool(client.exists(self._redis_key(key)))

    def clear(self, key: tuple[str, str]) -> None:
        client = get_redis_sync_client()
        client.delete(self._redis_key(key))


# --- Tool idempotency ---------------------------------------------------------

_IDEMPOTENCY_WINDOW_S = 60  # Matches tools/idempotency.py's local window.
_SUCCESS_PREFIX = "Email sent successfully"


def redis_is_duplicate_invocation(idempotency_key: str) -> Optional[str]:
    """Drop-in for tools.idempotency.is_duplicate_invocation."""
    try:
        client = get_redis_sync_client()
        raw = client.get(f"turtle:idem:{idempotency_key}")
        return raw if raw is not None else None
    except Exception as exc:
        print(f"LOG: Redis idempotency check failed ({exc}), treating as new invocation")
        return None


def redis_record_invocation(idempotency_key: str, result: str) -> None:
    """Drop-in for tools.idempotency.record_invocation. SETEX gives the same
    60s dedup horizon as the local SQLite version's created_at_s cutoff, with
    the expiry enforced by Redis itself instead of a WHERE clause."""
    if not str(result).startswith(_SUCCESS_PREFIX):
        return
    try:
        client = get_redis_sync_client()
        client.set(f"turtle:idem:{idempotency_key}", result, ex=_IDEMPOTENCY_WINDOW_S)
    except Exception as exc:
        print(f"LOG: Redis idempotency record failed ({exc}) — continuing without idempotency")
