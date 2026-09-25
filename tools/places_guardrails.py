"""
tools/places_guardrails.py
---------------------------
Caching + per-user daily call cap for tools/places_tool.py (WP 1.G / S-7.9).

Two independent primitives, each with a local (in-process) and a cloud
(Redis) implementation, selected via core/storage/factory.py's
get_places_cache() / get_places_call_limiter() the same way every other
cloud/local seam in this codebase is selected:

* Cache — search results (find_place) cached 10 minutes, details
  (place_details) cached 1 hour. Keyed GLOBALLY (turtle:places_cache:v1:...),
  never per-user or per-tenant. The underlying place/route data returned by
  Google is not user-specific — two users asking "where is the nearest
  Starbucks" should get the same cached answer — so scoping the key to a
  user would only add cross-user cache misses for zero privacy benefit.
  What WOULD be sensitive is the fact that a given user searched for a given
  query; that association only ever exists in the caller's own request logs
  (this module never writes user_id into a cache key or value), not here.

* Call cap — a per-user daily ceiling on actual upstream Google calls,
  counted AFTER a cache hit has already been excluded (a cache hit costs
  Google nothing, so it doesn't count against the budget). Own key
  namespace: turtle:places_cap:v1:<user_id>. This is deliberately NOT the
  same limiter or keyspace as core/guardrails.py's WebSocketRateLimiter /
  core/storage/factory.py's get_ws_rate_limiter() (turtle:ws_rate:<id>),
  which budgets INBOUND CHAT MESSAGES on an hour+day sliding window sized
  for chat cadence. Calling that limiter a second time per places call would
  double-count one inbound message against the message budget while using
  the wrong window semantics for a billed-API-call budget — see the WP
  brief's recon note. A refused call raises PlacesCallCapExceeded; callers
  turn that into a ToolResult.rate_limited() (never an unhandled exception).

Redis-unavailable posture (cloud mode only — local mode has no Redis at
all): FAIL OPEN, logged. A cache GET/SET that can't reach Redis is treated
as a miss / a no-op write; a call-cap check that can't reach Redis lets the
call through uncounted. This is a deliberate departure from
tools/idempotency.py's fail-closed posture for duplicate-email sends: that
primitive guards against a real, hard-to-undo harm (sending the same email
twice), where refusing on ambiguity is the safer default. Here, failing
closed would take down the entire Places/Routes toolset (a read-only,
non-destructive capability) for every user during a Redis blip, in exchange
for a strictly softer guarantee — a best-effort perf cache and a soft cost
ceiling, not a correctness or safety invariant. That trade favours
availability.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any, Optional

from core.config import settings

logger = logging.getLogger(__name__)

_CACHE_KEY_PREFIX = "turtle:places_cache:v1"
_CAP_KEY_PREFIX = "turtle:places_cap:v1"

# A little over 24h, mirroring RedisWebSocketRateLimiter's own key TTL choice
# — bounds an abandoned user's sorted-set key lifetime in Redis.
_CAP_KEY_TTL_S = 90_000


def cache_key(tool: str, params: dict[str, Any]) -> str:
    """Deterministic, GLOBAL cache key for one Places/Routes call.

    Prefixed by tool name so find_place/place_details/get_directions keys
    can never collide with each other even if their param dicts happened to
    hash the same way, and prefixed by the shared turtle:places_cache:v1:
    namespace so this can never collide with any other Redis-backed feature
    in this codebase (ws-rate, channel-gate, idempotency all use their own
    turtle:<feature>: prefixes). No user/tenant identifier is ever part of
    this key — see the module docstring for why the cache is global.
    """
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{_CACHE_KEY_PREFIX}:{tool}:{digest}"


class PlacesCallCapExceeded(RuntimeError):
    """Raised when a user has exhausted their daily Places/Routes call cap."""

    def __init__(self, user_id: str, limit: int) -> None:
        self.user_id = user_id
        self.limit = limit
        super().__init__(
            f"Places daily call cap exceeded for user {user_id}: {limit}/day."
        )


# ---------------------------------------------------------------------------
# Local mode: in-process implementations (no Redis available)
# ---------------------------------------------------------------------------


class InProcessPlacesCache:
    """Local-mode cache substitute: a per-process dict with per-entry expiry.

    Lost on process restart and not shared across workers — acceptable for
    local/dev mode, which already runs single-process (SQLite/FAISS are the
    same story). Cloud mode gets the real, persistent Redis cache below.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= now:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        with self._lock:
            self._store[key] = (time.time() + ttl_seconds, value)


class InProcessPlacesCallLimiter:
    """Local-mode per-user daily call counter, in-process.

    Mirrors core/guardrails.py's WebSocketRateLimiter's day-window sliding
    logic exactly, but under its own dict/lock — local mode has no Redis to
    share a keyspace with in the first place, so the "don't collide with
    ws_rate" concern is moot here; it only matters for the cloud
    implementation below.
    """

    def __init__(self, *, per_day: Optional[int] = None) -> None:
        self.per_day = int(
            per_day if per_day is not None else settings.places_daily_call_cap
        )
        self._events: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check_and_record(self, user_id: str) -> None:
        if self.per_day <= 0:
            return
        uid = user_id or "anonymous"
        now = time.time()
        day_cutoff = now - 86400
        with self._lock:
            history = [t for t in self._events.get(uid, []) if t > day_cutoff]
            if len(history) >= self.per_day:
                self._events[uid] = history
                raise PlacesCallCapExceeded(uid, self.per_day)
            history.append(now)
            self._events[uid] = history


# ---------------------------------------------------------------------------
# Cloud mode: Redis-backed implementations
# ---------------------------------------------------------------------------


class RedisPlacesCache:
    """Cloud-mode cache backed by the shared sync Redis client.

    Sync, not async: matches the existing sync-Redis-on-the-loop-thread
    pattern this codebase already uses for guardrails/channel-gate/
    idempotency (core.storage.cloud.get_redis_sync_client), and
    tools/places_tool.py's call sites are not awaited through a job queue —
    they run directly in the request path like those callers do.
    """

    def get(self, key: str) -> Optional[Any]:
        try:
            from core.storage.cloud import get_redis_sync_client

            client = get_redis_sync_client()
            raw = client.get(key)
        except Exception as exc:  # fail open — see module docstring
            logger.warning("Places cache GET failed (%s); treating as a miss.", exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        try:
            from core.storage.cloud import get_redis_sync_client

            client = get_redis_sync_client()
            client.set(key, json.dumps(value), ex=ttl_seconds)
        except Exception as exc:  # fail open — see module docstring
            logger.warning("Places cache SET failed (%s); skipping cache write.", exc)


class RedisPlacesCallLimiter:
    """Cloud-mode per-user daily call cap, sliding 24h window via a Redis
    sorted set — same technique as RedisWebSocketRateLimiter, but under its
    OWN turtle:places_cap:v1: namespace so it never shares state (or
    double-counts) with that limiter's turtle:ws_rate: keys. See the module
    docstring for the full rationale.
    """

    def __init__(self, *, per_day: Optional[int] = None) -> None:
        self.per_day = int(
            per_day if per_day is not None else settings.places_daily_call_cap
        )

    def check_and_record(self, user_id: str) -> None:
        if self.per_day <= 0:
            return
        uid = user_id or "anonymous"
        try:
            from core.storage.cloud import get_redis_sync_client

            client = get_redis_sync_client()
            key = f"{_CAP_KEY_PREFIX}:{uid}"
            now = time.time()
            day_cutoff = now - 86400
            client.zremrangebyscore(key, "-inf", day_cutoff)
            count = client.zcard(key)
            if count >= self.per_day:
                raise PlacesCallCapExceeded(uid, self.per_day)
            member = f"{now}:{id(object())}"
            client.zadd(key, {member: now})
            client.expire(key, _CAP_KEY_TTL_S)
        except PlacesCallCapExceeded:
            raise
        except Exception as exc:  # fail open — see module docstring
            logger.warning(
                "Places call cap check failed (%s); failing open (call allowed).",
                exc,
            )
