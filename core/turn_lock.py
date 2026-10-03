"""Per-user turn serialisation (ledger 6.3).

One user's turns -- web text, web voice, streamed mic, and every channel --
must never interleave: two concurrent turns resume the same session and the
last writer silently drops the other's messages.

The lock is ``SET turtle:turn_lock:{user_id} <token> NX EX 120``, released with
the compare-and-delete Lua script from the Redis ``SET`` documentation. A
GET-then-DEL release would be check-then-act against shared state: if the key
expired and another turn took it between the two commands, the DEL would free
someone else's lock.

Contention policy (decided in the P6-B1 brief, not in the ledger):

  * a contending turn waits a short, BOUNDED time (``wait_s``) for the holder;
  * if the lock is still held it is reported ``busy`` and the CALLER rejects
    that turn with an explicit message -- never a silent drop, never an
    unbounded block of the receive loop;
  * FAIL OPEN: if Redis is unreachable, slow or erroring the turn proceeds
    without the lock (``degraded``). This is a serialisation hint, not a
    security control; making it fail closed would let a Redis blip take the
    whole product down. Same posture and reasoning as
    apps/channels/discord.py::_claim_interaction_id.

Local mode has no Redis: an in-process table gives the same semantics for the
single-process case, so a web tab and a channel turn for one user still do not
interleave there.

Usage::

    async with turn_lock(user_id, "web") as lock:
        if lock.busy:
            ...reject the turn...
        else:
            ...run it...

The release runs in ``__aexit__``, i.e. a ``finally`` -- it executes on
``asyncio.CancelledError`` (a BaseException that escapes ``except Exception``),
so an interrupted turn does not leak the key for its whole TTL.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from core.config import settings

TURN_LOCK_KEY_PREFIX = "turtle:turn_lock:"
TURN_LOCK_TTL_S = 120
# How long a contending turn waits for the holder before being rejected. Kept
# short on purpose: the receive loop must stay responsive.
TURN_LOCK_WAIT_S = 2.0
_POLL_INTERVAL_S = 0.1
# Hard bound on one Redis round trip. The shared client already carries 1s
# socket timeouts; this guards the client construction / lock-acquire path too.
_REDIS_OP_TIMEOUT_S = 2.0

# Compare-and-delete, verbatim from the Redis SET documentation
# (https://redis.io/docs/latest/commands/set/ "Patterns: the Redlock-lite
# single-instance lock"). Atomic on the server: the DEL only happens if the key
# still holds OUR token.
RELEASE_LUA = (
    'if redis.call("get", KEYS[1]) == ARGV[1] then\n'
    '    return redis.call("del", KEYS[1])\n'
    "else\n"
    "    return 0\n"
    "end"
)


def turn_lock_key(user_id: str) -> str:
    return f"{TURN_LOCK_KEY_PREFIX}{user_id}"


def new_token(owner: str) -> str:
    """A per-acquisition token. Unique even when one connection (``owner``)
    takes the lock twice, so a stale release can never free a newer holder."""
    return f"{owner}:{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Backends. Two operations only -- the policy lives in ``TurnLock`` so it is
# testable without a Redis server.
# ---------------------------------------------------------------------------
class InProcessTurnLockBackend:
    """Local mode: a dict with expiry, guarded by the single-threaded loop."""

    def __init__(self) -> None:
        self._held: dict[str, tuple[str, float]] = {}

    async def try_acquire(self, key: str, token: str, ttl_s: int) -> bool:
        now = time.monotonic()
        cur = self._held.get(key)
        if cur is not None and cur[1] > now:
            return False
        self._held[key] = (token, now + ttl_s)
        return True

    async def release(self, key: str, token: str) -> bool:
        cur = self._held.get(key)
        if cur is not None and cur[0] == token:
            del self._held[key]
            return True
        return False


class RedisTurnLockBackend:
    """Cloud mode: SET NX EX to acquire, Lua compare-and-delete to release."""

    async def _client(self) -> Any:
        from core.storage.cloud import get_redis_client

        return await get_redis_client()

    async def try_acquire(self, key: str, token: str, ttl_s: int) -> bool:
        client = await self._client()
        return bool(await client.set(key, token, nx=True, ex=ttl_s))

    async def release(self, key: str, token: str) -> bool:
        client = await self._client()
        deleted = await client.eval(RELEASE_LUA, 1, key, token)
        return bool(deleted)


_in_process_backend = InProcessTurnLockBackend()
_redis_backend = RedisTurnLockBackend()


def get_turn_lock_backend() -> Any:
    """Redis in cloud mode, in-process locally (same pattern as
    core/storage/factory.py::get_ws_rate_limiter)."""
    return _redis_backend if settings.is_cloud else _in_process_backend


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class TurnLock:
    """Async context manager. After ``__aenter__`` exactly one of:

    ``acquired`` -- we hold the lock;
    ``degraded`` -- the backend failed, we proceed WITHOUT it (fail open);
    ``busy``     -- another turn holds it past ``wait_s``; the caller must
                    reject this turn.
    """

    def __init__(
        self,
        user_id: str,
        owner: str,
        *,
        wait_s: float | None = None,
        ttl_s: int = TURN_LOCK_TTL_S,
        backend: Any = None,
    ) -> None:
        self.key = turn_lock_key(user_id)
        self.token = new_token(owner)
        self.owner = owner
        # Resolved at call time (not def time) so tests/ops can tune the module constant.
        self.wait_s = TURN_LOCK_WAIT_S if wait_s is None else wait_s
        self.ttl_s = ttl_s
        self._backend = backend
        self.acquired = False
        self.degraded = False
        self.busy = False

    async def __aenter__(self) -> "TurnLock":
        backend = self._backend if self._backend is not None else get_turn_lock_backend()
        self._backend = backend
        deadline = time.monotonic() + self.wait_s
        while True:
            try:
                got = await asyncio.wait_for(
                    backend.try_acquire(self.key, self.token, self.ttl_s),
                    timeout=_REDIS_OP_TIMEOUT_S,
                )
            except Exception as exc:  # noqa: BLE001 - fail open on ANY backend error
                # CancelledError is a BaseException and is deliberately not
                # caught here: a cancelled turn must stay cancelled.
                print(f"LOG: turn lock unavailable, failing OPEN: {type(exc).__name__}: {exc}")
                self.degraded = True
                return self
            if got:
                self.acquired = True
                return self
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.busy = True
                return self
            await asyncio.sleep(min(_POLL_INTERVAL_S, remaining))

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()

    async def release(self) -> None:
        if not self.acquired:
            return
        self.acquired = False
        try:
            await asyncio.wait_for(
                self._backend.release(self.key, self.token),
                timeout=_REDIS_OP_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001
            # The TTL reclaims the key; a failed release must not mask the
            # turn's own outcome.
            print(f"LOG: turn lock release failed (TTL will reclaim): {type(exc).__name__}: {exc}")


def turn_lock(user_id: str, owner: str, **kwargs: Any) -> TurnLock:
    return TurnLock(user_id, owner, **kwargs)


BUSY_MESSAGE = (
    "Turtle is still working on your previous message. "
    "Please try again in a moment."
)
