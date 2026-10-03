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

SESSION LEASE (ledger 6.4) lives here too, on the same primitive and the same
two backends: ``SessionLease`` claims ``turtle:session_lease:{session_id}``
(``SET NX EX 90``), is refreshed by the connection's 30 s ping, and is released
by compare-and-delete. It exists because two connections that resume the same
session each write their whole in-memory history over the other's (Postgres
``put`` is a full overwrite), and one tab's disconnect finalises -- and
compacts -- the session under the other. See ``SessionLease`` for the one
addition the ledger's text lacks (identity-based takeover, so a client
reconnecting after an unclean drop is not locked out by its own stale lease).
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


# Session lease (ledger 6.4).
SESSION_LEASE_KEY_PREFIX = "turtle:session_lease:"
SESSION_LEASE_TTL_S = 90
# The client pings every 30 s; the lease TTL is three missed pings.

# Refresh only if the key still holds OUR token (an unconditional EXPIRE would
# extend somebody else's lease). Same compare-and-act shape as RELEASE_LUA.
REFRESH_LUA = (
    'if redis.call("get", KEYS[1]) == ARGV[1] then\n'
    '    return redis.call("expire", KEYS[1], ARGV[2])\n'
    "else\n"
    "    return 0\n"
    "end"
)

# Take over a lease held by a PREVIOUS connection of the same client identity.
# Values are "<identity>:<uuid>", so the identity prefix (ARGV[1], including the
# trailing colon) is compared atomically with the overwrite: no window in which
# another client's freshly-won lease could be replaced.
TAKEOVER_LUA = (
    'local cur = redis.call("get", KEYS[1])\n'
    "if cur and string.sub(cur, 1, string.len(ARGV[1])) == ARGV[1] then\n"
    '    redis.call("set", KEYS[1], ARGV[2], "EX", ARGV[3])\n'
    "    return 1\n"
    "end\n"
    "return 0"
)


def turn_lock_key(user_id: str) -> str:
    return f"{TURN_LOCK_KEY_PREFIX}{user_id}"


def session_lease_key(session_id: str) -> str:
    return f"{SESSION_LEASE_KEY_PREFIX}{session_id}"


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

    # -- session-lease operations (ledger 6.4) ------------------------------
    def _live(self, key: str) -> tuple[str, float] | None:
        cur = self._held.get(key)
        if cur is not None and cur[1] > time.monotonic():
            return cur
        return None

    async def holder(self, key: str) -> str | None:
        cur = self._live(key)
        return cur[0] if cur else None

    async def refresh(self, key: str, token: str, ttl_s: int) -> bool:
        cur = self._live(key)
        if cur is not None and cur[0] == token:
            self._held[key] = (token, time.monotonic() + ttl_s)
            return True
        return False

    async def takeover(self, key: str, prefix: str, token: str, ttl_s: int) -> bool:
        cur = self._live(key)
        if cur is not None and cur[0].startswith(prefix):
            self._held[key] = (token, time.monotonic() + ttl_s)
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

    # -- session-lease operations (ledger 6.4) ------------------------------
    async def holder(self, key: str) -> str | None:
        client = await self._client()
        value = await client.get(key)
        if value is None:
            return None
        return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)

    async def refresh(self, key: str, token: str, ttl_s: int) -> bool:
        client = await self._client()
        return bool(await client.eval(REFRESH_LUA, 1, key, token, ttl_s))

    async def takeover(self, key: str, prefix: str, token: str, ttl_s: int) -> bool:
        client = await self._client()
        return bool(await client.eval(TAKEOVER_LUA, 1, key, prefix, token, ttl_s))


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


# ---------------------------------------------------------------------------
# Session lease (ledger 6.4)
# ---------------------------------------------------------------------------
class SessionLease:
    """One WebSocket connection's claim on a session.

    ``SET turtle:session_lease:{session_id} <identity>:<uuid> NX EX 90``,
    refreshed by the 30 s ping, released by compare-and-delete. A second
    connection that finds the lease held opens a NEW session instead of
    resuming (and does not demote, sweep or compact the holder's); compaction
    and finalisation run only for the holder.

    Addition to the ledger's text -- IDENTITY. The ledger's bare NX has a
    hole: after an unclean drop (Vercel's 300 s cut, a dead network) the dead
    connection's lease lives for up to 90 s while the client retries every
    1-30 s, so the reconnecting client would find its OWN stale lease held and
    open a new session on every cut. The lease value therefore carries the
    client's identity (a per-tab id the client sends as ``?cid=``), and a
    claim by the same identity takes the lease over atomically (TAKEOVER_LUA).
    A different client (another tab, another device) is still refused. A
    connection with no client id uses its own connection id, which can never
    match a previous connection -- exactly the ledger's behaviour.

    Posture: FAIL OPEN, like the turn lock. If the backend errors the claim
    succeeds as ``degraded`` and the connection behaves as it did before the
    lease existed; a Redis blip must not take sessions down.
    """

    def __init__(
        self,
        identity: str,
        *,
        ttl_s: int = SESSION_LEASE_TTL_S,
        backend: Any = None,
    ) -> None:
        self.identity = identity
        self.token = new_token(identity)
        self._prefix = f"{identity}:"
        self.ttl_s = ttl_s
        self._backend = backend
        self.session_id: str | None = None
        self.held = False
        self.degraded = False
        self.lost = False

    def _be(self) -> Any:
        if self._backend is None:
            self._backend = get_turn_lock_backend()
        return self._backend

    @property
    def is_holder(self) -> bool:
        """True when this connection may write/finalise its session."""
        return (self.held or self.degraded) and not self.lost

    async def claim(self, session_id: str, *, takeover: bool = True) -> bool:
        """Claim ``session_id``. False means another client holds it.

        ``takeover`` lets a new connection of the SAME client identity replace
        its predecessor's lease. ``ensure`` passes False: a connection that has
        been superseded must not steal the lease back."""
        key = session_lease_key(session_id)
        backend = self._be()
        try:
            got = await asyncio.wait_for(
                backend.try_acquire(key, self.token, self.ttl_s),
                timeout=_REDIS_OP_TIMEOUT_S,
            )
            if not got and takeover:
                got = await asyncio.wait_for(
                    backend.takeover(key, self._prefix, self.token, self.ttl_s),
                    timeout=_REDIS_OP_TIMEOUT_S,
                )
        except Exception as exc:  # noqa: BLE001 - fail open on ANY backend error
            print(f"LOG: session lease unavailable, failing OPEN: {type(exc).__name__}: {exc}")
            self.degraded = True
            self.session_id = session_id
            return True
        if got:
            self.held = True
            self.lost = False
            self.session_id = session_id
        return bool(got)

    async def held_by_other(self, session_id: str) -> bool:
        """Read-only probe: is ``session_id`` leased by a DIFFERENT client?"""
        try:
            value = await asyncio.wait_for(
                self._be().holder(session_lease_key(session_id)),
                timeout=_REDIS_OP_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 - fail open
            print(f"LOG: session lease probe failed, assuming free: {type(exc).__name__}: {exc}")
            return False
        return value is not None and not value.startswith(self._prefix)

    async def ensure(self) -> bool:
        """Refresh (ping / turn boundaries). If the key vanished -- expiry
        during a stall, a Redis eviction -- try to re-claim it; if somebody
        else has it by now the lease is LOST and the caller must stop writing.
        Returns ``is_holder``."""
        if self.session_id is None or self.lost:
            return self.is_holder
        if not self.held:
            return self.is_holder  # degraded: nothing to refresh
        try:
            ok = await asyncio.wait_for(
                self._be().refresh(
                    session_lease_key(self.session_id), self.token, self.ttl_s
                ),
                timeout=_REDIS_OP_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 - fail open: keep holding
            print(f"LOG: session lease refresh failed (TTL decides): {type(exc).__name__}: {exc}")
            return True
        if ok:
            return True
        # Not ours any more.
        self.held = False
        if await self.claim(self.session_id, takeover=False):
            return True
        self.lost = True
        print(f"LOG: session lease lost for {self.session_id}")
        return False

    async def release(self) -> None:
        if not self.held or self.session_id is None:
            return
        self.held = False
        try:
            await asyncio.wait_for(
                self._be().release(session_lease_key(self.session_id), self.token),
                timeout=_REDIS_OP_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"LOG: session lease release failed (TTL will reclaim): {type(exc).__name__}: {exc}")


BUSY_MESSAGE = (
    "Turtle is still working on your previous message. "
    "Please try again in a moment."
)
