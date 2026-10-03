"""
test/p6_b1_turn_lock_test.py
-----------------------------
Ledger 6.3: the per-user turn lock (core/turn_lock.py).

WHAT THIS FILE CANNOT PROVE. The release path is a Lua compare-and-delete run
by a Redis server. fakeredis here has no Lua (``unknown command 'eval'``) and no
Redis server is installed, so NOTHING in this file executes the Lua. The fake
client below only records the ``eval`` call shape (script text, numkeys, key,
token); the script's behaviour is covered by
test/cloud_integration/turn_lock_cloud_test.py against a real Redis, which is
the only place it ever runs.

What IS covered here is the decision logic around it: bounded wait, busy,
fail-open on the driver's real errors, release on cancellation, and that a
stale release cannot free a newer holder.
"""
from __future__ import annotations

import asyncio
import time

import pytest
import redis.exceptions

from core import turn_lock as tl
from core.tenant_purge import _redis_patterns_for_user


class FakeRedis:
    """Records calls. ``set`` has real NX semantics; ``eval`` only RECORDS."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.set_calls: list[tuple] = []
        self.eval_calls: list[tuple] = []

    async def set(self, key, value, nx=False, ex=None):
        self.set_calls.append((key, value, nx, ex))
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def eval(self, script, numkeys, *keys_and_args):
        self.eval_calls.append((script, numkeys, keys_and_args))
        key, token = keys_and_args
        if self.store.get(key) == token:  # emulation of the script, NOT the script
            del self.store[key]
            return 1
        return 0


@pytest.fixture
def redis_backend(monkeypatch):
    fake = FakeRedis()

    async def _get():
        return fake

    monkeypatch.setattr("core.storage.cloud.get_redis_client", _get)
    return tl.RedisTurnLockBackend(), fake


def _run(coro):
    return asyncio.run(coro)


def test_acquire_uses_set_nx_ex_and_release_uses_the_documented_lua(redis_backend) -> None:
    backend, fake = redis_backend

    async def body():
        async with tl.turn_lock("usr_x", "web:abc", backend=backend) as lock:
            assert lock.acquired
            key, token, nx, ex = fake.set_calls[0]
            assert key == "turtle:turn_lock:usr_x"
            assert token.startswith("web:abc:")
            assert nx is True and ex == 120
        script, numkeys, args = fake.eval_calls[0]
        assert script == tl.RELEASE_LUA
        assert numkeys == 1
        assert args == ("turtle:turn_lock:usr_x", token)
        assert fake.store == {}

    _run(body())


def test_release_lua_is_compare_and_delete() -> None:
    # Pins the script text against an accidental edit to a GET-then-DEL or an
    # unconditional DEL. (It does not execute it -- see the module docstring.)
    assert 'redis.call("get", KEYS[1]) == ARGV[1]' in tl.RELEASE_LUA
    assert 'redis.call("del", KEYS[1])' in tl.RELEASE_LUA


def test_contending_turn_is_busy_after_a_bounded_wait(redis_backend) -> None:
    backend, fake = redis_backend

    async def body():
        async with tl.turn_lock("usr_x", "a", backend=backend) as first:
            assert first.acquired
            t0 = time.monotonic()
            async with tl.turn_lock("usr_x", "b", wait_s=0.3, backend=backend) as second:
                elapsed = time.monotonic() - t0
                assert second.busy and not second.acquired and not second.degraded
                assert 0.25 <= elapsed < 1.5, f"wait was not bounded: {elapsed:.2f}s"
            # The loser's exit must NOT release the winner's lock.
            assert fake.store["turtle:turn_lock:usr_x"] == first.token

    _run(body())


def test_waiter_gets_the_lock_when_the_holder_releases_in_time(redis_backend) -> None:
    backend, _ = redis_backend

    async def body():
        holder = tl.turn_lock("usr_x", "a", backend=backend)
        await holder.__aenter__()

        async def release_soon():
            await asyncio.sleep(0.2)
            await holder.__aexit__(None, None, None)

        rel = asyncio.create_task(release_soon())
        async with tl.turn_lock("usr_x", "b", wait_s=2.0, backend=backend) as second:
            assert second.acquired
        await rel

    _run(body())


@pytest.mark.parametrize(
    "exc",
    [redis.exceptions.ConnectionError("refused"), redis.exceptions.TimeoutError("slow"),
     redis.exceptions.ResponseError("boom")],
)
def test_fails_open_on_the_real_driver_errors(redis_backend, exc) -> None:
    backend, fake = redis_backend

    async def boom(*a, **k):
        raise exc

    fake.set = boom

    async def body():
        async with tl.turn_lock("usr_x", "a", backend=backend) as lock:
            assert lock.degraded and not lock.busy and not lock.acquired
        assert fake.eval_calls == [], "nothing was acquired, so nothing may be released"

    _run(body())


def test_fails_open_when_redis_stalls(redis_backend, monkeypatch) -> None:
    backend, fake = redis_backend
    monkeypatch.setattr(tl, "_REDIS_OP_TIMEOUT_S", 0.2)

    async def hang(*a, **k):
        await asyncio.sleep(30)

    fake.set = hang

    async def body():
        t0 = time.monotonic()
        async with tl.turn_lock("usr_x", "a", backend=backend) as lock:
            assert lock.degraded
        assert time.monotonic() - t0 < 2.0

    _run(body())


def test_unset_redis_url_in_cloud_mode_fails_open(monkeypatch) -> None:
    """The other real failure: get_redis_client() raising CloudBackendUnavailable."""
    from core.config import settings
    from core.storage.cloud import CloudBackendUnavailable
    import core.storage.cloud as cloud

    async def unavailable():
        raise CloudBackendUnavailable("REDIS_URL is not set")

    monkeypatch.setattr(cloud, "get_redis_client", unavailable)

    async def body():
        async with tl.turn_lock("usr_x", "a", backend=tl.RedisTurnLockBackend()) as lock:
            assert lock.degraded

    _run(body())


def test_lock_is_released_when_the_holder_is_cancelled(redis_backend) -> None:
    """CancelledError is a BaseException: an `except Exception` release would leak."""
    backend, fake = redis_backend

    async def body():
        started = asyncio.Event()

        async def turn():
            async with tl.turn_lock("usr_x", "a", backend=backend):
                started.set()
                await asyncio.sleep(30)

        t = asyncio.create_task(turn())
        await started.wait()
        assert "turtle:turn_lock:usr_x" in fake.store
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert fake.store == {}, "a cancelled turn leaked its lock for the full TTL"

    _run(body())


def test_stale_release_cannot_free_a_newer_holder() -> None:
    backend = tl.InProcessTurnLockBackend()

    async def body():
        key = tl.turn_lock_key("usr_x")
        assert await backend.try_acquire(key, "old", 120)
        # "old" expires; a new turn takes the key.
        backend._held[key] = ("old", time.monotonic() - 1)
        assert await backend.try_acquire(key, "new", 120)
        assert await backend.release(key, "old") is False
        assert backend._held[key][0] == "new"
        assert await backend.release(key, "new") is True

    _run(body())


def test_local_mode_uses_the_in_process_backend_not_redis(monkeypatch) -> None:
    from core.config import settings

    monkeypatch.setattr(settings, "deploy_mode", "local")
    assert tl.get_turn_lock_backend() is tl._in_process_backend
    monkeypatch.setattr(settings, "deploy_mode", "cloud")
    assert tl.get_turn_lock_backend() is tl._redis_backend


def test_turn_lock_key_is_registered_for_tenant_purge() -> None:
    """Nothing else pins _redis_patterns_for_user: forget-me must delete the lock."""
    assert tl.turn_lock_key("usr_abc") in _redis_patterns_for_user("usr_abc")
