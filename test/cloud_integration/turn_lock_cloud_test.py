"""
Real-Redis test for core/turn_lock.py (ledger 6.3).

This is the ONLY place the Lua compare-and-delete release ever executes: the
unit tests (test/p6_b1_turn_lock_test.py) cannot run Lua (no fakeredis+lupa, no
local server). Runs in the ``cloud-tests`` CI job; self-skips without
DATABASE_URL/REDIS_URL (see conftest.py).
"""
from __future__ import annotations

import time
import uuid

import pytest

from test.cloud_integration.conftest import run_async

pytestmark = pytest.mark.cloud


def _backend():
    from core.turn_lock import RedisTurnLockBackend

    return RedisTurnLockBackend()


async def _raw():
    from core.storage.cloud import get_redis_client

    return await get_redis_client()


def test_set_nx_then_compare_and_delete_release_roundtrip() -> None:
    from core.turn_lock import turn_lock_key

    uid = f"usr_{uuid.uuid4().hex[:12]}"
    key = turn_lock_key(uid)
    backend = _backend()

    async def body():
        client = await _raw()
        try:
            assert await backend.try_acquire(key, "tok-a", 30) is True
            assert await backend.try_acquire(key, "tok-b", 30) is False  # NX
            assert 0 < await client.ttl(key) <= 30
            # Wrong token: Lua must NOT delete.
            assert await backend.release(key, "tok-b") is False
            assert await client.get(key) == "tok-a"
            # Right token: deleted.
            assert await backend.release(key, "tok-a") is True
            assert await client.get(key) is None
            # Releasing again is a no-op, not an error.
            assert await backend.release(key, "tok-a") is False
        finally:
            await client.delete(key)

    run_async(body())


def test_stale_release_does_not_free_the_next_holder() -> None:
    """The exact hazard GET-then-DEL has: A's key expires, B takes it, A releases."""
    from core.turn_lock import turn_lock_key

    uid = f"usr_{uuid.uuid4().hex[:12]}"
    key = turn_lock_key(uid)
    backend = _backend()

    async def body():
        client = await _raw()
        try:
            assert await backend.try_acquire(key, "tok-a", 1) is True
            time.sleep(1.5)  # real TTL expiry
            assert await backend.try_acquire(key, "tok-b", 30) is True
            assert await backend.release(key, "tok-a") is False
            assert await client.get(key) == "tok-b"
        finally:
            await client.delete(key)

    run_async(body())


def test_two_turns_for_one_user_do_not_overlap() -> None:
    import asyncio

    from core.turn_lock import turn_lock

    uid = f"usr_{uuid.uuid4().hex[:12]}"
    active = 0
    max_active = 0
    busy = 0

    async def turn(name: str):
        nonlocal active, max_active, busy
        async with turn_lock(uid, name, wait_s=0.2) as lock:
            if lock.busy:
                busy += 1
                return
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.6)
            active -= 1

    async def body():
        await asyncio.gather(turn("a"), turn("b"), turn("c"))
        client = await _raw()
        assert await client.get(f"turtle:turn_lock:{uid}") is None

    run_async(body())
    assert max_active == 1
    assert busy == 2
