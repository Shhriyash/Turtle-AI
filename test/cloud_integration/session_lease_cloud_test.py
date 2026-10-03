"""
Real-Redis test for the session lease (ledger 6.4) and the turn-result buffer
(ledger 6.8).

This is the ONLY place REFRESH_LUA and TAKEOVER_LUA (and the lease's use of
RELEASE_LUA) ever run against a real Redis server: the unit tests
(test/p6_b2_lease_test.py) cannot run Lua without lupa and then only against
fakeredis' emulation. Runs in the ``cloud-tests`` CI job; self-skips without
DATABASE_URL/REDIS_URL (see conftest.py).
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from test.cloud_integration.conftest import run_async

pytestmark = pytest.mark.cloud


def _lease(identity: str, **kw):
    from core.turn_lock import RedisTurnLockBackend, SessionLease

    return SessionLease(identity, backend=RedisTurnLockBackend(), **kw)


async def _raw():
    from core.storage.cloud import get_redis_client

    return await get_redis_client()


def test_claim_is_nx_ex_90_and_a_second_client_is_refused() -> None:
    from core.turn_lock import session_lease_key

    sid = f"turtle_session_{uuid.uuid4().hex[:12]}"
    key = session_lease_key(sid)

    async def body():
        client = await _raw()
        try:
            a, b = _lease("tabA"), _lease("tabB")
            assert await a.claim(sid) is True
            assert 0 < await client.ttl(key) <= 90
            assert await client.get(key) == a.token
            assert await b.claim(sid) is False, "a different client must not take a held lease"
            assert await client.get(key) == a.token
            assert await b.held_by_other(sid) is True
            assert await a.held_by_other(sid) is False
        finally:
            await client.delete(key)

    run_async(body())


def test_same_client_takes_over_its_own_stale_lease_atomically() -> None:
    from core.turn_lock import session_lease_key

    sid = f"turtle_session_{uuid.uuid4().hex[:12]}"
    key = session_lease_key(sid)

    async def body():
        client = await _raw()
        try:
            old, new, other = _lease("tabA"), _lease("tabA"), _lease("tabB")
            assert await old.claim(sid)
            assert await new.claim(sid) is True, "TAKEOVER_LUA must accept the same identity"
            assert await client.get(key) == new.token
            assert 0 < await client.ttl(key) <= 90
            # the superseded connection cannot refresh, cannot steal it back,
            # and its release cannot free the new holder
            assert await old.ensure() is False and old.lost
            await old.release()
            assert await client.get(key) == new.token
            # an identity that merely SHARES A PREFIX must not match ("tabA" vs "tabAB")
            assert await _lease("tabAB").claim(sid) is False
            assert await other.claim(sid) is False
        finally:
            await client.delete(key)

    run_async(body())


def test_refresh_extends_only_our_own_lease_and_release_frees_it() -> None:
    from core.turn_lock import session_lease_key

    sid = f"turtle_session_{uuid.uuid4().hex[:12]}"
    key = session_lease_key(sid)

    async def body():
        client = await _raw()
        try:
            a = _lease("tabA", ttl_s=30)
            assert await a.claim(sid)
            await client.expire(key, 5)
            assert await a.ensure() is True
            assert await client.ttl(key) > 5, "refresh did not extend the TTL"
            await a.release()
            assert await client.get(key) is None
            # a vanished key is re-claimed rather than reported lost
            assert await a.claim(sid)
            await client.delete(key)
            assert await a.ensure() is True and a.held and not a.lost
            assert await client.get(key) == a.token
        finally:
            await client.delete(key)

    run_async(body())


def test_lease_expires_on_its_own() -> None:
    from core.turn_lock import session_lease_key

    sid = f"turtle_session_{uuid.uuid4().hex[:12]}"

    async def body():
        client = await _raw()
        try:
            a = _lease("tabA", ttl_s=1)
            assert await a.claim(sid)
            await asyncio.sleep(1.6)
            assert await _lease("tabB").claim(sid) is True
        finally:
            await client.delete(session_lease_key(sid))

    run_async(body())


def test_turn_result_buffer_roundtrip_and_ttl_on_real_redis() -> None:
    from core.session_store import TURN_RESULT_TTL_S, TurnResultBuffer, turn_result_key

    sid = f"turtle_session_{uuid.uuid4().hex[:12]}"

    async def body():
        client = await _raw()
        from core.session_store import _RedisResultBackend

        buf = TurnResultBuffer(backend=_RedisResultBackend())
        try:
            for n in (1, 2, 4):
                await buf.put(sid, f"{sid}_turn_{n}", {"n": n, "frame": {"content": f"a{n}"}})
            ttl = await client.ttl(turn_result_key(sid, f"{sid}_turn_1"))
            assert 0 < ttl <= TURN_RESULT_TTL_S == 120
            assert [r["n"] for r in await buf.after(sid, f"{sid}_turn_1")] == [2, 4]
            assert [r["n"] for r in await buf.after(sid, None)] == [1, 2, 4]
        finally:
            for n in (1, 2, 4):
                await client.delete(turn_result_key(sid, f"{sid}_turn_{n}"))

    run_async(body())
