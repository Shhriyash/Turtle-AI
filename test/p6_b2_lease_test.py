"""
test/p6_b2_lease_test.py
------------------------
P6-B2 (ledger 6.4 + 6.8) at the unit level: ``core.turn_lock.SessionLease``,
``SessionStore.start_or_restore(lease=...)`` and ``core.session_store.TurnResultBuffer``.

WHAT THIS FILE CANNOT PROVE. The cloud lease runs three Lua scripts
(RELEASE_LUA, REFRESH_LUA, TAKEOVER_LUA) on a Redis server. No Redis server is
installed here and fakeredis 2.38.0 has no Lua without ``lupa``, so:

  * the in-process backend (local mode) is exercised for real;
  * the Redis backend is exercised only for command SHAPE (a recording fake:
    ``SET NX EX`` is real, ``eval`` only records script/keys/args);
  * ``test_lua_scripts_under_lupa`` runs the three scripts for real through
    fakeredis' Lua support IF ``lupa`` is importable, and is skipped otherwise
    (it is not in requirements.txt);
  * the only run against a real Redis is
    test/cloud_integration/session_lease_cloud_test.py, in the ``cloud-tests``
    CI job.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
import redis.exceptions

from core import session_store as ss
from core import turn_lock as tl
from core.config import TurtleSettings
from core.storage import Session


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# SessionLease on the in-process backend
# ---------------------------------------------------------------------------
def _lease(identity: str, backend=None) -> tl.SessionLease:
    return tl.SessionLease(identity, backend=backend or tl.InProcessTurnLockBackend())


def test_second_client_is_refused_and_same_client_takes_over() -> None:
    async def body():
        be = tl.InProcessTurnLockBackend()
        a, b, a2 = _lease("tabA", be), _lease("tabB", be), _lease("tabA", be)
        assert await a.claim("s1") is True
        assert await b.claim("s1") is False and not b.is_holder
        assert await b.held_by_other("s1") is True
        assert await a.held_by_other("s1") is False
        # the dead connection's lease is still held: the same tab takes it over
        assert await a2.claim("s1") is True
        # ... and the superseded connection must NOT steal it back
        assert await a.ensure() is False and a.lost and not a.is_holder
        assert await a2.ensure() is True
        # a stale release cannot free the new holder
        await a.release()
        assert await b.held_by_other("s1") is True
        await a2.release()
        assert await b.claim("s1") is True

    _run(body())


def test_lease_expires_and_ensure_reclaims_a_vanished_key() -> None:
    async def body():
        be = tl.InProcessTurnLockBackend()
        a = tl.SessionLease("tabA", backend=be, ttl_s=1)
        assert await a.claim("s1")
        be._held.pop(tl.session_lease_key("s1"))  # eviction / expiry during a stall
        assert await a.ensure() is True, "nobody took it: re-claim, do not declare loss"
        assert a.held and not a.lost

        # but if someone else got it in the meantime the lease IS lost
        be._held.pop(tl.session_lease_key("s1"))
        assert await _lease("tabB", be).claim("s1")
        assert await a.ensure() is False and a.lost

    _run(body())


def test_anonymous_connections_never_match_each_other() -> None:
    """No client id -> a private identity: exactly the ledger's plain NX lease."""
    async def body():
        be = tl.InProcessTurnLockBackend()
        assert await _lease("c1abc", be).claim("s1")
        assert await _lease("c2def", be).claim("s1") is False

    _run(body())


class _Exploding:
    """The driver's REAL error, not a mocked unset-URL exception."""

    async def try_acquire(self, *a):
        raise redis.exceptions.ConnectionError("Error 111 connecting to redis:6379. Connection refused.")

    takeover = holder = refresh = release = try_acquire


def test_lease_fails_open_on_redis_errors() -> None:
    async def body():
        lease = _lease("tabA", _Exploding())
        assert await lease.claim("s1") is True
        assert lease.degraded and lease.is_holder, "a Redis blip must not take sessions down"
        assert await lease.ensure() is True
        assert await lease.held_by_other("s1") is False
        await lease.release()  # must not raise

    _run(body())


# ---------------------------------------------------------------------------
# Redis backend: command shapes
# ---------------------------------------------------------------------------
class RecordingRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.eval_calls: list[tuple] = []
        self.set_calls: list[tuple] = []

    async def set(self, key, value, nx=False, ex=None):
        self.set_calls.append((key, value, nx, ex))
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def get(self, key):
        return self.store.get(key)

    async def eval(self, script, numkeys, *args):
        self.eval_calls.append((script, numkeys, args))
        return 1


def test_redis_backend_command_shapes(monkeypatch) -> None:
    fake = RecordingRedis()

    async def _get():
        return fake

    monkeypatch.setattr("core.storage.cloud.get_redis_client", _get)

    async def body():
        a = tl.SessionLease("tabA", backend=tl.RedisTurnLockBackend())
        key = tl.session_lease_key("s1")
        assert key == "turtle:session_lease:s1"
        assert await a.claim("s1")
        assert fake.set_calls == [(key, a.token, True, 90)], "ledger: SET NX EX 90"
        # second client: NX fails, then the takeover script is tried with the
        # client's identity prefix, NOT with a bare overwrite
        b = tl.SessionLease("tabB", backend=tl.RedisTurnLockBackend())
        fake.eval_calls.clear()
        await b.claim("s1")
        (script, numkeys, args), = fake.eval_calls
        assert script == tl.TAKEOVER_LUA and numkeys == 1
        assert args == (key, "tabB:", b.token, 90)
        fake.eval_calls.clear()
        await a.ensure()
        (script, numkeys, args), = fake.eval_calls
        assert script == tl.REFRESH_LUA and args == (key, a.token, 90)
        fake.eval_calls.clear()
        await a.release()
        (script, numkeys, args), = fake.eval_calls
        assert script == tl.RELEASE_LUA and args == (key, a.token)
        assert await tl.SessionLease("tabC", backend=tl.RedisTurnLockBackend()).held_by_other("s1")

    _run(body())


def test_lua_scripts_under_lupa() -> None:
    """Runs the REAL Lua through fakeredis when lupa is available (it is not in
    requirements.txt; the author ran this once with lupa on PYTHONPATH)."""
    pytest.importorskip("lupa")
    from fakeredis import FakeAsyncRedis

    class B(tl.RedisTurnLockBackend):
        def __init__(self, c):
            self.c = c

        async def _client(self):
            return self.c

    async def body():
        be = B(FakeAsyncRedis(decode_responses=True))
        a, b, a2 = _lease("tabA", be), _lease("tabB", be), _lease("tabA", be)
        assert await a.claim("s1") and not await b.claim("s1")
        assert await a2.claim("s1")
        assert await a.ensure() is False and a.lost
        await a.release()
        assert await a2.ensure() is True
        await a2.release()
        assert await b.claim("s1")

    _run(body())


# ---------------------------------------------------------------------------
# SessionStore.start_or_restore(lease=...)
# ---------------------------------------------------------------------------
class MemBackend:
    def __init__(self) -> None:
        self.rows: dict[str, Session] = {}

    async def put(self, session: Session) -> None:
        self.rows[session.session_id] = Session(session_id=session.session_id, data=dict(session.data))

    async def get(self, session_id: str):
        return self.rows.get(session_id)

    async def list_sessions(self, status_filter=None, user_id=None):
        return [
            Session(session_id=s.session_id, data=dict(s.data)) for s in self.rows.values()
            if (status_filter is None or s.data.get("status") == status_filter)
            and (user_id is None or s.data.get("user_id") == user_id)
        ]


def _row(be: MemBackend, sid: str, status: str, age_s: float = 5, user: str = "u1") -> None:
    ts = time.time() - age_s
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
    be.rows[sid] = Session(session_id=sid, data={
        "status": status, "user_id": user, "messages": [], "updated_at": stamp,
    })


def test_leased_session_is_neither_resumed_nor_demoted() -> None:
    async def body():
        be, locks = MemBackend(), tl.InProcessTurnLockBackend()
        _row(be, "live", "active", age_s=5)
        _row(be, "stale_but_live", "active", age_s=99999)  # idle tab, still connected
        owner = _lease("tabA", locks)
        assert await owner.claim("live") and await owner.claim("stale_but_live")

        newcomer = _lease("tabB", locks)
        store = ss.SessionStore(backend=be, user_id="u1")
        result = await store.start_or_restore(mode="resume_if_active", lease=newcomer)
        assert result.restored is False and result.session_id not in ("live", "stale_but_live")
        assert result.previous_session_id is None, "nothing may be demoted under its owner"
        assert be.rows["live"].data["status"] == "active"
        assert be.rows["stale_but_live"].data["status"] == "active"
        assert await newcomer.held_by_other("live") and newcomer.is_holder
        assert newcomer.session_id == result.session_id

    _run(body())


def test_same_client_resumes_its_leased_session_and_it_is_claimed() -> None:
    async def body():
        be, locks = MemBackend(), tl.InProcessTurnLockBackend()
        _row(be, "mine", "active", age_s=5)
        be.rows["mine"].data["turn_counter"] = 7
        old = _lease("tabA", locks)
        assert await old.claim("mine")
        new = _lease("tabA", locks)
        store = ss.SessionStore(backend=be, user_id="u1")
        result = await store.start_or_restore(mode="resume_if_active", lease=new)
        assert result.restored and result.session_id == "mine"
        assert store.turn_counter == 7, "turn numbering must continue across a resume"
        assert new.is_holder and not await old.ensure()

    _run(body())


def test_without_a_lease_behaviour_is_unchanged() -> None:
    """Channels and scripts pass no lease: resume the newest, demote the stale."""
    async def body():
        be = MemBackend()
        _row(be, "recent", "active", age_s=5)
        _row(be, "old", "active", age_s=99999)
        store = ss.SessionStore(backend=be, user_id="u1")
        result = await store.start_or_restore(mode="resume_if_active")
        assert result.restored and result.session_id == "recent"

        be2 = MemBackend()
        _row(be2, "old", "active", age_s=99999)
        store2 = ss.SessionStore(backend=be2, user_id="u1")
        result2 = await store2.start_or_restore(mode="resume_if_active")
        assert not result2.restored
        assert be2.rows["old"].data["status"] == "pending_finalization"

    _run(body())


def test_pending_finalization_session_is_resumed_only_if_not_leased() -> None:
    async def body():
        be, locks = MemBackend(), tl.InProcessTurnLockBackend()
        _row(be, "p", "pending_finalization", age_s=5)
        assert await _lease("tabA", locks).claim("p")
        store = ss.SessionStore(backend=be, user_id="u1")
        result = await store.start_or_restore(mode="resume_if_active", lease=_lease("tabB", locks))
        assert not result.restored

    _run(body())


# ---------------------------------------------------------------------------
# TurnResultBuffer
# ---------------------------------------------------------------------------
def _buf() -> ss.TurnResultBuffer:
    return ss.TurnResultBuffer(backend=ss._InProcessResultBackend())


def test_turn_number_parsing() -> None:
    assert ss.turn_number("turtle_session_x_turn_12") == 12
    assert ss.turn_number(None) == 0 and ss.turn_number("garbage") == 0
    assert ss.turn_result_key("S", "S_turn_3") == "turtle:turn_result:S:S_turn_3"


def test_after_returns_newer_turns_in_order_and_tolerates_gaps() -> None:
    async def body():
        b = _buf()
        for n in (1, 2, 3, 5, 6):  # turn 4 never finished
            await b.put("S", f"S_turn_{n}", {"n": n})
        assert [r["n"] for r in await b.after("S", "S_turn_2")] == [3, 5, 6]
        assert [r["n"] for r in await b.after("S", None)] == [1, 2, 3, 5, 6]
        assert await b.after("S", "S_turn_6") == []
        assert await b.after("OTHER", None) == []

    _run(body())


def test_results_expire_after_the_ttl() -> None:
    async def body():
        b = ss.TurnResultBuffer(backend=ss._InProcessResultBackend(), ttl_s=1)
        await b.put("S", "S_turn_1", {"n": 1})
        assert len(await b.after("S", None)) == 1
        await asyncio.sleep(1.1)
        assert await b.after("S", None) == []
        assert ss.TURN_RESULT_TTL_S == 120, "ledger: buffered for 120 s"

    _run(body())


def test_replay_scan_fails_open_on_redis_errors() -> None:
    class Down:
        async def get(self, key):
            raise redis.exceptions.ConnectionError("Connection refused")

    async def body():
        assert await ss.TurnResultBuffer(backend=Down()).after("S", None) == []

    _run(body())


def test_redis_result_backend_uses_set_ex(monkeypatch) -> None:
    fake = RecordingRedis()

    async def _get():
        return fake

    monkeypatch.setattr("core.storage.cloud.get_redis_client", _get)

    async def body():
        b = ss.TurnResultBuffer(backend=ss._RedisResultBackend())
        await b.put("S", "S_turn_1", {"n": 1})
        key, value, nx, ex = fake.set_calls[0]
        assert key == "turtle:turn_result:S:S_turn_1" and ex == 120 and nx is False
        assert [r["n"] for r in await b.after("S", None)] == [1]

    _run(body())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_ws_max_duration_defaults_to_300_and_reads_the_env_alias(monkeypatch) -> None:
    monkeypatch.delenv("TURTLE_WS_MAX_DURATION_S", raising=False)
    assert TurtleSettings(_env_file=None).ws_max_duration_s == 300
    monkeypatch.setenv("TURTLE_WS_MAX_DURATION_S", "25")
    s = TurtleSettings(_env_file=None)
    assert s.ws_max_duration_s == 25 and "ws_max_duration_s" in s.model_fields_set
