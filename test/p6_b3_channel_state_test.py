"""
test/p6_b3_channel_state_test.py
---------------------------------
P6-B3 (ledger 6.6): in cloud, channel ``SharedState`` is rebuilt per request;
serialisation comes from the Redis per-user turn lock (6.3), not from the
per-instance cache + ``asyncio.Lock`` (which excluded nothing across instances).

The things a naive "just drop the cache" would silently break, each pinned here:

  * PERIODIC REFLECTION. ``PeriodicReflector`` keeps its turn counter on the
    state object, so a fresh state per request makes the counter 1 forever and
    Stage B extraction + the rolling summary would stop for every cloud channel
    user. Cloud channel states use a reflector that derives its counter from the
    session store's persisted ``turn_counter``.
  * ``_ACTIVE_STATES`` (shutdown-flush registry): a state that registers and
    never unregisters leaks one entry per channel message.
  * STALE HISTORY: the state must be built INSIDE the user's turn lock, or a turn
    that waited builds on the history it read before the holder finished.
  * the account-link merge must exclude an in-flight turn of the SOURCE user on
    ANOTHER instance, which the in-process channel lock cannot see.

Cloud is simulated: ``settings.deploy_mode == "cloud"`` and the turn-lock backend
is replaced by one shared in-process backend standing in for Redis (the shared
backend is what makes two "instances" contend). Real Redis/Postgres behaviour is
NOT exercised here.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import apps.turtle_server as ts
import core.identity as identity_mod
from apps.channels import TurtleEvent
from core import turn_lock as turn_lock_mod
from core.config import settings
from core.periodic_reflector import PeriodicReflector

USER = "usr_p6b3"
EVERY = 3


class _PersistedSessions:
    """Stands in for the session backend: survives across rebuilt states."""

    def __init__(self) -> None:
        self.turn_counter: dict[str, int] = {}


class _FakeSessionStore:
    persisted = _PersistedSessions()
    builds = 0

    def __init__(self, user_id: str, **_kw) -> None:
        type(self).builds += 1
        self.user_id = user_id
        self.session_id = "sess_1"
        self.message_history: list = []
        self.turn_counter = type(self).persisted.turn_counter.get(user_id, 0)

    async def start_or_restore(self, mode: str = "x", **_kw):
        return SimpleNamespace(session_id=self.session_id)

    async def replace_messages(self, messages) -> None:
        # _sync_to_backend persists the counter with the blob.
        type(self).persisted.turn_counter[self.user_id] = self.turn_counter


@pytest.fixture
def harness(monkeypatch):
    _FakeSessionStore.persisted = _PersistedSessions()
    _FakeSessionStore.builds = 0
    monkeypatch.setattr(settings, "deploy_mode", "cloud", raising=False)
    monkeypatch.setattr(settings, "reflect_enabled", True, raising=False)
    monkeypatch.setattr(settings, "reflect_every_turns", EVERY, raising=False)

    # One backend shared by every "instance" == Redis.
    shared = turn_lock_mod.InProcessTurnLockBackend()
    monkeypatch.setattr(turn_lock_mod, "get_turn_lock_backend", lambda: shared)

    class _AnyLimiter:
        def check_and_record(self, _key):
            return None

    monkeypatch.setattr(ts, "get_ws_rate_limiter", lambda: _AnyLimiter())

    # --- constructors used by the REAL _build_channel_state -----------------
    monkeypatch.setattr(ts, "SessionStore", _FakeSessionStore)
    monkeypatch.setattr(ts, "PersonalMemoryStore", MagicMock())
    monkeypatch.setattr(ts, "PersonalMemoryPromptBuilder", MagicMock())
    monkeypatch.setattr(ts, "JournalStore", MagicMock())
    gate = MagicMock()
    gate.next_prompt.return_value = None
    monkeypatch.setattr(ts, "ConfirmationGate", MagicMock(return_value=gate))
    rag = MagicMock()
    rag.start_session = AsyncMock(return_value="sess_1")
    rag._extract_turn_records_from_messages.return_value = []
    monkeypatch.setattr(ts, "TurtleRAGSystem", MagicMock(return_value=rag))
    monkeypatch.setattr(ts, "personal_memory_dir", lambda uid: __import__("pathlib").Path("nope") / uid)
    monkeypatch.setattr("core.storage.factory.get_vector_store", lambda: MagicMock())
    monkeypatch.setattr("core.storage.factory.get_confirmation_state_backend", lambda uid: MagicMock())
    monkeypatch.setattr("core.retrieval_broker.RetrievalBroker", MagicMock())

    monkeypatch.setattr(ts, "_CHANNEL_STATES", {})
    monkeypatch.setattr(ts, "_CHANNEL_STATE_LOCKS", {})
    before = dict(ts._ACTIVE_STATES)

    reflected: list[int] = []

    async def fake_reflect(self, state, *, session_id, message_history):
        sess = self._get(session_id)
        reflected.append(sess.turn_counter)
        sess.last_reflected_turn = sess.turn_counter  # what a successful _reflect does

    monkeypatch.setattr(PeriodicReflector, "_reflect", fake_reflect)

    seen_counter_at_entry: list[int] = []
    gauge = SimpleNamespace(now=0, peak=0, delay=0.0)

    async def fake_execute_turn(ws, state, text, history, *, channel, send_status=True):
        seen_counter_at_entry.append(state.session_store.turn_counter)
        gauge.now += 1
        gauge.peak = max(gauge.peak, gauge.now)
        try:
            if gauge.delay:
                await asyncio.sleep(gauge.delay)
            ts._new_turn_id(state)  # the real counter bookkeeping
            await state.session_store.replace_messages([])
            await state.reflector.on_turn(
                state, session_id=state.session_store.session_id, message_history=[]
            )
            await asyncio.sleep(0)  # let the fire-and-forget reflection task run
        finally:
            gauge.now -= 1
        return SimpleNamespace(reply_text="ok", output_text="ok")

    monkeypatch.setattr(ts, "_execute_turn", fake_execute_turn)

    build_calls = {"n": 0}
    real_build = ts._build_channel_state

    async def counting_build(user_id, channel):
        build_calls["n"] += 1
        return await real_build(user_id, channel)

    monkeypatch.setattr(ts, "_build_channel_state", counting_build)

    fake_identity = SimpleNamespace(resolve_user=AsyncMock(return_value=USER))
    monkeypatch.setattr(identity_mod, "identity_manager", fake_identity, raising=False)

    yield SimpleNamespace(
        reflected=reflected,
        builds=build_calls,
        entry_counters=seen_counter_at_entry,
        gauge=gauge,
        before_active=before,
        backend=shared,
    )

    # Whatever a test left registered must not leak into other tests.
    for key in list(ts._ACTIVE_STATES):
        if key not in before:
            ts._ACTIVE_STATES.pop(key, None)


def _event(channel="discord", cuid="chan_1"):
    return TurtleEvent(
        user_id=USER, channel=channel, modality="text", content="hello",
        message_id="m", thread_id="t", channel_user_id=cuid,
    )


def test_cloud_channel_state_is_built_per_request(harness):
    async def scenario():
        for _ in range(3):
            reply = await ts._channel_dispatch_handler(_event())
            assert reply.content == "ok"

    asyncio.run(scenario())
    assert harness.builds["n"] == 3, "cloud must rebuild the state for every request"
    assert ts._CHANNEL_STATES == {}, "cloud must not retain a per-instance state cache"


def test_cloud_reflection_still_fires_at_the_configured_interval(harness):
    """THE headline: N rebuilt-per-request channel turns still reflect every
    ``reflect_every_turns``. Without a durable counter every request would see
    turn 1 and this list would be empty."""
    async def scenario():
        for _ in range(7):
            await ts._channel_dispatch_handler(_event())
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert harness.reflected == [EVERY, 2 * EVERY], harness.reflected


def test_cloud_reflection_is_not_re_fired_between_intervals(harness):
    async def scenario():
        for _ in range(EVERY - 1):
            await ts._channel_dispatch_handler(_event())
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert harness.reflected == []


def test_cloud_does_not_leak_active_states(harness):
    async def scenario():
        for _ in range(5):
            await ts._channel_dispatch_handler(_event())

    asyncio.run(scenario())
    assert set(ts._ACTIVE_STATES) == set(harness.before_active), (
        "every per-request state registered for shutdown flush must be unregistered"
    )


def test_cloud_does_not_leak_active_states_when_the_turn_raises(harness, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("model fell over")

    monkeypatch.setattr(ts, "_execute_turn", boom)

    async def scenario():
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await ts._channel_dispatch_handler(_event())

    asyncio.run(scenario())
    assert set(ts._ACTIVE_STATES) == set(harness.before_active)


def test_cloud_concurrent_messages_across_instances_do_not_interleave(harness, monkeypatch):
    """Two Discord identities mapped to ONE user hold different in-process
    channel locks (the lock key is (channel, channel_user_id)), exactly like two
    serverless instances: only the shared turn lock can serialise them. The
    second turn must also see the first's persisted counter, i.e. its state was
    built AFTER it got the lock."""
    monkeypatch.setattr(turn_lock_mod, "TURN_LOCK_WAIT_S", 5.0)
    harness.gauge.delay = 0.2

    async def scenario():
        return await asyncio.gather(
            ts._channel_dispatch_handler(_event(cuid="chan_A")),
            ts._channel_dispatch_handler(_event(cuid="chan_B")),
        )

    replies = asyncio.run(scenario())
    assert [r.content for r in replies] == ["ok", "ok"]
    assert harness.gauge.peak == 1, "turns for one user ran concurrently"
    assert sorted(harness.entry_counters) == [0, 1], (
        "the second turn built its state before the first finished (stale history): "
        f"{harness.entry_counters}"
    )


def test_cloud_second_message_is_rejected_busy_not_dropped_or_run(harness, monkeypatch):
    monkeypatch.setattr(turn_lock_mod, "TURN_LOCK_WAIT_S", 0.1)
    harness.gauge.delay = 0.6

    async def scenario():
        return await asyncio.gather(
            ts._channel_dispatch_handler(_event(cuid="chan_A")),
            ts._channel_dispatch_handler(_event(cuid="chan_B")),
        )

    replies = asyncio.run(scenario())
    texts = sorted(r.content for r in replies)
    assert texts == sorted(["ok", turn_lock_mod.BUSY_MESSAGE])
    assert harness.gauge.peak == 1
    assert set(ts._ACTIVE_STATES) == set(harness.before_active)


def test_cloud_build_wires_the_durable_reflector_and_local_keeps_plain(harness, monkeypatch):
    async def scenario():
        cloud_state = await ts._build_channel_state(USER, "discord")
        monkeypatch.setattr(settings, "deploy_mode", "local", raising=False)
        monkeypatch.setattr("core.memory_sqlite.MemorySQLiteIndex", MagicMock(side_effect=RuntimeError("off")))
        local_state = await ts._build_channel_state(USER, "discord")
        return cloud_state, local_state

    cloud_state, local_state = asyncio.run(scenario())
    try:
        assert isinstance(cloud_state.reflector, PeriodicReflector)
        assert type(cloud_state.reflector) is not PeriodicReflector
        assert type(local_state.reflector) is PeriodicReflector
    finally:
        ts._unregister_shutdown_state(cloud_state)
        ts._unregister_shutdown_state(local_state)


def test_local_mode_keeps_the_state_cache(harness, monkeypatch):
    """Local is one process: the cache is a latency win and has no
    cross-instance problem. Behaviour must be unchanged."""
    monkeypatch.setattr(settings, "deploy_mode", "local", raising=False)
    monkeypatch.setattr("core.memory_sqlite.MemorySQLiteIndex", MagicMock(side_effect=RuntimeError("off")))
    shared = turn_lock_mod.InProcessTurnLockBackend()
    monkeypatch.setattr(turn_lock_mod, "get_turn_lock_backend", lambda: shared)

    async def scenario():
        for _ in range(3):
            await ts._channel_dispatch_handler(_event())

    asyncio.run(scenario())
    try:
        assert harness.builds["n"] == 1
        assert (USER, "discord") in ts._CHANNEL_STATES
    finally:
        for st, _ in ts._CHANNEL_STATES.values():
            ts._unregister_shutdown_state(st)


def test_cloud_prunes_idle_in_process_locks_so_sender_ids_cannot_grow_them_unbounded(harness):
    """Cloud has no cached states, and lock eviction used to be driven by the
    state cache; without this the lock dict grows once per distinct sender."""
    async def scenario():
        for i in range(ts._CHANNEL_STATE_CAP + 10):
            ts._channel_state_lock(("discord", f"sender_{i}"))
        held = ts._channel_state_lock(("discord", "busy_sender"))
        async with held:
            ts._evict_stale_channel_states(time.monotonic())
            assert ("discord", "busy_sender") in ts._CHANNEL_STATE_LOCKS, "a held lock must survive"
        assert len(ts._CHANNEL_STATE_LOCKS) == 1

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Link redemption must exclude an in-flight turn of the SOURCE user (any instance)
# ---------------------------------------------------------------------------
@pytest.fixture()
def link_env(tmp_path, monkeypatch):
    import core.auth_secret as m
    import core.paths as core_paths
    from core.account_linking import LinkCodeStore
    from core.identity import IdentityManager

    mgr = IdentityManager(db_path=tmp_path / "users.sqlite")
    monkeypatch.setattr(identity_mod, "identity_manager", mgr, raising=False)
    monkeypatch.setattr(settings, "deploy_mode", "local", raising=False)
    monkeypatch.setattr(settings, "dev_anon", False, raising=False)
    monkeypatch.setattr(settings, "auth_secret_key", None, raising=False)
    m._reset_for_tests()
    root = tmp_path / "personal"
    root.mkdir()
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root, raising=False)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snap", raising=False)
    asyncio.run(mgr.init_db())
    return mgr, LinkCodeStore(tmp_path / "users.sqlite")


def test_link_redeem_waits_out_an_in_flight_turn_of_the_source_user(link_env, monkeypatch):
    from fastapi.testclient import TestClient

    from apps.auth import create_session_token

    mgr, store = link_env
    monkeypatch.setattr(turn_lock_mod, "TURN_LOCK_WAIT_S", 0.2)
    source = asyncio.run(mgr.resolve_user("discord", "759"))
    target = asyncio.run(mgr.resolve_user("web_email", "t@example.com"))
    code = store.issue(channel="discord", channel_user_id="759", source_user_id=source).code
    headers = {"Authorization": f"Bearer {create_session_token(target)}"}

    backend = turn_lock_mod.get_turn_lock_backend()
    key = turn_lock_mod.turn_lock_key(source)
    backend._held[key] = ("someone-elses-token", time.monotonic() + 60)  # a turn is running
    try:
        with TestClient(ts.app) as client:
            busy = client.post("/api/account/link", json={"code": code}, headers=headers)
            assert busy.status_code == 503, busy.text
            assert asyncio.run(mgr.resolve_user("discord", "759")) == source, "mapping must be unchanged"

            backend._held.pop(key, None)  # the turn finished
            ok = client.post("/api/account/link", json={"code": code}, headers=headers)
            assert ok.status_code == 200, ok.text
        assert asyncio.run(mgr.resolve_user("discord", "759")) == target
    finally:
        backend._held.pop(key, None)
