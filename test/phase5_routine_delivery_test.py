"""
Phase 5 (W2) — routine delivery v1.

Covers the live-push delivery layered on top of the scheduler's journal write:

  (a) _fire_routine, given an injected delivery hook, writes the scheduled_fire
      journal event AND calls the hook with a routine WS frame carrying the
      routine name — the journal write is the source of truth, delivery rides
      on top.
  (b) deliver_routine_notice, with a real running app loop and a set of fake
      sockets, bridges the frame onto every one of the target user's sockets and
      leaves other users untouched.
  (c) with no live socket, the frame is stashed and drained exactly once (a
      second drain returns empty).
  (d) the per-user pending queue is capped (most-recent kept).
  (e) a delivery hook that raises does NOT corrupt the journal write and never
      lets an exception escape _fire_routine.

Journal isolation follows test/phase3_scheduler_test.py: the pm_root fixture
redirects all per-user memory/journal writes into a tmp dir. No running
scheduler is needed — _fire_routine is driven synchronously; the delivery helper
is exercised inside asyncio.run against a real loop.
"""
from __future__ import annotations

import asyncio

import pytest

import core.paths as core_paths
import core.routine_scheduler as rs
from core.memory_journal import JournalStore
from core.routine_scheduler import _build_routine_frame, _fire_routine


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def pm_root(tmp_path, monkeypatch):
    """Redirect all per-user memory/journal writes into a tmp dir (see phase3)."""
    root = tmp_path / "pm"
    root.mkdir()
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snapshots")
    # routine_scheduler binds PERSONAL_MEMORY_DIR at import time — patch its copy.
    monkeypatch.setattr(rs, "PERSONAL_MEMORY_DIR", root)
    return root


def _scheduled_fires(user_id: str, routine_key: str):
    events = JournalStore(user_id=user_id).load_all()
    return [e for e in events if e.key == f"workflow.scheduled_fire.{routine_key}"]


# ── (a) fire path: journal written AND delivery hook called ──────────────

def test_fire_routine_writes_journal_and_calls_delivery_hook(pm_root, monkeypatch):
    calls: list[tuple[str, dict]] = []

    def _fake_hook(user_id: str, frame: dict) -> bool:
        calls.append((user_id, frame))
        return True

    monkeypatch.setattr(rs, "_delivery_hook", _fake_hook)

    user_id = "usr_deliver"
    routine_key = "workflow.morning_routine"
    value = {"routine": "morning briefing", "items": ["Indore news"],
             "cadence": "daily", "time": "07:30"}

    _fire_routine(user_id, routine_key, value)

    # Journal (source of truth) written exactly once, applied=False audit record.
    fires = _scheduled_fires(user_id, routine_key)
    assert len(fires) == 1
    assert fires[0].applied is False
    assert fires[0].extractor == "scheduler"

    # Delivery hook called once with a routine frame carrying the routine name.
    assert len(calls) == 1
    called_user, frame = calls[0]
    assert called_user == user_id
    assert frame["type"] == "routine"
    assert frame["code"] == "routine_fire"
    assert frame["routine_key"] == routine_key
    assert "morning briefing" in frame["message"]
    # Items summary rides along in the humanized message.
    assert "Indore news" in frame["message"]


def test_build_routine_frame_falls_back_to_humanized_key():
    # No explicit routine name → derive a display name from the key tail.
    frame = _build_routine_frame("workflow.daily_briefing", {}, "2026-07-20T00:00:00+00:00")
    assert frame["type"] == "routine"
    assert "daily briefing" in frame["message"]


# ── (b) delivery helper: real loop + fake sockets, per-user scoping ───────

class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)


def test_delivery_helper_sends_to_all_user_sockets_only(pm_root, monkeypatch):
    import apps.turtle_server as ts

    ws_a1, ws_a2, ws_b = _FakeWS(), _FakeWS(), _FakeWS()

    async def _run():
        loop = asyncio.get_running_loop()
        # Point the helper's cross-thread bridge at THIS live loop, and stage a
        # clean registry: user A has two sockets, user B has one.
        monkeypatch.setattr(ts, "_APP_LOOP", loop)
        monkeypatch.setattr(ts, "_LIVE_SOCKETS", {
            "userA": {ws_a1, ws_a2},
            "userB": {ws_b},
        })
        monkeypatch.setattr(ts, "_PENDING_ROUTINE_NOTICES", {})

        frame = {"type": "routine", "code": "routine_fire", "message": "hi A"}
        result = ts.deliver_routine_notice("userA", frame)
        # Let the loop process the run_coroutine_threadsafe-scheduled sends.
        await asyncio.sleep(0.1)

        assert result is True
        assert ws_a1.sent == [frame]
        assert ws_a2.sent == [frame]
        # Other user untouched, and nothing queued (it was delivered live).
        assert ws_b.sent == []
        assert ts.pop_pending_routine_notices("userA") == []

    asyncio.run(_run())


# ── (c) no live socket: stash then drain exactly once ────────────────────

def test_no_socket_stashes_and_drains_once(pm_root, monkeypatch):
    import apps.turtle_server as ts

    monkeypatch.setattr(ts, "_APP_LOOP", None)
    monkeypatch.setattr(ts, "_LIVE_SOCKETS", {})
    monkeypatch.setattr(ts, "_PENDING_ROUTINE_NOTICES", {})

    frame = {"type": "routine", "code": "routine_fire", "message": "queued"}
    result = ts.deliver_routine_notice("ghost", frame)
    assert result is False

    drained = ts.pop_pending_routine_notices("ghost")
    assert drained == [frame]
    # Draining is destructive — a second pop is empty.
    assert ts.pop_pending_routine_notices("ghost") == []


def test_live_socket_but_no_loop_falls_back_to_stash(pm_root, monkeypatch):
    import apps.turtle_server as ts

    ws = _FakeWS()
    monkeypatch.setattr(ts, "_APP_LOOP", None)  # loop never captured
    monkeypatch.setattr(ts, "_LIVE_SOCKETS", {"u": {ws}})
    monkeypatch.setattr(ts, "_PENDING_ROUTINE_NOTICES", {})

    frame = {"type": "routine", "message": "no loop"}
    result = ts.deliver_routine_notice("u", frame)
    assert result is False
    assert ws.sent == []  # cannot bridge without a loop
    assert ts.pop_pending_routine_notices("u") == [frame]


# ── (d) pending queue cap ────────────────────────────────────────────────

def test_pending_cap_respected(pm_root, monkeypatch):
    import apps.turtle_server as ts

    monkeypatch.setattr(ts, "_APP_LOOP", None)
    monkeypatch.setattr(ts, "_LIVE_SOCKETS", {})
    monkeypatch.setattr(ts, "_PENDING_ROUTINE_NOTICES", {})

    cap = ts._PENDING_ROUTINE_MAX_PER_USER
    overflow = cap + 3
    for i in range(overflow):
        ts.deliver_routine_notice("u", {"type": "routine", "message": f"n{i}"})

    drained = ts.pop_pending_routine_notices("u")
    assert len(drained) == cap
    # Kept the most recent `cap` frames — oldest were evicted.
    assert drained[0]["message"] == f"n{overflow - cap}"
    assert drained[-1]["message"] == f"n{overflow - 1}"


# ── (e) delivery failure must not corrupt the journal write ──────────────

def test_delivery_failure_leaves_journal_intact_and_never_raises(pm_root, monkeypatch):
    def _boom(user_id: str, frame: dict) -> bool:
        raise RuntimeError("delivery down")

    monkeypatch.setattr(rs, "_delivery_hook", _boom)

    user_id = "usr_boom"
    routine_key = "workflow.morning_routine"
    value = {"routine": "morning briefing", "items": ["x"],
             "cadence": "daily", "time": "07:30"}

    # Must not raise even though the hook does.
    _fire_routine(user_id, routine_key, value)

    # The journal write (source of truth) survived the failed delivery.
    fires = _scheduled_fires(user_id, routine_key)
    assert len(fires) == 1
    assert fires[0].value["source_routine"] == routine_key


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
