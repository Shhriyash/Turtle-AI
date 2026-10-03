"""
test/p6_b2_resume_test.py
-------------------------
P6-B2 (ledger 6.8 + 6.4): the planned 1012 cut, the resume protocol and the
session lease, driven through the REAL ``apps.turtle_server.websocket_endpoint``
(local mode, real SessionStore on the suite's throwaway TURTLE_DATA_DIR) with a
scripted socket. Only auth and the model turn are faked; the receive loop, the
connection budget, the finalisation ``finally``, the lease, the turn lock and
the result buffer are production code.

The fake turn mirrors the real one in the ways that matter here: it takes a
``turn_id`` from ``_new_turn_id``, emits ``done`` THROUGH ``_ws_send_json`` (which
buffers it) and then persists history with ``replace_messages``.

What these tests do NOT cover: a real Redis (the Lua scripts and the cloud
backend run only in test/cloud_integration/session_lease_cloud_test.py), a real
browser, or Vercel's real 300 s kill.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

import apps.turtle_server as server
from core import turn_lock as turn_lock_mod
from core.config import settings
from core.session_store import SessionStore, turn_results
from test.p6_b1_turn_tasks_test import ScriptedWS, _until


class CutWS(ScriptedWS):
    """ScriptedWS + the query string and the close code."""

    def __init__(self, cid: str | None = None) -> None:
        super().__init__()
        self.query_params = {"cid": cid} if cid else {}
        self.close_code: int | None = None

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.close_code = code


class PersistingTurns:
    """Stand-in for ``_execute_turn`` that behaves like the real one around
    ids, the ``done`` frame and persistence."""

    def __init__(self, delay: float = 0.03, delays: dict[str, float] | None = None) -> None:
        self.delay = delay
        self.delays = delays or {}
        self.texts: list[str] = []
        self.histories: list[int] = []

    async def __call__(self, ws, state, text, history, *, channel, send_status=True):
        self.texts.append(text)
        self.histories.append(len(history or []))
        await server._ws_send_json(ws, {"type": "status", "status": "thinking"})
        await asyncio.sleep(self.delays.get(text, self.delay))
        turn_id = server._new_turn_id(state)
        new_history = list(history or []) + [
            ModelRequest(parts=[UserPromptPart(content=text)]),
            ModelResponse(parts=[TextPart(content=f"re:{text}")]),
        ]
        await server._ws_send_json(
            ws, {"type": "done", "content": f"re:{text}", "tool_urls": [], "turn_id": turn_id},
        )
        await asyncio.shield(state.session_store.replace_messages(new_history))
        return server.TurnOutcome(new_history, f"re:{text}", f"re:{text}")


@pytest.fixture(autouse=True)
def stage_b(monkeypatch):
    """Finalisation runs Stage B (an LLM call) over the transcript. Record it
    instead of calling out; the planned-cut tests assert it did NOT run."""
    calls: list[str] = []

    async def fake_stage_b(state, *, session_id, message_history):
        calls.append(session_id)

    monkeypatch.setattr(server, "run_stage_b_session_extractor", fake_stage_b)
    monkeypatch.setattr(server, "_sync_personal_memory_from_messages", lambda *a, **k: None)
    return calls


@pytest.fixture(autouse=True)
def _fresh_state():
    turn_lock_mod._in_process_backend._held.clear()
    # monkeypatch.setattr(settings, ...) leaves the field in pydantic's
    # model_fields_set even after it restores the value, and
    # server._ws_max_duration_s() reads "was it set explicitly?" from there.
    explicit_before = set(settings.model_fields_set)
    yield
    settings.__pydantic_fields_set__.intersection_update(explicit_before)
    turn_lock_mod._in_process_backend._held.clear()


def _run(body):
    asyncio.run(body())


def _patches(user: str, turns: PersistingTurns):
    return (
        patch.object(server, "authenticate_websocket", new=AsyncMock(return_value=user)),
        patch.object(server, "_execute_turn", new=turns),
    )


async def _open(ws: CutWS) -> "asyncio.Task":
    task = asyncio.create_task(server.websocket_endpoint(ws))
    await _until(lambda: any(f.get("status") == "ready" for f in ws.frames("status")))
    return task


async def _close(ws: CutWS, task: "asyncio.Task") -> None:
    ws.disconnect()
    await asyncio.wait_for(task, timeout=15)


async def _stored(user: str, session_id: str):
    return await SessionStore(user_id=user).backend.get(session_id)


async def _until_stored(user: str, session_id: str, n_messages: int) -> Any:
    """The fake turn persists AFTER emitting `done` (as the real one does), so a
    test that reads the row right after `done` would race it."""
    for _ in range(300):
        row = await _stored(user, session_id)
        if row is not None and len(row.data["messages"]) == n_messages:
            return row
        await asyncio.sleep(0.01)
    raise AssertionError(f"session {session_id} never reached {n_messages} messages")


def _ready(ws: CutWS) -> dict:
    return next(f for f in ws.frames("status") if f.get("status") == "ready")


# ---------------------------------------------------------------------------
# The headline: a long session with a short connection ceiling loses no turn.
# ---------------------------------------------------------------------------
class ProtocolClient:
    """A scriptable stand-in for web/js/websocket.js: stable cid, 1012 handling,
    `resume` first, resend of `unstarted`, `last_turn_id` bookkeeping."""

    def __init__(self, user: str, turns: PersistingTurns) -> None:
        self.user = user
        self.turns = turns
        self.cid = "tab_" + user[-8:].replace("_", "x")
        self.session_id: str | None = None
        self.last_turn_id: str | None = None
        self.answers: list[str] = []
        self.turn_ids: list[str] = []
        self.sessions_seen: list[str] = []
        self.closes: list[int | None] = []
        self.resend: list[dict] = []
        self.connections = 0

    def _absorb(self, ws: CutWS, seen: int) -> int:
        frames = ws.frames()
        for f in frames[seen:]:
            if f.get("type") == "done":
                self.answers.append(f["content"])
                tid = f.get("turn_id")
                if tid:
                    self.turn_ids.append(tid)
                    if self.session_id and tid.startswith(self.session_id + "_turn_"):
                        self.last_turn_id = tid
            elif f.get("type") == "status" and f.get("status") == "ready" and "session_id" in f:
                if self.session_id is None:
                    self.session_id = f["session_id"]
                self.sessions_seen.append(f["session_id"])
            elif f.get("type") == "resumed" and f.get("session_id") != self.session_id:
                self.session_id = f["session_id"]
                self.last_turn_id = None
            elif f.get("type") == "status" and f.get("status") == "reconnect":
                self.resend.extend(f.get("unstarted", []))
        return len(frames)

    async def run(self, messages: list[str]) -> None:
        todo = list(messages)
        while todo or self.resend:
            self.connections += 1
            ws = CutWS(self.cid)
            task = asyncio.create_task(server.websocket_endpoint(ws))
            await _until(lambda: ws.frames("status"), timeout=10)
            had_session = self.session_id is not None
            if had_session:
                ws.push({"type": "resume", "session_id": self.session_id,
                         "last_turn_id": self.last_turn_id})
            for item in self.resend:
                assert item["kind"] == "text"
                ws.push({"type": "text", "content": item["content"]})
            sent = len(self.resend)
            self.resend = []
            seen = 0
            answered_here = 0
            # Send one message at a time, like a person, until the server cuts.
            while not task.done():
                seen = self._absorb(ws, seen)
                if sent == answered_here and todo:
                    ws.push({"type": "text", "content": todo.pop(0)})
                    sent += 1
                elif sent == answered_here and not todo:
                    break
                answered_here = len([f for f in ws.frames("done")])
                await asyncio.sleep(0.01)
            seen = self._absorb(ws, seen)
            if task.done():
                self.closes.append(ws.close_code)
                await task
                # Anything the server refused is now in self.resend; messages
                # sent but never answered (and not echoed) would be a LOSS.
            else:
                await _close(ws, task)
                self.closes.append(None)
                self._absorb(ws, seen)


def test_long_session_with_short_ceiling_loses_no_turn_across_reconnects(monkeypatch) -> None:
    """Ledger acceptance: 'a 10-minute voice session with
    TURTLE_WS_MAX_DURATION_S=25 in test loses no turn across reconnects',
    time-compressed (the real clock, ceiling 1.5 s, deadline 0.1 s + 0.2 s margin)."""
    monkeypatch.setattr(settings, "ws_max_duration_s", 1.5)
    monkeypatch.setattr(server, "_TURN_DEADLINE_TEXT_S", 0.1)
    monkeypatch.setattr(server, "_WS_CUT_MARGIN_S", 0.2)
    user = "usr_p6b2_headline"
    turns = PersistingTurns(delay=0.03)
    N = 45
    messages = [f"m{i}" for i in range(1, N + 1)]

    async def body():
        a, b = _patches(user, turns)
        client = ProtocolClient(user, turns)
        with a, b:
            await client.run(messages)

        # every message answered exactly once, in order
        assert client.answers == [f"re:m{i}" for i in range(1, N + 1)]
        assert turns.texts == messages, "a message ran twice or was dropped"
        # the connection really was cut and resumed several times
        assert client.connections >= 3, f"only {client.connections} connections; ceiling never bit"
        assert client.closes.count(1012) >= 2, client.closes
        # ... into the SAME session every time (nothing was finalised away)
        assert len(set(client.sessions_seen)) == 1, client.sessions_seen
        # ... and the model saw the whole conversation: history grew by 2/turn
        assert turns.histories == [2 * i for i in range(N)], turns.histories
        # turn ids are unique across reconnects (counter continues on resume)
        assert len(client.turn_ids) == len(set(client.turn_ids)) == N
        nums = [int(t.rsplit("_", 1)[1]) for t in client.turn_ids]
        assert nums == sorted(nums)

    _run(body)


# ---------------------------------------------------------------------------
# The defect the ledger misses: a planned cut must not finalise the session.
# ---------------------------------------------------------------------------
def test_planned_cut_leaves_session_active_and_untruncated_then_resumable(monkeypatch, stage_b) -> None:
    monkeypatch.setattr(settings, "ws_max_duration_s", 3.0)
    monkeypatch.setattr(server, "_TURN_DEADLINE_TEXT_S", 0.05)
    monkeypatch.setattr(server, "_TURN_DEADLINE_VOICE_S", 0.05)
    monkeypatch.setattr(server, "_WS_CUT_MARGIN_S", 0.05)
    user = "usr_p6b2_nofinal"
    # "slow" is admitted early but outlives the ceiling; "late" is queued behind
    # it and, when "slow" finishes, can no longer finish itself.
    turns = PersistingTurns(delay=0.01, delays={"slow": 3.2})

    async def body():
        a, b = _patches(user, turns)
        with a, b:
            ws1 = CutWS("tabNofinal1")
            t1 = await _open(ws1)
            sid = _ready(ws1)["session_id"]
            # 8 turns = 16 messages > COMPLETED_SESSION_MESSAGE_TAIL (12): a
            # mark_finalized() on this row would visibly truncate it.
            for i in range(8):
                ws1.push({"type": "text", "content": f"t{i}"})
                await _until(lambda i=i: len(ws1.frames("done")) == i + 1)
            await _until_stored(user, sid, 16)
            ws1.push({"type": "text", "content": "slow"})
            await _until(lambda: "slow" in turns.texts)
            ws1.push({"type": "text", "content": "late"})
            await _until(lambda: ws1.frames("turn_queued"))
            await asyncio.wait_for(t1, timeout=15)
            assert [f["content"] for f in ws1.frames("done")][-1] == "re:slow", \
                "the in-flight turn must finish before the cut, not be cancelled"

            assert ws1.close_code == 1012
            assert stage_b == [], "Stage B ran over a half conversation at a planned cut"
            reconnect = ws1.frames("status")[-1]
            assert reconnect["status"] == "reconnect"
            assert reconnect["unstarted"] == [{"kind": "text", "content": "late"}]
            assert "late" not in turns.texts, "a turn that cannot finish must not start"

            row = await _stored(user, sid)
            assert row.data["status"] == "active", row.data["status"]
            assert len(row.data["messages"]) == 18, "the cut truncated the session (mark_finalized ran)"

            ws2 = CutWS("tabNofinal1")
            t2 = await _open(ws2)
            assert [f for f in ws2.frames("status") if f.get("status") == "restored"], \
                "reconnect after a planned cut must resume, not start fresh"
            assert _ready(ws2)["session_id"] == sid
            assert [f["message_count"] for f in ws2.frames("status") if f.get("status") == "restored"] == [18]
            await _close(ws2, t2)

    _run(body)


def test_idle_connection_is_cut_before_the_platform_kills_it(monkeypatch, stage_b) -> None:
    """The ledger gates turn STARTS, but the platform's hard kill also lands on an
    IDLE socket -- and the kill runs the full finalisation `finally`, compacting a
    session the client is about to resume. The server must cut an idle connection
    itself, once it is too old for any turn to finish."""
    monkeypatch.setattr(settings, "ws_max_duration_s", 1.5)
    monkeypatch.setattr(server, "_TURN_DEADLINE_TEXT_S", 0.1)
    monkeypatch.setattr(server, "_TURN_DEADLINE_VOICE_S", 0.1)
    monkeypatch.setattr(server, "_WS_CUT_MARGIN_S", 0.1)
    user = "usr_p6b2_idle"
    turns = PersistingTurns(delay=0.01)

    async def body():
        a, b = _patches(user, turns)
        with a, b:
            ws = CutWS("tabIdle000001")
            task = await _open(ws)
            sid = _ready(ws)["session_id"]
            ws.push({"type": "text", "content": "hello"})
            await _until(lambda: ws.frames("done"))
            await _until_stored(user, sid, 2)
            # no further traffic: the server must end the connection itself
            await asyncio.wait_for(task, timeout=5)
            assert ws.close_code == 1012
            last = ws.frames("status")[-1]
            assert last["status"] == "reconnect" and last["reason"] == "max_duration"
            assert last["unstarted"] == []
            assert stage_b == []
            assert (await _stored(user, sid)).data["status"] == "active"

    _run(body)


def test_real_disconnect_still_finalises(monkeypatch, stage_b) -> None:
    """The planned-cut exemption must not leak into ordinary disconnects."""
    user = "usr_p6b2_realclose"
    turns = PersistingTurns(delay=0.01)

    async def body():
        a, b = _patches(user, turns)
        with a, b:
            ws = CutWS("tabRealClose1")
            t = await _open(ws)
            sid = _ready(ws)["session_id"]
            for i in range(8):
                ws.push({"type": "text", "content": f"t{i}"})
                await _until(lambda i=i: len(ws.frames("done")) == i + 1)
            await _close(ws, t)
            row = await _stored(user, sid)
            assert row.data["status"] == "completed"
            assert len(row.data["messages"]) == 12, "finalisation compacts to the last 12"
            assert ws.close_code is None, "an ordinary disconnect is not a 1012 cut"
            assert stage_b == [sid]

    _run(body)


def test_no_cut_when_no_ceiling_is_configured_in_local_mode() -> None:
    """Local default: TURTLE_WS_MAX_DURATION_S unset -> no ceiling (no behaviour change)."""
    assert "ws_max_duration_s" not in settings.model_fields_set
    assert settings.is_cloud is False
    assert server._ws_max_duration_s() is None


# ---------------------------------------------------------------------------
# Admission is per turn: it uses the turn's own deadline.
# ---------------------------------------------------------------------------
def test_budget_admits_only_turns_that_can_finish(monkeypatch) -> None:
    monkeypatch.setattr(server, "_TURN_DEADLINE_TEXT_S", 60.0)
    monkeypatch.setattr(server, "_TURN_DEADLINE_VOICE_S", 15.0)
    monkeypatch.setattr(server, "_WS_CUT_MARGIN_S", 5.0)

    async def body():
        text = server._PendingTurn("text", "hi", 16000)
        b = server._ConnectionBudget(300.0)
        assert b.admit(text, voice=False)
        b.started -= 234  # 66 s left: covers 60 + 5
        assert b.admit(text, voice=False)
        b.started -= 2    # 64 s left: does not
        assert not b.admit(text, voice=False)
        assert b.requested and b.reason == "max_duration"
        assert b.unstarted == [{"kind": "text", "content": "hi"}]

        b = server._ConnectionBudget(300.0)
        assert b.admit(text, voice=False)
        b.started -= 270  # 30 s left: a voice turn (15 + 5) fits, a text turn does not
        assert b.admit(server._PendingTurn("audio", b"\x01\x02", 16000), voice=True)
        assert not b.admit(text, voice=False)

        # a ceiling shorter than one turn must not reconnect forever
        b = server._ConnectionBudget(10.0)
        assert b.admit(text, voice=False), "the first turn of a connection is always admitted"
        assert not b.admit(text, voice=False)

        # once a cut is requested nothing more starts; audio is returned base64
        b = server._ConnectionBudget(None)
        b.request("lease_lost")
        assert not b.admit(server._PendingTurn("audio", b"\x01\x02", 8000), voice=True)
        assert b.unstarted == [{
            "kind": "audio", "data": base64.b64encode(b"\x01\x02").decode(), "sample_rate": 8000,
        }]

    _run(body)


# ---------------------------------------------------------------------------
# Streamed-mic turns go through the same gate.
# ---------------------------------------------------------------------------
def test_streamed_mic_utterance_is_deferred_not_dropped_when_it_cannot_finish() -> None:
    class Ev:
        def __init__(self, kind, transcript=""):
            self.kind, self.transcript, self.raw = kind, transcript, None

    class FakeStt:
        def events(self):
            async def gen():
                yield Ev("end_of_turn", "book a table")
            return gen()

    async def body():
        ws = CutWS()
        budget = server._ConnectionBudget(1.0)
        budget.started -= 100  # long past the ceiling
        budget.admitted = 1
        session = server._MicStreamSession(
            FakeStt(), {"messages": None}, sample_rate=16000,
            finishing=asyncio.Event(), turn_started=asyncio.Event(),
        )
        session.admit = lambda text: budget.admit(
            server._PendingTurn("text", text, 16000), voice=True, source="mic")
        ran = []

        async def must_not_run(*a, **k):
            ran.append(1)

        with patch.object(server, "_run_streamed_turn", new=must_not_run):
            await asyncio.wait_for(server._flux_mic_consumer(ws, object(), session), timeout=5)
        assert ran == [], "a refused utterance started a turn"
        assert ws.frames("transcription") == [], "no user bubble for an utterance that did not run"
        assert budget.requested
        assert budget.unstarted == [{"kind": "text", "content": "book a table", "source": "mic"}]

    _run(body)


# ---------------------------------------------------------------------------
# Resume: replay of buffered results, ownership check, turn-id continuity.
# ---------------------------------------------------------------------------
def test_resume_replays_only_the_callers_turns_after_last_turn_id() -> None:
    user = "usr_p6b2_replay"

    async def body():
        sid = "turtle_session_replaytest"
        for n in (1, 2, 3):
            await turn_results.put(sid, f"{sid}_turn_{n}", {
                "user_id": user, "session_id": sid, "voice": False, "transcript": None,
                "frame": {"type": "done", "content": f"ans{n}", "tool_urls": [],
                          "turn_id": f"{sid}_turn_{n}"},
            })
        # someone else's turn 4 in the same session id: must never be replayed
        await turn_results.put(sid, f"{sid}_turn_4", {
            "user_id": "usr_other", "session_id": sid, "voice": False, "transcript": None,
            "frame": {"type": "done", "content": "SECRET", "tool_urls": [],
                      "turn_id": f"{sid}_turn_4"},
        })
        turns = PersistingTurns()
        a, b = _patches(user, turns)
        with a, b:
            ws = CutWS("tabReplay0001")
            t = await _open(ws)
            ws.push({"type": "resume", "session_id": sid, "last_turn_id": f"{sid}_turn_1"})
            await _until(lambda: ws.frames("resumed"))
            await _until(lambda: len(ws.frames("done")) == 2)
            assert [f["content"] for f in ws.frames("done")] == ["ans2", "ans3"]
            res = ws.frames("resumed")[0]
            assert res["replayed"] == 2 and res["same_session"] is False
            assert all(f["content"] != "SECRET" for f in ws.frames("done"))
            await _close(ws, t)

    _run(body)


def test_replayed_voice_turn_redraws_the_user_and_resynthesises_audio() -> None:
    user = "usr_p6b2_voice"

    async def body():
        sid = "turtle_session_voicereplay"
        await turn_results.put(sid, f"{sid}_turn_1", {
            "user_id": user, "session_id": sid, "voice": True, "transcript": "what time is it",
            "frame": {"type": "done", "content": "It is noon.", "tool_urls": [],
                      "turn_id": f"{sid}_turn_1"},
        })

        async def fake_tts(text, **kw):
            assert text.strip() == "It is noon."
            yield "s0", b"AUDIO"

        turns = PersistingTurns()
        a, b = _patches(user, turns)
        with a, b, patch("core.streaming_tts.stream_tts_from_text", new=fake_tts):
            ws = CutWS("tabVoice00001")
            t = await _open(ws)
            ws.push({"type": "resume", "session_id": sid, "last_turn_id": None})
            await _until(lambda: [p for _, k, p in ws.events if k == "bytes"])
            assert [p for _, k, p in ws.events if k == "bytes"] == [b"AUDIO"]
            assert [f["text"] for f in ws.frames("transcription")] == ["what time is it"]
            await _close(ws, t)

    _run(body)


def test_turn_ids_continue_after_resume_and_do_not_overwrite_buffered_results() -> None:
    """Without the persisted counter a resumed session restarts at _turn_1 and
    overwrites turtle:turn_result:{sid}:{sid}_turn_1."""
    user = "usr_p6b2_ids"

    async def body():
        turns = PersistingTurns(delay=0.0)
        a, b = _patches(user, turns)
        with a, b:
            ws1 = CutWS("tabIds0000001")
            t1 = await _open(ws1)
            sid = _ready(ws1)["session_id"]
            for i in range(3):
                ws1.push({"type": "text", "content": f"a{i}"})
                await _until(lambda i=i: len(ws1.frames("done")) == i + 1)
            ids1 = [f["turn_id"] for f in ws1.frames("done")]
            # Simulate an unplanned drop that left the session resumable (a crash
            # skips the finaliser): cancel without letting finally finalise.
            row = await _until_stored(user, sid, 6)
            assert row.data["turn_counter"] == 3
            await _close(ws1, t1)

            # The finaliser completed the row; put it back to `active` the way a
            # crash/planned cut leaves it, so this test isolates id continuity.
            row = await _stored(user, sid)
            row.data["status"] = "active"
            await SessionStore(user_id=user).backend.put(row)

            ws2 = CutWS("tabIds0000001")
            t2 = await _open(ws2)
            assert _ready(ws2)["session_id"] == sid
            ws2.push({"type": "text", "content": "b0"})
            await _until(lambda: ws2.frames("done"))
            ids2 = [f["turn_id"] for f in ws2.frames("done")]
            assert ids2 == [f"{sid}_turn_4"], (ids1, ids2)
            await _close(ws2, t2)
            first = await turn_results.after(sid, None)
            assert [r["frame"]["content"] for r in first][:3] == ["re:a0", "re:a1", "re:a2"]

    _run(body)


# ---------------------------------------------------------------------------
# Session lease (6.4) at the endpoint.
# ---------------------------------------------------------------------------
def test_second_tab_gets_its_own_session_and_the_first_is_not_finalised() -> None:
    """Reproduces the recon's two-tab clobber: before the lease, tab B resumed
    A's session and A's disconnect finalised it under B."""
    user = "usr_p6b2_twotabs"
    turns = PersistingTurns(delay=0.01)

    async def body():
        a, b = _patches(user, turns)
        with a, b:
            wsA = CutWS("tabAAAAAAAA1")
            tA = await _open(wsA)
            sidA = _ready(wsA)["session_id"]
            wsA.push({"type": "text", "content": "from A"})
            await _until(lambda: wsA.frames("done"))

            wsB = CutWS("tabBBBBBBBB1")
            tB = await _open(wsB)
            sidB = _ready(wsB)["session_id"]
            assert sidB != sidA, "tab B resumed tab A's leased session"
            assert not [f for f in wsB.frames("status") if f.get("status") == "restored"]
            # B's connect-time demotion/sweep must not have touched A's session.
            rowA = await _until_stored(user, sidA, 2)
            assert rowA.data["status"] == "active"

            await _close(wsA, tA)  # A leaves: finalises ITS session only
            rowA = await _stored(user, sidA)
            rowB = await _stored(user, sidB)
            assert rowA.data["status"] == "completed"
            assert rowB.data["status"] == "active", "A's disconnect finalised B's session"

            wsB.push({"type": "text", "content": "from B"})
            await _until(lambda: wsB.frames("done"))
            await _close(wsB, tB)

    _run(body)


def test_same_tab_reconnecting_over_its_own_stale_lease_resumes_its_session() -> None:
    """Unclean drop: the dead connection's lease is still held (90 s TTL) when the
    SAME tab reconnects. It must take it over, not open a new session."""
    user = "usr_p6b2_stale"
    turns = PersistingTurns(delay=0.01)

    async def body():
        a, b = _patches(user, turns)
        with a, b:
            old = CutWS("tabStale00001")
            told = await _open(old)
            sid = _ready(old)["session_id"]
            old.push({"type": "text", "content": "before"})
            await _until(lambda: old.frames("done"))
            await _until_stored(user, sid, 2)  # persistence follows `done`
            # `old` is NOT closed: its server task is still alive, lease held.
            new = CutWS("tabStale00001")
            tnew = await _open(new)
            assert _ready(new)["session_id"] == sid, "the tab was locked out by its own stale lease"
            assert [f["message_count"] for f in new.frames("status") if f.get("status") == "restored"] == [2]

            # The superseded connection finds out on its next ping, stops writing
            # and leaves WITHOUT finalising the session the new one now owns.
            old.push({"type": "ping"})
            await asyncio.wait_for(told, timeout=10)
            assert old.close_code == 1012
            assert old.frames("status")[-1]["reason"] == "lease_lost"
            row = await _stored(user, sid)
            assert row.data["status"] == "active", "the superseded connection finalised the live session"

            new.push({"type": "text", "content": "after"})
            await _until(lambda: new.frames("done"))
            row = await _until_stored(user, sid, 4)
            await _close(new, tnew)

    _run(body)


def test_ping_refreshes_the_lease() -> None:
    user = "usr_p6b2_refresh"
    turns = PersistingTurns()

    async def body():
        a, b = _patches(user, turns)
        with a, b:
            ws = CutWS("tabRefresh0001")
            t = await _open(ws)
            sid = _ready(ws)["session_id"]
            key = turn_lock_mod.session_lease_key(sid)
            be = turn_lock_mod._in_process_backend
            token, expiry = be._held[key]
            be._held[key] = (token, time.monotonic() + 1.0)  # nearly expired
            ws.push({"type": "ping"})
            await _until(lambda: ws.frames("pong"))
            await asyncio.sleep(0.05)
            assert be._held[key][1] - time.monotonic() > 60, "ping did not refresh the lease"
            await _close(ws, t)
            assert key not in be._held, "the lease must be released on disconnect"

    _run(body)


def test_lease_is_refreshed_by_a_ping_that_arrives_while_a_long_turn_runs() -> None:
    """A turn can outlast the 90 s lease; the receive loop (ledger 6.1) still
    reads pings mid-turn, and that ping is what keeps the lease alive."""
    user = "usr_p6b2_longturn"
    turns = PersistingTurns(delay=0.01, delays={"long": 1.5})

    async def body():
        a, b = _patches(user, turns)
        with a, b:
            ws = CutWS("tabLongTurn001")
            t = await _open(ws)
            sid = _ready(ws)["session_id"]
            key = turn_lock_mod.session_lease_key(sid)
            be = turn_lock_mod._in_process_backend
            ws.push({"type": "text", "content": "long"})
            await _until(lambda: "long" in turns.texts)
            token, _ = be._held[key]
            be._held[key] = (token, time.monotonic() + 1.0)
            ws.push({"type": "ping"})
            await _until(lambda: ws.frames("pong"))
            await asyncio.sleep(0.05)
            assert not ws.frames("done"), "the ping must be handled MID-turn"
            assert be._held[key][1] - time.monotonic() > 60
            await _until(lambda: ws.frames("done"))
            await _close(ws, t)

    _run(body)


def test_connect_sweep_never_finalises_or_compacts_a_leased_session() -> None:
    """A session can be flipped to pending_finalization under a live owner (a
    channel's new-session branch, a stale-window demotion). The newcomer's
    connect-time sweep would then run Stage B and mark_finalized (compacting) it
    -- ledger 6.4: finalisation and compaction run only for the lease holder."""
    user = "usr_p6b2_sweep"
    turns = PersistingTurns(delay=0.01)

    async def body():
        a, b = _patches(user, turns)
        with a, b:
            wsA = CutWS("tabSweepAAAA1")
            tA = await _open(wsA)
            sidA = _ready(wsA)["session_id"]
            for i in range(8):
                wsA.push({"type": "text", "content": f"a{i}"})
                await _until(lambda i=i: len(wsA.frames("done")) == i + 1)
            await _until_stored(user, sidA, 16)
            row = await _stored(user, sidA)
            row.data["status"] = "pending_finalization"  # demoted under A's feet
            await SessionStore(user_id=user).backend.put(row)

            wsB = CutWS("tabSweepBBBB1")
            tB = await _open(wsB)
            assert _ready(wsB)["session_id"] != sidA
            row = await _stored(user, sidA)
            assert row.data["status"] == "pending_finalization", "the sweep finalised a leased session"
            assert len(row.data["messages"]) == 16, "the sweep compacted a leased session"
            await _close(wsB, tB)
            await _close(wsA, tA)

    _run(body)
