"""
test/p6_b1_turn_tasks_test.py
------------------------------
P6-B1 (ledger 6.1 + 6.3): the WebSocket receive loop never awaits a turn.

Each turn is an asyncio task; ``interrupt`` cancels it and emits ``interrupted``;
a second message while a turn runs is queued ONE deep (a third replaces it);
and one user's turns -- across connections and surfaces -- are serialised by a
per-user lock.

These tests drive the REAL ``apps.turtle_server.websocket_endpoint`` (local mode:
real stores under the suite's throwaway TURTLE_DATA_DIR) with a scripted
duck-typed socket. Only the model turn (``_execute_turn``) and auth are faked --
the receive loop, the send lock, the turn tasks, the queue and the turn lock
are all production code.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable
from unittest.mock import AsyncMock, patch

import pytest

import apps.turtle_server as server
from core import turn_lock as turn_lock_mod

USER = "usr_p6b1_turns"


class ScriptedWS:
    """Starlette WebSocket stand-in whose writes take real (event-loop) time.

    ``_write`` yields mid-write, exactly like a real socket under backpressure,
    so two un-serialised writers genuinely overlap and ``max_writers`` exceeds 1.
    """

    def __init__(self) -> None:
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.events: list[tuple[float, str, Any]] = []
        self.writers = 0
        self.max_writers = 0

    async def accept(self) -> None:
        return None

    async def receive(self) -> dict:
        return await self.inbox.get()

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        return None

    async def send_json(self, data: dict) -> None:
        await self._write("json", data)

    async def send_bytes(self, data: bytes) -> None:
        await self._write("bytes", data)

    async def _write(self, kind: str, payload: Any) -> None:
        self.writers += 1
        self.max_writers = max(self.max_writers, self.writers)
        try:
            await asyncio.sleep(0.005)
            self.events.append((time.monotonic(), kind, payload))
        finally:
            self.writers -= 1

    # -- test helpers ------------------------------------------------------
    def push(self, obj: dict) -> None:
        self.inbox.put_nowait({"type": "websocket.receive", "text": json.dumps(obj)})

    def disconnect(self) -> None:
        self.inbox.put_nowait({"type": "websocket.disconnect"})

    def frames(self, type_: str | None = None) -> list[dict]:
        out = [p for _, k, p in self.events if k == "json"]
        return [f for f in out if type_ is None or f.get("type") == type_]

    def time_of(self, pred: Callable[[dict], bool]) -> float | None:
        for t, k, p in self.events:
            if k == "json" and pred(p):
                return t
        return None


async def _until(cond: Callable[[], bool], timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.01)


class FakeTurns:
    """Stand-in for ``_execute_turn``: slow, cancellable, records overlap."""

    def __init__(self, durations: dict[str, float] | None = None) -> None:
        self.durations = durations or {}
        self.calls: list[tuple[str, Any]] = []
        self.active = 0
        self.max_active = 0
        self.started: list[str] = []

    async def __call__(self, ws, state, text, history, *, channel, send_status=True):
        self.calls.append((text, history))
        self.started.append(text)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await server._ws_send_json(ws, {"type": "status", "status": "thinking"})
            await asyncio.sleep(self.durations.get(text, 0.05))
            await server._ws_send_json(ws, {"type": "done", "content": f"re:{text}"})
            return server.TurnOutcome((history or []) + [text], f"re:{text}", f"re:{text}")
        finally:
            self.active -= 1


@pytest.fixture(autouse=True)
def _fresh_local_lock():
    turn_lock_mod._in_process_backend._held.clear()
    yield
    turn_lock_mod._in_process_backend._held.clear()


def _run(body):
    asyncio.run(body())


async def _connect(ws: ScriptedWS) -> "asyncio.Task":
    task = asyncio.create_task(server.websocket_endpoint(ws))
    await _until(lambda: any(f.get("status") == "ready" for f in ws.frames("status")))
    return task


async def _close(ws: ScriptedWS, task: "asyncio.Task") -> None:
    ws.disconnect()
    await asyncio.wait_for(task, timeout=10)


def _patches(turns: FakeTurns):
    return (
        patch.object(server, "authenticate_websocket", new=AsyncMock(return_value=USER)),
        patch.object(server, "_execute_turn", new=turns),
    )


# ---------------------------------------------------------------------------
# 6.1 headline: interrupt is readable during a turn on the DEFAULT text path
# ---------------------------------------------------------------------------
def test_interrupt_mid_turn_emits_interrupted_fast_with_no_done_and_history_unchanged() -> None:
    turns = FakeTurns({"first": 5.0, "second": 0.05})

    async def body():
        ws = ScriptedWS()
        a, b = _patches(turns)
        with a, b:
            task = await _connect(ws)
            ws.push({"type": "text", "content": "first"})
            await _until(lambda: turns.active == 1)
            await asyncio.sleep(0.2)
            t_sent = time.monotonic()
            ws.push({"type": "interrupt"})
            await _until(lambda: ws.frames("interrupted"), timeout=3)
            t_int = ws.time_of(lambda f: f.get("type") == "interrupted")
            assert t_int - t_sent < 0.1, f"interrupted took {t_int - t_sent:.3f}s"
            assert ws.frames("done") == [], "an interrupted turn must not emit done"
            await _until(lambda: {"type": "status", "status": "ready"} in ws.frames("status"))
            await _until(lambda: turns.active == 0)

            # History is unchanged: the NEXT turn is handed the ORIGINAL (empty)
            # history, not anything the cancelled turn produced.
            ws.push({"type": "text", "content": "second"})
            await _until(lambda: ws.frames("done"))
            assert turns.calls[1][0] == "second"
            assert not turns.calls[1][1], f"history leaked from cancelled turn: {turns.calls[1][1]!r}"
            await _close(ws, task)

    _run(body)


def test_receive_loop_stays_live_during_a_turn() -> None:
    """ping is answered while a turn is still running (the loop never awaits it)."""
    turns = FakeTurns({"slow": 1.0})

    async def body():
        ws = ScriptedWS()
        a, b = _patches(turns)
        with a, b:
            task = await _connect(ws)
            ws.push({"type": "text", "content": "slow"})
            await _until(lambda: turns.active == 1)
            ws.push({"type": "ping"})
            await _until(lambda: ws.frames("pong"), timeout=0.5)
            assert ws.frames("done") == [], "pong must arrive BEFORE the turn finished"
            await _until(lambda: ws.frames("done"))
            await _close(ws, task)

    _run(body)


# ---------------------------------------------------------------------------
# 6.1: queued one deep, a third replaces the queued one
# ---------------------------------------------------------------------------
def test_second_message_is_queued_third_replaces_and_nothing_runs_concurrently() -> None:
    turns = FakeTurns({"A": 0.4, "B": 0.05, "C": 0.05})

    async def body():
        ws = ScriptedWS()
        a, b = _patches(turns)
        with a, b:
            task = await _connect(ws)
            ws.push({"type": "text", "content": "A"})
            await _until(lambda: turns.active == 1)
            ws.push({"type": "text", "content": "B"})
            ws.push({"type": "text", "content": "C"})
            await _until(lambda: len(ws.frames("turn_queued")) == 2)
            assert [f["content"] for f in ws.frames("turn_queued")] == ["B", "C"]
            assert turns.started == ["A"], "queued message must not start while A runs"

            await _until(lambda: len(ws.frames("done")) == 2)
            assert turns.started == ["A", "C"], "B was replaced; C must run after A"
            assert turns.max_active == 1, "turns ran concurrently"
            # C continues from A's history (queued turn picks up the new history).
            assert turns.calls[1][1] == ["A"]
            await _close(ws, task)

    _run(body)


def test_interrupt_cancels_running_turn_and_clears_queue() -> None:
    turns = FakeTurns({"A": 5.0, "B": 0.05})

    async def body():
        ws = ScriptedWS()
        a, b = _patches(turns)
        with a, b:
            task = await _connect(ws)
            ws.push({"type": "text", "content": "A"})
            await _until(lambda: turns.active == 1)
            ws.push({"type": "text", "content": "B"})
            await _until(lambda: ws.frames("turn_queued"))
            ws.push({"type": "interrupt"})
            await _until(lambda: ws.frames("interrupted"))
            await asyncio.sleep(0.3)
            assert turns.started == ["A"], "an interrupt must not let the queued turn run"
            assert ws.frames("turn_queue_cleared"), "client must be told the pending turn is gone"
            await _close(ws, task)

    _run(body)


# ---------------------------------------------------------------------------
# No interleaving on the socket: send_bytes shares the JSON send lock
# ---------------------------------------------------------------------------
def test_send_bytes_does_not_interleave_with_json_frames() -> None:
    async def body():
        ws = ScriptedWS()

        async def fake_tts(text, **kw):
            for i in range(5):
                yield f"s{i}", b"\x00" * 8

        async def fake_turn(ws_, state, text, history, *, channel, send_status=True):
            return server.TurnOutcome(history, "hello there. general kenobi.", "hello")

        with patch.object(server, "_execute_turn", new=fake_turn), \
                patch.object(server, "_voice_stream_llm_enabled", return_value=False), \
                patch("core.streaming_tts.stream_tts_from_text", new=fake_tts):
            speaking = asyncio.create_task(server._reply_and_speak(
                ws, object(), "hi", None, timings={}, overall_start=time.time(),
            ))
            # The receive loop answering pong/interrupted while audio streams.
            noise = [asyncio.create_task(server._ws_send_json(ws, {"type": "pong"})) for _ in range(20)]
            for _ in range(20):
                await asyncio.sleep(0.003)
                noise.append(asyncio.create_task(server._ws_send_json(ws, {"type": "pong"})))
            await asyncio.gather(speaking, *noise)
        assert [k for _, k, _ in ws.events].count("bytes") == 5
        assert ws.max_writers == 1, f"{ws.max_writers} writers were on the socket at once"

    _run(body)


# ---------------------------------------------------------------------------
# 6.3: the per-user lock
# ---------------------------------------------------------------------------
def test_second_connection_for_same_user_is_rejected_while_first_turn_runs(monkeypatch) -> None:
    monkeypatch.setattr(turn_lock_mod, "TURN_LOCK_WAIT_S", 0.3)
    turns = FakeTurns({"one": 1.0, "two": 0.05})

    async def body():
        ws1, ws2 = ScriptedWS(), ScriptedWS()
        a, b = _patches(turns)
        with a, b:
            t1 = await _connect(ws1)
            t2 = await _connect(ws2)
            ws1.push({"type": "text", "content": "one"})
            await _until(lambda: turns.active == 1)
            ws2.push({"type": "text", "content": "two"})
            await _until(lambda: ws2.frames("error"), timeout=3)
            err = ws2.frames("error")[0]
            assert err["code"] == "turn_in_progress"
            assert turns.started == ["one"], "the rejected turn must not run"
            await _until(lambda: {"type": "status", "status": "ready"} in ws2.frames("status"))

            await _until(lambda: ws1.frames("done"))
            ws2.push({"type": "text", "content": "two"})
            await _until(lambda: ws2.frames("done"))
            assert turns.max_active == 1
            await _close(ws1, t1)
            await _close(ws2, t2)

    _run(body)


def test_lock_is_released_when_the_turn_is_interrupted() -> None:
    turns = FakeTurns({"A": 5.0})

    async def body():
        ws = ScriptedWS()
        a, b = _patches(turns)
        with a, b:
            task = await _connect(ws)
            ws.push({"type": "text", "content": "A"})
            await _until(lambda: turns.active == 1)
            held = turn_lock_mod.turn_lock(USER, "probe", wait_s=0)
            async with held:
                assert held.busy, "lock must be held while the turn runs"
            ws.push({"type": "interrupt"})
            await _until(lambda: ws.frames("interrupted"))
            await _until(lambda: turns.active == 0)
            probe = turn_lock_mod.turn_lock(USER, "probe", wait_s=0)
            async with probe:
                assert probe.acquired, "an interrupted turn leaked the lock for its full TTL"
            await _close(ws, task)

    _run(body)


# ---------------------------------------------------------------------------
# 6.3 on the CHANNEL surface: a channel turn and a web turn never interleave
# ---------------------------------------------------------------------------
def test_channel_turn_is_rejected_while_a_web_turn_holds_the_user_lock(monkeypatch) -> None:
    from types import SimpleNamespace

    from apps.channels import TurtleEvent

    monkeypatch.setattr(turn_lock_mod, "TURN_LOCK_WAIT_S", 0.2)
    ran: list[str] = []

    async def fake_exec(ws, state, text, history, *, channel, send_status=True):
        ran.append(text)
        return server.TurnOutcome(history, "hi", "hi")

    state = SimpleNamespace(
        session_store=SimpleNamespace(message_history=None),
        confirmation_gate=SimpleNamespace(next_prompt=lambda: None),
        channel="", channel_user_id="", channel_is_private=False, http_client=None,
    )
    event = TurtleEvent(user_id=USER, channel="discord", modality="text", content="hello")

    async def body():
        server._CHANNEL_STATES[(USER, "discord")] = (state, time.monotonic())
        try:
            with patch.object(server, "_execute_turn", new=fake_exec):
                async with turn_lock_mod.turn_lock(USER, "web:other") as web:
                    assert web.acquired
                    reply = await server._channel_dispatch_handler(event)
                assert reply.content == turn_lock_mod.BUSY_MESSAGE
                assert ran == [], "the channel turn must not run while a web turn holds the lock"
                # Holder gone: the same channel message now runs.
                reply = await server._channel_dispatch_handler(event)
                assert ran == ["hello"]
        finally:
            server._CHANNEL_STATES.pop((USER, "discord"), None)

    _run(body)
