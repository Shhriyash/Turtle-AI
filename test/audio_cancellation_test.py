"""
P6-A1 (ledger 6.2): task cancellation must reach the audio layer.

The defect: ``stream_tts_from_*`` shielded each per-sentence synth task, had no
try/finally, and the synth itself ran a *sync* Deepgram/Groq call in an executor
thread. Cancelling the turn raised ``CancelledError`` in the outer await but
left every synth task (and its HTTP request, and its thread) running.

These tests do NOT mock the provider call. A real TCP server accepts the
connection and never answers; the real SDK client (sync pre-fix, async post-fix)
talks to it. "Cancellation reached the audio layer" is therefore measured at the
only place that matters: the server sees the client's socket close.

Not verifiable offline: a live Deepgram/Groq cancel (does the provider stop
billing, does their edge notice) needs real infrastructure.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import threading

import pytest


# ---------------------------------------------------------------------------
# A real server that accepts, reads the request, and never replies.
# ---------------------------------------------------------------------------

class _HangServer:
    def __init__(self) -> None:
        self.connected = 0
        self.open = 0
        self._server: asyncio.base_events.Server | None = None
        self._writers: list[asyncio.StreamWriter] = []

    async def start(self) -> int:
        self._server = await asyncio.start_server(self._on_conn, "127.0.0.1", 0)
        return self._server.sockets[0].getsockname()[1]

    async def _on_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connected += 1
        self.open += 1
        self._writers.append(writer)
        try:
            while True:
                data = await reader.read(65536)
                if not data:  # client closed its end
                    break
        except Exception:
            pass
        finally:
            self.open -= 1
            writer.close()

    async def stop(self) -> None:
        # Close every socket so a still-blocked client thread unblocks and the
        # event-loop shutdown (which joins the default executor) cannot hang.
        for w in list(self._writers):
            try:
                w.close()
            except Exception:
                pass
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()


async def _until(pred, timeout: float = 5.0) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return pred()


def _leftover_tasks() -> list[asyncio.Task]:
    """Pending tasks that are not the test itself or the hang server's plumbing."""
    me = asyncio.current_task()
    plumbing = ("_on_conn", "accept_coro", "wait_closed")
    out = []
    for t in asyncio.all_tasks():
        if t is me or t.done():
            continue
        if any(p in repr(t.get_coro()) for p in plumbing):
            continue
        out.append(t)
    return out


@pytest.fixture
def dg_hang(monkeypatch):
    """Point the real Deepgram SDK (sync AND async) at the hang server."""
    from deepgram.environment import DeepgramClientEnvironment

    monkeypatch.setenv("DEEPGRAM_API_KEY", "ci-dummy")
    monkeypatch.setenv("GROQ_API_KEY", "ci-dummy")
    monkeypatch.setenv("TTS_STREAM_USE_WS", "0")
    # Never fall through to Groq TTS during the test.
    monkeypatch.setattr(
        "core.openrouter_tts._synthesize_groq_bytes",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no groq in test")),
        raising=False,
    )

    def _point(port: int) -> None:
        monkeypatch.setattr(
            DeepgramClientEnvironment.PRODUCTION, "base", f"http://127.0.0.1:{port}"
        )

    return _point


SENTENCES = (
    "This is the first sentence of the reply. "
    "This is the second sentence of the reply. "
    "This is the third sentence of the reply."
)


# ---------------------------------------------------------------------------
# stream_tts_from_text
# ---------------------------------------------------------------------------

def test_cancelling_stream_tts_from_text_stops_every_synth_and_closes_sockets(dg_hang):
    from core.streaming_tts import stream_tts_from_text

    async def scenario():
        srv = _HangServer()
        dg_hang(await srv.start())
        try:
            async def consume():
                async for _ in stream_tts_from_text(SENTENCES, tts_timeout_s=30.0):
                    pass

            turn = asyncio.create_task(consume())
            assert await _until(lambda: srv.connected >= 3), "synth requests never started"
            turn.cancel()
            with pytest.raises(asyncio.CancelledError):
                await turn

            # The orphan check: nothing the generator spawned may survive.
            assert await _until(lambda: not _leftover_tasks(), 2.0), (
                f"orphaned synth tasks after cancel: {_leftover_tasks()!r}"
            )
            # The thing the ledger actually asks for: the HTTP request is closed.
            assert await _until(lambda: srv.open == 0, 3.0), (
                f"{srv.open} HTTP connection(s) still open after cancel"
            )
        finally:
            await srv.stop()

    asyncio.run(scenario())


def test_cancel_cleanup_lives_in_finally_not_except_exception():
    """CancelledError is a BaseException; cleanup in ``except Exception`` leaks.

    Structural guard on top of the behavioural test above: both generators must
    reach their cleanup via ``finally`` (or BaseException), and neither may
    still shield the synth task.
    """
    import core.streaming_tts as m

    for fn in (m.stream_tts_from_text, m.stream_tts_from_token_stream):
        src = inspect.getsource(fn)
        assert "asyncio.shield" not in src, f"{fn.__name__} still shields its synth task"
        assert "finally:" in src, f"{fn.__name__} has no finally cleanup"


def test_consumer_stuck_between_yields_then_aclose_cancels_outstanding():
    """Cancel while the generator is suspended at a ``yield`` (consumer busy
    sending the previous chunk). The generator never sees CancelledError there;
    ``aclose()`` is the only signal, and it must cancel the remaining tasks."""
    import core.streaming_tts as m

    started: list[str] = []
    cancelled: list[str] = []

    async def fake_synth(text, *, model=None, speed=None):
        started.append(text)
        if "first" in text:
            return b"RIFF-first"
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append(text)
            raise
        return b"never"

    async def scenario():
        orig = m._synthesize_sentence_async
        m._synthesize_sentence_async = fake_synth
        try:
            gen = m.stream_tts_from_text(SENTENCES, tts_timeout_s=30.0)
            first = await gen.__anext__()
            assert first[1] == b"RIFF-first"
            await gen.aclose()
            await asyncio.sleep(0)
            assert len(cancelled) == 2, f"siblings not cancelled on aclose: {cancelled}"
            assert not _leftover_tasks()
        finally:
            m._synthesize_sentence_async = orig

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# stream_tts_from_token_stream
# ---------------------------------------------------------------------------

def test_cancelling_token_stream_tts_stops_every_synth_and_closes_sockets(dg_hang):
    from core.streaming_tts import stream_tts_from_token_stream

    async def tokens():
        for word in SENTENCES.split(" "):
            yield word + " "
            await asyncio.sleep(0)
        # Keep the LLM "still generating" so cancel lands mid-stream.
        await asyncio.sleep(60)

    async def scenario():
        srv = _HangServer()
        dg_hang(await srv.start())
        try:
            async def consume():
                async for _ in stream_tts_from_token_stream(tokens(), tts_timeout_s=30.0):
                    pass

            turn = asyncio.create_task(consume())
            assert await _until(lambda: srv.connected >= 2), "synth requests never started"
            turn.cancel()
            with pytest.raises(asyncio.CancelledError):
                await turn
            assert await _until(lambda: not _leftover_tasks(), 2.0), (
                f"orphaned synth tasks after cancel: {_leftover_tasks()!r}"
            )
            assert await _until(lambda: srv.open == 0, 3.0), (
                f"{srv.open} HTTP connection(s) still open after cancel"
            )
        finally:
            await srv.stop()

    asyncio.run(scenario())


def test_token_stream_cancel_during_final_drain(dg_hang):
    """Cancel in the drain loop (token stream finished, awaiting the front task)."""
    from core.streaming_tts import stream_tts_from_token_stream

    async def tokens():
        for word in SENTENCES.split(" "):
            yield word + " "

    async def scenario():
        srv = _HangServer()
        dg_hang(await srv.start())
        try:
            async def consume():
                async for _ in stream_tts_from_token_stream(tokens(), tts_timeout_s=30.0):
                    pass

            turn = asyncio.create_task(consume())
            assert await _until(lambda: srv.connected >= 3)
            turn.cancel()
            with pytest.raises(asyncio.CancelledError):
                await turn
            assert await _until(lambda: not _leftover_tasks(), 2.0), _leftover_tasks()
            assert await _until(lambda: srv.open == 0, 3.0), srv.open
        finally:
            await srv.stop()

    asyncio.run(scenario())


def test_timeout_still_cancels_the_timed_out_synth(dg_hang):
    """Removing the shield must not lose the per-sentence timeout behaviour."""
    from core.streaming_tts import stream_tts_from_text

    async def scenario():
        srv = _HangServer()
        dg_hang(await srv.start())
        try:
            out = [x async for x in stream_tts_from_text(
                "Only one sentence here to time out.", tts_timeout_s=0.3
            )]
            assert out == []
            assert await _until(lambda: not _leftover_tasks(), 2.0), _leftover_tasks()
            assert await _until(lambda: srv.open == 0, 3.0), srv.open
        finally:
            await srv.stop()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Groq STT through the real voice entrypoint
# ---------------------------------------------------------------------------

class _FakeWS:
    def __init__(self) -> None:
        self.sent: list = []

    async def send_json(self, data, *a, **k):
        self.sent.append(data)

    async def send_text(self, data, *a, **k):
        self.sent.append(data)

    async def send_bytes(self, data, *a, **k):
        self.sent.append(data)


def test_cancelling_voice_turn_closes_the_groq_stt_request(monkeypatch):
    import numpy as np

    monkeypatch.setenv("GROQ_API_KEY", "ci-dummy")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "ci-dummy")

    import apps.turtle_server as ts
    from core.stt_fastrtc import FastRTCSTT

    audio = (np.arange(4000, dtype=np.int16) % 200).tobytes()

    async def scenario():
        srv = _HangServer()
        port = await srv.start()
        # Both groq.Groq and groq.AsyncGroq read GROQ_BASE_URL.
        monkeypatch.setenv("GROQ_BASE_URL", f"http://127.0.0.1:{port}")
        monkeypatch.setattr(ts.agents_mgr, "stt", FastRTCSTT())
        try:
            turn = asyncio.create_task(
                ts._handle_audio_message(_FakeWS(), object(), audio, None, sample_rate=16000)
            )
            assert await _until(lambda: srv.connected >= 1), "STT request never started"
            turn.cancel()
            with pytest.raises(asyncio.CancelledError):
                await turn
            assert await _until(lambda: srv.open == 0, 3.0), (
                f"{srv.open} Groq STT connection(s) still open after cancel"
            )
            assert await _until(lambda: not _leftover_tasks(), 2.0), _leftover_tasks()
        finally:
            await srv.stop()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Client caching (ledger 6.10, client half)
# ---------------------------------------------------------------------------

def test_async_clients_are_built_once_per_loop_with_explicit_timeouts(monkeypatch):
    import httpx

    monkeypatch.setenv("GROQ_API_KEY", "ci-dummy")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "ci-dummy")
    from tools.tts import client as c

    async def same_loop():
        g1, g2 = c.get_async_groq_client(), c.get_async_groq_client()
        d1, d2 = c.get_async_deepgram_client(), c.get_async_deepgram_client()
        assert g1 is g2 and d1 is d2
        assert isinstance(g1.timeout, httpx.Timeout)
        assert g1.timeout.read is not None and g1.timeout.connect is not None
        dg_http = d1._client_wrapper.httpx_client.httpx_client
        assert dg_http.timeout.connect is not None and dg_http.timeout.read is not None
        return g1, d1

    g_a, d_a = asyncio.run(same_loop())
    # A different loop must NOT reuse a client whose connections belong to a
    # closed loop (httpx pools are loop-bound).
    g_b, d_b = asyncio.run(same_loop())
    assert g_a is not g_b and d_a is not d_b


def test_async_client_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY2", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    from tools.tts import client as c

    async def go():
        with pytest.raises(RuntimeError):
            c.get_async_groq_client()
        with pytest.raises(RuntimeError):
            c.get_async_deepgram_client()

    asyncio.run(go())


# ---------------------------------------------------------------------------
# Flux streaming STT: start() cancelled mid-connect must not strand its thread
# ---------------------------------------------------------------------------

def test_flux_start_cancelled_signals_worker_to_stop(monkeypatch):
    from core.stt_streaming import FluxStreamingSTT

    monkeypatch.setenv("DEEPGRAM_API_KEY", "ci-dummy")
    release = threading.Event()

    def blocked_worker(self):
        # Stand-in for a sync websocket connect that is still in flight.
        release.wait(5)

    monkeypatch.setattr(FluxStreamingSTT, "_session_worker", blocked_worker)

    async def scenario():
        stt = FluxStreamingSTT()
        t = asyncio.create_task(stt.start(connect_timeout=30.0))
        await asyncio.sleep(0.1)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert stt._stop.is_set(), "cancelled start() left the session worker un-signalled"
        release.set()

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# Consumer-side wiring (ledger 6.2). The generators above reap their
# outstanding synth tasks in a `finally`, but a `finally` in an async
# generator only runs when the generator is CLOSED -- NOT when the consumer
# is cancelled mid-body. Demonstrated directly:
#
#   bare `async for`, consumer cancelled while awaiting in the loop body
#       -> the generator's finally does NOT run
#   same, wrapped in contextlib.aclosing
#       -> the finally runs immediately
#
# apps/turtle_server.py awaits ws.send_bytes() inside both TTS loops, which
# is precisely where a cancelled turn lands. So without aclosing the synth
# tasks outlive the cancelled turn and 6.2's reaping never fires -- the fix
# would be wired at the generator end only, which is this repo's single most
# repeated defect shape. These guard the consumer end.
# --------------------------------------------------------------------------

def test_bare_async_for_really_does_defer_generator_cleanup():
    """Pins the asyncio behaviour the aclosing wiring exists for.

    If a future Python made a bare `async for` close its generator on
    consumer cancellation, this test fails and the aclosing wrapping below
    becomes belt-and-braces rather than load-bearing. Either way we want to
    find out from a test rather than by shipping orphaned tasks.
    """
    import contextlib

    ran: list[str] = []

    async def gen():
        try:
            for i in range(5):
                yield i
        finally:
            ran.append("finally")

    async def drive(wrap: bool) -> list[str]:
        ran.clear()

        async def consume():
            if wrap:
                async with contextlib.aclosing(gen()) as g:
                    async for _ in g:
                        await asyncio.sleep(10)
            else:
                async for _ in gen():
                    await asyncio.sleep(10)

        t = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        return list(ran)

    assert asyncio.run(drive(wrap=False)) == [], (
        "a bare `async for` ran the generator's finally on cancel -- the "
        "aclosing wiring may no longer be needed, re-check 6.2"
    )
    assert asyncio.run(drive(wrap=True)) == ["finally"], (
        "aclosing failed to run the generator's finally on cancel"
    )


def test_turtle_server_consumes_both_tts_generators_under_aclosing():
    """Both consumer sites must close the generator deterministically.

    Structural rather than behavioural: driving the full streaming turn needs
    an LLM, a socket and a TTS provider. What can regress is someone
    unwrapping one of the two loops, so that is what this pins.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "apps" / "turtle_server.py"
    text = src.read_text(encoding="utf-8")

    # Imports are checked via the AST, not substring search: a comment
    # mentioning an import is not an import (this test tripped on its own
    # explanatory comment when it used `in text`).
    tree = ast.parse(text)
    imported: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.setdefault(node.module, set()).update(
                alias.name for alias in node.names
            )

    assert "aclosing" in imported.get("contextlib", set()), "aclosing import missing"
    assert "Groq" not in imported.get("groq", set()), (
        "the sync Groq import is dead after 6.2 moved STT to AsyncGroq"
    )

    for gen_name in ("stream_tts_from_text", "stream_tts_from_token_stream"):
        # The call must be the argument of an aclosing(...), not the iterable
        # of a bare `async for`.
        assert f"async for" not in text.split(f"{gen_name}(")[1].split(")")[0], (
            f"{gen_name} appears to be iterated directly"
        )
        idx = text.index(f"{gen_name}(\n") if f"{gen_name}(\n" in text else text.index(f"{gen_name}(")
        window = text[max(0, idx - 400):idx]
        assert "aclosing(" in window, (
            f"{gen_name} is consumed without contextlib.aclosing -- its synth "
            f"tasks would outlive a turn cancelled while awaiting ws.send_bytes"
        )
