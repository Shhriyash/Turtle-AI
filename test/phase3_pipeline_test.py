"""Phase 3 W3 — the channel-path-unification guarantee.

Every entrypoint (web text, web voice, and every channel adapter) now funnels
through ONE canonical turn pipeline, ``apps.turtle_server._execute_turn``. This
module drives that pipeline directly, fully offline (no LLM, no network, no live
keys), with a fake websocket that records frames, a stubbed model call
(``run_agent_with_fallbacks``), and a minimal ``SharedState`` built from fakes.

It pins the properties the unification has to preserve/grant:
  (a) the per-turn trace span "turtle.turn" is emitted for web, web_voice AND a
      channel dispatch (each carrying its ``channel`` attribute);
  (b) the ``ws=None`` path (how a channel adapter drives the pipeline) completes
      without error and still returns the reply;
  (c) the confirmation sidecar frame is emitted when the gate has a pending
      prompt (voice thereby gains the sidecar the old audio handler lacked);
  (d) the turn's history is persisted through the session store;
  (e) a model-call failure is reported through ``_classify_handler_error`` — a
      structured error frame with a code, never the raw exception string.

The turn pipeline is now a single ``run_agent_with_fallbacks`` call — no router,
no graph layer, no chat-intercept of a bare "yes" (a confirming turn reaches the
model like any other and is auto-promoted in the memory funnel instead).

The side-effect funnels (``_apply_explicit_facts_from_turn`` /
``_queue_confirmation_candidates_from_turn``) are owned by a concurrent
workstream and can spawn background tasks / touch the LLM, so they are
monkeypatched to no-ops here — this test is about pipeline *wiring*, not the
memory write funnel (that is pinned by phase2_e2e_memory_test.py).
"""
from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

import apps.turtle_server as ts
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeWS:
    """Records every frame the pipeline emits."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send_json(self, data: dict) -> None:
        self.frames.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append({"type": "__bytes__", "n": len(data)})

    def of_type(self, t: str) -> list[dict]:
        return [f for f in self.frames if f.get("type") == t]


class FakeSessionStore:
    """Just the surface _execute_turn touches: session_id + history + replace +
    the pending-email peek (used by the pending-email task_type bypass)."""

    def __init__(self, session_id: str = "s_test", pending_email: dict | None = None) -> None:
        self.session_id = session_id
        self.message_history: list = []
        self.replace_calls: list[list] = []
        self._pending_email = pending_email or {
            "recipients": [], "cc_recipients": [], "bcc_recipients": [],
            "subject": "", "content": "",
        }

    def get_pending_email(self) -> dict:
        return self._pending_email

    async def replace_messages(self, messages: list) -> None:
        self.message_history = list(messages)
        self.replace_calls.append(list(messages))


class FakeRAG:
    def __init__(self) -> None:
        self.conversations: list[tuple[str, str]] = []

    def add_conversation(self, user_text: str, output: str) -> None:
        self.conversations.append((user_text, output))


class StubGate:
    """Confirmation gate stand-in. ``prompt`` drives the sidecar path."""

    def __init__(self, prompt=None) -> None:
        self._prompt = prompt

    def next_prompt(self):
        return self._prompt

    def record_response(self, event_id, accepted):  # pragma: no cover
        return True


class FakeResponse:
    def __init__(self, output: str, msgs: list) -> None:
        self.output = output
        self._msgs = msgs

    def new_messages(self) -> list:
        return self._msgs

    def all_messages(self) -> list:
        return self._msgs


class SpanRecorder:
    """Stand-in for trace_sink — records every span name + attributes."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, dict]] = []

    @contextlib.contextmanager
    def span(self, name: str, **kwargs):
        self.spans.append((name, kwargs))
        yield None

    def channels_for(self, name: str) -> list[str]:
        return [kw.get("channel") for n, kw in self.spans if n == name]

    def intents_for(self, name: str) -> list[str]:
        return [kw.get("intent") for n, kw in self.spans if n == name]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def env(monkeypatch):
    """Neutralise every offline-hostile side effect and record the span sink."""
    recorder = SpanRecorder()
    monkeypatch.setattr(ts, "trace_sink", recorder)
    # Never touch Logfire in the model or error path.
    monkeypatch.setattr(ts, "_logfire_loaded", False)
    # Owned by a concurrent workstream / can spawn bg tasks — no-op here.
    monkeypatch.setattr(ts, "_apply_explicit_facts_from_turn", lambda *a, **k: None)
    monkeypatch.setattr(ts, "_queue_confirmation_candidates_from_turn", lambda *a, **k: 0)

    emits: list[tuple] = []
    def _emit_once(user_id, event, **fields):
        emits.append((user_id, event, fields))
        return True
    monkeypatch.setattr(ts, "emit_event_once", _emit_once)

    return SimpleNamespace(recorder=recorder, emits=emits)


def _make_state(gate=None, session_store=None) -> ts.SharedState:
    return ts.SharedState(
        http_client=None,
        session_store=session_store or FakeSessionStore(),
        personal_memory_store=None,
        personal_memory_prompt=None,
        journal_store=None,
        confirmation_gate=gate or StubGate(),
        task_history_store=None,
        rag_system=FakeRAG(),
        retrieval_broker=None,
        reflector=None,
        user_id="u_test",
    )


def _use_model(monkeypatch, output: str = "stub reply", raises: Exception | None = None) -> None:
    """Stub the ONE model call the pipeline now makes: run_agent_with_fallbacks."""
    async def _fake_run(primary_agent, fallback_agents, prompt, **kwargs):
        if raises is not None:
            raise raises
        msgs = [
            ModelRequest(parts=[UserPromptPart(content=prompt)]),
            ModelResponse(parts=[TextPart(content=output)]),
        ]
        return FakeResponse(output, msgs)
    monkeypatch.setattr(ts, "run_agent_with_fallbacks", _fake_run)


# ---------------------------------------------------------------------------
# (a) trace span per channel
# ---------------------------------------------------------------------------

def test_web_turn_emits_span_and_persists(env, monkeypatch):
    _use_model(monkeypatch, output="hi there")
    state = _make_state()
    ws = FakeWS()

    outcome = asyncio.run(ts._execute_turn(ws, state, "what's up", None, channel="web"))

    # (a) span emitted for web
    assert "web" in env.recorder.channels_for("turtle.turn")
    # (d) history persisted through the session store
    assert state.session_store.replace_calls, "replace_messages never called"
    assert state.session_store.message_history, "session store history is empty"
    # terminal done frame carries the model output
    done = ws.of_type("done")
    assert done and done[0]["content"] == "hi there"
    # first-message analytics fired with the channel
    assert ("u_test", "first_message_sent", {"channel": "web"}) in env.emits
    assert outcome.output_text == "hi there"
    assert outcome.reply_text == "hi there"


def test_voice_turn_emits_span(env, monkeypatch):
    _use_model(monkeypatch, output="spoken reply")
    state = _make_state()
    ws = FakeWS()

    outcome = asyncio.run(ts._execute_turn(ws, state, "hello", None, channel="web_voice"))

    assert "web_voice" in env.recorder.channels_for("turtle.turn")
    # Voice needs the model reply back so it can run TTS on it.
    assert outcome.output_text == "spoken reply"


def test_channel_dispatch_span_and_ws_none(env, monkeypatch):
    """(a) channel span + (b) ws=None path completes and still returns a reply."""
    _use_model(monkeypatch, output="channel reply")
    state = _make_state()

    outcome = asyncio.run(
        ts._execute_turn(None, state, "ping", None, channel="whatsapp", send_status=False)
    )

    assert "whatsapp" in env.recorder.channels_for("turtle.turn")
    # (b) no exception with ws=None; the terminal text comes back on reply_text
    assert outcome.reply_text == "channel reply"
    # (d) continuity: persisted through the session store, not a bare list
    assert state.session_store.message_history


# ---------------------------------------------------------------------------
# (c) confirmation sidecar
# ---------------------------------------------------------------------------

def test_confirmation_sidecar_emitted_when_pending(env, monkeypatch):
    _use_model(monkeypatch, output="the answer")
    prompt = SimpleNamespace(
        all_event_ids=["evt_1"],
        topic="preferences",
        key="preferences.editor",
        question="Want me to remember you like VS Code?",
    )
    state = _make_state(gate=StubGate(prompt=prompt))
    ws = FakeWS()

    # A non-yes/no turn: the pending question is surfaced as a sidecar and the
    # turn proceeds normally.
    asyncio.run(ts._execute_turn(ws, state, "tell me a joke", None, channel="web"))

    sidecars = ws.of_type("confirmation_prompt")
    assert sidecars, "no confirmation_prompt sidecar frame emitted"
    assert sidecars[0]["event_ids"] == ["evt_1"]
    assert sidecars[0]["topic"] == "preferences"
    # The turn still produced a normal reply after the sidecar.
    assert ws.of_type("done")


def test_yes_turn_reaches_model_not_intercepted(env, monkeypatch):
    """A bare "yes" is NOT intercepted: it reaches the model like any other turn.

    (The chat-intercept half of the gate was deleted — confirmations happen via
    the web UI's /api/memory/confirm panel, and a confirming chat turn is
    auto-promoted in the memory funnel, not short-circuited here.)
    """
    _use_model(monkeypatch, output="reached-the-model")
    prompt = SimpleNamespace(
        all_event_ids=["evt_2"],
        topic="preferences",
        key="preferences.editor",
        question="Remember it?",
    )
    state = _make_state(gate=StubGate(prompt=prompt))
    ws = FakeWS()

    outcome = asyncio.run(ts._execute_turn(ws, state, "yes", None, channel="web"))

    # The model WAS reached and its reply is the turn output — no interception.
    done = ws.of_type("done")
    assert done and done[0]["content"] == "reached-the-model"
    assert outcome.output_text == "reached-the-model"
    # The pending question is still surfaced as a sidecar for the UI panel.
    assert ws.of_type("confirmation_prompt")
    assert not ws.of_type("error")


# ---------------------------------------------------------------------------
# (e) classified error handling
# ---------------------------------------------------------------------------

def test_model_failure_is_classified_not_raw(env, monkeypatch):
    _use_model(monkeypatch, raises=RuntimeError("raw internal boom"))
    state = _make_state()
    ws = FakeWS()

    outcome = asyncio.run(ts._execute_turn(ws, state, "trigger error", None, channel="web"))

    errors = ws.of_type("error")
    assert errors, "no error frame emitted"
    err = errors[0]
    # Structured: a classification code, never the raw exception string.
    assert err.get("code") == "internal_error"
    assert "raw internal boom" not in err.get("message", "")
    # The pipeline swallows the error and returns cleanly.
    assert outcome.output_text is None
    assert outcome.reply_text == err["message"]


def test_channel_error_completes_with_ws_none(env, monkeypatch):
    """(b)+(e): a failing channel turn (ws=None) still returns a friendly reply."""
    _use_model(monkeypatch, raises=RuntimeError("kaboom"))
    state = _make_state()

    outcome = asyncio.run(
        ts._execute_turn(None, state, "boom", None, channel="slack", send_status=False)
    )

    # No websocket, no crash; the classified friendly message is relayed.
    assert outcome.output_text is None
    assert outcome.reply_text
    assert "kaboom" not in outcome.reply_text
