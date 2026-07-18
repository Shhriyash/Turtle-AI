"""Phase 4 (Option C convergence) — the collapsed single-agent turn pipeline.

Wave 1 deleted the router, the planner, the decorative graph layer, per-intent
tool scoping, and the gate's chat-intercept half. ``_execute_turn`` now makes
exactly ONE model call (``run_agent_with_fallbacks``), labels the turn from a
cheap heuristic, and offers every tool on every turn.

This module drives the pipeline fully offline (no LLM, no network, no keys) with
a fake websocket, a stubbed model call, and a minimal ``SharedState`` — the same
fixture/stub pattern as phase3_pipeline_test.py.

Pins the four convergence guarantees:
  (a) NO router: the deleted modules are gone, and the turn's span ``intent`` is
      the heuristic ``_detect_task_type`` — nothing overrides it.
  (b) pending-email bypass: a pending draft in the session store forces the turn
      label to "email" even when the words don't say "email".
  (c) all-tools-offered: no per-intent scoping symbols survive, and the main
      assistant offers the full 7-tool set on a bare turn.
  (d) a bare "yes" is NOT intercepted — it reaches the stubbed model.
"""
from __future__ import annotations

import asyncio
import contextlib
import types
from types import SimpleNamespace

import pytest

import apps.turtle_server as ts
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage


# ---------------------------------------------------------------------------
# Fakes (mirrors phase3_pipeline_test.py)
# ---------------------------------------------------------------------------

class FakeWS:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send_json(self, data: dict) -> None:
        self.frames.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append({"type": "__bytes__", "n": len(data)})

    def of_type(self, t: str) -> list[dict]:
        return [f for f in self.frames if f.get("type") == t]


class FakeSessionStore:
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
    def add_conversation(self, user_text: str, output: str) -> None:
        pass


class StubGate:
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
    def __init__(self) -> None:
        self.spans: list[tuple[str, dict]] = []

    @contextlib.contextmanager
    def span(self, name: str, **kwargs):
        self.spans.append((name, kwargs))
        yield None

    def intents_for(self, name: str) -> list[str]:
        return [kw.get("intent") for n, kw in self.spans if n == name]


@pytest.fixture()
def env(monkeypatch):
    recorder = SpanRecorder()
    monkeypatch.setattr(ts, "trace_sink", recorder)
    monkeypatch.setattr(ts, "_logfire_loaded", False)
    monkeypatch.setattr(ts, "_apply_explicit_facts_from_turn", lambda *a, **k: None)
    monkeypatch.setattr(ts, "_queue_confirmation_candidates_from_turn", lambda *a, **k: 0)
    monkeypatch.setattr(ts, "emit_event_once", lambda *a, **k: True)
    return SimpleNamespace(recorder=recorder)


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
# (a) NO router — the deleted layers are gone and the span intent is heuristic
# ---------------------------------------------------------------------------

def test_router_and_graph_layers_are_gone():
    # The whole router/graph/dream-pass layer was deleted, not just unwired.
    for mod in ("core.router", "core.graph", "core.dream_pass"):
        with pytest.raises(ModuleNotFoundError):
            __import__(mod)
    # And the server no longer holds the graph selector or the intent stub.
    assert not hasattr(ts, "_select_graph")


def test_turn_span_intent_is_heuristic_task_type(env, monkeypatch):
    _use_model(monkeypatch, output="ok")
    state = _make_state()

    text = "what's the latest news on AI"
    # Sanity: this is what the heuristic labels the turn — nothing else may
    # override it now that the router is gone.
    assert ts._detect_task_type(text) == "web"

    asyncio.run(ts._execute_turn(FakeWS(), state, text, None, channel="web"))

    assert env.recorder.intents_for("turtle.turn") == ["web"]


# ---------------------------------------------------------------------------
# (b) pending-email bypass forces the turn label to "email"
# ---------------------------------------------------------------------------

def test_pending_email_forces_email_task_type(env, monkeypatch):
    _use_model(monkeypatch, output="ok")

    # A turn whose words alone read as "general" (no email/mail keyword).
    text = "the subject is lunch"
    assert ts._detect_task_type(text) == "general"

    # With a pending draft in the session store, the bypass relabels it "email".
    store = FakeSessionStore(pending_email={
        "recipients": ["alice@example.com"], "cc_recipients": [], "bcc_recipients": [],
        "subject": "", "content": "",
    })
    state = _make_state(session_store=store)

    asyncio.run(ts._execute_turn(FakeWS(), state, text, None, channel="web"))

    assert env.recorder.intents_for("turtle.turn") == ["email"]


def test_no_pending_email_keeps_heuristic_label(env, monkeypatch):
    _use_model(monkeypatch, output="ok")
    # Same text, but an empty draft: the bypass does not fire, label stays.
    state = _make_state(session_store=FakeSessionStore())

    asyncio.run(ts._execute_turn(FakeWS(), state, "the subject is lunch", None, channel="web"))

    assert env.recorder.intents_for("turtle.turn") == ["general"]


# ---------------------------------------------------------------------------
# (c) all tools offered on every turn — no per-intent scoping survives
# ---------------------------------------------------------------------------

def test_no_tool_scoping_symbols_remain():
    assert not hasattr(ts, "_scope_tools_by_intent")
    assert not hasattr(ts, "_TOOL_NAMES_BY_INTENT")


def test_main_assistant_offers_full_toolset():
    # With prepare_tools removed, the assistant offers every tool regardless of
    # any turn label. Drive it with a TestModel that calls nothing and inspect
    # the tool set it was offered.
    tm = TestModel(call_tools=[])
    asyncio.run(
        ts.agents_mgr.main_assistant.run(
            "hi",
            model=tm,
            deps=types.SimpleNamespace(intent="", user_id=""),
            usage=RunUsage(),
        )
    )
    names = {t.name for t in tm.last_model_request_parameters.function_tools}
    expected = {
        "search_web", "search_url", "send_email_assistant", "recall",
        "calendar_create", "calendar_list", "remember",
    }
    assert expected <= names, f"missing {expected - names}"
    # history_tool folded into recall — it must not be registered anymore.
    assert "history_tool" not in names


# ---------------------------------------------------------------------------
# (d) a bare "yes" is not intercepted — it reaches the stubbed model
# ---------------------------------------------------------------------------

def test_yes_turn_reaches_model(env, monkeypatch):
    _use_model(monkeypatch, output="reached-the-model")
    prompt = SimpleNamespace(
        all_event_ids=["evt_1"],
        topic="preferences",
        key="preferences.editor",
        question="Remember it?",
    )
    state = _make_state(gate=StubGate(prompt=prompt))
    ws = FakeWS()

    outcome = asyncio.run(ts._execute_turn(ws, state, "yes", None, channel="web"))

    done = ws.of_type("done")
    assert done and done[0]["content"] == "reached-the-model"
    assert outcome.output_text == "reached-the-model"
    assert not ws.of_type("error")
