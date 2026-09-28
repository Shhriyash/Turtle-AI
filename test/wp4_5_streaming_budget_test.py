"""
WP4.5 (ledger 4.5) — close the streaming/voice daily-budget bypass.

Phase 1 built the daily spend ceiling (``_reserve_daily_spend`` /
``_finalize_daily_spend`` in apps/turtle_server.py, pinned by
test/wp1d2_daily_budget_test.py) but explicitly deferred wiring it into the
voice/streaming turn (``_execute_turn_streaming``) to "Phase 4" — see the
STREAMING GAP comment this module used to carry. Before this fix, a user
could exhaust the entire daily token ceiling by speaking instead of typing:
``stream_agent_text_with_fallbacks`` was never given a ``stats=`` to record
real cost into, and neither ``_reserve_daily_spend`` nor
``_finalize_daily_spend`` was ever called on that path.

This module:
  1. reproduces the bypass (a streaming turn touches no spend key at all),
  2. proves the fix reserves before the stream and reconciles to the ACTUAL
     token cost after,
  3. proves the reservation is released even when the stream is cancelled
     mid-flight (``asyncio.CancelledError`` — ``BaseException``, not
     ``Exception``),
  4. proves a user already at the cap is refused on the streaming path
     (via the pre-audio fallback to the batch path, which speaks the
     refusal — see ``_execute_turn_streaming``'s docstring for why),
  5. proves the batch path (``_execute_turn``) is byte-for-byte unchanged.

Reuses the exact fakes/fixtures test/wp1d2_daily_budget_test.py and
test/phase3_pipeline_test.py already established, rather than inventing a
second test harness for the same mechanism.
"""
from __future__ import annotations

import asyncio

import pytest

import apps.turtle_server as ts
import core.streaming_tts as streaming_tts_mod
from core.llm_client import CascadeStats, StreamCollector, stream_agent_text_with_fallbacks
from test.phase3_pipeline_test import (
    FakeRAG,
    FakeSessionStore,
    FakeWS,
    SpanRecorder,
    StubGate,
)
from test.wp1d2_daily_budget_test import _FakeRedis, _patch_redis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cloud_mode(monkeypatch):
    monkeypatch.setattr(ts.settings, "deploy_mode", "cloud", raising=False)
    monkeypatch.setattr(ts.settings, "daily_token_budget", 1_000_000, raising=False)
    monkeypatch.setattr(ts.settings, "unmetered_user_ids", "", raising=False)
    return None


@pytest.fixture()
def turn_env(monkeypatch):
    """Same offline-hostile neutralisation as phase3_pipeline_test.env, plus
    cloud mode + a fake Redis wired in so the real reserve/finalize calls
    land somewhere observable."""
    recorder = SpanRecorder()
    monkeypatch.setattr(ts, "trace_sink", recorder)
    monkeypatch.setattr(ts, "_logfire_loaded", False)
    monkeypatch.setattr(ts, "_apply_explicit_facts_from_turn", lambda *a, **k: None)
    monkeypatch.setattr(ts, "_queue_confirmation_candidates_from_turn", lambda *a, **k: 0)
    monkeypatch.setattr(ts, "emit_event_once", lambda *a, **k: True)

    monkeypatch.setattr(ts.settings, "deploy_mode", "cloud", raising=False)
    monkeypatch.setattr(ts.settings, "daily_token_budget", 1_000_000, raising=False)
    monkeypatch.setattr(ts.settings, "unmetered_user_ids", "", raising=False)
    client = _patch_redis(monkeypatch, _FakeRedis())
    return client


def _make_state() -> ts.SharedState:
    return ts.SharedState(
        http_client=None,
        session_store=FakeSessionStore(),
        personal_memory_store=None,
        personal_memory_prompt=None,
        journal_store=None,
        confirmation_gate=StubGate(),
        task_history_store=None,
        rag_system=FakeRAG(),
        retrieval_broker=None,
        reflector=None,
        user_id="usr_stream",
    )


def _stub_fake_tts(monkeypatch, *, sentences: list[str] | None = None):
    """Replace the real (network-hitting) TTS synthesiser with one that just
    drains the token iterator (so the stubbed LLM cascade actually runs and
    populates stats/collector) and yields one fake audio chunk per sentence,
    marking first_audio_sent True.
    """
    sentences = sentences if sentences is not None else ["ok"]

    async def _fake_tts_stream(token_iterator, **kwargs):
        async for _delta in token_iterator:
            pass
        for s in sentences:
            yield s, b"FAKE_AUDIO"

    monkeypatch.setattr(streaming_tts_mod, "stream_tts_from_token_stream", _fake_tts_stream)


def _stub_stream_agent(monkeypatch, *, input_tokens=0, output_tokens=0, raises=None, output="ok"):
    """Stub stream_agent_text_with_fallbacks so it mutates the REAL
    CascadeStats object passed in via stats=, mirroring
    wp1d2_daily_budget_test._stub_run_agent for the streaming runner."""

    async def _fake_stream(primary_agent, fallback_agents, prompt, *, deps, message_history,
                            usage_limits, collector, stats=None, **kwargs):
        if stats is not None:
            stats.input_tokens += input_tokens
            stats.output_tokens += output_tokens
        if raises is not None:
            raise raises
        collector.output = output
        collector.agent = primary_agent
        collector._new_messages = []
        for tok in output.split(" "):
            yield tok + " "

    monkeypatch.setattr(ts, "stream_agent_text_with_fallbacks", _fake_stream)


def _run_streaming_turn(state, ws, user_text="hello"):
    timings: dict = {}
    return asyncio.run(
        ts._execute_turn_streaming(
            ws, state, user_text, None,
            channel="web_voice", timings=timings, overall_start=0.0,
        )
    )


# ---------------------------------------------------------------------------
# 1. Reproduce the bypass directly against the real, unstubbed
#    stream_agent_text_with_fallbacks call in _execute_turn_streaming: the
#    call it makes must include stats=<something>, and that something's
#    tokens must be reconcilable against turtle:spend:{uid}:*.
# ---------------------------------------------------------------------------

def test_streaming_call_site_passes_stats_for_budget_reconciliation(turn_env, monkeypatch):
    """Today (after the fix) _execute_turn_streaming must pass stats= to
    stream_agent_text_with_fallbacks so real token cost can be reconciled
    against the daily spend key. Captures the actual kwargs the call site
    uses (not a mock's assumption) so this fails honestly if the call site
    regresses to dropping stats= again."""
    captured_kwargs = {}

    async def _capturing_stream(primary_agent, fallback_agents, prompt, *, collector, **kwargs):
        captured_kwargs.update(kwargs)
        collector.output = "hi"
        collector.agent = primary_agent
        collector._new_messages = []
        yield "hi "

    monkeypatch.setattr(ts, "stream_agent_text_with_fallbacks", _capturing_stream)
    _stub_fake_tts(monkeypatch)

    state = _make_state()
    ws = FakeWS()
    _run_streaming_turn(state, ws)

    assert "stats" in captured_kwargs, (
        "stream_agent_text_with_fallbacks was called with no stats= kwarg — "
        "the streaming path has no CascadeStats to reconcile real token cost "
        "against, so the daily budget silently never sees this turn's spend."
    )
    assert isinstance(captured_kwargs["stats"], ts.CascadeStats)


def test_streaming_turn_touches_the_spend_key(turn_env, monkeypatch):
    """Direct proof of the fix: a streaming turn with a non-zero token cost
    must leave a non-zero balance on turtle:spend:{uid}:{yyyymmdd}, not an
    untouched key."""
    client = turn_env
    _stub_stream_agent(monkeypatch, input_tokens=300, output_tokens=120)
    _stub_fake_tts(monkeypatch)
    state = _make_state()
    ws = FakeWS()

    _run_streaming_turn(state, ws)

    key = ts._spend_key("usr_stream")
    assert key in client.store, (
        f"streaming turn never touched {key} — the daily budget bypass is back"
    )
    assert client.store[key] == 420  # 300 + 120, the real cost, not the ceiling


# ---------------------------------------------------------------------------
# 2. Reservation reconciles to actual cost, not the worst-case ceiling.
# ---------------------------------------------------------------------------

def test_normal_streaming_completion_records_actual_spend_not_the_ceiling(turn_env, monkeypatch):
    client = turn_env
    _stub_stream_agent(monkeypatch, input_tokens=300, output_tokens=120)
    _stub_fake_tts(monkeypatch)
    state = _make_state()
    ws = FakeWS()

    outcome = _run_streaming_turn(state, ws)

    assert outcome.output_text == "ok"
    key = ts._spend_key("usr_stream")
    ceiling = ts.agents_mgr.usage_limits.total_tokens_limit
    assert client.store[key] == 420
    assert client.store[key] != ceiling


# ---------------------------------------------------------------------------
# 3. Cancellation mid-stream must not strand the reservation.
#    asyncio.CancelledError inherits BaseException, not Exception.
# ---------------------------------------------------------------------------

def test_streaming_turn_cancelled_mid_flight_still_finalizes_via_finally(turn_env, monkeypatch):
    client = turn_env
    _stub_stream_agent(monkeypatch, input_tokens=777, output_tokens=0, raises=asyncio.CancelledError())
    _stub_fake_tts(monkeypatch)
    state = _make_state()
    ws = FakeWS()

    with pytest.raises(asyncio.CancelledError):
        _run_streaming_turn(state, ws)

    key = ts._spend_key("usr_stream")
    # Reconciled to the actual (partial) spend recorded before cancellation,
    # not left stranded at the full reserved ceiling.
    assert client.store[key] == 777


def test_streaming_turn_that_raises_before_any_model_call_refunds_the_full_reservation(turn_env, monkeypatch):
    """An exception before the streamed cascade is ever reached (e.g. memory
    context resolution) must refund the FULL reservation."""
    client = turn_env

    async def _boom(*a, **k):
        raise RuntimeError("memory context blew up")

    monkeypatch.setattr(ts, "_resolve_memory_context", _boom)
    state = _make_state()
    ws = FakeWS()

    with pytest.raises(RuntimeError):
        _run_streaming_turn(state, ws)

    key = ts._spend_key("usr_stream")
    assert client.store.get(key, 0) == 0  # fully refunded — nothing was actually spent


# ---------------------------------------------------------------------------
# 4. A user already at the cap is refused on the streaming path.
# ---------------------------------------------------------------------------

def test_streaming_turn_over_budget_is_refused_before_any_audio(turn_env, monkeypatch):
    client = turn_env
    ceiling = ts.agents_mgr.usage_limits.total_tokens_limit
    key = ts._spend_key("usr_stream")
    client.store[key] = ceiling * 1_000_000  # already far over any reasonable budget

    stream_called = False

    async def _should_not_run(*a, **k):
        nonlocal stream_called
        stream_called = True
        yield "should never run"

    monkeypatch.setattr(ts, "stream_agent_text_with_fallbacks", _should_not_run)
    _stub_fake_tts(monkeypatch)
    state = _make_state()
    ws = FakeWS()

    with pytest.raises(ts._StreamPreAudioError):
        _run_streaming_turn(state, ws)

    assert not stream_called, "streaming ran the LLM cascade despite being over budget"
    # No frames sent — nothing was spoken/shown before the refusal (so the
    # caller's fallback to the batch path, which DOES speak the refusal via
    # TTS, is the only thing the user hears).
    assert ws.frames == []


def test_streaming_refusal_falls_back_to_batch_which_speaks_it(turn_env, monkeypatch):
    """End-to-end via _reply_and_speak: an over-budget user gets the
    refusal SPOKEN (through the batch path's TTS), not silence."""
    client = turn_env
    ceiling = ts.agents_mgr.usage_limits.total_tokens_limit
    key = ts._spend_key("usr_stream")
    client.store[key] = ceiling * 1_000_000

    monkeypatch.setattr(ts, "_voice_stream_llm_enabled", lambda: True)

    async def _should_not_run(*a, **k):
        raise AssertionError("streaming cascade must not run when over budget")
        yield  # pragma: no cover

    monkeypatch.setattr(ts, "stream_agent_text_with_fallbacks", _should_not_run)

    async def _fake_batch_run(primary_agent, fallback_agents, prompt, **kwargs):
        raise AssertionError("batch cascade must not run either — refusal happens pre-LLM")

    monkeypatch.setattr(ts, "run_agent_with_fallbacks", _fake_batch_run)

    # No real TTS network calls: stub the batch TTS helper the same way.
    async def _fake_text_tts(text, **kwargs):
        yield text, b"FAKE_AUDIO"

    import core.streaming_tts as _tts_mod
    monkeypatch.setattr(_tts_mod, "stream_tts_from_text", _fake_text_tts)

    state = _make_state()
    ws = FakeWS()
    timings: dict = {}

    asyncio.run(ts._reply_and_speak(ws, state, "hello", None, timings=timings, overall_start=0.0))

    done_frames = ws.of_type("done")
    assert done_frames, "no done frame — the refusal was never surfaced to the user at all"
    assert "usage limit" in done_frames[0]["content"].lower()
    # The refusal was spoken: at least one audio frame went out.
    assert any(f.get("type") == "__bytes__" for f in ws.frames)


# ---------------------------------------------------------------------------
# 5. The batch path is byte-for-byte unchanged.
# ---------------------------------------------------------------------------

def test_batch_path_reservation_and_finalize_unchanged(turn_env, monkeypatch):
    """Re-run of wp1d2_daily_budget_test's normal-completion case through
    _execute_turn (untouched by this WP) to pin that the batch path's
    observable spend accounting did not shift."""
    from test.wp1d2_daily_budget_test import _stub_run_agent

    client = turn_env
    _stub_run_agent(monkeypatch, input_tokens=300, output_tokens=120)
    state = _make_state()
    ws = FakeWS()

    asyncio.run(ts._execute_turn(ws, state, "hello", None, channel="web"))

    key = ts._spend_key("usr_stream")
    assert client.store[key] == 420


# ---------------------------------------------------------------------------
# 6. core/llm_client.py: a rung that streams some tokens and then fails
#    mid-stream counts as WASTED spend (already billed by the provider),
#    matching the batch runner's "wasted tokens count" policy — see the
#    WASTED TOKENS COUNT comment in apps/turtle_server.py. This is what lets
#    _execute_turn_streaming's finally reconcile to a real, non-zero cost
#    even when the stream "dies mid-way with partial usage" rather than
#    completing.
# ---------------------------------------------------------------------------

class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.requests = 1


class _FailingStreamResult:
    """Streams two tokens, then raises — modelling a provider that dies
    mid-response after already billing the tokens it sent."""

    def __init__(self, usage: _FakeUsage) -> None:
        self._usage = usage

    async def stream_text(self, delta: bool = True):
        yield "partial "
        raise RuntimeError("simulated mid-stream provider failure")

    def usage(self):
        return self._usage

    async def get_output(self):  # pragma: no cover - never reached (raises first)
        return "partial"

    def new_messages(self):  # pragma: no cover
        return []

    def all_messages(self):  # pragma: no cover
        return []


class _FailingStreamCtx:
    def __init__(self, usage: _FakeUsage) -> None:
        self._result = _FailingStreamResult(usage)

    async def __aenter__(self):
        return self._result

    async def __aexit__(self, *exc_info):
        return False


class _FailingAgent:
    """Only rung; no fallback — the failure must propagate (mirrors a
    mid-stream failure once tokens were already emitted to the user)."""

    def __init__(self, usage: _FakeUsage) -> None:
        self._usage = usage
        self.model_name = "fake-failing-model"

    def run_stream(self, *args, **kwargs):
        return _FailingStreamCtx(self._usage)


async def _drain_capturing_stats(agent):
    """Runs the streamed cascade to exhaustion (it raises), returning the
    CascadeStats object regardless — it's mutated in place, so it still
    reflects whatever was recorded up to the point of failure."""
    stats = CascadeStats()
    collector = StreamCollector()
    gen = stream_agent_text_with_fallbacks(
        agent, [], "hi", deps=None, message_history=None,
        usage_limits=None, collector=collector, stats=stats,
    )
    try:
        async for _delta in gen:
            pass
    except RuntimeError:
        pass
    return stats


def test_mid_stream_failure_counts_already_emitted_tokens_as_wasted(monkeypatch):
    from core import health_tracker
    monkeypatch.setattr(health_tracker, "is_cooling", lambda *a, **k: False)
    monkeypatch.setattr(health_tracker, "mark_failure", lambda *a, **k: None)
    monkeypatch.setattr(health_tracker, "mark_success", lambda *a, **k: None)

    agent = _FailingAgent(_FakeUsage(input_tokens=50, output_tokens=10))

    stats = asyncio.run(_drain_capturing_stats(agent))
    assert stats.wasted_input_tokens == 50, (
        "tokens the provider already streamed before failing were not "
        "counted as wasted spend — a mid-stream failure would be invisible "
        "to the daily budget reconciliation"
    )
    assert stats.wasted_output_tokens == 10
    assert stats.total_input_tokens == 50
    assert stats.total_output_tokens == 10
