"""
WP1.D2 (ledger 1a.4 part 4) — per-tenant daily token budget.

RESERVATION, NOT CHECK-THEN-ACT. An earlier version of this budget read the
user's spend, compared to the limit, ran the turn, then recorded the real
cost afterwards — the same shape wave 1's email idempotency bug had before
its SET-NX fix. Concurrent turns for one user (extra browser tabs, a
scripted client) could all pass a stale pre-check before any of them
recorded, multiplying the effective budget by however many turns ran at
once. Fixed by making Redis enforce the limit via an atomic INCRBY
reservation of the per-turn ceiling (``UsageLimits.total_tokens_limit``)
BEFORE the turn runs, refunded/reconciled by an actual-cost delta after —
see the module comment above ``_reserve_daily_spend`` in
apps/turtle_server.py for the full design.

Default 1,000,000 tokens/user/UTC-day (TURTLE_DAILY_TOKEN_BUDGET);
TURTLE_UNMETERED_USER_IDS exempt. Cloud-mode only — local has no Redis and is
never metered.
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

import apps.turtle_server as ts
from core.config import parse_unmetered_user_ids
from test.phase3_pipeline_test import (
    FakeRAG,
    FakeResponse,
    FakeSessionStore,
    FakeWS,
    SpanRecorder,
    StubGate,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart


# ---------------------------------------------------------------------------
# Fake Redis — thread-safe, so it can back a REAL-concurrency test (multiple
# OS threads hitting it at once), not just sequential calls dressed up as a
# concurrency test. A single lock around each pipeline.execute() mirrors
# Redis's own single-threaded command execution / MULTI-EXEC atomicity
# (transaction=True, the default the verifier confirmed the real pipeline
# uses) — that atomicity is the entire mechanism under test.
# ---------------------------------------------------------------------------

class _FakePipeline:
    def __init__(self, client):
        self._client = client
        self._ops = []

    def incrby(self, key, amount):
        self._ops.append(("incrby", key, amount))
        return self

    def expire(self, key, seconds):
        self._ops.append(("expire", key, seconds))
        return self

    def execute(self):
        results = []
        with self._client._lock:
            for op, key, val in self._ops:
                if op == "incrby":
                    self._client.store[key] = int(self._client.store.get(key, 0)) + val
                    results.append(self._client.store[key])
                elif op == "expire":
                    self._client.expiries[key] = val
                    results.append(True)
        self._ops = []
        return results


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, int] = {}
        self.expiries: dict[str, int] = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            val = self.store.get(key)
            return str(val) if val is not None else None

    def decrby(self, key, amount):
        with self._lock:
            self.store[key] = int(self.store.get(key, 0)) - amount
            return self.store[key]

    def pipeline(self):
        return _FakePipeline(self)


class _BrokenRedis:
    """Simulates Redis being unreachable."""

    def get(self, key):
        raise ConnectionError("simulated Redis outage")

    def decrby(self, key, amount):
        raise ConnectionError("simulated Redis outage")

    def pipeline(self):
        raise ConnectionError("simulated Redis outage")


@pytest.fixture()
def cloud_mode(monkeypatch):
    monkeypatch.setattr(ts.settings, "deploy_mode", "cloud", raising=False)
    monkeypatch.setattr(ts.settings, "daily_token_budget", 1000, raising=False)
    monkeypatch.setattr(ts.settings, "unmetered_user_ids", "", raising=False)
    return None


def _patch_redis(monkeypatch, client):
    import core.storage.cloud as cloud_mod

    monkeypatch.setattr(cloud_mod, "get_redis_sync_client", lambda: client, raising=False)
    return client


# ── local mode: always unmetered ─────────────────────────────────────────────

def test_local_mode_never_refuses_and_never_reserves(monkeypatch):
    monkeypatch.setattr(ts.settings, "deploy_mode", "local", raising=False)
    monkeypatch.setattr(ts.settings, "daily_token_budget", 1, raising=False)  # absurdly tight
    allowed, refusal, reserved, key = ts._reserve_daily_spend("usr_anyone")
    assert allowed is True
    assert refusal is None
    assert reserved == 0
    ts._finalize_daily_spend(key, reserved, 999999)  # must not raise, must not touch Redis


# ── cloud mode: reserve / refuse / reset time ────────────────────────────────

def test_under_budget_reserves_the_per_turn_ceiling(cloud_mode, monkeypatch):
    monkeypatch.setattr(ts.settings, "daily_token_budget", 1_000_000, raising=False)
    client = _patch_redis(monkeypatch, _FakeRedis())
    allowed, refusal, reserved, key = ts._reserve_daily_spend("usr_a")
    assert allowed is True
    assert refusal is None
    ceiling = ts.agents_mgr.usage_limits.total_tokens_limit
    assert reserved == ceiling
    assert client.store[key] == ceiling


def test_reservation_that_would_exceed_budget_is_refused_and_refunded(cloud_mode, monkeypatch):
    ceiling = ts.agents_mgr.usage_limits.total_tokens_limit
    monkeypatch.setattr(ts.settings, "daily_token_budget", ceiling - 1, raising=False)  # can't fit even one
    client = _patch_redis(monkeypatch, _FakeRedis())
    key = ts._spend_key("usr_a")

    allowed, refusal, reserved, returned_key = ts._reserve_daily_spend("usr_a")
    assert allowed is False
    assert reserved == 0
    assert returned_key == key
    assert refusal is not None
    assert "resets at" in refusal
    # The refusal-path refund must leave the counter exactly where it
    # started — no ceiling-sized hole left behind by a refused attempt.
    assert client.store.get(key, 0) == 0


def test_refusal_names_a_concrete_utc_reset_time(cloud_mode, monkeypatch):
    client = _patch_redis(monkeypatch, _FakeRedis())
    client.store[ts._spend_key("usr_a")] = 1000  # already at the 1000 limit
    _allowed, refusal, _reserved, _key = ts._reserve_daily_spend("usr_a")
    assert refusal is not None
    reset_str = refusal.split("resets at", 1)[1].strip().rstrip(".")
    assert "UTC" in reset_str
    parsed = datetime.strptime(reset_str, "%H:%M UTC on %Y-%m-%d")
    now = datetime.now(UTC).replace(tzinfo=None)
    assert now < parsed <= now + timedelta(days=1, minutes=1)


def test_unmetered_user_is_never_refused_and_never_reserves(cloud_mode, monkeypatch):
    monkeypatch.setattr(ts.settings, "unmetered_user_ids", "usr_owner, usr_other", raising=False)
    client = _patch_redis(monkeypatch, _FakeRedis())
    client.store[ts._spend_key("usr_owner")] = 10_000_000
    allowed, refusal, reserved, key = ts._reserve_daily_spend("usr_owner")
    assert allowed is True and refusal is None and reserved == 0
    # Nothing was touched on their behalf.
    assert client.store[ts._spend_key("usr_owner")] == 10_000_000


def test_budget_disabled_when_zero_or_negative_and_warns_once(cloud_mode, monkeypatch, capsys):
    monkeypatch.setattr(ts.settings, "daily_token_budget", 0, raising=False)
    monkeypatch.setattr(ts, "_BUDGET_DISABLED_WARNED", False, raising=False)
    client = _patch_redis(monkeypatch, _FakeRedis())
    client.store[ts._spend_key("usr_a")] = 999_999_999

    allowed, refusal, reserved, _key = ts._reserve_daily_spend("usr_a")
    assert allowed is True and refusal is None and reserved == 0
    out1 = capsys.readouterr().out
    assert "DISABLED" in out1

    # Second call: warning does not repeat (log once per process).
    ts._reserve_daily_spend("usr_a")
    out2 = capsys.readouterr().out
    assert "DISABLED" not in out2


# ── finalize: reconciling a reservation to actual spend ──────────────────────

def test_finalize_adjusts_reservation_down_to_actual_spend(cloud_mode, monkeypatch):
    ceiling = ts.agents_mgr.usage_limits.total_tokens_limit
    monkeypatch.setattr(ts.settings, "daily_token_budget", ceiling * 10, raising=False)
    client = _patch_redis(monkeypatch, _FakeRedis())
    _allowed, _refusal, reserved, key = ts._reserve_daily_spend("usr_a")
    assert client.store[key] == ceiling

    actual = 250  # the turn only really cost 250 tokens, far under the ceiling
    ts._finalize_daily_spend(key, reserved, actual)
    assert client.store[key] == actual


def test_finalize_is_a_noop_when_nothing_was_reserved(cloud_mode, monkeypatch):
    client = _patch_redis(monkeypatch, _FakeRedis())
    ts._finalize_daily_spend("", 0, 5000)  # nothing reserved -> nothing to adjust
    assert client.store == {}


def test_zero_reserved_tokens_do_not_touch_redis_on_finalize(cloud_mode, monkeypatch):
    client = _patch_redis(monkeypatch, _FakeRedis())
    ts._finalize_daily_spend("turtle:spend:usr_a:20260101", 0, 500)
    assert client.store == {}


# ── UTC day boundary ─────────────────────────────────────────────────────────

def test_spend_key_rolls_at_utc_midnight(monkeypatch):
    monkeypatch.setattr(ts, "_utc_day_str", lambda: "20260925")
    key_before = ts._spend_key("usr_a")
    monkeypatch.setattr(ts, "_utc_day_str", lambda: "20260926")
    key_after = ts._spend_key("usr_a")
    assert key_before != key_after
    assert key_before == "turtle:spend:usr_a:20260925"
    assert key_after == "turtle:spend:usr_a:20260926"


def test_spend_from_yesterday_does_not_count_against_today(cloud_mode, monkeypatch):
    ceiling = ts.agents_mgr.usage_limits.total_tokens_limit
    monkeypatch.setattr(ts.settings, "daily_token_budget", ceiling * 10, raising=False)
    client = _patch_redis(monkeypatch, _FakeRedis())
    monkeypatch.setattr(ts, "_utc_day_str", lambda: "20260101")
    _allowed, _refusal, reserved, key = ts._reserve_daily_spend("usr_a")
    ts._finalize_daily_spend(key, reserved, 999)  # yesterday, near/at limit

    monkeypatch.setattr(ts, "_utc_day_str", lambda: "20260102")
    # Today's key is fresh — under budget despite yesterday's near-max spend.
    allowed, refusal, _reserved2, _key2 = ts._reserve_daily_spend("usr_a")
    assert allowed is True and refusal is None


# ── Redis unavailable: fail OPEN (pinned posture, confirmed against a real
# unroutable socket by the verifier — this fake models the same outcome) ────

def test_redis_unavailable_reserve_fails_open(cloud_mode, monkeypatch):
    _patch_redis(monkeypatch, _BrokenRedis())
    allowed, refusal, reserved, key = ts._reserve_daily_spend("usr_a")
    assert allowed is True and refusal is None and reserved == 0


def test_redis_unavailable_finalize_does_not_raise(cloud_mode, monkeypatch):
    _patch_redis(monkeypatch, _BrokenRedis())
    ts._finalize_daily_spend("turtle:spend:usr_a:20260101", 100, 50)  # best-effort: swallow, log, move on


# ── per-turn UsageLimits ceiling ─────────────────────────────────────────────

def test_usage_limits_has_a_per_turn_token_ceiling():
    assert ts.agents_mgr.usage_limits.total_tokens_limit == 100_000
    assert ts.agents_mgr.usage_limits.request_limit == 30


# ── parse_unmetered_user_ids ─────────────────────────────────────────────────

def test_parse_unmetered_user_ids_trims_and_drops_empties():
    assert parse_unmetered_user_ids(" usr_a, usr_b ,,") == frozenset({"usr_a", "usr_b"})
    assert parse_unmetered_user_ids("") == frozenset()
    assert parse_unmetered_user_ids(None) == frozenset()


# ---------------------------------------------------------------------------
# THE MUST-FIX: real concurrency cannot multiply the budget.
#
# This is the exact scenario the coordinator ran against the OLD check-then-
# act code and got 20/20 passing a stale pre-check. Uses real OS threads
# (ThreadPoolExecutor), not sequential calls — _reserve_daily_spend is a sync
# function (it blocks briefly on a real Redis round trip in production, same
# as core.guardrails.ws_rate_limiter's sync Redis calls), so thread-level
# concurrency is what actually happens when N turns for one user run at once.
# ---------------------------------------------------------------------------

def test_concurrent_reservations_cannot_collectively_exceed_the_budget(cloud_mode, monkeypatch):
    ceiling = ts.agents_mgr.usage_limits.total_tokens_limit  # 100_000
    budget = int(ceiling * 2.5)  # room for exactly 2 reservations, never 3
    monkeypatch.setattr(ts.settings, "daily_token_budget", budget, raising=False)
    client = _patch_redis(monkeypatch, _FakeRedis())

    n = 20
    results: list[tuple] = [None] * n  # type: ignore[list-item]

    def worker(i):
        results[i] = ts._reserve_daily_spend("usr_concurrent")

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(worker, i) for i in range(n)]
        for f in futures:
            f.result()

    allowed_count = sum(1 for r in results if r[0])
    max_grantable = budget // ceiling  # = 2
    assert allowed_count <= max_grantable, (
        f"{allowed_count} of {n} concurrent turns were granted a reservation "
        f"(budget allows at most {max_grantable}) — the daily budget was "
        f"multiplied by concurrency, which is exactly the bug this fixes. "
        f"Under the old check-then-act design this assertion fails 20 > {max_grantable}."
    )
    assert allowed_count >= 1, "the fake Redis or reservation logic is broken, not just strict"

    key = ts._spend_key("usr_concurrent")
    # The counter must reflect exactly what was granted — no phantom
    # reservations left behind by refused attempts (each refusal refunds
    # itself before returning).
    assert client.store.get(key, 0) == allowed_count * ceiling


# ---------------------------------------------------------------------------
# Refund paths through the REAL _execute_turn pipeline: raise, cancel,
# normal completion. Each asserts the final recorded spend equals the turn's
# actual cost and no reservation is left stranded.
# ---------------------------------------------------------------------------

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
        user_id="usr_turn",
    )


@pytest.fixture()
def turn_env(monkeypatch):
    """Same offline-hostile neutralisation as phase3_pipeline_test.env, plus
    cloud mode + a fake Redis wired in so _execute_turn's real reserve/
    finalize calls land somewhere observable."""
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


def _stub_run_agent(monkeypatch, *, input_tokens=0, output_tokens=0, raises=None):
    """Stub run_agent_with_fallbacks so it mutates the REAL CascadeStats
    object _execute_turn passes in via stats=, so _finalize_daily_spend sees
    a genuine non-zero actual cost on a normal completion."""
    async def _fake_run(primary_agent, fallback_agents, prompt, **kwargs):
        stats = kwargs.get("stats")
        if stats is not None:
            stats.input_tokens += input_tokens
            stats.output_tokens += output_tokens
        if raises is not None:
            raise raises
        msgs = [
            ModelRequest(parts=[UserPromptPart(content=prompt)]),
            ModelResponse(parts=[TextPart(content="ok")]),
        ]
        return FakeResponse("ok", msgs)
    monkeypatch.setattr(ts, "run_agent_with_fallbacks", _fake_run)


def test_normal_completion_records_actual_spend_not_the_ceiling(turn_env, monkeypatch):
    client = turn_env
    _stub_run_agent(monkeypatch, input_tokens=300, output_tokens=120)
    state = _make_state()
    ws = FakeWS()

    asyncio.run(ts._execute_turn(ws, state, "hello", None, channel="web"))

    key = ts._spend_key("usr_turn")
    assert client.store[key] == 420  # 300 + 120, NOT the 100_000 ceiling


def test_turn_that_raises_records_actual_spend_and_strands_nothing(turn_env, monkeypatch):
    """This is the case the OLD design specifically called out as worth
    counting (an expensive failed cascade) — and the reservation scheme must
    still land on the real cost, not leak the reserved ceiling."""
    client = turn_env
    _stub_run_agent(monkeypatch, input_tokens=5000, output_tokens=1000, raises=RuntimeError("boom"))
    state = _make_state()
    ws = FakeWS()

    outcome = asyncio.run(ts._execute_turn(ws, state, "trigger error", None, channel="web"))
    assert outcome.output_text is None  # the model never answered

    key = ts._spend_key("usr_turn")
    assert client.store[key] == 6000  # 5000 + 1000, not the 100_000 ceiling, not 0


def test_turn_cancelled_mid_flight_still_finalizes_via_finally(turn_env, monkeypatch):
    """asyncio.CancelledError inherits from BaseException, not Exception — a
    bare `except Exception: ...finalize()` would miss it and strand the full
    reservation. The `finally` in _execute_turn must still run."""
    client = turn_env
    _stub_run_agent(monkeypatch, input_tokens=777, output_tokens=0, raises=asyncio.CancelledError())
    state = _make_state()
    ws = FakeWS()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ts._execute_turn(ws, state, "cancel me", None, channel="web"))

    key = ts._spend_key("usr_turn")
    # Reconciled to the actual (partial) spend recorded on the stats object
    # before cancellation, not left at the full reserved ceiling.
    assert client.store[key] == 777


def test_turn_that_raises_before_any_model_call_refunds_the_full_reservation(turn_env, monkeypatch):
    """An exception before run_agent_with_fallbacks is ever reached (e.g. in
    memory-context resolution) must refund the FULL reservation — actual
    spend is genuinely 0, since the LLM was never called."""
    client = turn_env

    async def _boom(*a, **k):
        raise RuntimeError("memory context blew up")

    monkeypatch.setattr(ts, "_resolve_memory_context", _boom)
    state = _make_state()
    ws = FakeWS()

    asyncio.run(ts._execute_turn(ws, state, "hi", None, channel="web"))

    key = ts._spend_key("usr_turn")
    assert client.store.get(key, 0) == 0  # fully refunded — nothing was actually spent
