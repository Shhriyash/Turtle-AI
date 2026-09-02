"""
core/llm_client cascade budget tests — per-rung deadline, in-rung retry, spend accounting.

Before this work the ONLY deadline on a turn was a single 60s wait_for wrapped
around the whole cascade in _execute_turn. That made the fallback chain mostly
decorative: a rung that hung for 55s consumed the entire budget and the user got
a timeout instead of the answer rung 2 would have produced. Measured traces
(data/traces/traces.jsonl, 80 turns) showed a 43.6s max with 26/80 over 10s.

These tests pin the three behaviours that fix it:
  1. a hung rung is abandoned at its deadline and the cascade ADVANCES
  2. a transient failure gets one in-rung retry before the cascade swaps model
  3. spend is accounted, including tokens burned by rungs that FAILED

Async tests use ``asyncio.run`` inside a sync test to match the house
convention (test/calendar_tool_test.py, test/phase1_episodic_wiring_test.py) —
the suite deliberately carries no pytest-asyncio dependency.

All fakes — no provider is contacted.
"""
from __future__ import annotations

import asyncio

import pytest

from core.llm_client import (
    CascadeStats,
    _is_timeout_error,
    _retry_same_rung,
    run_agent_with_fallbacks,
)


class _Usage:
    def __init__(self, tin: int = 0, tout: int = 0, reqs: int = 1) -> None:
        self.input_tokens = tin
        self.output_tokens = tout
        self.requests = reqs


class _Result:
    def __init__(self, output: str, tin: int = 0, tout: int = 0) -> None:
        self.output = output
        self._usage = _Usage(tin, tout)

    def usage(self) -> _Usage:
        return self._usage


class _FakeModel:
    def __init__(self, name: str) -> None:
        self.model_name = name


class _FakeAgent:
    """Minimal stand-in for a pydantic-ai Agent.

    ``behaviour`` is a coroutine invoked per attempt with the 1-based attempt
    count, so a test can make the first attempt fail and the second succeed.
    """

    def __init__(self, name: str, behaviour) -> None:
        self.model = _FakeModel(name)
        self._behaviour = behaviour
        self.calls = 0

    async def run(self, *args, **kwargs):
        self.calls += 1
        return await self._behaviour(self.calls)


def _http_error(status: int, message: str):
    """A REAL ModelHTTPError.

    test/llm_fallback_classification_test.py uses MagicMock(spec=ModelHTTPError)
    because it only ever passes the object to a classifier. These tests must
    ``raise`` it, and a MagicMock is not a BaseException subclass — so construct
    the genuine article instead.
    """
    from pydantic_ai.exceptions import ModelHTTPError

    return ModelHTTPError(status_code=status, model_name="test-model", body=message)


@pytest.fixture(autouse=True)
def _no_cooldowns(monkeypatch):
    """health_tracker is process-global; keep cooldowns from leaking between tests."""
    from core import health_tracker

    monkeypatch.setattr(health_tracker, "is_cooling", lambda _a: False)
    monkeypatch.setattr(health_tracker, "mark_failure", lambda _a, _e: None)
    monkeypatch.setattr(health_tracker, "mark_success", lambda _a: None)


class TestTimeoutClassification:
    def test_asyncio_timeout_is_detected(self):
        """asyncio.TimeoutError stringifies to '' — the substring scan in
        _is_retryable_upstream_error cannot see it, which is exactly why
        _is_timeout_error exists and is listed separately in should_fallback.
        Without it the new per-rung deadline would turn a slow provider into a
        hard turn failure instead of a fallback."""
        assert str(asyncio.TimeoutError()) == ""
        assert _is_timeout_error(asyncio.TimeoutError()) is True

    def test_quota_error_is_not_retried_in_rung(self):
        """Retrying the SAME key on a 429 is guaranteed to fail again; it must
        swap models rather than spend the budget twice."""
        assert _retry_same_rung(_http_error(429, "rate limit exceeded")) is False

    def test_5xx_is_retried_in_rung(self):
        assert _retry_same_rung(_http_error(503, "service unavailable")) is True


class TestPerRungDeadline:
    def test_hung_rung_does_not_consume_the_cascade(self):
        """THE regression this work exists for.

        Rung 1 hangs far longer than the budget. The cascade must abandon it at
        the per-rung deadline and let rung 2 answer, rather than letting one bad
        provider convert a recoverable turn into a timeout.
        """

        async def hangs(_n):
            await asyncio.sleep(30)
            return _Result("never reached")

        async def answers(_n):
            return _Result("from the healthy rung", tin=100, tout=20)

        slow = _FakeAgent("slow-model", hangs)
        healthy = _FakeAgent("healthy-model", answers)
        stats = CascadeStats()

        async def run():
            return await run_agent_with_fallbacks(
                slow, [healthy],
                "prompt",
                rung_timeout_s=0.2,
                rung_retries=0,
                stats=stats,
            )

        result = asyncio.run(run())

        assert result.output == "from the healthy rung"
        assert stats.timed_out_attempts == 1, "the hung rung should be recorded as a timeout"
        assert stats.winning_rung == 1
        assert "healthy-model" in stats.winning_model

    def test_total_budget_stops_the_cascade(self):
        """A long chain of slow rungs must not serially add up past the total
        budget the caller was willing to wait for."""

        async def slow(_n):
            await asyncio.sleep(5)
            return _Result("too late")

        agents = [_FakeAgent(f"slow-{i}", slow) for i in range(6)]
        stats = CascadeStats()

        async def run():
            return await run_agent_with_fallbacks(
                agents[0], agents[1:],
                "prompt",
                deadline_s=1.0,
                rung_timeout_s=0.3,
                rung_retries=0,
                stats=stats,
            )

        with pytest.raises(Exception):
            asyncio.run(run())

        assert stats.attempts < 6, (
            f"cascade ignored the total budget: {stats.attempts} attempts"
        )


class TestInRungRetry:
    """In-rung retry is deliberately scoped to the LAST rung only.

    When an untried rung remains, advancing beats retrying: the next rung is a
    fresh connection (usually the same model on a different API key), costs no
    backoff sleep, and is at least as likely to succeed. Retrying earlier rungs
    would add latency to every transient blip for no gain. This is what
    test/phase8_fallback_cascade_test.py::test_transient_error_does_not_poison_sibling_keys
    pins from the other direction.
    """

    def test_last_rung_retries_on_transient_failure(self):
        """With nothing left to fall to, one retry is strictly better than
        failing the turn."""

        async def flaky(n):
            if n == 1:
                raise _http_error(503, "service unavailable")
            return _Result("recovered on retry", tin=50, tout=10)

        only = _FakeAgent("only-rung", flaky)
        stats = CascadeStats()

        async def run():
            return await run_agent_with_fallbacks(
                only, [],
                "prompt",
                rung_timeout_s=5,
                rung_retries=1,
                stats=stats,
            )

        result = asyncio.run(run())

        assert result.output == "recovered on retry"
        assert only.calls == 2, "the last rung should have been retried in-rung"
        assert stats.winning_rung == 0

    def test_earlier_rung_advances_instead_of_retrying(self):
        """A transient failure with a sibling available must ADVANCE, not sleep
        and retry — the sibling is a free, fresh attempt."""

        async def flaky(_n):
            raise _http_error(503, "service unavailable")

        async def answers(_n):
            return _Result("from the sibling key")

        primary = _FakeAgent("primary", flaky)
        sibling = _FakeAgent("sibling", answers)
        stats = CascadeStats()

        async def run():
            return await run_agent_with_fallbacks(
                primary, [sibling],
                "prompt",
                rung_timeout_s=5,
                rung_retries=1,
                stats=stats,
            )

        result = asyncio.run(run())

        assert result.output == "from the sibling key"
        assert primary.calls == 1, (
            "an earlier rung must not burn a backoff sleep when a sibling is untried"
        )
        assert stats.winning_rung == 1

    def test_quota_failure_swaps_immediately_without_retry(self):
        """A 429 must NOT be retried in-rung — the same key cannot recover."""

        async def quota_dead(_n):
            raise _http_error(429, "rate limit exceeded")

        async def answers(_n):
            return _Result("from fallback")

        primary = _FakeAgent("quota-dead", quota_dead)
        backup = _FakeAgent("backup", answers)
        stats = CascadeStats()

        async def run():
            return await run_agent_with_fallbacks(
                primary, [backup],
                "prompt",
                rung_timeout_s=5,
                rung_retries=1,
                stats=stats,
            )

        result = asyncio.run(run())

        assert result.output == "from fallback"
        assert primary.calls == 1, "a 429 must not be retried on the same key"


class TestSpendAccounting:
    def test_usage_is_recorded(self):
        """Turtle recorded ZERO token data before this: observability.py defined
        ATTR_TOKENS_IN/OUT and emit_span accepted tokens_in/tokens_out, but no
        call site ever passed them, so every turtle.turn span on disk carried
        latency and nothing about spend. CascadeStats closes that loop."""

        async def answers(_n):
            return _Result("ok", tin=1234, tout=56)

        agent = _FakeAgent("m", answers)
        stats = CascadeStats()

        asyncio.run(run_agent_with_fallbacks(agent, [], "prompt", stats=stats))

        assert stats.input_tokens == 1234
        assert stats.output_tokens == 56
        assert stats.attempts == 1
        assert stats.failed_attempts == 0

    def test_failed_rungs_are_counted(self):
        """A failed rung still billed its input. Attributing it as waste is the
        whole point — it is what a flapping provider actually costs."""

        async def dies(_n):
            raise _http_error(503, "service unavailable")

        async def answers(_n):
            return _Result("ok", tin=10, tout=5)

        stats = CascadeStats()

        asyncio.run(run_agent_with_fallbacks(
            _FakeAgent("dead", dies), [_FakeAgent("alive", answers)],
            "prompt",
            rung_retries=0,
            stats=stats,
        ))

        assert stats.failed_attempts == 1
        assert stats.attempts == 2
        assert stats.winning_rung == 1

    def test_span_attrs_are_flat_and_prefixed(self):
        """as_span_attrs feeds core/observability.py, which expects flat
        turtle.* keys — OTel silently drops nested values."""
        stats = CascadeStats(attempts=3, failed_attempts=2, winning_rung=2)
        attrs = stats.as_span_attrs()

        assert all(k.startswith("turtle.") for k in attrs)
        assert all(isinstance(v, (int, float, str)) for v in attrs.values())
        assert attrs["turtle.cascade_attempts"] == 3
        assert attrs["turtle.cascade_failed_attempts"] == 2
