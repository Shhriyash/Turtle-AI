"""
Phase 8 — fallback-cascade resilience (found via the live Discord turn trace).

The Discord "@Turtle remember my fav color is teal" turn produced only a canned
error reply even though the fact persisted: Gemini hit its function-call-adjacency
400, the loop then re-tried the SAME gemini family two more times, and
OpenRouter's 402 (out of credits) was classified as non-fallbackable — so the
cascade aborted before reaching the Groq llama rescue rung.

Covers the two fixes in core/llm_client.py:
  - 402 is fallback-eligible, so the cascade continues past a broke provider.
  - a model family cooled mid-call is skipped for its later rungs, so the loop
    doesn't burn identical, already-doomed calls (the 3x Gemini waste).
"""
from __future__ import annotations

import asyncio

import pytest

from core import health_tracker
from core.llm_client import run_agent_with_fallbacks
from pydantic_ai.exceptions import ModelHTTPError

# Gemini's strict tool-turn ordering rejection — a harmony/tool-render 400 that
# is fallback-eligible AND cools the whole model family (bucket scope).
_HARMONY_MSG = "function response turn comes immediately after a function call"


@pytest.fixture(autouse=True)
def _clear_cooldowns():
    with health_tracker._lock:
        health_tracker._cooldown_until.clear()
    yield
    with health_tracker._lock:
        health_tracker._cooldown_until.clear()


def _http_error(status_code: int, message: str = ""):
    body = message or f"HTTP {status_code}"
    try:
        return ModelHTTPError(status_code=status_code, model_name="m", body=body)
    except TypeError:
        return ModelHTTPError(status_code, "m", body)


# Distinct classes → distinct health_tracker family buckets ({cls}:{model_name}).
class _GeminiModel:
    model_name = "gemini-2.5-flash"


class _OpenRouterModel:
    model_name = "google/gemini-2.5-flash"


class _GroqModel:
    model_name = "llama-3.3-70b-versatile"


class _FakeAgent:
    def __init__(self, model, *, raises: Exception | None = None, result=None):
        self.model = model
        self._raises = raises
        self._result = result
        self.calls = 0

    async def run(self, *a, **k):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._result


def test_cascade_continues_past_402_to_groq_rescue():
    """A 402 on OpenRouter must NOT abort the cascade — Groq must still be tried."""
    primary = _FakeAgent(_GeminiModel(), raises=_http_error(400, _HARMONY_MSG))
    openrouter = _FakeAgent(_OpenRouterModel(), raises=_http_error(402, "requires more credits"))
    groq = _FakeAgent(_GroqModel(), result="teal-noted")

    out = asyncio.run(run_agent_with_fallbacks(primary, [openrouter, groq]))

    assert out == "teal-noted"
    assert primary.calls == 1
    assert openrouter.calls == 1
    assert groq.calls == 1  # reached the rescue rung — the whole point


def test_cooled_family_rungs_are_skipped_midloop():
    """After the first Gemini rung cools the family, later Gemini rungs are
    skipped rather than re-called with the identical doomed request."""
    g1 = _FakeAgent(_GeminiModel(), raises=_http_error(400, _HARMONY_MSG))
    g2 = _FakeAgent(_GeminiModel(), raises=_http_error(400, _HARMONY_MSG))
    g3 = _FakeAgent(_GeminiModel(), raises=_http_error(400, _HARMONY_MSG))
    groq = _FakeAgent(_GroqModel(), result="OK")

    out = asyncio.run(run_agent_with_fallbacks(g1, [g2, g3, groq]))

    assert out == "OK"
    assert g1.calls == 1      # primary is always tried
    assert g2.calls == 0      # skipped — same cooled family as g1
    assert g3.calls == 0      # skipped — same cooled family as g1
    assert groq.calls == 1    # distinct family, not cooling → tried


def test_402_first_still_reaches_a_working_rung():
    """Even if the very first rung is a 402, the cascade proceeds."""
    openrouter = _FakeAgent(_OpenRouterModel(), raises=_http_error(402, "requires more credits"))
    groq = _FakeAgent(_GroqModel(), result="done")

    out = asyncio.run(run_agent_with_fallbacks(openrouter, [groq]))

    assert out == "done"
    assert groq.calls == 1
