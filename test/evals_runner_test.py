"""
evals/runner.py scoring + frozen prompt-set tests.

Migrated from test_tier1_verification.py (TestH1EvalHarness). All behavior:
score_result's tool-accuracy / hallucination-risk / pass rules and the
tier1_baseline.json schema.

Schema assertions check field *presence* and category coverage only — they do
NOT hardcode any specific expected tool name (e.g. history_tool vs recall), so
they stay green if the baseline's expected_tool_calls values are retuned.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BASELINE = ROOT / "evals" / "prompts" / "tier1_baseline.json"


class TestH1EvalHarness:
    """Eval harness runner and frozen prompt set are well-formed."""

    def test_baseline_prompts_are_valid_json(self):
        assert _BASELINE.exists(), "tier1_baseline.json not found"
        prompts = json.loads(_BASELINE.read_text(encoding="utf-8"))
        assert isinstance(prompts, list) and len(prompts) > 0

    def test_all_prompts_have_required_fields(self):
        prompts = json.loads(_BASELINE.read_text(encoding="utf-8"))
        required = {"id", "category", "prompt", "expected_tool_calls"}
        for p in prompts:
            missing = required - p.keys()
            assert not missing, f"Prompt {p.get('id')} missing fields: {missing}"

    def test_categories_cover_all_types(self):
        prompts = json.loads(_BASELINE.read_text(encoding="utf-8"))
        categories = {p["category"] for p in prompts}
        required_cats = {"chitchat", "web_search", "email", "adversarial"}
        for cat in required_cats:
            assert cat in categories, f"Missing category {cat!r}"

    def test_runner_score_result_correct_for_no_tools(self):
        from evals.runner import score_result
        result = {
            "id": "ch_001", "category": "chitchat",
            "prompt": "hey", "response": "Hello!",
            "tool_calls_observed": [],
            "expected_tool_calls": [],
            "latency_ms": 500, "timings": {}, "error": None,
        }
        scored = score_result(result)
        assert scored["tool_accuracy"] == 1.0
        assert scored["hallucination_risk"] is False
        assert scored["pass"] is True

    def test_runner_score_flags_missing_tool_call(self):
        from evals.runner import score_result
        result = {
            "id": "web_001", "category": "web_search",
            "prompt": "Bitcoin price?", "response": "It is $50000.",
            "tool_calls_observed": [],        # No tool called — FAIL
            "expected_tool_calls": ["search_web"],
            "latency_ms": 800, "timings": {}, "error": None,
        }
        scored = score_result(result)
        assert scored["tool_accuracy"] == 0.0
        assert scored["pass"] is False

    def test_runner_score_flags_hallucination_risk(self):
        """Response with no citation for a search-required query is a hallucination risk."""
        from evals.runner import score_result
        result = {
            "id": "web_002", "category": "web_search",
            "prompt": "Latest AI news?", "response": "OpenAI released GPT-5.",
            "tool_calls_observed": ["search_web"],
            "expected_tool_calls": ["search_web"],
            "latency_ms": 900, "timings": {}, "error": None,
        }
        scored = score_result(result)
        assert scored["hallucination_risk"] is True

    def test_runner_score_passes_with_citation(self):
        from evals.runner import score_result
        result = {
            "id": "web_003", "category": "web_search",
            "prompt": "Bitcoin price?",
            "response": "Bitcoin is $65,000. Source: https://coinmarketcap.com",
            "tool_calls_observed": ["search_web"],
            "expected_tool_calls": ["search_web"],
            "latency_ms": 1200, "timings": {}, "error": None,
        }
        scored = score_result(result)
        assert scored["tool_accuracy"] == 1.0
        assert scored["has_citation"] is True
        assert scored["hallucination_risk"] is False
        assert scored["pass"] is True

    def test_runner_module_importable(self):
        import asyncio
        from evals.runner import run_single_prompt, run_eval, score_result
        assert asyncio.iscoroutinefunction(run_single_prompt)
        assert asyncio.iscoroutinefunction(run_eval)
        assert callable(score_result)
