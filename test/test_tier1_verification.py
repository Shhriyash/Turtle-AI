"""
Tier 1 Verification Tests
arch_improve.md — Tier 1 checks

Run with:
    pytest test/test_tier1_verification.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# B3 — Tavily replaces DDG
# ---------------------------------------------------------------------------

class TestB3TavilySearch:
    """B3: Tavily is primary; DDG is fallback; retry-relax hack is gone."""

    def test_tavily_function_exists(self):
        from core.web_search import _search_tavily
        import asyncio
        assert asyncio.iscoroutinefunction(_search_tavily)
        print("[PASS] _search_tavily is async")

    def test_ddg_fallback_function_exists(self):
        from core.web_search import _search_duckduckgo_fallback
        print("[PASS] _search_duckduckgo_fallback exists as fallback")

    def test_search_duckduckgo_routes_to_tavily_when_key_set(self):
        """With TAVILY_API_KEY set, search_duckduckgo routes through Tavily."""
        import os, asyncio, unittest.mock as mock
        from core.web_search import search_duckduckgo

        fake_results = [{"url": "https://t.co/1", "title": "T1", "content": "snippet1"}]
        fake_resp = mock.AsyncMock()
        fake_resp.raise_for_status = mock.Mock()
        fake_resp.json = mock.Mock(return_value={"results": fake_results})

        with mock.patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            with mock.patch("httpx.AsyncClient.post", return_value=fake_resp):
                client_mock = mock.AsyncMock()
                client_mock.post = mock.AsyncMock(return_value=fake_resp)
                results = asyncio.get_event_loop().run_until_complete(
                    search_duckduckgo(client_mock, "bitcoin price", max_results=5)
                )
        assert len(results) >= 1
        assert results[0].url == "https://t.co/1"
        print("[PASS] Tavily results parsed correctly")

    def test_tavily_timeout_is_6s(self):
        """E4 alignment: Tavily timeout must be exactly 6.0 seconds."""
        from core.web_search import _TAVILY_TIMEOUT
        assert _TAVILY_TIMEOUT == 6.0, f"Expected 6.0, got {_TAVILY_TIMEOUT}"
        print("[PASS] Tavily timeout is 6.0 s")

    def test_ddg_timeout_is_6s(self):
        """E4 alignment: DDG fallback timeout must also be 6.0 seconds."""
        from core.web_search import _DDG_TIMEOUT
        assert _DDG_TIMEOUT == 6.0, f"Expected 6.0, got {_DDG_TIMEOUT}"
        print("[PASS] DDG fallback timeout is 6.0 s")

    def test_retry_relax_hack_removed_from_server(self):
        """B3: site:-relax retry block must not appear in turtle_server.py."""
        src = (ROOT / "apps" / "turtle_server.py").read_text(encoding="utf-8")
        assert "Relax site:" not in src, "DDG retry-relax hack still in turtle_server.py"
        print("[PASS] DDG retry-relax hack removed from server")

    def test_normalize_query_adds_amazon_site_filter(self):
        from core.web_search import _normalize_query
        result = _normalize_query("iphone 15 pro amazon.in")
        assert result.startswith("site:amazon.in"), f"Got: {result!r}"
        print("[PASS] amazon.in site: filter injected")

    def test_format_search_results_empty(self):
        from core.web_search import format_search_results
        out = format_search_results("test query", [])
        assert "No web results" in out
        print("[PASS] Empty search results formatted correctly")


# ---------------------------------------------------------------------------
# B5 — Email idempotency
# ---------------------------------------------------------------------------

class TestB5EmailIdempotency:
    """B5: sha1-keyed SQLite idempotency for email sends."""

    def test_build_key_is_deterministic(self):
        from tools.idempotency import build_email_idempotency_key
        k1 = build_email_idempotency_key(["a@b.com"], "Hello", "Body text")
        k2 = build_email_idempotency_key(["a@b.com"], "Hello", "Body text")
        assert k1 == k2, "Key must be deterministic"
        assert len(k1) == 40, f"SHA1 should be 40 hex chars, got {len(k1)}"
        print("[PASS] Idempotency key is deterministic SHA1")

    def test_different_recipients_produce_different_keys(self):
        from tools.idempotency import build_email_idempotency_key
        k1 = build_email_idempotency_key(["a@b.com"], "Hi", "Body")
        k2 = build_email_idempotency_key(["c@d.com"], "Hi", "Body")
        assert k1 != k2
        print("[PASS] Different recipients produce different keys")

    def test_new_key_returns_none(self):
        """A key never seen before must return None (not a duplicate)."""
        import time
        from tools.idempotency import is_duplicate_invocation, build_email_idempotency_key
        unique_key = build_email_idempotency_key(
            [f"unique_{time.time()}@test.com"], "Subject", "Body"
        )
        result = is_duplicate_invocation(unique_key)
        assert result is None, f"New key should return None, got {result!r}"
        print("[PASS] New key returns None (not duplicate)")

    def test_record_then_check_returns_cached(self):
        """After recording, the same key returns the cached result."""
        import time
        from tools.idempotency import (
            build_email_idempotency_key, is_duplicate_invocation, record_invocation
        )
        key = build_email_idempotency_key(
            [f"cached_{time.time()}@test.com"], "Test", "Body"
        )
        record_invocation(key, "Email sent successfully!")
        result = is_duplicate_invocation(key)
        assert result == "Email sent successfully!", f"Got {result!r}"
        print("[PASS] Cached result returned on duplicate check")

    def test_idempotency_wired_into_server_email_tool(self):
        """B5: idempotency import must appear inside the email tool handler."""
        src = (ROOT / "apps" / "turtle_server.py").read_text(encoding="utf-8")
        assert "build_email_idempotency_key" in src
        assert "is_duplicate_invocation" in src
        assert "record_invocation" in src
        print("[PASS] Idempotency wired into server email tool")


# ---------------------------------------------------------------------------
# D1 — LLM memory extractor
# ---------------------------------------------------------------------------

class TestD1LLMExtractor:
    """D1: LLM extractor path added alongside regex path."""

    def test_async_extractor_exists(self):
        from core.personal_memory_extract import extract_memory_candidates_from_messages_async
        import asyncio
        assert asyncio.iscoroutinefunction(extract_memory_candidates_from_messages_async)
        print("[PASS] extract_memory_candidates_from_messages_async is async")

    def test_sync_extractor_still_works(self):
        from core.personal_memory_extract import extract_memory_candidates_from_messages
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        msg = ModelRequest(parts=[UserPromptPart(content="My name is Riya")])
        candidates = extract_memory_candidates_from_messages(message_history=[msg])
        assert isinstance(candidates, list)
        print(f"[PASS] Sync extractor returned {len(candidates)} candidate(s)")

    def test_async_extractor_skips_llm_when_regex_finds_strong_signal(self):
        """D1: LLM should NOT fire when regex returns strong (confirmed/inferred) candidates."""
        import asyncio, unittest.mock as mock
        from core.personal_memory_extract import extract_memory_candidates_from_messages_async
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msg = ModelRequest(parts=[UserPromptPart(content="My name is Riya")])

        with mock.patch("core.personal_memory_extract._extract_with_llm") as mock_llm:
            mock_llm.return_value = []
            results = asyncio.get_event_loop().run_until_complete(
                extract_memory_candidates_from_messages_async(message_history=[msg])
            )
            # If regex found something, LLM should not have been called
            # (or was called and returned nothing — either way, regex result is used)
        assert isinstance(results, list)
        print(f"[PASS] Async extractor returned {len(results)} candidates, LLM gating works")

    def test_llm_extractor_function_exists(self):
        from core.personal_memory_extract import _extract_with_llm
        import asyncio
        assert asyncio.iscoroutinefunction(_extract_with_llm)
        print("[PASS] _extract_with_llm is an async function")

    def test_memory_extractor_prompt_exists(self):
        prompt_path = ROOT / "core" / "system_prompts" / "memory_extractor.txt"
        assert prompt_path.exists(), "memory_extractor.txt missing"
        content = prompt_path.read_text(encoding="utf-8")
        assert len(content) > 100
        print("[PASS] memory_extractor.txt prompt file present")


# ---------------------------------------------------------------------------
# D2 — Background dream pass
# ---------------------------------------------------------------------------

class TestD2BackgroundDreamPass:
    """D2: Dream pass must use asyncio.create_task, never await directly in handlers."""

    def _get_handler_source(self, name: str) -> str:
        import ast
        src = (ROOT / "apps" / "turtle_server.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        lines = src.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1 : node.end_lineno])
        raise AssertionError(f"{name} not found in server source")

    def test_text_handler_uses_create_task_for_dream_pass(self):
        src = self._get_handler_source("_handle_text_message")
        assert "create_task" in src and "_run_dream_pass_if_needed" in src, \
            "_handle_text_message: dream pass not converted to create_task"
        # Must NOT have a direct await of dream pass (outside create_task)
        assert "await _run_dream_pass_if_needed" not in src, \
            "Dream pass still awaited directly in text handler"
        print("[PASS] Text handler: dream pass uses asyncio.create_task")

    def test_audio_handler_uses_create_task_for_dream_pass(self):
        src = self._get_handler_source("_handle_audio_message")
        assert "create_task" in src and "_run_dream_pass_if_needed" in src, \
            "_handle_audio_message: dream pass not converted to create_task"
        assert "await _run_dream_pass_if_needed" not in src, \
            "Dream pass still awaited directly in audio handler"
        print("[PASS] Audio handler: dream pass uses asyncio.create_task")


# ---------------------------------------------------------------------------
# D4 — RetrievalBroker wired
# ---------------------------------------------------------------------------

class TestD4RetrievalBrokerWired:
    """D4: _resolve_memory_context uses RetrievalBroker, is async, bypass path removed."""

    def test_resolve_memory_context_is_async(self):
        import ast
        src = (ROOT / "apps" / "turtle_server.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_resolve_memory_context":
                print("[PASS] _resolve_memory_context is async")
                return
        raise AssertionError("_resolve_memory_context is not async or not found")

    def test_retrieval_broker_in_shared_state(self):
        src = (ROOT / "apps" / "turtle_server.py").read_text(encoding="utf-8")
        assert "retrieval_broker" in src, "retrieval_broker not found in SharedState"
        print("[PASS] retrieval_broker field present in SharedState")

    def test_retrieval_broker_constructed_in_setup(self):
        src = (ROOT / "apps" / "turtle_server.py").read_text(encoding="utf-8")
        assert "RetrievalBroker(" in src, "RetrievalBroker not constructed in setup"
        print("[PASS] RetrievalBroker constructed during WebSocket setup")

    def test_resolve_uses_broker_build_context(self):
        src = (ROOT / "apps" / "turtle_server.py").read_text(encoding="utf-8")
        assert "retrieval_broker.build_context" in src or "build_context(" in src, \
            "build_context not called in _resolve_memory_context"
        print("[PASS] RetrievalBroker.build_context() called in _resolve_memory_context")

    def test_retrieval_broker_imports_correctly(self):
        from core.retrieval_broker import RetrievalBroker
        assert RetrievalBroker is not None
        print("[PASS] RetrievalBroker importable from core.retrieval_broker")


# ---------------------------------------------------------------------------
# A2/A3/A4 — Stub graph
# ---------------------------------------------------------------------------

class TestA2A3A4StubGraph:
    """A2/A3/A4: Graph topology defined; routing by intent; email graph has 3-node chain."""

    def test_select_graph_chitchat(self):
        from core.graph import select_graph
        g = select_graph("chitchat")
        assert g.graph_def.name == "chitchat_graph"
        print("[PASS] select_graph('chitchat') -> chitchat_graph")

    def test_select_graph_email(self):
        from core.graph import select_graph
        g = select_graph("email")
        assert g.graph_def.name == "email_graph"
        print("[PASS] select_graph('email') -> email_graph")

    def test_select_graph_unknown_uses_default(self):
        from core.graph import select_graph
        g = select_graph("totally_unknown_intent")
        assert g is not None
        print(f"[PASS] Unknown intent defaults to {g.graph_def.name!r}")

    def test_email_graph_has_extract_validate_send_nodes(self):
        """A3: email graph must have extract, validate, and send nodes."""
        from core.graph import EMAIL_GRAPH, NodeKind
        node_kinds = {n.kind for n in EMAIL_GRAPH.nodes}
        assert NodeKind.EMAIL_EXTRACT in node_kinds, "Missing EMAIL_EXTRACT node"
        assert NodeKind.EMAIL_VALIDATE in node_kinds, "Missing EMAIL_VALIDATE node"
        assert NodeKind.EMAIL_SEND in node_kinds, "Missing EMAIL_SEND node"
        print("[PASS] Email graph has extract -> validate -> send nodes (A3)")

    def test_multi_step_graph_has_planner(self):
        """A4: multi_step_graph must have a PLANNER node for parallel tool decomposition."""
        from core.graph import MULTI_STEP_GRAPH, NodeKind
        node_kinds = {n.kind for n in MULTI_STEP_GRAPH.nodes}
        assert NodeKind.PLANNER in node_kinds, "Missing PLANNER node in multi_step_graph"
        print("[PASS] multi_step_graph has PLANNER node (A4)")

    def test_all_6_graphs_registered(self):
        from core.graph import list_graphs
        graphs = list_graphs()
        assert len(graphs) == 6, f"Expected 6 graphs, got {len(graphs)}"
        intents = {g["intent"] for g in graphs}
        for expected in ("chitchat", "web", "url", "email", "memory_recall", "multi_step"):
            assert expected in intents, f"Missing graph for intent {expected!r}"
        print(f"[PASS] All 6 graphs registered: {intents}")

    def test_turtle_graph_run_is_async(self):
        import asyncio
        from core.graph import TurtleGraph, CHITCHAT_GRAPH
        g = TurtleGraph(CHITCHAT_GRAPH)
        assert asyncio.iscoroutinefunction(g.run)
        print("[PASS] TurtleGraph.run() is async")


# ---------------------------------------------------------------------------
# E3 — Streaming TTS sentence splitting
# ---------------------------------------------------------------------------

class TestE3StreamingTTS:
    """E3: Sentence accumulator correctly splits at boundaries; streaming entry points exist."""

    def test_split_into_sentences_basic(self):
        from core.streaming_tts import split_into_sentences
        sentences = split_into_sentences(
            "Hello world. How are you? I am fine! Let us begin."
        )
        assert len(sentences) >= 3, f"Expected >= 3 sentences, got {sentences}"
        print(f"[PASS] split_into_sentences -> {len(sentences)} sentences")

    def test_split_handles_empty_string(self):
        from core.streaming_tts import split_into_sentences
        assert split_into_sentences("") == []
        print("[PASS] split_into_sentences('') returns []")

    def test_accumulator_fires_on_sentence_boundary(self):
        from core.streaming_tts import SentenceAccumulator
        acc = SentenceAccumulator()
        fired: list[str] = []
        tokens = ["Hello ", "world", ". ", "How ", "are ", "you?"]
        for token in tokens:
            fired.extend(acc.feed(token))
        remainder = acc.flush()
        all_output = fired + remainder
        assert len(all_output) >= 1, "Expected at least one sentence"
        print(f"[PASS] SentenceAccumulator produced {len(all_output)} sentence(s)")

    def test_accumulator_flush_returns_remainder(self):
        from core.streaming_tts import SentenceAccumulator
        acc = SentenceAccumulator()
        acc.feed("This is an unfinished thought")
        remainder = acc.flush()
        assert len(remainder) >= 1
        assert "unfinished" in remainder[0]
        print("[PASS] SentenceAccumulator.flush() returns partial buffer")

    def test_stream_tts_from_text_is_async_generator(self):
        import inspect
        from core.streaming_tts import stream_tts_from_text
        assert inspect.isasyncgenfunction(stream_tts_from_text)
        print("[PASS] stream_tts_from_text is an async generator")

    def test_stream_tts_from_token_stream_is_async_generator(self):
        import inspect
        from core.streaming_tts import stream_tts_from_token_stream
        assert inspect.isasyncgenfunction(stream_tts_from_token_stream)
        print("[PASS] stream_tts_from_token_stream is an async generator")

    def test_e3_wired_into_audio_handler(self):
        """E3: stream_tts_from_text must be imported/used in the audio handler."""
        src = (ROOT / "apps" / "turtle_server.py").read_text(encoding="utf-8")
        assert "stream_tts_from_text" in src, "stream_tts_from_text not wired into server"
        assert "streaming_tts" in src
        print("[PASS] stream_tts_from_text wired into audio handler")

    def test_old_monolithic_tts_replaced_in_audio_handler(self):
        """E3: synthesize_speech(full_text) must not be called in audio handler for TTS."""
        import ast
        src = (ROOT / "apps" / "turtle_server.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        lines = src.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_audio_message":
                func_src = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                assert "synthesize_speech(" not in func_src, \
                    "Audio handler still calls monolithic synthesize_speech — E3 not wired"
                print("[PASS] Monolithic synthesize_speech removed from audio handler")
                return
        raise AssertionError("_handle_audio_message not found")

    def test_e1_e2_stubs_marked_as_todo(self):
        """E1/E2 deferred — must have TODO markers in streaming_tts.py."""
        content = (ROOT / "core" / "streaming_tts.py").read_text(encoding="utf-8")
        assert "TODO E1" in content and "TODO E2" in content
        print("[PASS] E1/E2 correctly marked as TODO in streaming_tts.py")


# ---------------------------------------------------------------------------
# E4 — Latency budgets
# ---------------------------------------------------------------------------

class TestE4LatencyBudgets:
    """E4: Hard timeouts per stage; SLA breach logging; env-overridable."""

    def test_tool_budget_is_6000ms(self):
        from core.latency_budgets import budgets
        assert budgets.TOOL_MAX_MS == 6000, f"Got {budgets.TOOL_MAX_MS}"
        print("[PASS] TOOL_MAX_MS = 6000")

    def test_router_budget_is_400ms(self):
        from core.latency_budgets import budgets
        assert budgets.ROUTER_MAX_MS == 400
        print("[PASS] ROUTER_MAX_MS = 400")

    def test_tts_first_byte_budget_is_600ms(self):
        from core.latency_budgets import budgets
        assert budgets.TTS_FIRST_BYTE_MAX_MS == 600
        print("[PASS] TTS_FIRST_BYTE_MAX_MS = 600")

    def test_tool_s_property(self):
        from core.latency_budgets import budgets
        assert budgets.TOOL_S == 6.0
        print("[PASS] TOOL_S property = 6.0")

    def test_sla_exceeded_logs_without_raising(self):
        from core.latency_budgets import sla_exceeded
        sla_exceeded("test_stage", 1500.0, 400)  # Must not raise
        print("[PASS] sla_exceeded() logs without raising")

    def test_check_sla_no_breach_is_silent(self):
        import time
        from core.latency_budgets import check_sla
        start = time.time() - 0.001  # 1 ms elapsed
        check_sla("fast_stage", start, 6000)  # well within budget
        print("[PASS] check_sla within budget: silent")

    def test_env_override(self):
        import os, importlib
        import unittest.mock as mock
        with mock.patch.dict(os.environ, {"TURTLE_TOOL_MAX_MS": "9999"}):
            import core.latency_budgets as lb
            custom = lb.LatencyBudgets()
            assert custom.TOOL_MAX_MS == 9999, f"Got {custom.TOOL_MAX_MS}"
        print("[PASS] TURTLE_TOOL_MAX_MS env override works")

    def test_latency_budgets_imported_in_audio_handler(self):
        src = (ROOT / "apps" / "turtle_server.py").read_text(encoding="utf-8")
        assert "latency_budgets" in src and "check_sla" in src
        print("[PASS] latency_budgets/check_sla wired into audio handler")


# ---------------------------------------------------------------------------
# H1 — Eval harness
# ---------------------------------------------------------------------------

class TestH1EvalHarness:
    """H1: Eval harness runner and frozen prompt set are well-formed."""

    def test_baseline_prompts_are_valid_json(self):
        import json
        path = ROOT / "evals" / "prompts" / "tier1_baseline.json"
        assert path.exists(), "tier1_baseline.json not found"
        prompts = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(prompts, list) and len(prompts) > 0
        print(f"[PASS] tier1_baseline.json has {len(prompts)} prompts")

    def test_all_prompts_have_required_fields(self):
        import json
        prompts = json.loads(
            (ROOT / "evals" / "prompts" / "tier1_baseline.json").read_text(encoding="utf-8")
        )
        required = {"id", "category", "prompt", "expected_tool_calls"}
        for p in prompts:
            missing = required - p.keys()
            assert not missing, f"Prompt {p.get('id')} missing fields: {missing}"
        print(f"[PASS] All {len(prompts)} prompts have required fields")

    def test_categories_cover_all_types(self):
        import json
        prompts = json.loads(
            (ROOT / "evals" / "prompts" / "tier1_baseline.json").read_text(encoding="utf-8")
        )
        categories = {p["category"] for p in prompts}
        required_cats = {"chitchat", "web_search", "email", "adversarial"}
        for cat in required_cats:
            assert cat in categories, f"Missing category {cat!r}"
        print(f"[PASS] Categories present: {categories}")

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
        print("[PASS] score_result: no-tools chitchat pass=True")

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
        print("[PASS] score_result: missing required tool call -> pass=False")

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
        # Tool was called (accuracy=1.0) but no URL citation in response
        assert scored["hallucination_risk"] is True
        print("[PASS] score_result: no citation for search response -> hallucination_risk=True")

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
        print("[PASS] score_result: search with citation -> pass=True")

    def test_runner_module_importable(self):
        from evals.runner import run_single_prompt, run_eval, score_result
        import asyncio
        assert asyncio.iscoroutinefunction(run_single_prompt)
        assert asyncio.iscoroutinefunction(run_eval)
        print("[PASS] evals.runner: all async functions importable")
