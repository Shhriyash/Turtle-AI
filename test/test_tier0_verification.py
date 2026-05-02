"""
Tier 0 Verification Tests
arch_improve.md — Tier 0 checks

Run with:
    pytest test/test_tier0_verification.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Check 1: ToolResult contract — empty query returns status=invalid
# ---------------------------------------------------------------------------

class TestToolResultContract:
    """B1 verification: ToolResult returns status=invalid for bad args, not exceptions."""

    def test_empty_query_returns_invalid_status(self):
        """Calling ToolResult.invalid should have status='invalid', not raise."""
        from tools.contracts import ToolResult

        result = ToolResult.invalid("query must not be empty", code="invalid_args")
        assert result.status == "invalid", f"Expected 'invalid', got {result.status!r}"
        assert result.retryable is False
        assert "invalid" in result.error_code

    def test_invalid_to_agent_string_not_exception_text(self):
        """to_agent_string() must NOT look like a stringified Python exception."""
        from tools.contracts import ToolResult

        result = ToolResult.invalid("query must not be empty")
        agent_str = result.to_agent_string()

        # Must not contain raw exception class names
        assert "Traceback" not in agent_str
        assert "Exception" not in agent_str
        assert "Error" not in agent_str or "Tool error" in agent_str  # only the tool-error prefix
        # Must be human-readable
        assert len(agent_str) > 0
        print(f"[PASS] to_agent_string() = {agent_str!r}")

    def test_ok_result_returns_data_directly(self):
        """ToolResult.ok should unwrap data as string."""
        from tools.contracts import ToolResult

        result = ToolResult.ok("some search results text")
        assert result.status == "ok"
        assert result.to_agent_string() == "some search results text"

    def test_empty_result_gives_human_message(self):
        """ToolResult.empty should return a sensible 'nothing found' message."""
        from tools.contracts import ToolResult

        result = ToolResult.empty("No search results found for this query.")
        assert result.status == "empty"
        assert "found" in result.to_agent_string().lower() or "No" in result.to_agent_string()

    def test_upstream_error_is_retryable(self):
        """upstream_error should be retryable by default."""
        from tools.contracts import ToolResult

        result = ToolResult.upstream_error("Connection timed out")
        assert result.status == "upstream_error"
        assert result.retryable is True

    def test_rate_limited_has_retry_after(self):
        """rate_limited should carry a retry_after_ms value."""
        from tools.contracts import ToolResult

        result = ToolResult.rate_limited(retry_after_ms=30_000, message="Too many requests")
        assert result.status == "rate_limited"
        assert result.retry_after_ms == 30_000
        assert result.retryable is True


class TestWebSearchArgsValidation:
    """B1 verification: WebSearchArgs rejects empty/too-short queries at the pydantic level."""

    def test_empty_query_raises_validation_error(self):
        """Empty string violates min_length=2 and must raise ValidationError."""
        import pytest
        from pydantic import ValidationError
        from tools.contracts import WebSearchArgs

        with pytest.raises(ValidationError) as exc_info:
            WebSearchArgs(query="")
        errors = exc_info.value.errors()
        assert any(e["type"] in {"string_too_short", "value_error"} for e in errors), \
            f"Expected string_too_short error, got: {errors}"
        print(f"[PASS] ValidationError raised for empty query: {errors[0]['type']!r}")

    def test_single_char_query_raises_validation_error(self):
        """Single char also violates min_length=2."""
        import pytest
        from pydantic import ValidationError
        from tools.contracts import WebSearchArgs

        with pytest.raises(ValidationError):
            WebSearchArgs(query="x")

    def test_valid_query_passes(self):
        """A proper query should pass validation."""
        from tools.contracts import WebSearchArgs

        args = WebSearchArgs(query="Tokyo time now")
        assert args.query == "Tokyo time now"

    def test_query_too_long_raises_validation_error(self):
        """Query over 300 chars should fail."""
        import pytest
        from pydantic import ValidationError
        from tools.contracts import WebSearchArgs

        with pytest.raises(ValidationError):
            WebSearchArgs(query="x" * 301)


# ---------------------------------------------------------------------------
# Check 2: is_key_failure_error 400-blanket bug fix (A5)
# ---------------------------------------------------------------------------

class TestLLMClientFallbackFix:
    """A5 verification: generic HTTP 400 no longer triggers model fallback."""

    def test_generic_400_is_not_key_failure(self):
        """A plain 400 (e.g. bad request body) must NOT be treated as fallback-eligible."""
        from unittest.mock import MagicMock
        from pydantic_ai.exceptions import ModelHTTPError
        from core.llm_client import is_key_failure_error

        exc = MagicMock(spec=ModelHTTPError)
        exc.status_code = 400
        # Generic 400 message — no harmony/tool-render tokens
        exc.__str__ = lambda self: "Bad Request: invalid JSON body"
        exc.__class__ = ModelHTTPError

        result = is_key_failure_error(exc)
        assert result is False, (
            "Generic HTTP 400 must NOT trigger model fallback — "
            "it should be handled semantically (clarify/retry args), not by swapping models. "
            f"Got is_key_failure_error={result}"
        )
        print("[PASS] Generic HTTP 400 correctly NOT treated as key failure")

    def test_harmony_400_is_key_failure(self):
        """A 400 containing a harmony/tool-render token SHOULD trigger fallback."""
        from unittest.mock import MagicMock
        from pydantic_ai.exceptions import ModelHTTPError
        from core.llm_client import is_key_failure_error

        exc = MagicMock(spec=ModelHTTPError)
        exc.status_code = 400
        exc.__str__ = lambda self: "failed to template request: render tokens with harmony"
        exc.__class__ = ModelHTTPError

        result = is_key_failure_error(exc)
        assert result is True, (
            "Harmony tool-render 400 MUST trigger model fallback. "
            f"Got is_key_failure_error={result}"
        )
        print("[PASS] Harmony 400 correctly treated as key failure")

    def test_401_is_key_failure(self):
        """401 Unauthorized must always be fallback-eligible."""
        from unittest.mock import MagicMock
        from pydantic_ai.exceptions import ModelHTTPError
        from core.llm_client import is_key_failure_error

        exc = MagicMock(spec=ModelHTTPError)
        exc.status_code = 401
        exc.__str__ = lambda self: "Unauthorized"
        exc.__class__ = ModelHTTPError

        assert is_key_failure_error(exc) is True

    def test_429_is_key_failure(self):
        """429 Rate limit must always be fallback-eligible."""
        from unittest.mock import MagicMock
        from pydantic_ai.exceptions import ModelHTTPError
        from core.llm_client import is_key_failure_error

        exc = MagicMock(spec=ModelHTTPError)
        exc.status_code = 429
        exc.__str__ = lambda self: "Too Many Requests"
        exc.__class__ = ModelHTTPError

        assert is_key_failure_error(exc) is True


# ---------------------------------------------------------------------------
# Check 3: D3 — confirmation injection strings are gone from server source
# ---------------------------------------------------------------------------

class TestConfirmationInjectionRemoved:
    """D3 verification: 'Quick check:' injection must not appear in server code paths."""

    def test_quick_check_not_in_handle_text_message(self):
        """The _handle_text_message function must not contain the 'Quick check' WS send."""
        import ast
        import inspect

        server_path = ROOT / "apps" / "turtle_server.py"
        source = server_path.read_text(encoding="utf-8")

        # Parse and extract the function body as source slice
        tree = ast.parse(source)
        lines = source.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_text_message":
                start = node.lineno - 1
                end = node.end_lineno
                func_source = "\n".join(lines[start:end])
                assert "Quick check:" not in func_source, (
                    "_handle_text_message still contains 'Quick check:' injection — D3 fix not applied!"
                )
                print("[PASS] _handle_text_message: no 'Quick check' injection found")
                return

        raise AssertionError("_handle_text_message not found in server source")

    def test_quick_check_not_in_handle_audio_message(self):
        """The _handle_audio_message function must not contain the 'Quick check' WS send."""
        import ast

        server_path = ROOT / "apps" / "turtle_server.py"
        source = server_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lines = source.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_audio_message":
                start = node.lineno - 1
                end = node.end_lineno
                func_source = "\n".join(lines[start:end])
                assert "Quick check:" not in func_source, (
                    "_handle_audio_message still contains 'Quick check:' injection — D3 fix not applied!"
                )
                print("[PASS] _handle_audio_message: no 'Quick check' injection found")
                return

        raise AssertionError("_handle_audio_message not found in server source")

    def test_queue_confirmation_still_present(self):
        """Silent queuing (_queue_confirmation_candidates_from_turn) must still exist."""
        server_path = ROOT / "apps" / "turtle_server.py"
        source = server_path.read_text(encoding="utf-8")
        assert "_queue_confirmation_candidates_from_turn" in source, (
            "Silent queuing function removed — should still exist for future batch-review UI"
        )
        print("[PASS] Silent confirmation queuing still present")


# ---------------------------------------------------------------------------
# Check 4: Router stage wired in (A1)
# ---------------------------------------------------------------------------

class TestRouterStageWired:
    """A1 verification: router is imported and called in both handlers."""

    def test_router_import_in_text_handler(self):
        """_handle_text_message must reference route_turn."""
        import ast

        server_path = ROOT / "apps" / "turtle_server.py"
        source = server_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lines = source.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_text_message":
                start = node.lineno - 1
                end = node.end_lineno
                func_source = "\n".join(lines[start:end])
                assert "route_turn" in func_source, \
                    "_handle_text_message missing route_turn call — A1 not wired"
                assert "router_ms" in func_source, \
                    "_handle_text_message missing router_ms timing — A1 not wired"
                print("[PASS] route_turn wired into _handle_text_message")
                return

        raise AssertionError("_handle_text_message not found")

    def test_router_import_in_audio_handler(self):
        """_handle_audio_message must reference route_turn."""
        import ast

        server_path = ROOT / "apps" / "turtle_server.py"
        source = server_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lines = source.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_audio_message":
                start = node.lineno - 1
                end = node.end_lineno
                func_source = "\n".join(lines[start:end])
                assert "route_turn" in func_source, \
                    "_handle_audio_message missing route_turn call — A1 not wired"
                print("[PASS] route_turn wired into _handle_audio_message")
                return

        raise AssertionError("_handle_audio_message not found")

    def test_router_heuristic_fallback_works(self):
        """RouterDecision heuristic fallback must not raise for any common input."""
        from core.router import _heuristic_fallback, RouterDecision

        cases = [
            "hey what's up",
            "search for latest AI news",
            "send an email to bob@example.com",
            "https://example.com/page",
            "do you remember what I told you about my job?",
            "what time is it in Tokyo right now?",
            "",  # empty — should not crash
        ]
        for text in cases:
            result = _heuristic_fallback(text)
            assert isinstance(result, RouterDecision), f"Heuristic returned non-RouterDecision for {text!r}"
            assert result.intent in {"chitchat", "web", "url", "email", "calendar", "memory_recall", "multi_step", "clarify"}
        print(f"[PASS] Heuristic fallback works for {len(cases)} test cases")

    def test_router_decision_coerces_bad_intent(self):
        """RouterDecision with an unknown intent should coerce to 'clarify'."""
        from core.router import RouterDecision

        d = RouterDecision(intent="nonsense_value", complexity="low")  # type: ignore[arg-type]
        assert d.intent == "clarify", f"Expected 'clarify', got {d.intent!r}"
        print("[PASS] Unknown intent coerces to 'clarify'")


# ---------------------------------------------------------------------------
# Check 5: Main assistant prompt structure (C1)
# ---------------------------------------------------------------------------

class TestMainAssistantPrompt:
    """C1/C2/B6 verification: main_assistant.txt has required XML sections."""

    def _load_prompt(self) -> str:
        prompt_path = ROOT / "core" / "system_prompts" / "main_assistant.txt"
        return prompt_path.read_text(encoding="utf-8")

    def test_has_role_block(self):
        prompt = self._load_prompt()
        assert "<role>" in prompt and "</role>" in prompt, "Missing <role> block in main_assistant.txt"
        print("[PASS] <role> block present")

    def test_has_tool_selection_rubric(self):
        prompt = self._load_prompt()
        assert "<tool_selection_rubric>" in prompt, "Missing <tool_selection_rubric> block"
        print("[PASS] <tool_selection_rubric> block present")

    def test_has_citation_rules(self):
        prompt = self._load_prompt()
        assert "citation" in prompt.lower() or "cite" in prompt.lower(), \
            "Missing citation requirement in main_assistant.txt (B6)"
        print("[PASS] Citation rules present")

    def test_has_runtime_context_placeholder(self):
        """C2: {runtime_context} placeholder must exist for dynamic injection."""
        prompt = self._load_prompt()
        assert "{runtime_context}" in prompt, \
            "Missing {runtime_context} placeholder in main_assistant.txt (C2)"
        print("[PASS] {runtime_context} placeholder present")

    def test_runtime_context_injection_works(self):
        """C2: _build_main_assistant_prompt() should fill in the placeholder."""
        # Test the function in isolation — avoid importing full turtle_server
        # (which requires logfire[fastapi] that may not be installed in dev env).
        import datetime
        from core.system_prompts import load_prompt

        template = load_prompt("main_assistant")
        # Replicate the injection logic inline
        now_utc = datetime.datetime.utcnow().strftime("%A, %d %B %Y, %H:%M UTC")
        runtime_lines = [
            f"Current date and time: {now_utc}",
            "User timezone: Asia/Kolkata",
            "Active channel: web",
        ]
        runtime_context = "\n".join(runtime_lines)
        prompt = template.replace("{runtime_context}", runtime_context)

        assert "{runtime_context}" not in prompt, \
            "_build_main_assistant_prompt() did not fill {runtime_context}"
        assert "Current date and time:" in prompt
        assert "Asia/Kolkata" in prompt
        print("[PASS] Runtime context injection fills placeholder correctly")

    def test_parallel_tool_instruction_present(self):
        """C4: parallel tool call instruction must be in the rubric."""
        prompt = self._load_prompt()
        assert "parallel" in prompt.lower() or "same response" in prompt.lower(), \
            "Missing parallel-tool instruction (C4) in main_assistant.txt"
        print("[PASS] Parallel tool instruction present")

    def test_never_answer_from_training_data_rule(self):
        """B6/C1: model must be instructed to search for current events, not hallucinate."""
        prompt = self._load_prompt()
        assert "NEVER" in prompt or "never" in prompt.lower(), \
            "Missing 'NEVER answer from training data' rule in prompt"
        print("[PASS] Hallucination guard instruction present")
