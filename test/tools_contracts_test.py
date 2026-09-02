"""
tools/contracts.py behavior tests.

Migrated from test_tier0_verification.py (TestToolResultContract +
TestWebSearchArgsValidation) — the real behavior assertions, no source-grep.

Covers the ToolResult status envelope (ok/empty/invalid/rate_limited/
upstream_error), its retryable/retry_after semantics, the agent-facing string
rendering, and pydantic-level validation of WebSearchArgs.
"""
from __future__ import annotations


class TestToolResultContract:
    """ToolResult returns typed statuses for bad args, never raw exceptions."""

    def test_empty_query_returns_invalid_status(self):
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

        assert "Traceback" not in agent_str
        assert "Exception" not in agent_str
        assert "Error" not in agent_str or "Tool error" in agent_str
        assert len(agent_str) > 0

    def test_ok_result_returns_data_directly(self):
        from tools.contracts import ToolResult

        result = ToolResult.ok("some search results text")
        assert result.status == "ok"
        assert result.to_agent_string() == "some search results text"

    def test_empty_result_gives_human_message(self):
        from tools.contracts import ToolResult

        result = ToolResult.empty("No search results found for this query.")
        assert result.status == "empty"
        assert "found" in result.to_agent_string().lower() or "No" in result.to_agent_string()

    def test_upstream_error_is_retryable(self):
        from tools.contracts import ToolResult

        result = ToolResult.upstream_error("Connection timed out")
        assert result.status == "upstream_error"
        assert result.retryable is True

    def test_rate_limited_has_retry_after(self):
        from tools.contracts import ToolResult

        result = ToolResult.rate_limited(retry_after_ms=30_000, message="Too many requests")
        assert result.status == "rate_limited"
        assert result.retry_after_ms == 30_000
        assert result.retryable is True


class TestWebSearchArgsValidation:
    """WebSearchArgs rejects empty/too-short/too-long queries at the pydantic level."""

    def test_empty_query_raises_validation_error(self):
        import pytest
        from pydantic import ValidationError
        from tools.contracts import WebSearchArgs

        with pytest.raises(ValidationError) as exc_info:
            WebSearchArgs(query="")
        errors = exc_info.value.errors()
        assert any(e["type"] in {"string_too_short", "value_error"} for e in errors), \
            f"Expected string_too_short error, got: {errors}"

    def test_single_char_query_raises_validation_error(self):
        import pytest
        from pydantic import ValidationError
        from tools.contracts import WebSearchArgs

        with pytest.raises(ValidationError):
            WebSearchArgs(query="x")

    def test_valid_query_passes(self):
        from tools.contracts import WebSearchArgs

        args = WebSearchArgs(query="Tokyo time now")
        assert args.query == "Tokyo time now"

    def test_query_too_long_raises_validation_error(self):
        import pytest
        from pydantic import ValidationError
        from tools.contracts import WebSearchArgs

        with pytest.raises(ValidationError):
            WebSearchArgs(query="x" * 301)
