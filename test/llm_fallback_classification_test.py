"""
core/llm_client.is_key_failure_error classification tests.

Migrated from test_tier0_verification.py (TestLLMClientFallbackFix) — the A5
fix that a generic HTTP 400 must NOT trigger a model swap, while harmony/
tool-render 400s and auth/rate-limit statuses must.

Pure classification behavior over mocked ModelHTTPError; no server/router import.
"""
from __future__ import annotations


class TestLLMClientFallbackFix:
    """Generic HTTP 400 no longer triggers model fallback; auth/rate/harmony do."""

    def test_generic_400_is_not_key_failure(self):
        """A plain 400 (bad request body) must NOT be treated as fallback-eligible."""
        from unittest.mock import MagicMock
        from pydantic_ai.exceptions import ModelHTTPError
        from core.llm_client import is_key_failure_error

        exc = MagicMock(spec=ModelHTTPError)
        exc.status_code = 400
        exc.__str__ = lambda self: "Bad Request: invalid JSON body"
        exc.__class__ = ModelHTTPError

        result = is_key_failure_error(exc)
        assert result is False, (
            "Generic HTTP 400 must NOT trigger model fallback — "
            "it should be handled semantically (clarify/retry args), not by swapping models. "
            f"Got is_key_failure_error={result}"
        )

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

    def test_402_out_of_credits_is_key_failure(self):
        """402 (provider out of credits/quota, e.g. OpenRouter) MUST be
        fallback-eligible — otherwise the cascade aborts before the Groq rescue
        rung and a tool turn fails outright. Regression for the live Discord turn
        where OpenRouter's 402 stranded llama-3.3-70b and the turn failed."""
        from unittest.mock import MagicMock
        from pydantic_ai.exceptions import ModelHTTPError
        from core.llm_client import is_key_failure_error

        exc = MagicMock(spec=ModelHTTPError)
        exc.status_code = 402
        exc.__str__ = lambda self: "This request requires more credits, or fewer max_tokens."
        exc.__class__ = ModelHTTPError

        assert is_key_failure_error(exc) is True
