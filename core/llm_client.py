from __future__ import annotations

import os
from typing import Any

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

OPENROUTER_DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
GROQ_DEFAULT_PRIMARY_MODEL = "llama-3.3-70b-versatile"
GROQ_DEFAULT_FALLBACK_MODEL = "llama-3.1-8b-instant"
OPENROUTER_KEY_ENV_VARS = [
    "OPEN_ROUTER_API_KEY_1",
    "OPEN_ROUTER_API_KEY_2",
    "OPEN_ROUTER_API_KEY_3",
]


def _flatten_exception_messages(exc: Exception) -> list[str]:
    messages: list[str] = [str(exc).lower()]
    nested = getattr(exc, "exceptions", None)
    if isinstance(nested, tuple):
        for child in nested:
            if isinstance(child, Exception):
                messages.extend(_flatten_exception_messages(child))
            else:
                messages.append(str(child).lower())
    return messages


def get_openrouter_keys() -> list[str]:
    keys: list[str] = []
    for name in OPENROUTER_KEY_ENV_VARS:
        value = os.getenv(name)
        if value:
            keys.append(value)

    # Optional single-key fallback for legacy setups.
    single_key = os.getenv("OPENROUTER_API_KEY")
    if single_key and single_key not in keys:
        keys.append(single_key)

    return keys


def get_openrouter_models(model_name: str | None = None, settings: ModelSettings | None = None) -> list[OpenRouterModel]:
    model = model_name or os.getenv("OPEN_ROUTER_MODEL", os.getenv("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL))
    app_url = os.getenv("OPENROUTER_APP_URL")
    app_title = os.getenv("OPENROUTER_APP_TITLE")
    models: list[OpenRouterModel] = []

    for api_key in get_openrouter_keys():
        provider = OpenRouterProvider(api_key=api_key, app_url=app_url, app_title=app_title)
        models.append(OpenRouterModel(model, provider=provider, settings=settings))

    return models


def get_groq_model(model_name: str | None = None, settings: ModelSettings | None = None) -> GroqModel | None:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2")
    if not api_key:
        return None
    model = model_name or os.getenv("GROQ_PRIMARY_MODEL", GROQ_DEFAULT_PRIMARY_MODEL)
    return GroqModel(model, settings=settings)


def get_groq_fallback_model(model_name: str | None = None, settings: ModelSettings | None = None) -> GroqModel | None:
    api_key = os.getenv("GROQ_API_KEY2") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return GroqModel(model_name or os.getenv("GROQ_FALLBACK_MODEL", GROQ_DEFAULT_FALLBACK_MODEL), settings=settings)


def is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, ModelHTTPError) and exc.status_code == 429:
        return True
    if isinstance(exc, ModelAPIError):
        message = str(exc).lower()
        return "rate limit" in message or "rate_limit" in message
    message = str(exc).lower()
    return "rate limit" in message or "rate_limit" in message


def is_key_failure_error(exc: Exception) -> bool:
    harmony_tool_render_tokens = [
        "tools should have a name",
        "failed to template request",
        "render tokens with harmony",
        "harmonyerror",
        "encodingerror",
    ]
    if isinstance(exc, ModelHTTPError):
        if exc.status_code in {401, 403, 404, 429}:
            return True
        if exc.status_code == 400:
            # Only treat 400 as fallback-eligible when it's a known provider-level
            # tool-render / template-compatibility failure.  Generic validation
            # errors (bad args, missing fields) must NOT trigger a model swap —
            # they should surface as semantic errors to the caller.
            message = str(exc).lower()
            return any(token in message for token in harmony_tool_render_tokens)
        return False
    if isinstance(exc, ModelAPIError):
        messages = _flatten_exception_messages(exc)
        return any(
            token in message
            for message in messages
            for token in [
                "rate limit",
                "rate_limit",
                "invalid api key",
                "unauthorized",
                "tool_choice",
                "no endpoints found",
                *harmony_tool_render_tokens,
            ]
        )
    messages = _flatten_exception_messages(exc)
    return any(
        token in message
        for message in messages
        for token in [
            "rate limit",
            "rate_limit",
            "invalid api key",
            "unauthorized",
            "tool_choice",
            "no endpoints found",
            *harmony_tool_render_tokens,
        ]
    )


def _fallback_log(exc: Exception) -> None:
    print(f"LOG: Model failed ({exc.__class__.__name__}), trying next fallback")


def _is_retryable_upstream_error(exc: Exception) -> bool:
    """True for transient 5xx / network errors that warrant retrying another model."""
    if isinstance(exc, ModelHTTPError):
        return exc.status_code >= 500
    message = str(exc).lower()
    return any(token in message for token in ["connection", "timeout", "eof", "reset", "service unavailable"])


async def run_agent_with_fallbacks(primary_agent: Any, fallback_agents: list[Any], *args: Any, **kwargs: Any):
    """Run the primary agent, falling over to fallbacks on key/rate/tool-render failures.

    Semantic fallback strategy (A5):
    - is_key_failure_error (401/403/404/429/harmony 400) → model swap
    - 5xx / transient network → model swap
    - Other 400 (bad args, validation) → propagate immediately; caller handles
      semantically (clarify, ask again) rather than burning another model.
    """
    agents = [primary_agent] + (fallback_agents or [])
    last_exc: Exception | None = None
    for idx, agent in enumerate(agents):
        try:
            return await agent.run(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            should_fallback = is_key_failure_error(exc) or _is_retryable_upstream_error(exc)
            if idx < len(agents) - 1 and should_fallback:
                _fallback_log(exc)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("No agent available for execution")


def run_agent_sync_with_fallbacks(primary_agent: Any, fallback_agents: list[Any], *args: Any, **kwargs: Any):
    """Sync variant — same fallback strategy as run_agent_with_fallbacks."""
    agents = [primary_agent] + (fallback_agents or [])
    last_exc: Exception | None = None
    for idx, agent in enumerate(agents):
        try:
            return agent.run_sync(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            should_fallback = is_key_failure_error(exc) or _is_retryable_upstream_error(exc)
            if idx < len(agents) - 1 and should_fallback:
                _fallback_log(exc)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("No agent available for execution")
