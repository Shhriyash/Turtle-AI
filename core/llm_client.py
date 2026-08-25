from __future__ import annotations

import os
from typing import Any

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

try:
    import logfire as _logfire  # type: ignore
except Exception:
    _logfire = None  # type: ignore

# Roster (2026-08-25): Gemini 2.5 Flash is the primary main/email model and Groq
# gpt-oss-20b is the second rung, each fanned across every available API key for
# that provider; Gemini via OpenRouter is the last-resort rung, so this default
# names the Gemini slug (ox-alpha was discarded for intermittently returning
# empty output on multi-tool turns). The previous Groq defaults
# (llama-3.3-70b-versatile / llama-3.1-8b-instant) were decommissioned by Groq.
OPENROUTER_DEFAULT_MODEL = "google/gemini-2.5-flash"
GROQ_DEFAULT_PRIMARY_MODEL = "openai/gpt-oss-20b"
GROQ_DEFAULT_FALLBACK_MODEL = "openai/gpt-oss-20b"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
GROQ_KEY_ENV_VARS = [
    "GROQ_API_KEY",
    "GROQ_API_KEY2",
]
OPENROUTER_KEY_ENV_VARS = [
    "OPEN_ROUTER_API_KEY_1",
    "OPEN_ROUTER_API_KEY_2",
    "OPEN_ROUTER_API_KEY_3",
]
GEMINI_KEY_ENV_VARS = [
    "GEMINI_API_KEY",
    "GEMINI_API_KEY_2",
    "GEMINI_API_KEY_3",
]

# Substrings that identify a provider-level tool-render / template-compat
# failure (chiefly Groq's Harmony path for gpt-oss models, but also Gemini's
# strict function-call/response adjacency). Used both to decide a model swap
# and to log the exact failure when it happens.
_HARMONY_TOKENS = [
    "tools should have a name",
    "failed to template request",
    "render tokens with harmony",
    "harmonyerror",
    "encodingerror",
    # Gemini direct is strict about function-call / function-response
    # adjacency in message_history. Groq/OpenRouter tolerate gaps; Google
    # 400s. Treat as a provider-compat failure so the cascade can skip past
    # Gemini-direct rungs to a more lenient backend (Groq llama).
    "function response turn comes immediately after a function call",
    "function call turn",
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

    or_settings = dict(settings) if settings else {}
    # The same gemini-2.5-flash served via OpenRouter runs with thinking enabled;
    # a small max_tokens budget gets consumed by hidden reasoning, yielding
    # empty/truncated replies on exactly the rungs that serve during
    # Gemini-direct cooldowns. 2048 is the production-surviving floor
    # documented in problems/2026-05-30.
    or_settings["max_tokens"] = max(2048, int(or_settings.get("max_tokens", 0) or 0))

    for api_key in get_openrouter_keys():
        provider = OpenRouterProvider(api_key=api_key, app_url=app_url, app_title=app_title)
        models.append(OpenRouterModel(model, provider=provider, settings=ModelSettings(**or_settings)))

    return models


def get_gemini_keys() -> list[str]:
    """Direct Google AI Studio (generativelanguage.googleapis.com) keys."""
    keys: list[str] = []
    for name in GEMINI_KEY_ENV_VARS:
        value = os.getenv(name)
        if value:
            keys.append(value)
    # Optional Google-canonical alias for the single-key case.
    single_key = os.getenv("GOOGLE_API_KEY")
    if single_key and single_key not in keys:
        keys.append(single_key)
    return keys


def get_google_models(
    model_name: str | None = None,
    settings: ModelSettings | None = None,
) -> list[GoogleModel]:
    """One GoogleModel per available Gemini API key.

    Going direct to Google avoids the OpenRouter latency hop and is materially
    cheaper than the same model via OpenRouter. Returned in env-var order so
    callers can stack them as round-robin fallbacks.
    """
    model = model_name or os.getenv("GEMINI_MODEL", GEMINI_DEFAULT_MODEL)
    # Gemini 2.5 models are *thinking* models: by default they spend part of the
    # maxOutputTokens budget on internal reasoning. Turtle's agents are
    # conversational + tool-routing, not deep-reasoning, so thinking just burns
    # output budget — and with a small max_tokens it can consume the *entire*
    # allowance, yielding empty/truncated responses (finish_reason length/error)
    # that never reach a tool call. Disable it explicitly (thinking_budget=0,
    # supported by gemini-2.5-flash) so the full token budget goes to the answer.
    google_settings = GoogleModelSettings(
        **(dict(settings) if settings else {}),
        google_thinking_config={"thinking_budget": 0, "include_thoughts": False},
    )
    models: list[GoogleModel] = []
    for api_key in get_gemini_keys():
        provider = GoogleProvider(api_key=api_key)
        models.append(GoogleModel(model, provider=provider, settings=google_settings))
    return models


def get_groq_keys() -> list[str]:
    """All distinct, non-empty Groq API keys, in env-var order.

    Mirrors get_openrouter_keys(): one entry per key so a caller can build one
    model per key and fan a single model family across every available quota
    bucket (the "on all of their api keys available" roster directive).
    """
    keys: list[str] = []
    for name in GROQ_KEY_ENV_VARS:
        value = os.getenv(name)
        if value and value not in keys:
            keys.append(value)
    return keys


def get_groq_models(model_name: str | None = None, settings: ModelSettings | None = None) -> list[GroqModel]:
    """One GroqModel per available Groq API key, in env-var order.

    Lets the agent cascade retry the SAME model on a second key before it
    changes models — a rate-limited key fails over to the next key's quota
    rather than immediately dropping to a different (slower / less-capable)
    provider.
    """
    model = model_name or os.getenv("GROQ_PRIMARY_MODEL", GROQ_DEFAULT_PRIMARY_MODEL)
    return [
        GroqModel(model, provider=GroqProvider(api_key=api_key), settings=settings)
        for api_key in get_groq_keys()
    ]


def get_groq_model(model_name: str | None = None, settings: ModelSettings | None = None) -> GroqModel | None:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2")
    if not api_key:
        return None
    model = model_name or os.getenv("GROQ_PRIMARY_MODEL", GROQ_DEFAULT_PRIMARY_MODEL)
    return GroqModel(model, provider=GroqProvider(api_key=api_key), settings=settings)


def get_groq_fallback_model(model_name: str | None = None, settings: ModelSettings | None = None) -> GroqModel | None:
    # Prefer KEY2 so the fallback rung really is a separate quota bucket.
    api_key = os.getenv("GROQ_API_KEY2") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return GroqModel(
        model_name or os.getenv("GROQ_FALLBACK_MODEL", GROQ_DEFAULT_FALLBACK_MODEL),
        provider=GroqProvider(api_key=api_key),
        settings=settings,
    )


def is_rate_limit_error(exc: Exception) -> bool:
    # 413 = Groq "Request too large" for the per-minute token (TPM) budget —
    # it carries code 'rate_limit_exceeded'. It's a capacity limit, not a bad
    # request, so treat it like 429.
    if isinstance(exc, ModelHTTPError) and exc.status_code in {413, 429}:
        return True
    if isinstance(exc, ModelAPIError):
        message = str(exc).lower()
        return "rate limit" in message or "rate_limit" in message
    message = str(exc).lower()
    return "rate limit" in message or "rate_limit" in message


def is_key_failure_error(exc: Exception) -> bool:
    if isinstance(exc, ModelHTTPError):
        # 402 = provider/key out of credits or quota (e.g. OpenRouter "requires
        # more credits"). 413 = TPM "request too large" (rate_limit_exceeded).
        # Both are capacity/quota signals for THIS provider, not client errors —
        # fall over to a model on a different provider/limit rather than aborting
        # the cascade (a 402 on OpenRouter must not strand the Groq rescue rung).
        if exc.status_code in {401, 402, 403, 404, 413, 429}:
            return True
        if exc.status_code == 400:
            # Only treat 400 as fallback-eligible when it's a known provider-level
            # tool-render / template-compatibility failure.  Generic validation
            # errors (bad args, missing fields) must NOT trigger a model swap —
            # they should surface as semantic errors to the caller.
            message = str(exc).lower()
            return any(token in message for token in _HARMONY_TOKENS)
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
                *_HARMONY_TOKENS,
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
            *_HARMONY_TOKENS,
        ]
    )


def _is_harmony_error(exc: Exception) -> bool:
    """True when the failure looks like a provider tool-render/template bug
    (Groq Harmony for gpt-oss, or Gemini function-call adjacency)."""
    return any(
        token in message
        for message in _flatten_exception_messages(exc)
        for token in _HARMONY_TOKENS
    )


def _log_harmony_error(agent: Any, exc: Exception) -> None:
    """Capture the *exact* harmony/tool-render error so we can finally see which
    variant gpt-oss-120b is hitting. Cheap, fires only on a confirmed match."""
    if not _is_harmony_error(exc):
        return
    model = getattr(agent, "model", None)
    model_name = getattr(model, "model_name", None) or getattr(model, "name", None) or str(model)
    status = getattr(exc, "status_code", None)
    print(f"LOG: HARMONY tool-render failure on {model_name} (status={status}): {exc}")
    if _logfire is not None:
        try:
            _logfire.error(
                "harmony_tool_render_error",
                model_name=str(model_name),
                status_code=status,
                error_class=exc.__class__.__name__,
                error=str(exc)[:2000],
            )
        except Exception:
            pass


def _fallback_log(exc: Exception) -> None:
    print(f"LOG: Model failed ({exc.__class__.__name__}), trying next fallback")


def _is_retryable_upstream_error(exc: Exception) -> bool:
    """True for transient 5xx / network errors that warrant retrying another model."""
    if isinstance(exc, ModelHTTPError):
        return exc.status_code >= 500
    message = str(exc).lower()
    return any(token in message for token in ["connection", "timeout", "eof", "reset", "service unavailable"])


def _is_output_validation_error(exc: Exception) -> bool:
    """True when pydantic_ai gave up on a model's structured output / tool-call args.

    The model returned syntactically OK content that didn't match the declared
    schema, and pydantic_ai exhausted its in-band retries. Falling over to a
    different provider is the right move — same-model retries rarely fix
    schema-shape problems, but a different provider's serializer often will.
    """
    if isinstance(exc, UnexpectedModelBehavior):
        return True
    message = str(exc).lower()
    return (
        "exceeded maximum retries" in message
        and "output validation" in message
    )


def _model_family(agent: Any) -> str:
    """Stable ``ModelClass:model_name`` identifier for an agent's model.

    Two cascade rungs that differ ONLY by API key share a family (e.g.
    ``stealth/ox-alpha`` on three OpenRouter keys). Used to skip the sibling
    rungs of a family that already failed DETERMINISTICALLY this call — an
    output-validation / empty-output failure repeats identically on every key,
    so replaying it per key just burns the turn's time budget before the
    cascade reaches a different, possibly-working model.
    """
    model = getattr(agent, "model", agent)
    name = getattr(model, "model_name", None) or getattr(model, "name", None) or ""
    return f"{model.__class__.__name__}:{name}"


def _sanitize_message_history(messages: list[ModelMessage] | None) -> list[ModelMessage] | None:
    """Phase 5 / A5: drop ToolCallPart entries with empty tool_name.

    pydantic-ai's message round-tripping can occasionally serialize a tool
    call whose function name is the empty string (observed against Groq's
    Harmony tool-render path → 400 'tools should have a name'). One such
    message in the history wipes the *entire* fallback chain — every
    OpenAI-compat provider rejects the same body. This sanitizer drops the
    broken part and any orphaned ToolReturnPart that refers to its
    tool_call_id, so the cascade survives the bug regardless of which model
    is downstream.

    Returns the (possibly trimmed) list, or the original `messages` value
    when no change is needed.
    """
    if not messages:
        return messages

    dropped_tool_call_ids: set[str] = set()
    changed = False
    cleaned: list[ModelMessage] = []

    for msg in messages:
        if isinstance(msg, ModelResponse):
            new_parts = []
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and not (part.tool_name or "").strip():
                    if part.tool_call_id:
                        dropped_tool_call_ids.add(part.tool_call_id)
                    changed = True
                    if _logfire is not None:
                        try:
                            _logfire.error(
                                "tool_call_missing_name",
                                tool_call_id=part.tool_call_id,
                                args=str(part.args)[:200],
                            )
                        except Exception:
                            pass
                    print(
                        f"LOG: sanitizer dropped ToolCallPart with empty tool_name "
                        f"(tool_call_id={part.tool_call_id!r})"
                    )
                    continue
                new_parts.append(part)
            if not new_parts:
                # Empty response after sanitization — drop the whole message
                # rather than emit a malformed empty ModelResponse.
                continue
            if len(new_parts) != len(msg.parts):
                cleaned.append(ModelResponse(parts=new_parts))
            else:
                cleaned.append(msg)
        elif isinstance(msg, ModelRequest):
            new_parts = []
            for part in msg.parts:
                if (
                    isinstance(part, ToolReturnPart)
                    and part.tool_call_id in dropped_tool_call_ids
                ):
                    changed = True
                    print(
                        f"LOG: sanitizer dropped orphan ToolReturnPart "
                        f"(tool_call_id={part.tool_call_id!r})"
                    )
                    continue
                new_parts.append(part)
            if not new_parts:
                continue
            if len(new_parts) != len(msg.parts):
                cleaned.append(ModelRequest(parts=new_parts))
            else:
                cleaned.append(msg)
        else:
            cleaned.append(msg)

    return cleaned if changed else messages


async def run_agent_with_fallbacks(primary_agent: Any, fallback_agents: list[Any], *args: Any, **kwargs: Any):
    """Run the primary agent, falling over to fallbacks on key/rate/tool-render failures.

    Semantic fallback strategy (A5):
    - is_key_failure_error (401/403/404/429/harmony 400) → model swap
    - 5xx / transient network → model swap
    - Other 400 (bad args, validation) → propagate immediately; caller handles
      semantically (clarify, ask again) rather than burning another model.

    A2: agents in cooldown (recent transient/deterministic failure) are
    skipped; only fully-cooled-down agents are attempted.
    """
    from core import health_tracker

    # A5: sanitize outgoing message_history before any provider sees it.
    if "message_history" in kwargs:
        kwargs["message_history"] = _sanitize_message_history(kwargs["message_history"])

    agents = [primary_agent] + (fallback_agents or [])
    eligible = [a for a in agents if not health_tracker.is_cooling(a)]
    if not eligible:
        # Every model is cooling — bypass cooldowns rather than fail outright,
        # but warn so the operator sees it.
        print("LOG: all agents in cooldown; bypassing health tracker for this call")
        eligible = agents

    last_exc: Exception | None = None
    # Families that failed DETERMINISTICALLY this call (output-validation /
    # empty-output). health_tracker deliberately does NOT cool these (they're not
    # a provider/quota signal), so without this set the cascade would replay the
    # same doomed model on each of its API keys — three ox-alpha rungs ahead of
    # the first different model — and the per-turn timeout can fire before a
    # working model is ever reached. Per-call only: an intermittent failure never
    # benches the family for later turns.
    poisoned_families: set[str] = set()
    for idx, agent in enumerate(eligible):
        # An earlier failure THIS call may have cooled a model family that
        # recurs later in the chain (the same model appears once per API key —
        # e.g. gemini-2.5-flash on three Google keys). Re-check and skip such a
        # rung rather than burn an identical, already-doomed call. Never skip the
        # primary (idx 0); if everything remaining is cooling the final
        # `raise last_exc` still fires.
        if idx > 0 and last_exc is not None and health_tracker.is_cooling(agent):
            continue
        # Same idea for a deterministic output-validation failure (see
        # poisoned_families above): skip sibling keys of a family that already
        # failed that way so the cascade drops to a different model in-budget.
        if idx > 0 and _model_family(agent) in poisoned_families:
            print(
                f"LOG: skipping {_model_family(agent)} rung — "
                f"same model already failed output-validation this call"
            )
            continue
        try:
            result = await agent.run(*args, **kwargs)
            health_tracker.mark_success(agent)
            return result
        except Exception as exc:
            last_exc = exc
            _log_harmony_error(agent, exc)
            health_tracker.mark_failure(agent, exc)
            if _is_output_validation_error(exc):
                poisoned_families.add(_model_family(agent))
            should_fallback = (
                is_key_failure_error(exc)
                or _is_retryable_upstream_error(exc)
                or _is_output_validation_error(exc)
            )
            if idx < len(eligible) - 1 and should_fallback:
                _fallback_log(exc)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("No agent available for execution")


def run_agent_sync_with_fallbacks(primary_agent: Any, fallback_agents: list[Any], *args: Any, **kwargs: Any):
    """Sync variant — same fallback strategy as run_agent_with_fallbacks."""
    from core import health_tracker

    if "message_history" in kwargs:
        kwargs["message_history"] = _sanitize_message_history(kwargs["message_history"])

    agents = [primary_agent] + (fallback_agents or [])
    eligible = [a for a in agents if not health_tracker.is_cooling(a)]
    if not eligible:
        print("LOG: all agents in cooldown; bypassing health tracker for this call")
        eligible = agents

    last_exc: Exception | None = None
    poisoned_families: set[str] = set()
    for idx, agent in enumerate(eligible):
        # Skip a rung whose model family was cooled by an earlier failure this
        # call (see run_agent_with_fallbacks for the rationale).
        if idx > 0 and last_exc is not None and health_tracker.is_cooling(agent):
            continue
        # Skip sibling keys of a family that already failed output-validation
        # this call (health_tracker doesn't cool that class — see the async twin).
        if idx > 0 and _model_family(agent) in poisoned_families:
            print(
                f"LOG: skipping {_model_family(agent)} rung — "
                f"same model already failed output-validation this call"
            )
            continue
        try:
            result = agent.run_sync(*args, **kwargs)
            health_tracker.mark_success(agent)
            return result
        except Exception as exc:
            last_exc = exc
            _log_harmony_error(agent, exc)
            health_tracker.mark_failure(agent, exc)
            if _is_output_validation_error(exc):
                poisoned_families.add(_model_family(agent))
            should_fallback = (
                is_key_failure_error(exc)
                or _is_retryable_upstream_error(exc)
                or _is_output_validation_error(exc)
            )
            if idx < len(eligible) - 1 and should_fallback:
                _fallback_log(exc)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("No agent available for execution")
