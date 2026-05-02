"""
core/router.py
--------------
Tier 0 — A1: Router stage.

A small, fast LLM (groq:llama-3.1-8b-instant, ~150ms p50) classifies every
incoming user turn into a RouterDecision.  The graph executor (Tier 1: A2)
consumes this to pick the right graph; today the server uses it to skip the
tool-heavy agent for pure chitchat and to set the correct model tier.

Usage::

    from core.router import RouterDecision, route_turn

    decision = await route_turn(user_text, http_client=...)
    # decision.intent in {"chitchat","web","url","email","calendar",
    #                     "memory_recall","multi_step","clarify"}
"""

from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Router decision schema
# ---------------------------------------------------------------------------

Intent = Literal[
    "chitchat",
    "web",
    "url",
    "email",
    "calendar",
    "memory_recall",
    "multi_step",
    "clarify",
]

Complexity = Literal["low", "med", "high"]


class RouterDecision(BaseModel):
    """Structured output from the router LLM call."""

    intent: Intent
    complexity: Complexity
    suggested_tools: list[str] = Field(default_factory=list)
    needs_paid_model: bool = False
    reason: str = ""

    @field_validator("intent", mode="before")
    @classmethod
    def _coerce_intent(cls, v: str) -> str:
        valid = {"chitchat", "web", "url", "email", "calendar", "memory_recall", "multi_step", "clarify"}
        v = str(v).lower().strip()
        return v if v in valid else "clarify"

    @field_validator("complexity", mode="before")
    @classmethod
    def _coerce_complexity(cls, v: str) -> str:
        valid = {"low", "med", "high"}
        v = str(v).lower().strip()
        return v if v in valid else "low"


# ---------------------------------------------------------------------------
# Router implementation
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM_PROMPT: str | None = None


def _load_router_prompt() -> str:
    global _ROUTER_SYSTEM_PROMPT
    if _ROUTER_SYSTEM_PROMPT is None:
        from core.system_prompts import load_prompt
        _ROUTER_SYSTEM_PROMPT = load_prompt("router")
    return _ROUTER_SYSTEM_PROMPT


def _parse_router_response(text: str) -> RouterDecision:
    """
    Parse the raw LLM string into a RouterDecision.
    Tolerates minor JSON formatting issues (fenced blocks, trailing prose).
    """
    import re
    text = text.strip()

    # Strip ```json fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        # Grab first JSON object in the output
        obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if obj_match:
            text = obj_match.group(0)

    try:
        data = json.loads(text)
        return RouterDecision.model_validate(data)
    except Exception:
        # Graceful degrade — treat as clarify/low
        return RouterDecision(intent="clarify", complexity="low", reason="router parse failed")


async def route_turn(
    user_text: str,
    *,
    timeout_ms: int = 4000,
) -> RouterDecision:
    """
    Route a user turn using a small, fast Groq model.

    Falls back to a heuristic RouterDecision if the LLM call fails
    (network error, missing key, etc.) so the main agent always proceeds.

    Args:
        user_text: The raw user message to classify.
        timeout_ms: Hard timeout in milliseconds (default 4 s).

    Returns:
        RouterDecision with intent, complexity, suggested_tools, needs_paid_model.
    """
    import asyncio
    from groq import AsyncGroq

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2")
    if not api_key:
        return _heuristic_fallback(user_text)

    try:
        client = AsyncGroq(api_key=api_key)
        system_prompt = _load_router_prompt()

        async def _call() -> RouterDecision:
            response = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=256,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            return _parse_router_response(raw)

        timeout_s = timeout_ms / 1000.0
        decision = await asyncio.wait_for(_call(), timeout=timeout_s)
        print(f"LOG: Router -> intent={decision.intent}, complexity={decision.complexity}, paid={decision.needs_paid_model}")
        return decision

    except Exception as exc:
        print(f"LOG: Router failed ({exc.__class__.__name__}: {exc}), using heuristic fallback")
        return _heuristic_fallback(user_text)


def _heuristic_fallback(user_text: str) -> RouterDecision:
    """
    Cheap regex/keyword heuristic used when the LLM router is unavailable.
    Keeps the system functional even without a Groq key.
    """
    lowered = user_text.lower()

    if any(kw in lowered for kw in ["http://", "https://", ".com/", ".org/"]):
        return RouterDecision(intent="url", complexity="med", suggested_tools=["search_url"], reason="URL detected")

    if any(kw in lowered for kw in ["send email", "email to", "mail to", "send a mail"]):
        return RouterDecision(intent="email", complexity="med", suggested_tools=["send_email_assistant"], reason="Email keyword")

    if any(kw in lowered for kw in ["remember", "last time", "previously", "you told me", "do you know my"]):
        return RouterDecision(intent="memory_recall", complexity="low", suggested_tools=["history_tool"], reason="Memory recall keyword")

    if any(kw in lowered for kw in ["search", "latest", "news", "price", "weather", "time in", "what time"]):
        return RouterDecision(intent="web", complexity="low", suggested_tools=["search_web"], reason="Search keyword")

    return RouterDecision(intent="chitchat", complexity="low", reason="Default: no strong signal")
