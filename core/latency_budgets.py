"""
core/latency_budgets.py
-----------------------
E4: Hard latency budgets per pipeline stage.

Values are read from environment so they can be tuned per-deployment.
Exceeded budgets → graceful degrade message + SLA breach log.

Stage budgets (milliseconds):
  STT_MAX_MS           2000   Full-utterance transcription budget
  ROUTER_MAX_MS         400   Intent classification (already enforced in router.py)
  PLANNER_MAX_MS       8000   Multi-step planning
  TTS_FIRST_BYTE_MAX_MS 600   Time-to-first audio chunk from TTS start
  TOOL_MAX_MS          6000   Any single tool call (web, url, email — Tavily 6s cap)
  LLM_MAX_MS          25000   Full LLM response (soft; hard timeouts are in graph.py)

Usage::

    from core.latency_budgets import budgets, sla_exceeded

    async with asyncio.timeout(budgets.TOOL_MAX_MS / 1000):
        results = await run_tool(...)
"""
from __future__ import annotations

import os
import time


def _ms(env_key: str, default: int) -> int:
    try:
        return max(100, int(os.getenv(env_key, str(default))))
    except Exception:
        return default


class LatencyBudgets:
    """Per-stage latency budget configuration (milliseconds).

    All values are read from environment variables at construction time so
    they can be overridden per-deployment without code changes.
    """

    def __init__(self) -> None:
        # E4 values
        self.STT_MAX_MS: int           = _ms("TURTLE_STT_MAX_MS",            2000)
        self.ROUTER_MAX_MS: int        = _ms("TURTLE_ROUTER_MAX_MS",          400)
        self.PLANNER_MAX_MS: int       = _ms("TURTLE_PLANNER_MAX_MS",        8000)
        self.TTS_FIRST_BYTE_MAX_MS: int = _ms("TURTLE_TTS_FIRST_BYTE_MAX_MS", 600)
        self.TOOL_MAX_MS: int          = _ms("TURTLE_TOOL_MAX_MS",           6000)
        self.LLM_MAX_MS: int           = _ms("TURTLE_LLM_MAX_MS",           25000)

    @property
    def TOOL_S(self) -> float:
        return self.TOOL_MAX_MS / 1000.0

    @property
    def LLM_S(self) -> float:
        return self.LLM_MAX_MS / 1000.0

    @property
    def ROUTER_S(self) -> float:
        return self.ROUTER_MAX_MS / 1000.0

    @property
    def TTS_FIRST_BYTE_S(self) -> float:
        return self.TTS_FIRST_BYTE_MAX_MS / 1000.0


# Singleton budget object
budgets = LatencyBudgets()


# ---------------------------------------------------------------------------
# SLA breach logging
# ---------------------------------------------------------------------------

def sla_exceeded(stage: str, elapsed_ms: float, budget_ms: int) -> None:
    """Log an SLA breach span.  In Tier 2 (G5 OTel) this will emit a real span."""
    print(
        f"LOG: SLA breach stage={stage!r} elapsed={elapsed_ms:.0f}ms budget={budget_ms}ms"
        f" overage={elapsed_ms - budget_ms:.0f}ms"
    )


def check_sla(stage: str, start_s: float, budget_ms: int) -> None:
    """Call after a stage completes to log SLA breaches."""
    elapsed_ms = (time.time() - start_s) * 1000
    if elapsed_ms > budget_ms:
        sla_exceeded(stage, elapsed_ms, budget_ms)


# ---------------------------------------------------------------------------
# Degrade message for voice channel
# ---------------------------------------------------------------------------

DEGRADE_VOICE_MESSAGE = (
    "One moment, that's taking a bit longer than expected. "
    "I'll have your answer shortly."
)
