"""
core/graph.py
-------------
A2/A3/A4: Graph executor with real parallel tool execution for multi_step.

For multi_step intent, TurtleGraph runs a 3-stage pipeline:
  1. Planner  — fast Groq model extracts a typed list of independent steps
  2. Parallel — independent steps run concurrently via asyncio.gather,
                each as its own agent.run() so tool context is preserved
  3. Synthesis — main agent composes the final response from gathered results

All other intents delegate directly to run_agent_with_fallbacks (unchanged).

Graph selection is driven by RouterDecision.intent from core/router.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage


# ---------------------------------------------------------------------------
# Planner types (A4)
# ---------------------------------------------------------------------------

class PlannedStep(BaseModel):
    index: int
    tool: str
    args: dict[str, Any]
    depends_on: list[int]
    purpose: str


class PlannerOutput(BaseModel):
    steps: list[PlannedStep]
    summary: str


# ---------------------------------------------------------------------------
# Planner agent (lazy-initialised)
# ---------------------------------------------------------------------------

_PLANNER_PROMPT_PATH = Path(__file__).parent / "system_prompts" / "planner.txt"

_planner_agent: Agent | None = None


def _get_planner_agent() -> Agent | None:
    """Return (and cache) a fast Groq planner agent for step extraction."""
    global _planner_agent
    if _planner_agent is not None:
        return _planner_agent
    try:
        from core.llm_client import get_groq_model
        model = get_groq_model("llama-3.1-8b-instant")
        if model is None:
            return None
        prompt_text = _PLANNER_PROMPT_PATH.read_text(encoding="utf-8") if _PLANNER_PROMPT_PATH.exists() else ""
        _planner_agent = Agent(model, output_type=PlannerOutput, instructions=prompt_text)
        return _planner_agent
    except Exception as exc:
        print(f"LOG: Planner agent init failed: {exc}")
        return None


async def _run_planner(user_text: str) -> PlannerOutput | None:
    """Call the fast planner and return a typed step list, or None on failure."""
    agent = _get_planner_agent()
    if agent is None:
        return None
    try:
        result = await asyncio.wait_for(
            agent.run(user_text, usage=RunUsage()),
            timeout=8.0,
        )
        return result.output
    except Exception as exc:
        print(f"LOG: Planner failed ({exc.__class__.__name__}): {exc}")
        return None


def _step_to_prompt(step: PlannedStep) -> str:
    """Convert a PlannedStep into a self-contained agent prompt."""
    tool_prompts = {
        "search_web": lambda a: f"Search the web for: {a.get('query', '')}",
        "search_url":  lambda a: f"Fetch and summarise this URL: {a.get('url', '')}",
        "send_email_assistant": lambda a: f"Email task: {a.get('query', '')}",
        "history_tool": lambda a: f"Recall from conversation history: {a.get('query', '')}",
        "calendar_create": lambda a: f"Create a calendar event: {json.dumps(a)}",
        "calendar_list": lambda a: f"List upcoming calendar events: {json.dumps(a)}",
    }
    builder = tool_prompts.get(step.tool)
    if builder:
        return builder(step.args)
    return f"Execute {step.tool} with args {json.dumps(step.args)}"


def _build_synthesis_prompt(original_prompt: str, results: list[tuple[PlannedStep, str]]) -> str:
    parts = [f"The user asked: {original_prompt}\n\nHere are the results gathered in parallel:\n"]
    for step, output in results:
        parts.append(f"[{step.tool} — {step.purpose}]\n{output}\n")
    parts.append("\nUsing all of the above results, compose a complete, accurate response to the user.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Graph node definitions (stub topology)
# ---------------------------------------------------------------------------

class NodeKind(str, Enum):
    """All node kinds that may appear in a Turtle graph."""
    ROUTER         = "router"           # Intent classification (already done, feeds graph select)
    PLANNER        = "planner"          # Multi-step task decomposition
    LLM_CALL       = "llm_call"         # Single LLM agent call
    TOOL_CALL      = "tool_call"        # Direct tool invocation (A4: parallel-eligible)
    EMAIL_EXTRACT  = "email_extract"    # A3: email field extraction node
    EMAIL_VALIDATE = "email_validate"   # A3: validation node
    EMAIL_SEND     = "email_send"       # A3: actual send node
    CALENDAR_CREATE = "calendar_create" # F4: create a calendar event via Google Calendar API
    CALENDAR_LIST  = "calendar_list"    # F4: list upcoming calendar events
    RESPONSE       = "response"         # Format and return result
    ERROR          = "error"            # Graceful degrade


@dataclass(frozen=True)
class GraphNode:
    """A single node in a Turtle execution graph."""
    kind: NodeKind
    name: str
    description: str = ""
    # In Tier 2 this will hold a real Pydantic AI Graph node reference
    _impl: Any = field(default=None, compare=False, repr=False)


@dataclass
class GraphEdge:
    """Directed edge between two graph nodes."""
    from_node: str
    to_node: str
    condition: str = "always"  # "always" | "on_missing_fields" | "on_success" | ...


@dataclass
class TurtleGraphDef:
    """Static graph topology definition."""
    name: str
    intent: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    description: str = ""


# ---------------------------------------------------------------------------
# Graph topologies
# ---------------------------------------------------------------------------

CHITCHAT_GRAPH = TurtleGraphDef(
    name="chitchat_graph",
    intent="chitchat",
    description="Simple single LLM call — no tools.",
    nodes=[
        GraphNode(NodeKind.LLM_CALL, "llm_call", "Direct LLM response"),
        GraphNode(NodeKind.RESPONSE, "response", "Return result"),
    ],
    edges=[
        GraphEdge("llm_call", "response"),
    ],
)

WEB_SEARCH_GRAPH = TurtleGraphDef(
    name="web_search_graph",
    intent="web",
    description="LLM call with search_web tool available.",
    nodes=[
        GraphNode(NodeKind.LLM_CALL, "llm_call", "LLM with search tools"),
        GraphNode(NodeKind.TOOL_CALL, "search_web", "Web search tool"),
        GraphNode(NodeKind.RESPONSE, "response", "Return cited result"),
    ],
    edges=[
        GraphEdge("llm_call", "search_web", condition="on_tool_call"),
        GraphEdge("search_web", "llm_call", condition="always"),
        GraphEdge("llm_call", "response", condition="on_final_answer"),
    ],
)

URL_GRAPH = TurtleGraphDef(
    name="url_graph",
    intent="url",
    description="Fetch a URL and summarise.",
    nodes=[
        GraphNode(NodeKind.TOOL_CALL, "search_url", "URL fetch tool"),
        GraphNode(NodeKind.LLM_CALL, "llm_call", "Summarise fetched content"),
        GraphNode(NodeKind.RESPONSE, "response", "Return summary"),
    ],
    edges=[
        GraphEdge("search_url", "llm_call"),
        GraphEdge("llm_call", "response"),
    ],
)

# A3: Email graph — replaces the fake "Email Specialist" tool-calling approach
# with a proper extract → validate → send node chain.
EMAIL_GRAPH = TurtleGraphDef(
    name="email_graph",
    intent="email",
    description="A3: Email as a graph: extract → validate → send. Cuts one LLM round-trip.",
    nodes=[
        GraphNode(NodeKind.EMAIL_EXTRACT, "extract", "Extract To/Cc/Bcc/Subject/Body"),
        GraphNode(NodeKind.EMAIL_VALIDATE, "validate", "Validate fields, check missing"),
        GraphNode(NodeKind.EMAIL_SEND, "send", "Send via SMTP + idempotency check"),
        GraphNode(NodeKind.RESPONSE, "response", "Confirm send or ask for missing fields"),
        GraphNode(NodeKind.ERROR, "error", "Report failure"),
    ],
    edges=[
        GraphEdge("extract", "validate"),
        GraphEdge("validate", "send", condition="on_valid"),
        GraphEdge("validate", "response", condition="on_missing_fields"),
        GraphEdge("send", "response", condition="on_success"),
        GraphEdge("send", "error", condition="on_failure"),
    ],
)

# F4: Calendar graph — create events and list upcoming schedule
CALENDAR_GRAPH = TurtleGraphDef(
    name="calendar_graph",
    intent="calendar",
    description="F4: Create calendar events or list upcoming schedule via Google Calendar API.",
    nodes=[
        GraphNode(NodeKind.LLM_CALL, "extract", "Extract event details from user request"),
        GraphNode(NodeKind.CALENDAR_CREATE, "calendar_create", "Create event via Google Calendar API"),
        GraphNode(NodeKind.CALENDAR_LIST, "calendar_list", "List upcoming events via Google Calendar API"),
        GraphNode(NodeKind.LLM_CALL, "synthesise", "Format result into natural language response"),
        GraphNode(NodeKind.RESPONSE, "response", "Return event confirmation or schedule"),
        GraphNode(NodeKind.ERROR, "error", "Report missing credentials or API failure"),
    ],
    edges=[
        GraphEdge("extract", "calendar_create", condition="on_create_intent"),
        GraphEdge("extract", "calendar_list", condition="on_list_intent"),
        GraphEdge("calendar_create", "synthesise", condition="on_success"),
        GraphEdge("calendar_create", "error", condition="on_failure"),
        GraphEdge("calendar_list", "synthesise", condition="on_success"),
        GraphEdge("calendar_list", "error", condition="on_failure"),
        GraphEdge("synthesise", "response"),
    ],
)

MEMORY_RECALL_GRAPH = TurtleGraphDef(
    name="memory_recall_graph",
    intent="memory_recall",
    description="LLM call with history_tool available.",
    nodes=[
        GraphNode(NodeKind.TOOL_CALL, "history_tool", "History retrieval"),
        GraphNode(NodeKind.LLM_CALL, "llm_call", "Synthesise retrieved memory"),
        GraphNode(NodeKind.RESPONSE, "response", "Return recalled info"),
    ],
    edges=[
        GraphEdge("history_tool", "llm_call"),
        GraphEdge("llm_call", "response"),
    ],
)

# A4: Multi-step graph uses planner to emit parallel-eligible tool calls
MULTI_STEP_GRAPH = TurtleGraphDef(
    name="multi_step_graph",
    intent="multi_step",
    description="A4: Planner emits typed plan, executor runs independent steps in parallel.",
    nodes=[
        GraphNode(NodeKind.PLANNER, "planner", "Decompose into typed tool step list"),
        GraphNode(NodeKind.TOOL_CALL, "parallel_tools", "Execute independent steps concurrently"),
        GraphNode(NodeKind.LLM_CALL, "synthesise", "Combine results into final answer"),
        GraphNode(NodeKind.RESPONSE, "response", "Return answer"),
    ],
    edges=[
        GraphEdge("planner", "parallel_tools"),
        GraphEdge("parallel_tools", "synthesise"),
        GraphEdge("synthesise", "response"),
    ],
)

_GRAPH_REGISTRY: dict[str, TurtleGraphDef] = {
    g.intent: g for g in [
        CHITCHAT_GRAPH,
        WEB_SEARCH_GRAPH,
        URL_GRAPH,
        EMAIL_GRAPH,
        CALENDAR_GRAPH,
        MEMORY_RECALL_GRAPH,
        MULTI_STEP_GRAPH,
    ]
}

# Fallback for unknown intents
_DEFAULT_GRAPH = WEB_SEARCH_GRAPH


# ---------------------------------------------------------------------------
# Graph executor
# ---------------------------------------------------------------------------

class TurtleGraph:
    """Graph executor.

    For multi_step intent: runs the A4 planner → parallel → synthesis pipeline.
    All other intents: delegates directly to run_agent_with_fallbacks.
    """

    def __init__(self, graph_def: TurtleGraphDef) -> None:
        self.graph_def = graph_def

    async def run(
        self,
        primary_agent: Any,
        prompt: str,
        *,
        fallback_agents: list[Any] | None = None,
        timeout_s: float = 45.0,
        **kwargs: Any,
    ) -> Any:
        from core.llm_client import run_agent_with_fallbacks

        print(f"LOG: Graph executor -> {self.graph_def.name} (intent={self.graph_def.intent})")

        _fallbacks = fallback_agents or []

        if self.graph_def.intent == "multi_step":
            try:
                return await asyncio.wait_for(
                    self._run_parallel(primary_agent, _fallbacks, prompt, **kwargs),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                raise RuntimeError(f"Graph {self.graph_def.name} exceeded timeout ({timeout_s:.0f}s)")

        try:
            return await asyncio.wait_for(
                run_agent_with_fallbacks(primary_agent, _fallbacks, prompt, **kwargs),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"Graph {self.graph_def.name} exceeded timeout ({timeout_s:.0f}s)")

    async def _run_parallel(
        self,
        primary_agent: Any,
        fallback_agents: list[Any],
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        """A4: Planner → parallel tool execution → synthesis."""
        from core.llm_client import run_agent_with_fallbacks

        # ── 1. Plan ──────────────────────────────────────────────────────────
        plan = await _run_planner(prompt)

        independent = [s for s in plan.steps if not s.depends_on] if plan else []

        if len(independent) < 2:
            # Nothing to parallelise — fall through to normal agent run.
            print("LOG: Parallel planner: ≤1 independent step, running single agent")
            return await run_agent_with_fallbacks(primary_agent, fallback_agents, prompt, **kwargs)

        print(f"LOG: Parallel planner: {len(independent)} independent steps — running concurrently")

        # ── 2. Parallel execution ─────────────────────────────────────────────
        # Each sub-task gets its own fresh RunUsage so there are no race
        # conditions on the shared usage counter.  message_history is omitted
        # so each sub-agent starts without prior turns (it only needs to call
        # one tool, not carry the full conversation).
        sub_kwargs = {k: v for k, v in kwargs.items() if k not in ("message_history", "usage")}

        async def run_step(step: PlannedStep) -> tuple[PlannedStep, str]:
            sub_prompt = _step_to_prompt(step)
            try:
                result = await run_agent_with_fallbacks(
                    primary_agent,
                    fallback_agents,
                    sub_prompt,
                    usage=RunUsage(),
                    **sub_kwargs,
                )
                return step, result.output
            except Exception as exc:
                return step, f"[error: {exc}]"

        t0 = time.perf_counter()
        step_results: list[tuple[PlannedStep, str]] = list(
            await asyncio.gather(*[run_step(s) for s in independent])
        )
        elapsed = time.perf_counter() - t0
        print(f"LOG: Parallel execution finished: {len(independent)} tools in {elapsed:.2f}s")

        # ── 3. Synthesise ─────────────────────────────────────────────────────
        synthesis_prompt = _build_synthesis_prompt(prompt, step_results)
        # Synthesis uses the original message_history so the reply is
        # contextually grounded in the conversation.
        return await run_agent_with_fallbacks(
            primary_agent,
            fallback_agents,
            synthesis_prompt,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"TurtleGraph({self.graph_def.name!r}, intent={self.graph_def.intent!r})"


# ---------------------------------------------------------------------------
# Graph selection
# ---------------------------------------------------------------------------

def select_graph(intent: str) -> TurtleGraph:
    """Return the appropriate TurtleGraph for the given RouterDecision intent."""
    graph_def = _GRAPH_REGISTRY.get(intent, _DEFAULT_GRAPH)
    return TurtleGraph(graph_def)


def list_graphs() -> list[dict[str, str]]:
    """Return a summary of all registered graphs (for dev mode / introspection)."""
    return [
        {
            "name": g.name,
            "intent": g.intent,
            "description": g.description,
            "nodes": str(len(g.nodes)),
            "edges": str(len(g.edges)),
        }
        for g in _GRAPH_REGISTRY.values()
    ]
