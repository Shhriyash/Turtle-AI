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

try:
    import logfire as _logfire  # type: ignore
except Exception:
    _logfire = None  # type: ignore


class _RepairedHistoryResult:
    """Wraps a synthesis run result so its ``all_messages()`` extends the real
    conversation instead of returning only the internal synthesis exchange.

    The parallel-planner synthesis call runs without ``message_history`` (to
    keep raw tool-call records away from fallback models). Callers persist
    ``response.all_messages()`` as the canonical history, so without this
    repair every parallel multi-step turn would discard the conversation and
    store the synthesis prompt as the user's last message. We rebuild the
    history as ``prior + [real user turn, synthesized reply turns]``.

    Only ``.output`` and ``.all_messages()`` are consumed by callers; both are
    proxied here.
    """

    def __init__(self, inner: Any, prior_history: list[Any], user_prompt: str) -> None:
        from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart

        self._inner = inner
        inner_messages = list(inner.all_messages())
        # Keep only the final assistant reply. Intermediate tool-call responses
        # (and their request/return turns) from the synthesis run would orphan
        # in the stored history; the conversation only needs the visible answer.
        responses = [m for m in inner_messages if isinstance(m, ModelResponse)]
        reply_messages = responses[-1:] if responses else []
        real_user_turn = ModelRequest(parts=[UserPromptPart(content=user_prompt)])
        self._new_messages = [real_user_turn] + reply_messages
        self._messages = list(prior_history) + self._new_messages

    @property
    def output(self) -> Any:
        return self._inner.output

    def all_messages(self) -> list[Any]:
        return self._messages

    def new_messages(self) -> list[Any]:
        # A real method wins over __getattr__: proxying new_messages() to the
        # inner synthesis run would leak the internal synthesis prompt as a
        # user turn into persisted history.
        return list(self._new_messages)

    def __getattr__(self, name: str) -> Any:
        # Proxy any other attribute access (e.g. usage) to the wrapped result.
        return getattr(self._inner, name)


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

_planner_primary: Agent | None = None
_planner_fallbacks: list[Agent] | None = None


def _get_planner_agents() -> tuple[Agent | None, list[Agent]]:
    """Return (primary, fallbacks) planner agents, cached.

    Cascade: Gemini direct → Llama (Groq) → Gemini via OpenRouter.

    The planner emits a strict typed PlannerOutput. Gemini's structured-output
    discipline is the most reliable on the first pass, so it leads. Llama on
    Groq is the cheap-and-fast cross-provider backup (separate cloud, separate
    rate-limit bucket), and OpenRouter Gemini is the final rung if direct
    Google quota is exhausted.
    """
    global _planner_primary, _planner_fallbacks
    if _planner_primary is not None or _planner_fallbacks is not None:
        return _planner_primary, _planner_fallbacks or []

    try:
        from core.llm_client import (
            get_google_models, get_groq_model, get_openrouter_models,
        )
        prompt_text = (
            _PLANNER_PROMPT_PATH.read_text(encoding="utf-8")
            if _PLANNER_PROMPT_PATH.exists() else ""
        )

        def _agent(model: Any) -> Agent:
            return Agent(model, output_type=PlannerOutput, instructions=prompt_text)

        ordered_models: list[Any] = []
        ordered_models.extend(get_google_models())
        llama = get_groq_model("llama-3.3-70b-versatile")
        if llama is not None:
            ordered_models.append(llama)
        ordered_models.extend(get_openrouter_models())

        if not ordered_models:
            return None, []

        # A3: cap total planner cascade size. Avoids 7-attempt fan-outs when
        # the pool has 3 Gemini keys + 3 OpenRouter keys + Groq.
        try:
            from core.config import settings as _settings
            cap = max(1, int(_settings.planner_max_agents))
        except Exception:
            cap = 4
        if len(ordered_models) > cap:
            print(
                f"LOG: Planner cascade capped at {cap} (had {len(ordered_models)} models available)"
            )
            ordered_models = ordered_models[:cap]

        _planner_primary = _agent(ordered_models[0])
        _planner_fallbacks = [_agent(m) for m in ordered_models[1:]]
        return _planner_primary, _planner_fallbacks
    except Exception as exc:
        print(f"LOG: Planner agent init failed: {exc}")
        return None, []


async def _run_planner(user_text: str) -> PlannerOutput | None:
    """Call the planner cascade and return a typed step list, or None on failure.

    The planner's dominant failure mode is `UnexpectedModelBehavior` from
    output-schema validation — not an HTTP/auth error — so the shared
    run_agent_with_fallbacks helper (which only falls over on key/5xx errors)
    wouldn't trigger. We loop over the cascade ourselves and treat ANY
    exception per-agent as fallback-eligible, since planner output is
    schema-validated by pydantic-ai and a bad agent either returns valid JSON
    or raises — there's no silent-corruption case to worry about.
    """
    primary, fallbacks = _get_planner_agents()
    if primary is None:
        return None
    agents = [primary] + (fallbacks or [])

    async def _attempt(agent: Agent) -> PlannerOutput:
        result = await asyncio.wait_for(
            agent.run(user_text, usage=RunUsage()), timeout=8.0,
        )
        return result.output

    last_exc: Exception | None = None
    for idx, agent in enumerate(agents):
        try:
            return await _attempt(agent)
        except Exception as exc:
            last_exc = exc
            if idx < len(agents) - 1:
                print(
                    f"LOG: Planner attempt {idx + 1}/{len(agents)} failed "
                    f"({exc.__class__.__name__}), trying next model"
                )
                continue
            print(
                f"LOG: Planner failed on all {len(agents)} model(s) "
                f"({exc.__class__.__name__}): {exc}"
            )
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


def _schedule_waves(steps: list[PlannedStep]) -> list[list[PlannedStep]] | None:
    """Group steps into dependency waves via Kahn's algorithm.

    Returns a list of waves where each wave is a list of steps that can run
    concurrently (all of their dependencies are satisfied by earlier waves).
    Returns None if the dependency graph has a cycle or dangling reference.
    """
    by_index: dict[int, PlannedStep] = {s.index: s for s in steps}
    remaining_deps: dict[int, set[int]] = {
        s.index: {d for d in s.depends_on if d in by_index} for s in steps
    }
    waves: list[list[PlannedStep]] = []
    settled: set[int] = set()

    while len(settled) < len(by_index):
        wave_indices = [
            idx for idx, deps in remaining_deps.items()
            if idx not in settled and deps.issubset(settled)
        ]
        if not wave_indices:
            return None  # cycle or unreachable step
        wave_indices.sort()
        waves.append([by_index[i] for i in wave_indices])
        settled.update(wave_indices)

    return waves


def _step_to_prompt_with_context(
    step: PlannedStep, prior_results: dict[int, str]
) -> str:
    """Build a sub-agent prompt, appending dependency outputs as context."""
    base = _step_to_prompt(step)
    if not step.depends_on or not prior_results:
        return base
    context_lines = ["", "Context from prior steps:"]
    for dep_idx in step.depends_on:
        if dep_idx in prior_results:
            context_lines.append(f"[step {dep_idx}]\n{prior_results[dep_idx]}")
    return base + "\n".join(context_lines)


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
        timeout_s: float = 60.0,
        **kwargs: Any,
    ) -> Any:
        from core.llm_client import run_agent_with_fallbacks

        print(f"LOG: Graph executor -> {self.graph_def.name} (intent={self.graph_def.intent})")

        _fallbacks = fallback_agents or []

        # A6: single logical span per turn — child model spans nest under it
        # so the planner cascade is visible as one unit in Logfire.
        if _logfire is not None:
            span_cm = _logfire.span(
                "turtle.turn",
                graph=self.graph_def.name,
                intent=self.graph_def.intent,
            )
        else:
            from contextlib import nullcontext
            span_cm = nullcontext()

        with span_cm:
            # A1: only multi_step actually benefits from the planner cascade.
            # web (and every other graph) goes straight to the single agent
            # loop with search tools available — the planner adds nothing
            # but cost on a single-tool query.
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
        steps = plan.steps if plan else []

        waves = _schedule_waves(steps) if steps else None
        if not waves:
            # Cycle, empty plan, or planner failure — fall through.
            print("LOG: Parallel planner: no usable plan, running single agent")
            return await run_agent_with_fallbacks(primary_agent, fallback_agents, prompt, **kwargs)

        max_wave_width = max(len(w) for w in waves)
        if len(steps) < 2 or max_wave_width < 2:
            # Either trivial plan, or fully sequential (each wave has one step).
            # No parallelism to gain — let the normal agent loop handle it so it
            # can react to intermediate tool results turn-by-turn.
            print(
                f"LOG: Parallel planner: {len(steps)} step(s) in {len(waves)} wave(s), "
                f"max width {max_wave_width} — running single agent"
            )
            return await run_agent_with_fallbacks(primary_agent, fallback_agents, prompt, **kwargs)

        print(
            f"LOG: Parallel planner: {len(steps)} steps across {len(waves)} wave(s) "
            f"(widths={[len(w) for w in waves]}) — running with dependency scheduling"
        )

        # ── 2. Wave-based execution ───────────────────────────────────────────
        # Each sub-task gets its own fresh RunUsage so there are no race
        # conditions on the shared usage counter.  message_history is omitted
        # so each sub-agent starts without prior turns (it only needs to call
        # one tool, not carry the full conversation).
        sub_kwargs = {k: v for k, v in kwargs.items() if k not in ("message_history", "usage")}
        results_by_index: dict[int, str] = {}
        step_results: list[tuple[PlannedStep, str]] = []

        async def run_step(step: PlannedStep) -> tuple[PlannedStep, str]:
            sub_prompt = _step_to_prompt_with_context(step, results_by_index)
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
        for wave_num, wave in enumerate(waves):
            wave_t0 = time.perf_counter()
            wave_results = list(await asyncio.gather(*[run_step(s) for s in wave]))
            for step, output in wave_results:
                results_by_index[step.index] = output
                step_results.append((step, output))
            print(
                f"LOG: Wave {wave_num + 1}/{len(waves)} "
                f"({len(wave)} step(s)) finished in {time.perf_counter() - wave_t0:.2f}s"
            )
        elapsed = time.perf_counter() - t0
        print(f"LOG: Parallel execution finished: {len(steps)} tools in {elapsed:.2f}s")

        # ── 3. Synthesise ─────────────────────────────────────────────────────
        # The synthesis prompt already inlines the original user question AND
        # every tool result as plain text, so the model has all the info it
        # needs without seeing any tool-call message records. We deliberately
        # drop `message_history` here so that if synthesis fails over to a
        # fallback model, the fallback never has to interpret raw tool-call
        # objects — it just composes from text. Prior conversational context
        # is a nice-to-have for synthesis, but losing it is far cheaper than
        # a fallback model regurgitating "search_web(query=...)" as the reply.
        synthesis_prompt = _build_synthesis_prompt(prompt, step_results)
        synth_kwargs = {k: v for k, v in kwargs.items() if k != "message_history"}
        synth_result = await run_agent_with_fallbacks(
            primary_agent,
            fallback_agents,
            synthesis_prompt,
            **synth_kwargs,
        )

        # History repair: the synthesis call deliberately runs WITHOUT
        # message_history (so a fallback model never sees raw tool-call
        # records). But that means synth_result.all_messages() contains only
        # the internal synthesis prompt + reply — not the conversation. If the
        # caller stores that verbatim it WIPES the conversation and records the
        # tool-result blob as the user's last turn. Reattach the real prior
        # history and substitute the user's actual request for the synthesis
        # prompt, so all_messages() extends the conversation correctly.
        original_history = kwargs.get("message_history") or []
        return _RepairedHistoryResult(synth_result, original_history, prompt)

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
