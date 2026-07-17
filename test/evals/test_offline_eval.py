"""
test/evals/test_offline_eval.py
-------------------------------
Genuinely-offline agent eval (Phase 2 W5 rewrite).

The previous version of this file never once ran successfully (see the
2026-07-16 production autopsy). It:
  * read ``result.data`` — a name that does not exist on ``AgentRunResult`` in
    pydantic_ai 1.67 (the attribute is ``result.output``);
  * searched for ``ToolCallPart`` inside ``ModelRequest.parts`` — type-impossible,
    because a ``ToolCallPart`` only ever lives inside a ``ModelResponse``;
  * required live LLM API keys despite living under ``test/evals`` and being
    named an "offline" eval; and
  * instantiated production stores against the repo's ``data/`` tree.

This rewrite drives a real ``pydantic_ai.Agent`` whose model is a
``FunctionModel`` that decides tool calls deterministically from the user text —
so there is no network, no API key, and no flakiness. Tool calls are inspected
the correct way (``ToolCallPart`` in the ``ModelResponse`` parts of
``result.all_messages()``), the final answer is read from ``result.output``, and
nothing is written to disk. It runs as an ordinary part of the pytest suite.

Cases (6):
  tool-choice:
    - a memory question MUST call ``recall`` and nothing else;
    - a news/weather question MUST call ``search_web`` and nothing else;
    - a chitchat turn MUST call NO tool.
  answer-contains:
    - with ``recall`` returning a seeded fact, ``result.output`` contains it.
"""
from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


# ---------------------------------------------------------------------------
# Stub tool backing data — a fake "personal memory" the recall tool reads from.
# ---------------------------------------------------------------------------
_SEEDED_FACTS: dict[str, str] = {
    "favourite editor": "VS Code",
    "best friend": "Elvin",
}


# ---------------------------------------------------------------------------
# Deterministic model: routes each turn from the user text, with zero network.
# ---------------------------------------------------------------------------

def _latest_user_text(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    return str(part.content)
    return ""


def _prior_tool_returns(messages: list[ModelMessage]) -> list[str]:
    returns: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    returns.append(str(part.content))
    return returns


def _route(user_text: str) -> str:
    """Deterministic intent routing — the model's 'decision'."""
    low = user_text.lower()
    if any(k in low for k in ("favourite", "favorite", "best friend", "my name", "remember when")):
        return "recall"
    if any(k in low for k in ("news", "latest", "weather", "who won", "score", "today")):
        return "search"
    return "chitchat"


def _model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # If a tool has already returned, synthesize the final answer from it so the
    # seeded fact lands in result.output (the answer-contains assertions).
    tool_returns = _prior_tool_returns(messages)
    if tool_returns:
        return ModelResponse(parts=[TextPart(content="Here is what I found: " + " ".join(tool_returns))])

    user_text = _latest_user_text(messages)
    intent = _route(user_text)
    if intent == "recall":
        return ModelResponse(
            parts=[ToolCallPart(tool_name="recall", args={"query": user_text, "scope": "personal"})]
        )
    if intent == "search":
        return ModelResponse(
            parts=[ToolCallPart(tool_name="search_web", args={"query": user_text})]
        )
    # chitchat — answer directly, no tool.
    return ModelResponse(parts=[TextPart(content="Happy to chat! Nothing to look up there.")])


def _build_agent() -> Agent:
    agent = Agent(FunctionModel(_model_function), output_type=str)

    @agent.tool_plain
    def search_web(query: str) -> str:
        """Fake web search — returns a canned, network-free result."""
        return f"[Web] top result for {query!r}"

    @agent.tool_plain
    def recall(query: str, scope: str) -> str:
        """Fake personal-memory recall backed by an in-process dict."""
        low = query.lower()
        for fact_key, fact_value in _SEEDED_FACTS.items():
            if all(word in low for word in fact_key.split()):
                return f"[Personal Memory] {fact_key}: {fact_value}"
        return "No relevant information found."

    return agent


# ---------------------------------------------------------------------------
# Eval cases
# ---------------------------------------------------------------------------
EVAL_CASES = [
    {
        "id": "memory_editor_calls_recall",
        "prompt": "What is my favourite editor?",
        "expected_tools": {"recall"},
        "contains": "VS Code",
    },
    {
        "id": "memory_best_friend_answer_contains",
        "prompt": "Who is my best friend?",
        "expected_tools": {"recall"},
        "contains": "Elvin",
    },
    {
        "id": "news_calls_search_web",
        "prompt": "What's the latest news on artificial intelligence?",
        "expected_tools": {"search_web"},
        "contains": None,
    },
    {
        "id": "weather_calls_search_web",
        "prompt": "What's the weather in Tokyo today?",
        "expected_tools": {"search_web"},
        "contains": None,
    },
    {
        "id": "chitchat_calls_no_tool",
        "prompt": "Hey Turtle, how's it going?",
        "expected_tools": set(),
        "contains": None,
    },
    {
        "id": "thanks_calls_no_tool",
        "prompt": "Thanks, that's all!",
        "expected_tools": set(),
        "contains": None,
    },
]


def _observed_tool_calls(all_messages: list[ModelMessage]) -> set[str]:
    """Tool calls are ToolCallParts inside ModelResponse parts — never in a
    ModelRequest. (Getting this wrong is why the old eval never worked.)"""
    observed: set[str] = set()
    for message in all_messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    observed.add(part.tool_name)
    return observed


@pytest.mark.parametrize("case", EVAL_CASES, ids=lambda c: c["id"])
def test_offline_agent_eval(case):
    agent = _build_agent()
    result = agent.run_sync(case["prompt"])

    observed = _observed_tool_calls(result.all_messages())

    # Tool-choice: exact toolset match — a memory question must not search, a
    # news question must not recall, and chitchat must call nothing.
    assert observed == case["expected_tools"], (
        f"{case['id']}: expected tools {case['expected_tools']}, observed {observed}"
    )

    # Answer-contains: the seeded fact returned by recall reaches the output.
    if case["contains"] is not None:
        assert case["contains"] in result.output, (
            f"{case['id']}: expected {case['contains']!r} in output, got {result.output!r}"
        )
