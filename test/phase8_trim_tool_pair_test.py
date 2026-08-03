"""
Phase 8 — the Gemini function-call-adjacency 400 root cause.

Live symptom: every tool turn whose prompt+memory-context exceeded the history
token budget failed on gemini-2.5-flash with HTTP 400 "Please ensure that
function response turn comes immediately after a function call turn", forcing the
slow llama fallback and a ~60s TimeoutError.

Root cause: _trim_history_for_context, when the in-turn history exceeds
TURTLE_HISTORY_MAX_TOKENS, dropped the (bloated) UserPromptPart AND then dropped
the leading ModelResponse(ToolCallPart) — orphaning its ToolReturnPart. The
window handed to Gemini was a lone function_response with no preceding
function_call, which Gemini rejects. pydantic-ai's empty-user-turn prepend
repairs a leading function_CALL but NOT a leading function_RESPONSE.

Fix: the front-normalization is now pair-aware — it never drops a
ModelResponse(tool_call) whose ToolReturnPart survives, and never emits a window
that starts with an orphan tool-return (falling back to a valid tail instead).
"""
from __future__ import annotations

import apps.turtle_server as ts
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    TextPart,
)


def _U(text): return ModelRequest(parts=[UserPromptPart(content=text)])
def _C(cid): return ModelResponse(parts=[ToolCallPart(tool_name="remember", args={"a": 1}, tool_call_id=cid)])
def _R(cid): return ModelRequest(parts=[ToolReturnPart(tool_name="remember", content="Stored", tool_call_id=cid)])
def _AT(text): return ModelResponse(parts=[TextPart(content=text)])

# A prompt guaranteed to exceed the token budget → forces the trimming path.
_BIG = "x" * (ts.ACTIVE_HISTORY_MAX_TOKENS * 4 + 100)


def _leads_with_orphan_return(history) -> bool:
    if not history:
        return False
    head = history[0]
    return (
        isinstance(head, ModelRequest)
        and bool(head.parts)
        and all(isinstance(p, ToolReturnPart) for p in head.parts)
    )


def _has_user_turn(history) -> bool:
    return any(
        isinstance(m, ModelRequest) and any(isinstance(p, UserPromptPart) for p in m.parts)
        for m in history
    )


def test_over_budget_tool_turn_is_not_orphaned():
    """The exact failing case: [big_user, call, return] over budget must NOT
    collapse to a lone orphan ToolReturnPart."""
    out = ts._trim_history_for_context([_U(_BIG), _C("x1"), _R("x1")])
    assert out, "trim must never return an empty window"
    assert not _leads_with_orphan_return(out), (
        "window starts with an orphan function_response — Gemini 400s on this"
    )
    # the call/return pair must not be split (return kept without its call)
    call_ids = {p.tool_call_id for m in out if isinstance(m, ModelResponse)
                for p in m.parts if isinstance(p, ToolCallPart)}
    return_ids = {p.tool_call_id for m in out if isinstance(m, ModelRequest)
                  for p in m.parts if isinstance(p, ToolReturnPart)}
    assert return_ids <= call_ids, "a ToolReturnPart survived without its ToolCallPart"


def test_trim_then_sanitize_never_leads_with_orphan():
    """Through the REAL processor chain (trim → sanitize), several shapes must
    never yield a window that leads with an orphan tool-return."""
    cases = [
        [_U(_BIG), _C("a"), _R("a")],                         # observed
        [_U("t1"), _AT("a1"), _U(_BIG), _C("b"), _R("b")],    # over-budget multi-turn
        [_C("c"), _R("c"), _U(_BIG)],                         # no leading user
        [_R("gone"), _U("q"), _C("d"), _R("d")],              # pre-existing leading orphan
    ]
    for h in cases:
        out = ts._sanitize_tool_pairs(ts._trim_history_for_context(h))
        assert out, f"empty window for {h}"
        assert not _leads_with_orphan_return(out), f"orphan-led window for {h}: {out}"


def test_short_history_is_untouched():
    """The common path — short, under-budget history — is returned unchanged."""
    h = [_U("hello"), _AT("hi there")]
    assert ts._trim_history_for_context(h) == h


def test_over_budget_window_keeps_a_user_turn():
    """Trimming must not strip the conversation down to tool-only turns."""
    out = ts._trim_history_for_context([_U(_BIG), _C("x1"), _R("x1")])
    assert _has_user_turn(out)
