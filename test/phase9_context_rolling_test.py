"""
Phase 9 — production in-session context rolling.

Three defects that together made Turtle "forget" mid-conversation, especially on
channels (Discord):

1. TOKEN ESTIMATOR counted `len(str(message)) // 4` — the pydantic-ai dataclass
   REPR, not the content. Every message dragged ~130-155 chars of
   `datetime.datetime(...)` / `RequestUsage()` scaffolding => ~35 phantom tokens
   each, ~1,380 across a 40-message window (35% of the 4000 budget). Real turns
   were evicted 6-10x too early.

2. CHANNEL STATE TTL (1800s) equalled SessionStore's resume window (1800s), so
   idle-eviction was deterministically the one case that could NOT resume: the
   session's age was always just past the window => brand-new empty session.

3. SUMMARY CARRYOVER only scanned "completed" sessions, but channel-only users
   never hit the WebSocket sweep that finalizes, so their sessions sit in
   "pending_finalization" forever => zero carryover, every conversation cold.
"""
from __future__ import annotations

import apps.turtle_server as ts
from core.session_store import CARRYOVER_MAX_AGE_S
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)


def _U(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _A(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


# ── 1. token estimator ───────────────────────────────────────────────────────

def test_estimator_counts_content_not_repr():
    """A short turn must cost a handful of tokens, not ~36 tokens of timestamp."""
    msg = _U("hi there")
    repr_based = len(str(msg)) // 4
    assert ts._estimate_message_tokens(msg) < 10
    # and it must be dramatically cheaper than the old repr-based count
    assert ts._estimate_message_tokens(msg) * 3 < repr_based


def test_estimator_scales_with_real_content():
    small = ts._estimate_message_tokens(_U("hi"))
    big = ts._estimate_message_tokens(_U("word " * 400))
    assert big > small * 20


def test_estimator_ignores_timestamp_scaffolding():
    """Two messages with identical content cost the same regardless of the
    repr's variable-length timestamp/usage fields."""
    a = ts._estimate_message_tokens(_A("same body"))
    b = ts._estimate_message_tokens(_A("same body"))
    assert a == b


# ── 2. the trim keeps a realistic conversation ───────────────────────────────

def test_realistic_conversation_is_not_trimmed():
    """20 short turns (40 messages) is an ordinary chat. Under the old repr
    estimator this scored ~1600 phantom tokens and got trimmed; it must now
    survive intact so the model can actually see the conversation."""
    history: list = []
    for i in range(20):
        history.append(_U(f"question number {i} about my day"))
        history.append(_A(f"answer number {i}, here is a short helpful reply"))
    out = ts._trim_history_for_context(history)
    assert len(out) == len(history), "a normal 20-turn chat must not be evicted"


def test_genuinely_oversized_history_is_still_trimmed():
    """The budget must still bind on real content — no runaway context."""
    history: list = []
    for i in range(30):
        history.append(_U("x " * 2000))   # ~1000 tokens each of REAL content
        history.append(_A("y " * 2000))
    out = ts._trim_history_for_context(history)
    assert len(out) < len(history)
    total = sum(ts._estimate_message_tokens(m) for m in out)
    # allow the final message to overshoot; the window must still be bounded
    assert total <= ts.ACTIVE_HISTORY_MAX_TOKENS + 1200


# ── 3. eviction must not be the un-resumable case ────────────────────────────

def test_channel_ttl_is_below_session_resume_window():
    """Invariant: a channel state evicted for idleness must still fall INSIDE
    SessionStore's resume window, or the next message starts cold."""
    import inspect
    from core.session_store import SessionStore

    sig = inspect.signature(SessionStore.start_or_restore)
    resume_window = sig.parameters["resume_window_seconds"].default
    assert ts._CHANNEL_STATE_IDLE_TTL_S < resume_window, (
        f"channel TTL {ts._CHANNEL_STATE_IDLE_TTL_S}s must be < resume window "
        f"{resume_window}s, else eviction guarantees a cold session"
    )


# ── 4. carryover reaches channel (never-finalized) sessions ──────────────────

def test_carryover_age_cap_is_sane():
    """A bounded window: long enough for 'same user, next day', short enough
    that a crash-orphaned session can't leak stale context forever."""
    assert 3600 <= CARRYOVER_MAX_AGE_S <= 7 * 24 * 3600


def test_carryover_scans_pending_finalization(monkeypatch):
    """The channel case: the only prior session is pending_finalization (never
    finalized because there is no WS sweep on Discord). Carryover must find it."""
    import asyncio
    from core.session_store import SessionStore

    store = SessionStore.__new__(SessionStore)   # bypass __init__/backend
    store.session_id = "cur"
    store._summary = []

    class _S:
        def __init__(self, sid, data):
            self.session_id = sid
            self.data = data

    pending = _S("prev", {"summary": [{"bullets": ["talked about Goa"]}],
                          "updated_at": "recent"})

    async def _list(status):
        return [pending] if status == "pending_finalization" else []

    store.backend = type("B", (), {"list_sessions": lambda self: None})()
    store._list_sessions_for_user = _list                      # type: ignore
    store.get_summary_tail = lambda max_entries=6: []          # type: ignore
    store._seconds_since = lambda ts_str: 60.0                 # type: ignore

    out = asyncio.run(store.get_summary_tail_with_carryover(max_entries=6))
    assert out and out[0]["bullets"] == ["talked about Goa"]


def test_carryover_rejects_stale_pending_session(monkeypatch):
    """A long-abandoned pending session must NOT leak into a fresh conversation."""
    import asyncio
    from core.session_store import SessionStore

    store = SessionStore.__new__(SessionStore)
    store.session_id = "cur"
    store._summary = []

    class _S:
        def __init__(self, sid, data):
            self.session_id = sid
            self.data = data

    stale = _S("old", {"summary": [{"bullets": ["ancient context"]}],
                       "updated_at": "long ago"})

    async def _list(status):
        return [stale] if status == "pending_finalization" else []

    store.backend = type("B", (), {"list_sessions": lambda self: None})()
    store._list_sessions_for_user = _list                          # type: ignore
    store.get_summary_tail = lambda max_entries=6: []              # type: ignore
    store._seconds_since = lambda ts_str: CARRYOVER_MAX_AGE_S + 1  # type: ignore

    out = asyncio.run(store.get_summary_tail_with_carryover(max_entries=6))
    assert out == []
