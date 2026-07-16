"""
test/session_memory_continuity_test.py
--------------------------------------
Covers two in-session-memory fixes:

1. Parallel multi_step synthesis previously dropped message_history and the
   caller stored the synthesis-only result as the canonical conversation,
   wiping prior turns. _RepairedHistoryResult must re-extend the conversation.

2. resume_if_active must rejoin a recently-disconnected (pending_finalization)
   session so a dropped/refreshed WebSocket keeps its history, while leaving
   stale sessions to be finalized normally.
"""
import asyncio
import unittest
from datetime import UTC, datetime, timedelta

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from core.graph import _RepairedHistoryResult
from core.session_store import SessionStore
from core.storage import Session


class _FakeSynthResult:
    """Minimal stand-in for a pydantic-ai run result (synthesis call)."""

    def __init__(self, output: str, messages: list) -> None:
        self.output = output
        self._messages = messages
        self.usage = "sentinel-usage"

    def all_messages(self) -> list:
        return self._messages


class RepairedHistoryResultTests(unittest.TestCase):
    def _prior(self) -> list:
        return [
            ModelRequest(parts=[UserPromptPart(content="my name is Shriyash")]),
            ModelResponse(parts=[TextPart(content="Nice to meet you, Shriyash.")]),
        ]

    def test_all_messages_extends_conversation(self) -> None:
        prior = self._prior()
        # The synthesis run's own history: an internal synthesis prompt + reply.
        synth_messages = [
            ModelRequest(parts=[UserPromptPart(content="SYNTHESIS PROMPT with tool blobs")]),
            ModelResponse(parts=[TextPart(content="Here is your answer.")]),
        ]
        inner = _FakeSynthResult("Here is your answer.", synth_messages)

        wrapped = _RepairedHistoryResult(inner, prior, user_prompt="what's the weather and news?")
        out = wrapped.all_messages()

        # Prior conversation is preserved.
        self.assertEqual(out[0], prior[0])
        self.assertEqual(out[1], prior[1])
        # The real user request is recorded — NOT the internal synthesis prompt.
        user_turn = out[2]
        self.assertIsInstance(user_turn, ModelRequest)
        self.assertEqual(user_turn.parts[0].content, "what's the weather and news?")
        self.assertNotIn(
            "SYNTHESIS PROMPT",
            " ".join(str(p) for m in out for p in m.parts),
        )
        # Final assistant reply is present and last.
        self.assertIsInstance(out[-1], ModelResponse)
        self.assertEqual(out[-1].parts[0].content, "Here is your answer.")
        self.assertEqual(len(out), 4)

    def test_keeps_only_final_response_drops_tool_turns(self) -> None:
        # Synthesis that called a tool: response(tool call) -> request(return) -> response(text)
        synth_messages = [
            ModelRequest(parts=[UserPromptPart(content="SYNTH PROMPT")]),
            ModelResponse(parts=[TextPart(content="(intermediate tool-call turn)")]),
            ModelRequest(parts=[UserPromptPart(content="(tool return)")]),
            ModelResponse(parts=[TextPart(content="final answer")]),
        ]
        inner = _FakeSynthResult("final answer", synth_messages)
        wrapped = _RepairedHistoryResult(inner, [], user_prompt="do the thing")
        out = wrapped.all_messages()

        # Only [real user turn, final response] — no orphaned tool turns.
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].parts[0].content, "do the thing")
        self.assertEqual(out[1].parts[0].content, "final answer")

    def test_output_and_attr_proxy(self) -> None:
        inner = _FakeSynthResult("answer", [ModelResponse(parts=[TextPart(content="answer")])])
        wrapped = _RepairedHistoryResult(inner, [], user_prompt="q")
        self.assertEqual(wrapped.output, "answer")
        # Unknown attributes proxy to the wrapped result (e.g. usage).
        self.assertEqual(wrapped.usage, "sentinel-usage")


class _FakeBackend:
    """In-memory SessionStoreProtocol with list_sessions + init_db."""

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    async def init_db(self) -> None:
        return None

    async def get(self, session_id: str):
        return self.sessions.get(session_id)

    async def put(self, session: Session) -> None:
        self.sessions[session.session_id] = session

    async def list_sessions(self, status_filter: str | None = None) -> list[Session]:
        out = list(self.sessions.values())
        if status_filter is not None:
            out = [s for s in out if s.data.get("status") == status_filter]
        return out


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


class ResumeIfActiveTests(unittest.TestCase):
    def _seed(self, backend: _FakeBackend, session_id: str, *, status: str, age_seconds: int, msg: str) -> None:
        updated = _iso(datetime.now(UTC) - timedelta(seconds=age_seconds))
        messages = [
            {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": msg}]},
        ]
        backend.sessions[session_id] = Session(
            session_id=session_id,
            data={"status": status, "messages": messages, "updated_at": updated},
        )

    def test_resumes_recent_pending_finalization_session(self) -> None:
        async def run():
            backend = _FakeBackend()
            self._seed(backend, "recent", status="pending_finalization", age_seconds=10, msg="hello earlier")
            store = SessionStore(backend=backend)
            result = await store.start_or_restore(mode="resume_if_active", resume_window_seconds=1800)
            return store, backend, result

        store, backend, result = asyncio.run(run())
        self.assertTrue(result.restored)
        self.assertEqual(store.session_id, "recent")
        self.assertEqual(result.message_count, 1)
        # Flipped back to active so the connect-time finalizer skips it.
        self.assertEqual(backend.sessions["recent"].data["status"], "active")

    def test_does_not_resume_stale_pending_session(self) -> None:
        async def run():
            backend = _FakeBackend()
            self._seed(backend, "stale", status="pending_finalization", age_seconds=7200, msg="old chat")
            store = SessionStore(backend=backend)
            result = await store.start_or_restore(mode="resume_if_active", resume_window_seconds=1800)
            return store, result

        store, result = asyncio.run(run())
        # New empty session created; stale one left pending for finalization.
        self.assertFalse(result.restored)
        self.assertEqual(store.message_history, [])
        self.assertNotEqual(store.session_id, "stale")

    def test_prefers_active_over_pending(self) -> None:
        async def run():
            backend = _FakeBackend()
            self._seed(backend, "active1", status="active", age_seconds=60, msg="active chat")
            self._seed(backend, "pending1", status="pending_finalization", age_seconds=5, msg="pending chat")
            store = SessionStore(backend=backend)
            result = await store.start_or_restore(mode="resume_if_active")
            return store, result

        store, result = asyncio.run(run())
        self.assertTrue(result.restored)
        self.assertEqual(store.session_id, "active1")


if __name__ == "__main__":
    unittest.main()
