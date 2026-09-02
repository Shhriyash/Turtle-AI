"""
test/session_memory_continuity_test.py
--------------------------------------
Covers the in-session-memory continuity fix:

resume_if_active must rejoin a recently-disconnected (pending_finalization)
session so a dropped/refreshed WebSocket keeps its history, while leaving
stale sessions to be finalized normally.
"""
import asyncio
import unittest
from datetime import UTC, datetime, timedelta

from core.session_store import SessionStore
from core.storage import Session


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
