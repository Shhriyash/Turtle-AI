"""
test/phase5_compaction_test.py
------------------------------
Phase 5 W3: session compaction at finalization.

SessionStore.mark_finalized now, in addition to flipping a session to
"completed", compacts the row: it truncates the messages blob to its last
COMPLETED_SESSION_MESSAGE_TAIL entries and resets pending_email to the
default-empty shape. The rolling summary + personal-memory journal carry the
durable memory; the retained tail is only for trace_replay reconstruct fidelity.

These tests pin:
  (a) a >tail completed session is truncated to the last N, summary + user_id kept
  (b) pending_email is reset to the default-empty shape on finalize
  (c) the cross-tenant refusal returns before compaction — foreign blob untouched
  (d) get_summary_tail_with_carryover still reads a compacted completed session
  (e) a session at/below the tail size passes through with its messages intact
"""
import asyncio

from core.session_store import COMPLETED_SESSION_MESSAGE_TAIL, SessionStore
from core.storage import Session
from core.storage.local.sqlite_store import SQLiteSessionStore


def run(coro):
    return asyncio.run(coro)


def _messages(n: int) -> list[dict]:
    # Opaque markers — mark_finalized only slices the list, never validates it,
    # so identifiable dicts let us assert exactly which tail survived.
    return [{"n": i} for i in range(n)]


def _summary() -> list[dict]:
    return [
        {"timestamp": "2026-07-20T00:00:00Z", "turn_id_range": [1, 2], "bullets": ["user likes tea"]},
    ]


def test_finalize_truncates_to_tail_keeps_summary_and_user(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await backend.init_db()

        total = COMPLETED_SESSION_MESSAGE_TAIL + 8  # 20
        await backend.put(
            Session(
                session_id="sess",
                data={
                    "status": "pending_finalization",
                    "user_id": "usr_a",
                    "messages": _messages(total),
                    "pending_email": SessionStore._default_pending_email(),
                    "summary": _summary(),
                },
            )
        )

        store = SessionStore(backend, user_id="usr_a")
        await store.mark_finalized("sess")

        row = await backend.get("sess")
        assert row is not None
        assert row.data["status"] == "completed"
        # Only the last N messages survive, in order.
        assert len(row.data["messages"]) == COMPLETED_SESSION_MESSAGE_TAIL
        expected_tail = _messages(total)[-COMPLETED_SESSION_MESSAGE_TAIL:]
        assert row.data["messages"] == expected_tail
        # Durable memory is preserved.
        assert row.data["summary"] == _summary()
        assert row.data["user_id"] == "usr_a"

    run(scenario())


def test_finalize_resets_pending_email(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await backend.init_db()

        await backend.put(
            Session(
                session_id="sess",
                data={
                    "status": "pending_finalization",
                    "user_id": "usr_a",
                    "messages": _messages(3),
                    # A live draft that must NOT survive into a future session.
                    "pending_email": {
                        "recipients": ["boss@corp.com"],
                        "cc_recipients": [],
                        "bcc_recipients": [],
                        "subject": "Q3 numbers",
                        "content": "here they are",
                    },
                    "summary": _summary(),
                },
            )
        )

        store = SessionStore(backend, user_id="usr_a")
        await store.mark_finalized("sess")

        row = await backend.get("sess")
        assert row is not None
        assert row.data["pending_email"] == SessionStore._default_pending_email()

    run(scenario())


def test_cross_tenant_refusal_leaves_blob_untouched(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await backend.init_db()

        total = COMPLETED_SESSION_MESSAGE_TAIL + 8
        foreign_email = {
            "recipients": ["b@b.com"],
            "cc_recipients": [],
            "bcc_recipients": [],
            "subject": "hi",
            "content": "body",
        }
        await backend.put(
            Session(
                session_id="owned_by_b",
                data={
                    "status": "pending_finalization",
                    "user_id": "usr_b",
                    "messages": _messages(total),
                    "pending_email": foreign_email,
                    "summary": _summary(),
                },
            )
        )

        # usr_a must not finalize — and must not compact — usr_b's session.
        store = SessionStore(backend, user_id="usr_a")
        await store.mark_finalized("owned_by_b")

        row = await backend.get("owned_by_b")
        assert row is not None
        assert row.data["status"] == "pending_finalization"  # not flipped
        assert len(row.data["messages"]) == total  # not truncated
        assert row.data["pending_email"] == foreign_email  # not reset

    run(scenario())


def test_carryover_reads_compacted_completed_session(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await backend.init_db()

        await backend.put(
            Session(
                session_id="prev",
                data={
                    "status": "pending_finalization",
                    "user_id": "usr_a",
                    "messages": _messages(COMPLETED_SESSION_MESSAGE_TAIL + 5),
                    "pending_email": SessionStore._default_pending_email(),
                    "summary": _summary(),
                },
            )
        )

        # Finalize -> completed + compacted, summary intact.
        finalizer = SessionStore(backend, user_id="usr_a")
        await finalizer.mark_finalized("prev")

        # A brand-new session for the same user (empty own summary) must still
        # seed [Recent Summary] from the compacted completed session.
        fresh = SessionStore(backend, user_id="usr_a")
        await fresh.init_backend()
        tail = await fresh.get_summary_tail_with_carryover(max_entries=6)
        assert tail == _summary()

    run(scenario())


def test_below_tail_passes_through_unchanged(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await backend.init_db()

        small = _messages(5)  # < COMPLETED_SESSION_MESSAGE_TAIL
        await backend.put(
            Session(
                session_id="sess",
                data={
                    "status": "pending_finalization",
                    "user_id": "usr_a",
                    "messages": small,
                    "pending_email": SessionStore._default_pending_email(),
                    "summary": _summary(),
                },
            )
        )

        store = SessionStore(backend, user_id="usr_a")
        await store.mark_finalized("sess")

        row = await backend.get("sess")
        assert row is not None
        assert row.data["status"] == "completed"
        assert row.data["messages"] == small  # every message retained, in order

    run(scenario())


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
