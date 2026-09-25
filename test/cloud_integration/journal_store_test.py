"""
DDL-and-roundtrip test for core/storage/cloud/journal_store.py
(table: journal_events) against a real Postgres.
"""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.cloud


def test_journal_backend_append_iter_roundtrip() -> None:
    from core.memory_journal import MemoryEvent
    from core.storage.cloud.journal_store import PostgresJournalBackend, list_user_ids_pg

    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    backend = PostgresJournalBackend(user_id)

    event = MemoryEvent(
        event_id=f"evt_{uuid.uuid4().hex[:12]}",
        session_id="sess_1",
        turn_id="turn_1",
        observed_at="2026-01-01T00:00:00+00:00",
        kind="fact",
        topic="identity",
        key="name",
        value={"name": "Test User"},
        confidence=0.9,
        source="test",
        extractor="test_extractor",
    )

    assert backend.event_exists(event.event_id) is False

    backend.append_line(event)
    assert backend.event_exists(event.event_id) is True

    # Idempotent — ON CONFLICT DO NOTHING must not raise or duplicate.
    backend.append_line(event)

    events = list(backend.iter_events())
    assert len(events) == 1
    assert events[0].event_id == event.event_id
    assert events[0].value == {"name": "Test User"}

    assert backend.created_at_timestamp() is not None
    assert backend.total_bytes() > 0

    assert user_id in list_user_ids_pg()

    from core.storage.cloud import get_pg_sync_pool

    pool = get_pg_sync_pool()
    with pool.connection() as conn:
        conn.execute("DELETE FROM journal_events WHERE user_id = %s", (user_id,))
