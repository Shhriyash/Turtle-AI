"""
WP3.D (ledger 3.8, transaction half): proves core/memory_replayer.py::replay()
commits all topic writes in ONE real Postgres transaction, against a real
Postgres — an injected failure on the 5th backend write_topic() call must
leave NO row for this test's user_id in personal_memory_topics.

Companion to test/wp3d_projection_transaction_test.py, which covers the same
scenario with a fake pool (this environment has no live Postgres to run
this file against; it is written to run in the `cloud-tests` CI job where
DATABASE_URL/REDIS_URL are set, and self-skips cleanly here).
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.cloud


def _make_event(topic: str, key: str, statement: str, event_id: str):
    from core.memory_journal import MemoryEvent

    return MemoryEvent(
        event_id=event_id,
        session_id="s1",
        turn_id="t1",
        observed_at="2026-09-20T10:00:00Z",
        kind="fact",
        topic=topic,
        key=key,
        value={},
        confidence=0.9,
        source="explicit",
        extractor="llm_turn",
        applied=True,
        statement=statement,
    )


def test_replay_injected_failure_on_fifth_write_commits_nothing() -> None:
    from core.memory_replayer import replay
    from core.personal_memory_store import PersonalMemoryStore
    from core.storage.cloud import get_pg_sync_pool
    from core.storage.cloud.personal_memory_store import PostgresPersonalMemoryBackend

    user_id = f"usr_{uuid.uuid4().hex[:12]}"

    with patch("core.personal_memory_store.settings") as fake_settings:
        fake_settings.is_cloud = True
        fake_settings.user_storage_cap_mb = 0
        store = PersonalMemoryStore(user_id=user_id)

        events = [
            _make_event("identity", "identity.name", "Name: Alice", "ev1"),
            _make_event("preferences", "preferences.response_style", "Response style: concise", "ev2"),
            _make_event(
                "workflow", "workflow.prefers_draft_before_send", "Prefers draft before send: true", "ev3"
            ),
            _make_event(
                "contacts", "contacts.frequent_recipient.bob", "Frequent recipient: bob@example.com", "ev4"
            ),
            _make_event("projects", "projects.turtle", "Project: Turtle", "ev5"),
            _make_event("corrections", "corrections.name_role", "Correction: role is engineer", "ev6"),
        ]

        real_write_topic = PostgresPersonalMemoryBackend.write_topic
        call_count = {"n": 0}

        def flaky_write_topic(self, topic_name, content):
            call_count["n"] += 1
            if call_count["n"] == 5:
                raise RuntimeError("simulated crash on the 5th topic write")
            return real_write_topic(self, topic_name, content)

        try:
            with patch.object(PostgresPersonalMemoryBackend, "write_topic", flaky_write_topic):
                with pytest.raises(RuntimeError):
                    replay(events, store=store, reference_time=datetime(2026, 9, 26, tzinfo=UTC))

            pool = get_pg_sync_pool()
            with pool.connection() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM personal_memory_topics WHERE user_id = %s", (user_id,)
                ).fetchone()
            assert row[0] == 0, "an injected failure on the 5th topic write must commit NO topics"
        finally:
            pool = get_pg_sync_pool()
            with pool.connection() as conn:
                conn.execute("DELETE FROM personal_memory_topics WHERE user_id = %s", (user_id,))
                conn.execute("DELETE FROM personal_memory_daily_logs WHERE user_id = %s", (user_id,))


def test_replay_without_injected_failure_commits_every_topic() -> None:
    """Sanity: the transaction wrapper must not suppress a normal, successful
    replay — every topic with content lands in the real table."""
    from core.memory_replayer import replay
    from core.personal_memory_store import PersonalMemoryStore
    from core.storage.cloud import get_pg_sync_pool

    user_id = f"usr_{uuid.uuid4().hex[:12]}"

    with patch("core.personal_memory_store.settings") as fake_settings:
        fake_settings.is_cloud = True
        fake_settings.user_storage_cap_mb = 0
        store = PersonalMemoryStore(user_id=user_id)

        events = [
            _make_event("identity", "identity.name", "Name: Alice", "ev1"),
            _make_event("preferences", "preferences.response_style", "Response style: concise", "ev2"),
        ]

        try:
            result = replay(events, store=store, reference_time=datetime(2026, 9, 26, tzinfo=UTC))
            assert set(result.written_topics) == {"identity", "preferences"}

            pool = get_pg_sync_pool()
            with pool.connection() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM personal_memory_topics WHERE user_id = %s", (user_id,)
                ).fetchone()
            # +1 for the MEMORY.md index row written by update_index_entry().
            assert row[0] == 3
        finally:
            pool = get_pg_sync_pool()
            with pool.connection() as conn:
                conn.execute("DELETE FROM personal_memory_topics WHERE user_id = %s", (user_id,))
                conn.execute("DELETE FROM personal_memory_daily_logs WHERE user_id = %s", (user_id,))
