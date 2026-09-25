"""
DDL-and-roundtrip test for core/storage/cloud/personal_memory_store.py
(tables: personal_memory_topics, personal_memory_daily_logs) against a real
Postgres.
"""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.cloud


def test_personal_memory_backend_topics_index_and_daily_log_roundtrip() -> None:
    from core.storage.cloud.personal_memory_store import PostgresPersonalMemoryBackend

    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    backend = PostgresPersonalMemoryBackend(user_id)

    assert backend.read_topic("identity") is None

    backend.write_topic("identity", "# Identity\n- Name: Test User\n")
    assert backend.read_topic("identity") == "# Identity\n- Name: Test User\n"

    backend.write_topic("identity", "# Identity\n- Name: Updated User\n")
    assert backend.read_topic("identity") == "# Identity\n- Name: Updated User\n"

    assert backend.read_index() is None
    backend.write_index("# MEMORY.md\n- identity\n")
    assert backend.read_index() == "# MEMORY.md\n- identity\n"

    assert backend.read_daily_log("2026-01-01") is None
    backend.write_daily_log("2026-01-01", "- did a thing\n")
    assert backend.read_daily_log("2026-01-01") == "- did a thing\n"

    assert backend.total_bytes() > 0

    assert backend.delete_topic("identity") is True
    assert backend.read_topic("identity") is None
    assert backend.delete_topic("identity") is False  # already gone

    from core.storage.cloud import get_pg_sync_pool

    pool = get_pg_sync_pool()
    with pool.connection() as conn:
        conn.execute("DELETE FROM personal_memory_topics WHERE user_id = %s", (user_id,))
        conn.execute("DELETE FROM personal_memory_daily_logs WHERE user_id = %s", (user_id,))
