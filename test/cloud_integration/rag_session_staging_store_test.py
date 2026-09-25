"""
DDL-and-roundtrip test for core/storage/cloud/rag_session_staging_store.py
(table: rag_session_staging) against a real Postgres.
"""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.cloud


def test_rag_session_staging_write_read_clear_roundtrip() -> None:
    from core.storage.cloud.rag_session_staging_store import PostgresRagSessionStaging

    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    store = PostgresRagSessionStaging(user_id)

    assert store.read() is None

    session_data = {
        "session_id": "rag_sess_1",
        "creation_time": "2026-01-01T00:00:00+00:00",
        "conversations": [{"role": "user", "content": "hi"}],
    }
    store.write(session_data)

    read_back = store.read()
    assert read_back is not None
    assert read_back["session_id"] == "rag_sess_1"
    assert read_back["conversations"] == [{"role": "user", "content": "hi"}]

    updated = dict(session_data, conversations=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    store.write(updated)
    assert len(store.read()["conversations"]) == 2

    store.clear()
    assert store.read() is None
