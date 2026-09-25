"""
DDL-and-roundtrip test for core/storage/cloud/postgres_store.py
(table: sessions) against a real Postgres.
"""
from __future__ import annotations

import uuid

import pytest

from test.cloud_integration.conftest import run_async

pytestmark = pytest.mark.cloud


def test_postgres_session_store_put_get_list_delete_roundtrip() -> None:
    from core.storage import Session
    from core.storage.cloud.postgres_store import PostgresSessionStore

    store = PostgresSessionStore()
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    user_id = f"usr_{uuid.uuid4().hex[:12]}"

    async def _run():
        await store.init_db()

        assert await store.get(session_id) is None

        session = Session(session_id=session_id, data={"user_id": user_id, "status": "active"})
        await store.put(session)

        fetched = await store.get(session_id)
        assert fetched is not None
        assert fetched.data["user_id"] == user_id
        assert fetched.data["status"] == "active"

        listed = await store.list_sessions(user_id=user_id)
        assert any(s.session_id == session_id for s in listed)

        filtered = await store.list_sessions(status_filter="active", user_id=user_id)
        assert any(s.session_id == session_id for s in filtered)

        await store.delete(session_id)
        assert await store.get(session_id) is None

    run_async(_run())
