"""
DDL-and-roundtrip test for core/storage/cloud/confirmation_state_store.py
(table: confirmation_state) against a real Postgres.
"""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.cloud


def test_confirmation_state_load_save_roundtrip() -> None:
    from core.storage.cloud.confirmation_state_store import PostgresConfirmationState

    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    store = PostgresConfirmationState(user_id)

    # No row yet — load() must return the empty-pending default.
    assert store.load() == {"pending": []}

    event_ids = ["evt_1", "evt_2", "evt_3"]
    store.save({"pending": event_ids})
    assert store.load() == {"pending": event_ids}

    # ON CONFLICT DO UPDATE path.
    store.save({"pending": ["evt_4"]})
    assert store.load() == {"pending": ["evt_4"]}

    from core.storage.cloud import get_pg_sync_pool

    pool = get_pg_sync_pool()
    with pool.connection() as conn:
        conn.execute("DELETE FROM confirmation_state WHERE user_id = %s", (user_id,))
