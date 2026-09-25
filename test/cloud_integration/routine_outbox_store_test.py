"""
DDL-and-roundtrip test for core/storage/cloud/routine_outbox_store.py
(table: routine_outbox) against a real Postgres.
"""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.cloud


def test_outbox_save_load_and_empty_clears_row() -> None:
    from core.storage.cloud.routine_outbox_store import load_outbox_pg, save_outbox_pg

    user_id = f"usr_{uuid.uuid4().hex[:12]}"

    assert load_outbox_pg(user_id) == []

    frames = [{"type": "routine_notice", "text": "reminder 1"}, {"type": "routine_notice", "text": "reminder 2"}]
    save_outbox_pg(user_id, frames, max_frames=10)

    loaded = load_outbox_pg(user_id)
    assert loaded == frames

    # max_frames caps the tail.
    many_frames = [{"type": "routine_notice", "text": f"reminder {i}"} for i in range(5)]
    save_outbox_pg(user_id, many_frames, max_frames=2)
    assert load_outbox_pg(user_id) == many_frames[-2:]

    # Saving an empty list deletes the row.
    save_outbox_pg(user_id, [], max_frames=10)
    assert load_outbox_pg(user_id) == []
