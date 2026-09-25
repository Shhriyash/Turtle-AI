"""
DDL-and-roundtrip test for core/storage/cloud/routine_last_fired_store.py
(table: routine_last_fired) against a real Postgres.
"""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.cloud


def test_try_claim_fire_is_exactly_once() -> None:
    from core.storage.cloud.routine_last_fired_store import (
        prune_older_than,
        try_claim_fire,
    )
    from core.storage.cloud import get_pg_sync_pool

    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    routine_key = "morning_briefing"
    fire_bucket = "2026-01-01T08:00"

    first = try_claim_fire(user_id, routine_key, fire_bucket)
    assert first is True

    second = try_claim_fire(user_id, routine_key, fire_bucket)
    assert second is False

    # A different bucket for the same routine is a separate claim.
    other_bucket = try_claim_fire(user_id, routine_key, "2026-01-02T08:00")
    assert other_bucket is True

    # prune_older_than's own DELETE, read back via a direct row count.
    pool = get_pg_sync_pool()
    with pool.connection() as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM routine_last_fired WHERE user_id = %s", (user_id,)
        ).fetchone()[0]
    assert before == 2

    deleted = prune_older_than("2099-01-01T00:00:00")  # everything is "older"
    assert deleted >= 2

    with pool.connection() as conn:
        after = conn.execute(
            "SELECT COUNT(*) FROM routine_last_fired WHERE user_id = %s", (user_id,)
        ).fetchone()[0]
    assert after == 0
