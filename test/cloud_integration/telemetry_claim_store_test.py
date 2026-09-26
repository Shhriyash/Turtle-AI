"""
DDL-and-roundtrip test for core/storage/cloud/telemetry_claim_store.py
(table: telemetry_once) against a real Postgres. See
test/cloud_integration/routine_last_fired_store_test.py for the sibling
store this mirrors.
"""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.cloud


def test_try_claim_once_is_exactly_once() -> None:
    from core.storage.cloud import get_pg_sync_pool
    from core.storage.cloud.telemetry_claim_store import try_claim_once

    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    event = "onboarding_start"

    first = try_claim_once(user_id, event)
    assert first is True

    second = try_claim_once(user_id, event)
    assert second is False

    # A different event for the same user is a separate claim.
    other_event = try_claim_once(user_id, "onboarding_complete")
    assert other_event is True

    pool = get_pg_sync_pool()
    with pool.connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM telemetry_once WHERE user_id = %s", (user_id,)
        ).fetchone()[0]
    assert count == 2
