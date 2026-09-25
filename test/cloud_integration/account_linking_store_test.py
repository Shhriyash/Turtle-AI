"""
DDL-and-roundtrip test for core/storage/cloud/account_linking_store.py
(table: link_codes) against a real Postgres. See test/cloud_integration/conftest.py
for the live-services skip gate.
"""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.cloud


def test_link_code_store_issue_peek_reserve_consume_roundtrip() -> None:
    from core.storage.cloud.account_linking_store import PostgresLinkCodeStore

    store = PostgresLinkCodeStore()
    channel = "test_channel"
    channel_user_id = f"chan_{uuid.uuid4().hex[:12]}"
    source_user_id = f"usr_{uuid.uuid4().hex[:12]}"

    issued = store.issue(
        channel=channel, channel_user_id=channel_user_id, source_user_id=source_user_id
    )
    assert issued.channel == channel
    assert issued.source_user_id == source_user_id

    peeked = store.peek(issued.code)
    assert peeked is not None
    assert peeked.code == issued.code
    assert peeked.channel_user_id == channel_user_id

    target_user_id = f"usr_{uuid.uuid4().hex[:12]}"
    status, claim = store.reserve(issued.code, target_user_id)
    assert status == "ok"
    assert claim is not None
    assert claim.source_user_id == source_user_id

    consumed = store.consume(issued.code)
    assert consumed is not None
    assert consumed.code == issued.code

    # Already consumed — a second consume must find nothing.
    assert store.consume(issued.code) is None

    # Cleanup: purge this test's row so repeated runs stay tidy. purge_expired
    # only deletes expired rows, so delete directly via the store's own pool.
    from core.storage.cloud import get_pg_sync_pool

    pool = get_pg_sync_pool()
    with pool.connection() as conn:
        conn.execute("DELETE FROM link_codes WHERE code = %s", (issued.code,))
