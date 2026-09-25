"""
DDL-and-roundtrip test for core/storage/cloud/identity_store.py — the 4
tables users, channel_mappings, claimed_tokens, account_markers — against a
real Postgres.
"""
from __future__ import annotations

import uuid

import pytest

from test.cloud_integration.conftest import run_async

pytestmark = pytest.mark.cloud


def test_identity_store_full_roundtrip() -> None:
    from core.storage.cloud.identity_store import (
        PostgresIdentityManager,
        write_account_marker_pg,
    )
    from core.storage.cloud import get_pg_sync_pool

    manager = PostgresIdentityManager()
    channel = f"test_chan_{uuid.uuid4().hex[:8]}"
    channel_user_id = f"cu_{uuid.uuid4().hex[:12]}"

    async def _run() -> tuple[str, str]:
        # users + channel_mappings, via resolve_user's own SQL.
        user_id = await manager.resolve_user(channel, channel_user_id)
        assert user_id.startswith("usr_")

        # Calling again must resolve to the SAME user_id (channel_mappings read).
        again = await manager.resolve_user(channel, channel_user_id)
        assert again == user_id

        # claimed_tokens, via mark_token_claimed's own SQL.
        jti = f"jti_{uuid.uuid4().hex[:12]}"
        first_claim = await manager.mark_token_claimed(jti, user_id)
        assert first_claim is True
        second_claim = await manager.mark_token_claimed(jti, user_id)
        assert second_claim is False  # PK conflict -> already claimed

        # link_channel: rebind a second channel identity onto the same user.
        other_channel_user_id = f"cu2_{uuid.uuid4().hex[:12]}"
        previous = await manager.link_channel(
            user_id=user_id, channel=channel, channel_user_id=other_channel_user_id
        )
        assert previous is None
        resolved_other = await manager.resolve_user(channel, other_channel_user_id)
        assert resolved_other == user_id

        return user_id, jti

    user_id, jti = run_async(_run())

    # account_markers, via write_account_marker_pg's own SQL (sync/psycopg).
    email = f"user-{uuid.uuid4().hex[:8]}@example.invalid"
    write_account_marker_pg(user_id, email, True, channel=channel)

    pool = get_pg_sync_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT email, email_verified FROM account_markers WHERE user_id = %s",
            (user_id,),
        ).fetchone()
        assert row == (email, True)

        # Cleanup all 4 rows this test created.
        conn.execute("DELETE FROM account_markers WHERE user_id = %s", (user_id,))
        conn.execute("DELETE FROM claimed_tokens WHERE jti = %s", (jti,))
        conn.execute("DELETE FROM channel_mappings WHERE user_id = %s", (user_id,))
        conn.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
