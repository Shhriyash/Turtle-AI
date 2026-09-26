"""
test/cloud_integration/tenant_purge_cloud_test.py
---------------------------------------------------
WP 2.A (ledger 2.3/2.4) — real-Postgres/real-Redis proof that
core/tenant_purge.py::purge_user actually erases a user in cloud mode.

Before this WP, /admin/users, /forget-me, and /forget-me/confirm all reached
for aiosqlite.connect(identity_manager.db_path) — an attribute that only
exists on the LOCAL IdentityManager. In cloud mode identity_manager is a
PostgresIdentityManager, which has no db_path at all, so every one of those
three routes raised AttributeError outright. See test_admin_routes_cloud_*
below, which exercises the route handlers directly (not over HTTP) against
this package's live Postgres/Redis.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from test.cloud_integration.conftest import run_async

pytestmark = pytest.mark.cloud

_EMB_DIM = 1024


def _uid() -> str:
    return f"usr_{uuid.uuid4().hex[:12]}"


async def _ensure_all_tables() -> None:
    """Create every table this test seeds, using each store module's own DDL
    constants — read-only imports, nothing here edits those modules."""
    from core.storage.cloud import get_pg_pool
    from core.storage.cloud.account_linking_store import _CREATE_TABLE_SQL as _LINK_CODES_SQL
    from core.storage.cloud.calendar_token_store import _CREATE_TABLE_SQL as _CAL_TOKENS_SQL
    from core.storage.cloud.confirmation_state_store import _CREATE_TABLE_SQL as _CONFIRM_SQL
    from core.storage.cloud.identity_store import _CREATE_TABLES_SQL as _IDENTITY_SQLS
    from core.storage.cloud.journal_store import _CREATE_TABLE_SQL as _JOURNAL_SQL
    from core.storage.cloud.personal_memory_store import _CREATE_TABLES_SQL as _PM_SQLS
    from core.storage.cloud.pgvector_store import (
        _CREATE_VECTOR_DOCS_SQL,
        _CREATE_VECTOR_CHUNKS_SQL,
    )
    from core.storage.cloud.postgres_store import _CREATE_TABLE_SQL as _SESSIONS_SQL
    from core.storage.cloud.rag_session_staging_store import _CREATE_TABLE_SQL as _RAG_STAGE_SQL
    from core.storage.cloud.routine_last_fired_store import _CREATE_TABLE_SQL as _RLF_SQL
    from core.storage.cloud.routine_outbox_store import _CREATE_TABLE_SQL as _OUTBOX_SQL

    statements = list(_IDENTITY_SQLS) + list(_PM_SQLS) + [
        _LINK_CODES_SQL,
        _CAL_TOKENS_SQL,
        _CONFIRM_SQL,
        _JOURNAL_SQL,
        _CREATE_VECTOR_DOCS_SQL,
        _CREATE_VECTOR_CHUNKS_SQL,
        _SESSIONS_SQL,
        _RAG_STAGE_SQL,
        _RLF_SQL,
        _OUTBOX_SQL,
    ]
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        for stmt in statements:
            await conn.execute(stmt)


async def _seed_all_tables(user_id: str, *, other_user_id: str) -> None:
    """Seed exactly one row per table for user_id (and, for link_codes, a
    SECOND row where user_id is only the reserved_for side)."""
    from core.storage.cloud import get_pg_pool

    pool = await get_pg_pool()
    zero_vec = [0.0] * _EMB_DIM
    now = datetime.now(UTC)
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO users (user_id) VALUES ($1)", user_id)
        await conn.execute(
            "INSERT INTO channel_mappings (channel, channel_user_id, user_id) "
            "VALUES ('test_chan', $1, $2)",
            f"cu_{uuid.uuid4().hex[:10]}", user_id,
        )
        await conn.execute(
            "INSERT INTO claimed_tokens (jti, user_id) VALUES ($1, $2)",
            f"jti_{uuid.uuid4().hex[:10]}", user_id,
        )
        await conn.execute(
            "INSERT INTO account_markers (user_id, email, email_verified, created_at, channel) "
            "VALUES ($1, $2, true, $3, 'web_email')",
            user_id, f"{user_id}@example.invalid", now,
        )
        # link_codes: one row where user_id is the SOURCE, one where it is
        # only the RESERVED-FOR side (two-column purge, ledger 1b.6/2.3).
        await conn.execute(
            "INSERT INTO link_codes (code, channel, channel_user_id, source_user_id, expires_at) "
            "VALUES ($1, 'test_chan', $2, $3, $4)",
            f"CODE{uuid.uuid4().hex[:6].upper()}", f"cu_{uuid.uuid4().hex[:10]}",
            user_id, now + timedelta(minutes=15),
        )
        await conn.execute(
            "INSERT INTO link_codes (code, channel, channel_user_id, source_user_id, "
            "expires_at, reserved_for) VALUES ($1, 'test_chan', $2, $3, $4, $5)",
            f"CODE{uuid.uuid4().hex[:6].upper()}", f"cu_{uuid.uuid4().hex[:10]}",
            other_user_id, now + timedelta(minutes=15), user_id,
        )
        await conn.execute(
            "INSERT INTO calendar_tokens (user_id, token_json) VALUES ($1, $2)",
            user_id, json.dumps({"refresh_token": "fake-refresh-token-for-test"}),
        )
        await conn.execute(
            "INSERT INTO sessions (session_id, data, user_id) VALUES ($1, $2::jsonb, $3)",
            f"sess_{uuid.uuid4().hex[:10]}", json.dumps({}), user_id,
        )
        await conn.execute(
            "INSERT INTO personal_memory_topics (user_id, topic_name, content) "
            "VALUES ($1, 'identity', 'test content')",
            user_id,
        )
        await conn.execute(
            "INSERT INTO personal_memory_daily_logs (user_id, log_date, content) "
            "VALUES ($1, '2026-09-26', 'test log')",
            user_id,
        )
        await conn.execute(
            "INSERT INTO routine_outbox (user_id, frames) VALUES ($1, '[]'::jsonb)",
            user_id,
        )
        await conn.execute(
            "INSERT INTO rag_session_staging (user_id, session_id, creation_time) "
            "VALUES ($1, $2, $3)",
            user_id, f"sess_{uuid.uuid4().hex[:10]}", now.isoformat(),
        )
        await conn.execute(
            "INSERT INTO vector_docs (user_id, doc_id, text, embedding) "
            "VALUES ($1, $2, 'test doc', $3)",
            user_id, f"doc_{uuid.uuid4().hex[:10]}", zero_vec,
        )
        await conn.execute(
            "INSERT INTO vector_chunks (user_id, chunk_id, content, embedding) "
            "VALUES ($1, $2, 'test chunk', $3)",
            user_id, f"chunk_{uuid.uuid4().hex[:10]}", zero_vec,
        )
        await conn.execute(
            "INSERT INTO routine_last_fired (user_id, routine_key, fire_bucket) "
            "VALUES ($1, 'daily_digest', '2026-09-26')",
            user_id,
        )
        await conn.execute(
            "INSERT INTO journal_events (user_id, event_id, observed_at, payload) "
            "VALUES ($1, $2, $3, '{}'::jsonb)",
            user_id, f"evt_{uuid.uuid4().hex[:10]}", now,
        )
        await conn.execute(
            "INSERT INTO confirmation_state (user_id, pending) VALUES ($1, '[]'::jsonb)",
            user_id,
        )


async def _table_row_count(table: str, columns: tuple[str, ...], user_id: str) -> int:
    from core.storage.cloud import get_pg_pool

    pool = await get_pg_pool()
    where = " OR ".join(f"{c} = $1" for c in columns)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", user_id)
    return row["n"]


async def _cleanup_control_user(user_id: str) -> None:
    from core.tenant_purge import table_enumeration
    from core.storage.cloud import get_pg_pool

    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            for table, columns in table_enumeration().items():
                where = " OR ".join(f"{c} = $1" for c in columns)
                await conn.execute(f"DELETE FROM {table} WHERE {where}", user_id)


def test_purge_leaves_zero_rows_in_every_table_including_link_codes_both_columns() -> None:
    from core.tenant_purge import table_enumeration, purge_user

    user_id = _uid()
    other_user_id = _uid()

    async def _run():
        await _ensure_all_tables()
        await _seed_all_tables(user_id, other_user_id=other_user_id)
        await _seed_all_tables(other_user_id, other_user_id=user_id)  # control: must survive

        # Sanity: every table actually has >=1 row for this user before purge.
        before = {}
        for table, columns in table_enumeration().items():
            before[table] = await _table_row_count(table, columns, user_id)
        assert all(n >= 1 for n in before.values()), before

        result = await purge_user(user_id)

        after = {}
        for table, columns in table_enumeration().items():
            after[table] = await _table_row_count(table, columns, user_id)
        return after, result

    try:
        after, result = run_async(_run())
        for table, columns in table_enumeration().items():
            assert after[table] == 0, f"table {table!r} still has rows after purge: {after[table]}"
        assert isinstance(result["tables"], dict)
        assert set(result["tables"].keys()) == set(table_enumeration().keys())

        # Control user survives.
        control_after = run_async(
            _table_row_count("users", ("user_id",), other_user_id)
        )
        assert control_after == 1
    finally:
        run_async(_cleanup_control_user(other_user_id))


def test_redis_keys_purged_but_global_and_ephemeral_families_survive() -> None:
    from core.storage.cloud import get_redis_sync_client
    from core.tenant_purge import purge_user

    user_id = _uid()
    client = get_redis_sync_client()

    per_user_keys = [
        f"turtle:spend:{user_id}:20260926",
        f"turtle:places_cap:v1:{user_id}",
        f"turtle:ws_rate:{user_id}",
        f"turtle:live:{user_id}",
        f"turtle:gate:{user_id}:web_email",
        f"turtle:idem:{user_id}:cal:deadbeef",
    ]
    survivor_keys = [
        "turtle:places_cache:v1:find_place:deadbeef",  # deliberately global
        "turtle:nonce:some-random-nonce",                # ephemeral, not user-keyed
        "turtle:job:some-job-id",                         # ephemeral, not user-keyed
        "turtle:discord-interaction:some-interaction-id",  # ephemeral, not user-keyed
    ]
    for key in per_user_keys + survivor_keys:
        client.set(key, "1", ex=120)

    try:
        run_async(purge_user(user_id))

        for key in per_user_keys:
            assert client.get(key) is None, f"{key} should have been purged"
        for key in survivor_keys:
            assert client.get(key) == "1", f"{key} must survive the purge"
    finally:
        for key in per_user_keys + survivor_keys:
            client.delete(key)


def test_revoke_called_before_any_table_delete_and_revoke_failure_does_not_abort_purge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patches core.tenant_purge's OWN references to the imported Phase 1
    helpers (imported inside _revoke_calendar_token, so we patch the source
    module apps.calendar_oauth_routes — the only place that import resolves
    from) to: (a) prove revoke runs before any row is deleted, by having the
    fake revoke assert the row still exists at call time, and (b) prove a
    revoke failure (returns False, simulating Google unreachable) does not
    abort the purge — the table row is still gone afterwards.
    """
    from core.tenant_purge import purge_user

    user_id = _uid()
    revoke_called: list[bool] = []

    async def fake_read_token(uid: str) -> str | None:
        assert uid == user_id
        return json.dumps({"refresh_token": "fake-refresh-token-for-test"})

    async def fake_revoke_at_google(token_json: str) -> bool:
        # Ordering assertion: the user's `users` row must still exist right
        # now — revoke must run BEFORE the delete transaction.
        still_there = await _table_row_count("users", ("user_id",), user_id)
        assert still_there == 1, "revoke ran AFTER the user row was deleted"
        revoke_called.append(True)
        return False  # simulate Google unreachable / non-200/400 status

    async def _run():
        await _ensure_all_tables()
        await _seed_all_tables(user_id, other_user_id=_uid())

        import apps.calendar_oauth_routes as cal_routes

        monkeypatch.setattr(cal_routes, "_read_token", fake_read_token)
        monkeypatch.setattr(cal_routes, "_revoke_at_google", fake_revoke_at_google)

        return await purge_user(user_id)

    result = run_async(_run())
    assert revoke_called == [True]
    assert result["revoked"] is False  # reported, not swallowed
    assert result["tables"]["users"] == 1  # purge still completed despite revoke failure


def test_purge_log_row_is_content_free() -> None:
    from core.storage.cloud import get_pg_pool
    from core.tenant_purge import purge_user
    from core.storage.cloud.purge_log_store import hash_user_id

    user_id = _uid()

    async def _run():
        await _ensure_all_tables()
        await _seed_all_tables(user_id, other_user_id=_uid())
        result = await purge_user(user_id)

        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT user_id_sha256, counts FROM purge_log "
                "WHERE user_id_sha256 = $1 ORDER BY id DESC LIMIT 1",
                hash_user_id(user_id),
            )
        return result, row

    result, row = run_async(_run())
    assert row is not None
    assert row["user_id_sha256"] == hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    counts_text = json.dumps(row["counts"]) if not isinstance(row["counts"], str) else row["counts"]
    assert user_id not in counts_text  # plaintext id appears nowhere in the row
    assert user_id not in row["user_id_sha256"]


def test_admin_routes_cloud_users_forget_me_and_confirm_all_work() -> None:
    """Before WP 2.A: all three routes raised AttributeError in cloud mode
    (identity_manager.db_path doesn't exist on PostgresIdentityManager). This
    exercises the route handlers directly against a live Postgres/Redis and
    asserts they now succeed."""
    import jwt
    from pydantic import SecretStr

    from apps import admin_routes
    from core.config import settings
    from core.storage.cloud.identity_store import PostgresIdentityManager

    orig_admin_token = settings.admin_token
    orig_identity_manager = admin_routes.identity_manager
    settings.admin_token = SecretStr("fake-admin-token-for-cloud-test")
    admin_routes.identity_manager = PostgresIdentityManager()

    email = f"purge-{uuid.uuid4().hex[:10]}@example.invalid"

    async def _run():
        user_id = await admin_routes.identity_manager.resolve_user("web_email", email)

        users_resp = await admin_routes.admin_users(x_admin_token="fake-admin-token-for-cloud-test")
        users_payload = json.loads(users_resp.body)
        assert any(u["user_id"] == user_id for u in users_payload["users"])
        # Cloud mode reports null filesystem stats rather than crashing.
        matched = next(u for u in users_payload["users"] if u["user_id"] == user_id)
        assert matched["storage_bytes"] is None

        token = jwt.encode(
            {"sub": user_id, "kind": "forget_me", "email": email,
             "exp": datetime.now(UTC) + timedelta(minutes=5)},
            admin_routes._secret(), algorithm=admin_routes.ALGORITHM,
        )
        confirm_resp = await admin_routes.forget_me_confirm(token=token)
        assert confirm_resp.status_code == 200

        # The user is really gone now.
        gone = await admin_routes.identity_manager.lookup_user("web_email", email)
        return gone

    try:
        gone = run_async(_run())
        assert gone is None
    finally:
        settings.admin_token = orig_admin_token
        admin_routes.identity_manager = orig_identity_manager
