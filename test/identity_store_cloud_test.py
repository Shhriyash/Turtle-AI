"""
test/identity_store_cloud_test.py
------------------------------------
Unit coverage for core/storage/cloud/identity_store.py (Vercel migration
Phase 1e): PostgresIdentityManager (async/asyncpg, drop-in for
core.identity.IdentityManager) and write_account_marker_pg (sync/psycopg,
the cloud body of core.identity.write_account_marker). Uses lightweight fake
pools — no live Postgres reachable in this environment (same caveat as the
rest of this migration's unit tests).
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from core.storage.cloud.identity_store import (
    PostgresIdentityManager,
    write_account_marker_pg,
)


# --- Fake asyncpg pool -------------------------------------------------------

class _FakeAsyncConn:
    def __init__(self, db: dict):
        self._db = db  # {"users": {}, "channel_mappings": {}, "claimed_tokens": {}, "account_markers": {}}

    async def execute(self, sql: str, *args):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("CREATE TABLE") or sql_norm.startswith("CREATE INDEX"):
            return
        if sql_norm.startswith("INSERT INTO claimed_tokens"):
            jti, user_id = args
            if jti in self._db["claimed_tokens"]:
                import asyncpg

                raise asyncpg.UniqueViolationError("duplicate")
            self._db["claimed_tokens"][jti] = user_id
            return
        if sql_norm.startswith("INSERT INTO users (user_id) VALUES ($1) ON CONFLICT"):
            user_id = args[0]
            self._db["users"].setdefault(user_id, {"primary_email": None})
            return
        if sql_norm.startswith("INSERT INTO users (user_id) VALUES ($1)"):
            user_id = args[0]
            self._db["users"][user_id] = {"primary_email": None}
            return
        if sql_norm.startswith("UPDATE users SET primary_email = $1 WHERE user_id = $2 AND"):
            email, user_id = args
            row = self._db["users"].get(user_id)
            if row is not None and row["primary_email"] is None:
                row["primary_email"] = email
            return
        if sql_norm.startswith("UPDATE users SET primary_email"):
            email, user_id = args
            row = self._db["users"].get(user_id)
            if row is not None:
                row["primary_email"] = email
            return
        if sql_norm.startswith("INSERT INTO channel_mappings"):
            channel, channel_user_id, user_id = args
            self._db["channel_mappings"][(channel, channel_user_id)] = user_id
            return

    async def fetchrow(self, sql: str, *args):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("SELECT user_id FROM channel_mappings"):
            channel, channel_user_id = args
            user_id = self._db["channel_mappings"].get((channel, channel_user_id))
            return {"user_id": user_id} if user_id else None
        if sql_norm.startswith("SELECT user_id, email_verified FROM account_markers"):
            email = args[0]
            for uid, marker in self._db["account_markers"].items():
                if marker["email"] == email:
                    return {"user_id": uid, "email_verified": marker["email_verified"]}
            return None
        return None

    def transaction(self):
        return _NullAsyncCtx()


class _NullAsyncCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeAcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakeAsyncPool:
    def __init__(self):
        self._db = {
            "users": {}, "channel_mappings": {}, "claimed_tokens": {}, "account_markers": {},
        }

    def acquire(self):
        return _FakeAcquireCtx(_FakeAsyncConn(self._db))

    def seed_marker(self, user_id: str, email: str, verified: bool) -> None:
        self._db["account_markers"][user_id] = {"email": email, "email_verified": verified}


class PostgresIdentityManagerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.pool = _FakeAsyncPool()
        patcher = patch(
            "core.storage.cloud.identity_store.get_pg_pool",
            new_callable=AsyncMock,
            return_value=self.pool,
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.manager = PostgresIdentityManager()

    async def test_resolve_user_mints_new_id_for_unknown_channel_handle(self) -> None:
        user_id = await self.manager.resolve_user("discord", "12345")
        self.assertTrue(user_id.startswith("usr_"))

    async def test_resolve_user_returns_same_id_on_repeat_lookup(self) -> None:
        first = await self.manager.resolve_user("discord", "12345")
        second = await self.manager.resolve_user("discord", "12345")
        self.assertEqual(first, second)

    async def test_resolve_user_normalizes_email_channel(self) -> None:
        first = await self.manager.resolve_user("web_email", "  Alice@Example.com  ")
        second = await self.manager.resolve_user("web_email", "alice@example.com")
        self.assertEqual(first, second)

    async def test_resolve_user_populates_primary_email(self) -> None:
        user_id = await self.manager.resolve_user("web_email", "bob@example.com")
        self.assertEqual(self.pool._db["users"][user_id]["primary_email"], "bob@example.com")

    async def test_different_channels_get_different_ids_for_same_handle_string(self) -> None:
        a = await self.manager.resolve_user("discord", "same_handle")
        b = await self.manager.resolve_user("telegram", "same_handle")
        self.assertNotEqual(a, b)

    async def test_mark_token_claimed_first_time_true_second_time_false(self) -> None:
        first = await self.manager.mark_token_claimed("jti1", "usr_a")
        second = await self.manager.mark_token_claimed("jti1", "usr_a")
        self.assertTrue(first)
        self.assertFalse(second)

    async def test_link_channel_repoints_and_returns_previous(self) -> None:
        await self.manager.resolve_user("discord", "handle1")  # binds to some usr_x
        previous = await self.manager.link_channel(
            user_id="usr_target", channel="discord", channel_user_id="handle1"
        )
        self.assertIsNotNone(previous)
        resolved = await self.manager.resolve_user("discord", "handle1")
        self.assertEqual(resolved, "usr_target")

    async def test_rebind_from_verified_marker(self) -> None:
        self.pool.seed_marker("usr_original", "carol@example.com", verified=True)
        resolved = await self.manager.resolve_user("web_email", "carol@example.com")
        self.assertEqual(resolved, "usr_original")

    async def test_unverified_marker_does_not_rebind(self) -> None:
        self.pool.seed_marker("usr_original", "dave@example.com", verified=False)
        resolved = await self.manager.resolve_user("web_email", "dave@example.com")
        self.assertNotEqual(resolved, "usr_original")
        self.assertTrue(resolved.startswith("usr_"))


# --- Fake psycopg (sync) pool for write_account_marker_pg -------------------

class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeSyncConn:
    def __init__(self, table: dict):
        self._table = table

    def execute(self, sql: str, params=None):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("SELECT created_at FROM account_markers"):
            row = self._table.get(params[0])
            return _FakeCursor((row["created_at"],) if row else None)
        if sql_norm.startswith("INSERT INTO account_markers"):
            user_id, email, verified, created_at, channel = params
            self._table[user_id] = {
                "email": email, "email_verified": verified,
                "created_at": created_at, "channel": channel,
            }
            return _FakeCursor(None)
        return _FakeCursor(None)  # CREATE TABLE / CREATE INDEX


class _FakeSyncConnCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


class _FakeSyncPool:
    def __init__(self):
        self.table: dict = {}

    def connection(self):
        return _FakeSyncConnCtx(_FakeSyncConn(self.table))


class WriteAccountMarkerPgTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = _FakeSyncPool()
        import core.storage.cloud.identity_store as identity_store

        identity_store._markers_initialized = False
        patcher = patch(
            "core.storage.cloud.identity_store.get_pg_sync_pool", return_value=self.pool
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_write_then_read_back(self) -> None:
        write_account_marker_pg("usr_a", "eve@example.com", True, channel="web_email")
        row = self.pool.table["usr_a"]
        self.assertEqual(row["email"], "eve@example.com")
        self.assertTrue(row["email_verified"])

    def test_created_at_preserved_across_rewrites(self) -> None:
        write_account_marker_pg("usr_a", "eve@example.com", False, channel="web_email")
        first_created = self.pool.table["usr_a"]["created_at"]
        write_account_marker_pg("usr_a", "eve@example.com", True, channel="web_email")
        second_created = self.pool.table["usr_a"]["created_at"]
        self.assertEqual(first_created, second_created)
        self.assertTrue(self.pool.table["usr_a"]["email_verified"])


if __name__ == "__main__":
    unittest.main()
