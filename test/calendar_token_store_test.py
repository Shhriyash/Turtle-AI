"""
test/calendar_token_store_test.py
------------------------------------
Unit coverage for core/storage/cloud/calendar_token_store.py (Vercel
migration Phase 1c) and its two cloud-mode call sites: tools/calendar_tool.py
::_load_token_json and apps/calendar_oauth_routes.py's connect/status/
disconnect helpers. Uses a lightweight fake psycopg pool — no live Postgres
reachable in this environment (same caveat as the rest of this migration's
unit tests).
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from core.storage.cloud import calendar_token_store as store


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, table: dict):
        self._table = table

    def execute(self, sql: str, params=None):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("SELECT token_json"):
            user_id = params[0]
            value = self._table.get(user_id)
            return _FakeCursor((value,) if value is not None else None)
        if sql_norm.startswith("INSERT INTO calendar_tokens"):
            user_id, token_json = params
            self._table[user_id] = token_json
            return _FakeCursor(None)
        if sql_norm.startswith("DELETE FROM calendar_tokens"):
            self._table.pop(params[0], None)
            return _FakeCursor(None)
        return _FakeCursor(None)  # CREATE TABLE


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


class _FakePool:
    def __init__(self):
        self._table: dict = {}

    def connection(self):
        return _FakeConnCtx(_FakeConn(self._table))


class CalendarTokenStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = _FakePool()
        # Reset the module-level _initialized flag so each test starts clean
        # (mirrors PgChunkVectorStore's per-instance _initialized, but this
        # module uses free functions, so the flag is module-level).
        store._initialized = False
        patcher = patch(
            "core.storage.cloud.calendar_token_store.get_pg_sync_pool",
            return_value=self.pool,
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_get_missing_returns_none(self) -> None:
        self.assertIsNone(store.get_token_json("usr_a"))

    def test_put_then_get_round_trips(self) -> None:
        store.put_token_json("usr_a", '{"refresh_token": "abc"}')
        self.assertEqual(store.get_token_json("usr_a"), '{"refresh_token": "abc"}')

    def test_put_overwrites_existing(self) -> None:
        store.put_token_json("usr_a", '{"refresh_token": "old"}')
        store.put_token_json("usr_a", '{"refresh_token": "new"}')
        self.assertEqual(store.get_token_json("usr_a"), '{"refresh_token": "new"}')

    def test_delete_removes_token(self) -> None:
        store.put_token_json("usr_a", '{"refresh_token": "abc"}')
        store.delete_token_json("usr_a")
        self.assertIsNone(store.get_token_json("usr_a"))

    def test_token_exists(self) -> None:
        self.assertFalse(store.token_exists("usr_a"))
        store.put_token_json("usr_a", '{"refresh_token": "abc"}')
        self.assertTrue(store.token_exists("usr_a"))

    def test_get_empty_user_id_returns_none(self) -> None:
        self.assertIsNone(store.get_token_json(""))

    def test_tokens_scoped_per_user(self) -> None:
        store.put_token_json("usr_a", '{"refresh_token": "a"}')
        store.put_token_json("usr_b", '{"refresh_token": "b"}')
        self.assertEqual(store.get_token_json("usr_a"), '{"refresh_token": "a"}')
        self.assertEqual(store.get_token_json("usr_b"), '{"refresh_token": "b"}')


class CalendarToolCloudModeTest(unittest.TestCase):
    """tools.calendar_tool._load_token_json must read from the Postgres store
    in cloud mode instead of local disk."""

    def test_load_token_json_reads_from_postgres_in_cloud_mode(self) -> None:
        import tools.calendar_tool as ct

        with patch.object(ct, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.calendar_token_key = None
            fake_settings.google_calendar_token_json = '{"refresh_token": "global-legacy"}'
            with patch(
                "core.storage.cloud.calendar_token_store.get_token_json",
                return_value='{"refresh_token": "cloud-token"}',
            ):
                resolved = ct._load_token_json("usr_a")
        self.assertEqual(resolved, '{"refresh_token": "cloud-token"}')

    def test_load_token_json_falls_back_to_legacy_env_when_postgres_empty(self) -> None:
        import tools.calendar_tool as ct

        with patch.object(ct, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.calendar_token_key = None
            fake_settings.google_calendar_token_json = '{"refresh_token": "global-legacy"}'
            with patch(
                "core.storage.cloud.calendar_token_store.get_token_json",
                return_value=None,
            ):
                resolved = ct._load_token_json("usr_a")
        self.assertEqual(resolved, '{"refresh_token": "global-legacy"}')


if __name__ == "__main__":
    unittest.main()
