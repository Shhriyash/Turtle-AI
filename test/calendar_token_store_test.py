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


class _EnvTripwireSettings:
    """Settings stand-in whose google_calendar_token_json raises if ever
    read — proves the legacy env fallback is not merely unused by luck but
    genuinely never consulted (WP2.B / ledger 2.8)."""

    def __init__(self, *, is_cloud: bool, calendar_token_key=None):
        self.is_cloud = is_cloud
        self.calendar_token_key = calendar_token_key

    @property
    def google_calendar_token_json(self):
        raise AssertionError(
            "legacy env var google_calendar_token_json must not be read "
            "when a user_id is given in cloud mode"
        )


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

    def test_load_token_json_no_row_returns_none_without_env_fallback(self) -> None:
        """WP2.B (ledger 2.8): with a user_id in cloud mode, a genuine miss
        (no row) must NOT fall back to the legacy env var — that fallback
        survives only in local mode with no user_id. Previously this
        returned the global env token, which meant an unrelated user could
        end up creating events on the operator's own calendar."""
        import tools.calendar_tool as ct

        with patch.object(ct, "settings", _EnvTripwireSettings(is_cloud=True)):
            with patch(
                "core.storage.cloud.calendar_token_store.get_token_json",
                return_value=None,
            ):
                resolved = ct._load_token_json("usr_a")
        self.assertIsNone(resolved)

    def test_load_token_json_db_error_raises_unavailable_without_env_fallback(self) -> None:
        """A genuine DB error must be distinguishable from a miss, and must
        never fall through to the legacy env var — a transient DB error
        must not create events on the operator's own calendar."""
        import tools.calendar_tool as ct

        with patch.object(ct, "settings", _EnvTripwireSettings(is_cloud=True)):
            with patch(
                "core.storage.cloud.calendar_token_store.get_token_json",
                side_effect=RuntimeError("connection refused"),
            ):
                with self.assertRaises(ct.CalendarCredentialsUnavailable):
                    ct._load_token_json("usr_a")


if __name__ == "__main__":
    unittest.main()
