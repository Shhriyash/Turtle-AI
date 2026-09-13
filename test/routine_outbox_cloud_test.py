"""
test/routine_outbox_cloud_test.py
------------------------------------
Unit coverage for core/storage/cloud/routine_outbox_store.py (Vercel
migration Phase 2 prerequisite) and core/routine_outbox.py's cloud branch.
Uses a lightweight fake psycopg pool — no live Postgres reachable in this
environment (same caveat as the rest of this migration's unit tests).
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from core.storage.cloud import routine_outbox_store as store


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
        if sql_norm.startswith("SELECT frames FROM routine_outbox"):
            row = self._table.get(params[0])
            return _FakeCursor((row,) if row is not None else None)
        if sql_norm.startswith("INSERT INTO routine_outbox"):
            user_id, frames_json = params
            self._table[user_id] = frames_json
            return _FakeCursor(None)
        if sql_norm.startswith("DELETE FROM routine_outbox"):
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
        self.table: dict = {}

    def connection(self):
        return _FakeConnCtx(_FakeConn(self.table))


class RoutineOutboxStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = _FakePool()
        store._initialized = False
        patcher = patch(
            "core.storage.cloud.routine_outbox_store.get_pg_sync_pool", return_value=self.pool
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_load_missing_returns_empty_list(self) -> None:
        self.assertEqual(store.load_outbox_pg("usr_a"), [])

    def test_save_then_load_round_trips(self) -> None:
        frames = [{"type": "routine", "code": "routine_fire", "message": "hi"}]
        store.save_outbox_pg("usr_a", frames, max_frames=5)
        self.assertEqual(store.load_outbox_pg("usr_a"), frames)

    def test_save_caps_to_max_frames(self) -> None:
        frames = [{"n": i} for i in range(10)]
        store.save_outbox_pg("usr_a", frames, max_frames=3)
        loaded = store.load_outbox_pg("usr_a")
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded, frames[-3:])

    def test_save_empty_deletes_the_row(self) -> None:
        store.save_outbox_pg("usr_a", [{"n": 1}], max_frames=5)
        store.save_outbox_pg("usr_a", [], max_frames=5)
        self.assertEqual(store.load_outbox_pg("usr_a"), [])
        self.assertNotIn("usr_a", self.pool.table)

    def test_outboxes_scoped_per_user(self) -> None:
        store.save_outbox_pg("usr_a", [{"n": 1}], max_frames=5)
        store.save_outbox_pg("usr_b", [{"n": 2}], max_frames=5)
        self.assertEqual(store.load_outbox_pg("usr_a"), [{"n": 1}])
        self.assertEqual(store.load_outbox_pg("usr_b"), [{"n": 2}])


class RoutineOutboxCloudBranchTest(unittest.TestCase):
    """core.routine_outbox.load_outbox/save_outbox must delegate to the
    Postgres store in cloud mode."""

    def test_load_outbox_delegates_in_cloud_mode(self) -> None:
        import core.routine_outbox as outbox

        with patch.object(outbox, "settings") as fake_settings:
            fake_settings.is_cloud = True
            with patch(
                "core.storage.cloud.routine_outbox_store.load_outbox_pg",
                return_value=[{"n": 1}],
            ) as fake_load:
                result = outbox.load_outbox("usr_a")
        self.assertEqual(result, [{"n": 1}])
        fake_load.assert_called_once_with("usr_a")

    def test_save_outbox_delegates_in_cloud_mode(self) -> None:
        import core.routine_outbox as outbox

        with patch.object(outbox, "settings") as fake_settings:
            fake_settings.is_cloud = True
            with patch(
                "core.storage.cloud.routine_outbox_store.save_outbox_pg"
            ) as fake_save:
                outbox.save_outbox("usr_a", [{"n": 1}])
        fake_save.assert_called_once_with("usr_a", [{"n": 1}], max_frames=outbox._MAX_FRAMES)


if __name__ == "__main__":
    unittest.main()
