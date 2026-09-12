"""
test/journal_store_cloud_test.py
-----------------------------------
Unit coverage for core/memory_journal.py's cloud-mode selection (Vercel
migration Phase 1d) and core/storage/cloud/journal_store.PostgresJournalBackend.
Uses a lightweight fake psycopg pool — no live Postgres reachable in this
environment (same caveat as the rest of this migration's unit tests).
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from core.memory_journal import JournalStore, make_event
from core.storage.cloud.journal_store import PostgresJournalBackend


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, table: dict):
        self._table = table  # (user_id, event_id) -> (observed_at_dt, payload_json)

    def execute(self, sql: str, params=None):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("INSERT INTO journal_events"):
            user_id, event_id, observed_at, payload_json = params
            key = (user_id, event_id)
            if key not in self._table:  # ON CONFLICT DO NOTHING
                dt = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
                self._table[key] = (dt, payload_json)
            return _FakeCursor([])
        if sql_norm.startswith("SELECT 1 FROM journal_events"):
            user_id, event_id = params
            found = (user_id, event_id) in self._table
            return _FakeCursor([(1,)] if found else [])
        if sql_norm.startswith("SELECT payload FROM journal_events"):
            user_id = params[0]
            rows = [
                (payload,)
                for (uid, _eid), (_dt, payload) in sorted(
                    self._table.items(), key=lambda kv: kv[1][0]
                )
                if uid == user_id
            ]
            return _FakeCursor(rows)
        if sql_norm.startswith("SELECT COALESCE(SUM"):
            user_id = params[0]
            total = sum(len(payload) for (uid, _), (_, payload) in self._table.items() if uid == user_id)
            return _FakeCursor([(total,)])
        if sql_norm.startswith("SELECT MIN(observed_at)"):
            user_id = params[0]
            dts = [dt for (uid, _), (dt, _) in self._table.items() if uid == user_id]
            return _FakeCursor([(min(dts),)] if dts else [(None,)])
        return _FakeCursor([])  # CREATE TABLE / CREATE INDEX


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


def _make_test_event(key="preferences.tone", value=None, event_id=None):
    return make_event(
        kind="preference",
        topic="preferences",
        key=key,
        value=value or {"tone": "casual"},
        confidence=0.9,
        source="explicit",
        extractor="llm_turn",
        session_id="sess1",
        turn_id="t1",
        event_id=event_id,
    )


class PostgresJournalBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = _FakePool()
        patcher = patch(
            "core.storage.cloud.journal_store.get_pg_sync_pool", return_value=self.pool
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.backend = PostgresJournalBackend(user_id="usr_a")

    def test_append_then_iter_round_trips(self) -> None:
        event = _make_test_event()
        self.backend.append_line(event)
        events = list(self.backend.iter_events())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, event.event_id)
        self.assertEqual(events[0].value, {"tone": "casual"})

    def test_event_exists(self) -> None:
        event = _make_test_event()
        self.assertFalse(self.backend.event_exists(event.event_id))
        self.backend.append_line(event)
        self.assertTrue(self.backend.event_exists(event.event_id))

    def test_append_is_idempotent_by_event_id(self) -> None:
        event = _make_test_event(event_id="EVT1")
        self.backend.append_line(event)
        self.backend.append_line(event)  # duplicate append
        events = list(self.backend.iter_events())
        self.assertEqual(len(events), 1)

    def test_events_scoped_per_user(self) -> None:
        other = PostgresJournalBackend(user_id="usr_b")
        self.backend.append_line(_make_test_event(event_id="EVT_A"))
        other.append_line(_make_test_event(event_id="EVT_B"))
        self.assertEqual([e.event_id for e in self.backend.iter_events()], ["EVT_A"])
        self.assertEqual([e.event_id for e in other.iter_events()], ["EVT_B"])

    def test_created_at_timestamp_reflects_earliest_event(self) -> None:
        self.assertIsNone(self.backend.created_at_timestamp())
        self.backend.append_line(_make_test_event(event_id="EVT1"))
        ts = self.backend.created_at_timestamp()
        self.assertIsNotNone(ts)
        self.assertLessEqual(abs(ts - datetime.now(timezone.utc).timestamp()), 5)

    def test_total_bytes_grows_with_appends(self) -> None:
        self.assertEqual(self.backend.total_bytes(), 0)
        self.backend.append_line(_make_test_event(event_id="EVT1"))
        self.assertGreater(self.backend.total_bytes(), 0)


class JournalStoreCloudSelectionTest(unittest.TestCase):
    """JournalStore must select the Postgres backend only when is_cloud AND
    no explicit journal_dir was given (test isolation must survive cloud
    mode unchanged)."""

    def setUp(self) -> None:
        self.pool = _FakePool()
        patcher = patch(
            "core.storage.cloud.journal_store.get_pg_sync_pool", return_value=self.pool
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_cloud_mode_without_explicit_dir_uses_postgres(self) -> None:
        with patch("core.memory_journal.settings") as fake_settings:
            fake_settings.is_cloud = True
            store = JournalStore(user_id="usr_a")
            self.assertIsInstance(store._backend, PostgresJournalBackend)

    def test_explicit_journal_dir_forces_local_even_in_cloud_mode(self, tmp_path_factory=None) -> None:
        import tempfile
        from core.memory_journal import _LocalJournalBackend

        with patch("core.memory_journal.settings") as fake_settings:
            fake_settings.is_cloud = True
            with tempfile.TemporaryDirectory() as tmp:
                from pathlib import Path

                store = JournalStore(user_id="usr_a", journal_dir=Path(tmp))
                self.assertIsInstance(store._backend, _LocalJournalBackend)

    def test_append_and_load_all_work_through_cloud_backend(self) -> None:
        with patch("core.memory_journal.settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.user_storage_cap_mb = 0  # disable cap for this test
            store = JournalStore(user_id="usr_a")
            event = _make_test_event(event_id="EVT1")
            store.append(event)
            loaded = store.load_all()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].event_id, "EVT1")

    def test_storage_cap_enforced_in_cloud_mode(self) -> None:
        from core.guardrails import StorageCapExceededError

        with patch("core.memory_journal.settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.user_storage_cap_mb = 1  # 1 MB cap, easy to exceed
            store = JournalStore(user_id="usr_cap")
            # Pre-fill the fake table with enough bytes to already be over cap.
            store._backend.total_bytes = lambda: 2 * 1024 * 1024  # type: ignore[method-assign]
            with self.assertRaises(StorageCapExceededError):
                store.append(_make_test_event(event_id="EVT_BIG"))


if __name__ == "__main__":
    unittest.main()
