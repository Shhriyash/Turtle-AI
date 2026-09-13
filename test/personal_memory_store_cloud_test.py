"""
test/personal_memory_store_cloud_test.py
--------------------------------------------
Unit coverage for core/storage/cloud/personal_memory_store.py and
PersonalMemoryStore's cloud-mode selection (post-migration audit fix): the
topic markdown files core/personal_memory_prompt.py renders into every chat
turn's prompt were the one thing left entirely on local disk with zero
is_cloud awareness. Uses a lightweight fake psycopg pool — no live Postgres
reachable in this environment (same caveat as the rest of this migration's
unit tests).
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from core.personal_memory_store import PersonalMemoryStore, _LocalPersonalMemoryBackend
from core.storage.cloud.personal_memory_store import PostgresPersonalMemoryBackend


class _FakeCursor:
    def __init__(self, row, rowcount=0):
        self._row = row
        self.rowcount = rowcount

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, topics: dict, logs: dict):
        self.topics = topics  # (user_id, topic_name) -> content
        self.logs = logs      # (user_id, log_date) -> content

    def execute(self, sql: str, params=None):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("SELECT content FROM personal_memory_topics"):
            content = self.topics.get((params[0], params[1]))
            return _FakeCursor((content,) if content is not None else None)
        if sql_norm.startswith("INSERT INTO personal_memory_topics"):
            user_id, topic_name, content = params
            self.topics[(user_id, topic_name)] = content
            return _FakeCursor(None)
        if sql_norm.startswith("DELETE FROM personal_memory_topics"):
            existed = (params[0], params[1]) in self.topics
            self.topics.pop((params[0], params[1]), None)
            return _FakeCursor(None, rowcount=1 if existed else 0)
        if sql_norm.startswith("SELECT content FROM personal_memory_daily_logs"):
            content = self.logs.get((params[0], params[1]))
            return _FakeCursor((content,) if content is not None else None)
        if sql_norm.startswith("INSERT INTO personal_memory_daily_logs"):
            user_id, log_date, content = params
            self.logs[(user_id, log_date)] = content
            return _FakeCursor(None)
        if sql_norm.startswith("SELECT COALESCE(SUM(LENGTH(content)), 0) FROM personal_memory_topics"):
            total = sum(len(v) for (uid, _), v in self.topics.items() if uid == params[0])
            return _FakeCursor((total,))
        if sql_norm.startswith("SELECT COALESCE(SUM(LENGTH(content)), 0) FROM personal_memory_daily_logs"):
            total = sum(len(v) for (uid, _), v in self.logs.items() if uid == params[0])
            return _FakeCursor((total,))
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
        self.topics: dict = {}
        self.logs: dict = {}

    def connection(self):
        return _FakeConnCtx(_FakeConn(self.topics, self.logs))


def _patch_pool():
    import core.storage.cloud.personal_memory_store as pms

    pms._initialized = False
    return patch("core.storage.cloud.personal_memory_store.get_pg_sync_pool")


class PostgresPersonalMemoryBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = _FakePool()
        patcher = _patch_pool()
        self.addCleanup(patcher.stop)
        mock_get = patcher.start()
        mock_get.return_value = self.pool
        self.backend = PostgresPersonalMemoryBackend("usr_a")

    def test_read_missing_topic_returns_none(self) -> None:
        self.assertIsNone(self.backend.read_topic("identity"))

    def test_write_then_read_topic_round_trips(self) -> None:
        self.backend.write_topic("identity", "---\ntopic: identity\n---\n- Name: Alice")
        self.assertEqual(
            self.backend.read_topic("identity"), "---\ntopic: identity\n---\n- Name: Alice"
        )

    def test_index_is_stored_separately_from_topics(self) -> None:
        self.backend.write_index("- [Identity](identity.md) - name")
        self.backend.write_topic("identity", "content")
        self.assertEqual(self.backend.read_index(), "- [Identity](identity.md) - name")
        self.assertEqual(self.backend.read_topic("identity"), "content")

    def test_daily_log_round_trips(self) -> None:
        self.backend.write_daily_log("2026-09-12", "- fact one")
        self.assertEqual(self.backend.read_daily_log("2026-09-12"), "- fact one")

    def test_total_bytes_sums_topics_and_logs(self) -> None:
        self.backend.write_topic("identity", "12345")
        self.backend.write_daily_log("2026-09-12", "67")
        self.assertEqual(self.backend.total_bytes(), 5 + 2)

    def test_scoped_per_user(self) -> None:
        other = PostgresPersonalMemoryBackend("usr_b")
        self.backend.write_topic("identity", "a's content")
        other.write_topic("identity", "b's content")
        self.assertEqual(self.backend.read_topic("identity"), "a's content")
        self.assertEqual(other.read_topic("identity"), "b's content")

    def test_requires_user_id(self) -> None:
        with self.assertRaises(ValueError):
            PostgresPersonalMemoryBackend("")


class PersonalMemoryStoreCloudSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = _FakePool()
        patcher = _patch_pool()
        self.addCleanup(patcher.stop)
        mock_get = patcher.start()
        mock_get.return_value = self.pool

    def test_cloud_mode_without_explicit_paths_uses_postgres_backend(self) -> None:
        with patch("core.personal_memory_store.settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.user_storage_cap_mb = 50
            store = PersonalMemoryStore(user_id="usr_a")
            self.assertNotIsInstance(store._backend, _LocalPersonalMemoryBackend)

    def test_default_tenant_never_uses_cloud_backend(self) -> None:
        # "default"/"" are test/legacy single-tenant constructs — cross-tenant
        # collapse risk if pointed at a shared cloud row, matching write_topic's
        # own embed-job skip rule for the same tenants.
        with patch("core.personal_memory_store.settings") as fake_settings:
            fake_settings.is_cloud = True
            store = PersonalMemoryStore(user_id="default")
            self.assertIsInstance(store._backend, _LocalPersonalMemoryBackend)

    def test_explicit_base_dir_forces_local_even_in_cloud_mode(self, tmp_path_factory=None) -> None:
        import tempfile
        from pathlib import Path

        with patch("core.personal_memory_store.settings") as fake_settings:
            fake_settings.is_cloud = True
            with tempfile.TemporaryDirectory() as tmp:
                store = PersonalMemoryStore(user_id="usr_a", base_dir=Path(tmp))
                self.assertIsInstance(store._backend, _LocalPersonalMemoryBackend)

    def test_write_topic_then_load_topic_round_trips_in_cloud_mode(self) -> None:
        with patch("core.personal_memory_store.settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.user_storage_cap_mb = 0  # disable cap for this test
            store = PersonalMemoryStore(user_id="usr_a")
            store.write_topic("identity", ["- Name: Alice"], {"title": "Identity"})
            doc = store.load_topic("identity")
            self.assertEqual(doc.lines, ["- Name: Alice"])

    def test_load_index_and_save_index_round_trip_in_cloud_mode(self) -> None:
        from core.personal_memory_store import PersonalMemoryIndexEntry

        with patch("core.personal_memory_store.settings") as fake_settings:
            fake_settings.is_cloud = True
            store = PersonalMemoryStore(user_id="usr_a")
            store.save_index([
                PersonalMemoryIndexEntry(title="Identity", file_name="identity.md", summary="who they are")
            ])
            entries = store.load_index()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].file_name, "identity.md")


class ReplayerClearTopicCloudFixTest(unittest.TestCase):
    """Direct before/after demonstration of the memory_replayer.py fix: a
    topic with no remaining live facts must actually disappear in cloud
    mode, not silently stay stuck (the old code called Path.unlink() on a
    path that never existed there)."""

    def setUp(self) -> None:
        self.pool = _FakePool()
        patcher = _patch_pool()
        self.addCleanup(patcher.stop)
        mock_get = patcher.start()
        mock_get.return_value = self.pool

    def test_delete_topic_removes_content_in_cloud_mode(self) -> None:
        with patch("core.personal_memory_store.settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.user_storage_cap_mb = 0
            store = PersonalMemoryStore(user_id="usr_a")
            store.write_topic("identity", ["- Name: Alice"])
            self.assertTrue(store.topic_exists("identity"))

            deleted = store.delete_topic("identity")

            self.assertTrue(deleted)
            self.assertFalse(store.topic_exists("identity"))
            self.assertEqual(store.load_topic("identity").lines, [])

    def test_delete_topic_returns_false_when_nothing_to_delete(self) -> None:
        with patch("core.personal_memory_store.settings") as fake_settings:
            fake_settings.is_cloud = True
            store = PersonalMemoryStore(user_id="usr_a")
            self.assertFalse(store.delete_topic("identity"))


if __name__ == "__main__":
    unittest.main()
