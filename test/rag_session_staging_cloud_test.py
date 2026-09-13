"""
test/rag_session_staging_cloud_test.py
------------------------------------------
Unit coverage for core/storage/cloud/rag_session_staging_store.py and
TurtleRAGSystem's cloud-mode staging selection (post-migration audit fix):
current_session.json was assumed to be same-request scratch space, but
start_session()/add_conversation()/end_session() actually read it back
ACROSS requests to accumulate a conversation buffer before indexing. Uses a
lightweight fake psycopg pool — no live Postgres reachable in this
environment (same caveat as the rest of this migration's unit tests).
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from core.storage.cloud.rag_session_staging_store import PostgresRagSessionStaging


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, table: dict):
        self.table = table

    def execute(self, sql: str, params=None):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("SELECT session_id, creation_time, conversations"):
            row = self.table.get(params[0])
            if row is None:
                return _FakeCursor(None)
            return _FakeCursor((row["session_id"], row["creation_time"], row["conversations"]))
        if sql_norm.startswith("INSERT INTO rag_session_staging"):
            user_id, session_id, creation_time, conversations = params
            self.table[user_id] = {
                "session_id": session_id, "creation_time": creation_time, "conversations": conversations,
            }
            return _FakeCursor(None)
        if sql_norm.startswith("DELETE FROM rag_session_staging"):
            self.table.pop(params[0], None)
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


class PostgresRagSessionStagingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = _FakePool()
        import core.storage.cloud.rag_session_staging_store as rss

        rss._initialized = False
        patcher = patch(
            "core.storage.cloud.rag_session_staging_store.get_pg_sync_pool", return_value=self.pool
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.staging = PostgresRagSessionStaging("usr_a")

    def test_read_missing_returns_none(self) -> None:
        self.assertIsNone(self.staging.read())

    def test_write_then_read_round_trips(self) -> None:
        data = {"session_id": "s1", "creation_time": "2026-09-13T00:00:00", "conversations": [{"user_query": "hi"}]}
        self.staging.write(data)
        self.assertEqual(self.staging.read(), data)

    def test_write_overwrites_previous_session(self) -> None:
        self.staging.write({"session_id": "s1", "creation_time": "t1", "conversations": []})
        self.staging.write({"session_id": "s2", "creation_time": "t2", "conversations": [{"a": 1}]})
        result = self.staging.read()
        self.assertEqual(result["session_id"], "s2")
        self.assertEqual(result["conversations"], [{"a": 1}])

    def test_clear_removes_staging(self) -> None:
        self.staging.write({"session_id": "s1", "creation_time": "t1", "conversations": []})
        self.staging.clear()
        self.assertIsNone(self.staging.read())

    def test_scoped_per_user(self) -> None:
        other = PostgresRagSessionStaging("usr_b")
        self.staging.write({"session_id": "s1", "creation_time": "t1", "conversations": []})
        self.assertIsNone(other.read())

    def test_requires_user_id(self) -> None:
        with self.assertRaises(ValueError):
            PostgresRagSessionStaging("")


class TurtleRAGSystemCloudStagingTest(unittest.IsolatedAsyncioTestCase):
    """Exercises TurtleRAGSystem's actual start_session/add_conversation/
    end_session flow against the Postgres-backed staging — the direct
    demonstration that a session's buffer now survives across separate
    TurtleRAGSystem instantiations (simulating separate serverless
    invocations), which the local-file version could not do."""

    async def asyncSetUp(self) -> None:
        self.pool = _FakePool()
        import core.storage.cloud.rag_session_staging_store as rss

        rss._initialized = False
        patcher = patch(
            "core.storage.cloud.rag_session_staging_store.get_pg_sync_pool", return_value=self.pool
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def _make_system(self, user_id: str):
        from rag.system.complete_rag import TurtleRAGSystem, _CloudSessionStaging

        with patch("rag.system.complete_rag.settings") as fake_settings:
            fake_settings.is_cloud = True
            with patch("rag.system.complete_rag.get_vector_storage", return_value=object()):
                system = TurtleRAGSystem(user_id=user_id)
        self.assertIsInstance(system._staging, _CloudSessionStaging)
        return system

    async def test_conversation_buffer_survives_across_separate_instances(self) -> None:
        # "Invocation A" starts a session and adds one turn.
        system_a = self._make_system("usr_a")
        session_id = await system_a.start_session()
        system_a.add_conversation("hello", "hi there")

        # "Invocation B" — a totally separate TurtleRAGSystem instance,
        # simulating a later turn landing on a different serverless
        # instance. It must recover the SAME session and conversation.
        system_b = self._make_system("usr_a")
        recovered_session_id = await system_b.start_session()

        self.assertEqual(recovered_session_id, session_id)
        self.assertEqual(len(system_b.session_conversations), 1)
        self.assertEqual(system_b.session_conversations[0]["user_query"], "hello")

    async def test_end_session_clears_staging(self) -> None:
        system = self._make_system("usr_a")
        await system.start_session()
        system.add_conversation("hi", "hello")
        with patch.object(system, "_index_session_conversations", return_value=True):
            await system.end_session()

        summary_exists = system._staging.read()
        self.assertIsNone(summary_exists)


if __name__ == "__main__":
    unittest.main()
