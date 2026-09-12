"""
test/cloud_storage_test.py
---------------------------
Unit coverage for the cloud (TURTLE_DEPLOY=cloud) storage backends added in
the Vercel migration's Phase 1 — core/storage/cloud/. These run against
lightweight FAKE asyncpg/psycopg pools (no live Postgres in this environment),
so they verify SQL-independent behavior: JSON round-tripping, cosine-score
math, protocol conformance, and the settings.is_cloud selection switch in
core/storage/factory.py. A real Postgres integration pass (Neon preview,
ISSUE tracked in the plan's Verification section) still needs to run once a
database is reachable — these tests do not replace that.
"""
from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np

from core.storage import Session
from core.storage.cloud.pgvector_store import PgChunkVectorStore, PgVectorStore, _normalize
from core.storage.cloud.postgres_store import PostgresSessionStore


# ---------------------------------------------------------------------------
# Fake asyncpg pool/connection — enough surface for PostgresSessionStore and
# PgVectorStore: acquire() as an async context manager, execute/fetch/
# fetchrow, and a transaction() context manager. Backed by an in-memory dict
# so get/put/list/delete round-trip exactly like a real table would.
# ---------------------------------------------------------------------------
class _FakeAsyncpgConn:
    def __init__(self, sessions: dict, vector_docs: list):
        self._sessions = sessions
        self._vector_docs = vector_docs

    async def execute(self, sql: str, *args) -> None:
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("INSERT INTO sessions"):
            session_id, data_json, user_id = args
            self._sessions[session_id] = {
                "data": data_json, "user_id": user_id,
            }
        elif sql_norm.startswith("DELETE FROM sessions"):
            self._sessions.pop(args[0], None)
        elif sql_norm.startswith("UPDATE vector_docs SET deleted"):
            user_id, doc_id = args
            for row in self._vector_docs:
                if row["user_id"] == user_id and row["doc_id"] == doc_id and not row["deleted"]:
                    row["deleted"] = True
        elif sql_norm.startswith("INSERT INTO vector_docs"):
            user_id, doc_id, text, metadata_json, emb = args
            self._vector_docs.append({
                "user_id": user_id, "doc_id": doc_id, "text": text,
                "metadata": metadata_json, "embedding": emb, "deleted": False,
            })
        # CREATE TABLE / CREATE INDEX: no-op against the fake.

    async def fetchrow(self, sql: str, *args):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("SELECT data FROM sessions"):
            row = self._sessions.get(args[0])
            return {"data": row["data"]} if row else None
        return None

    async def fetch(self, sql: str, *args):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("SELECT session_id, data FROM sessions"):
            if "WHERE user_id" in sql_norm:
                user_id = args[0]
                return [
                    {"session_id": sid, "data": row["data"]}
                    for sid, row in self._sessions.items()
                    if row["user_id"] == user_id
                ]
            return [{"session_id": sid, "data": row["data"]} for sid, row in self._sessions.items()]
        if sql_norm.startswith("SELECT doc_id, text, metadata"):
            emb, user_id, k = args
            candidates = [r for r in self._vector_docs if r["user_id"] == user_id and not r["deleted"]]
            query = np.array(emb)
            scored = []
            for row in candidates:
                stored = np.array(row["embedding"])
                cosine_sim = float(np.dot(query, stored))
                scored.append((cosine_sim, row))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            return [
                {"doc_id": r["doc_id"], "text": r["text"], "metadata": r["metadata"], "score": score}
                for score, r in scored[:k]
            ]
        return []

    def transaction(self):
        return _NullAsyncCtx()


class _NullAsyncCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeAcquireCtx:
    def __init__(self, conn: _FakeAsyncpgConn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePgPool:
    def __init__(self):
        self._sessions: dict = {}
        self._vector_docs: list = []

    def acquire(self):
        return _FakeAcquireCtx(_FakeAsyncpgConn(self._sessions, self._vector_docs))


class PostgresSessionStoreTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.pool = _FakePgPool()
        patcher = patch(
            "core.storage.cloud.postgres_store.get_pg_pool",
            new_callable=AsyncMock,
            return_value=self.pool,
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.store = PostgresSessionStore()

    async def test_put_then_get_round_trips(self) -> None:
        await self.store.init_db()
        session = Session(session_id="s1", data={"user_id": "usr_a", "status": "active", "n": 1})
        await self.store.put(session)
        fetched = await self.store.get("s1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.data["user_id"], "usr_a")
        self.assertEqual(fetched.data["n"], 1)

    async def test_get_missing_returns_none(self) -> None:
        result = await self.store.get("does-not-exist")
        self.assertIsNone(result)

    async def test_list_sessions_scopes_by_user_id(self) -> None:
        await self.store.put(Session(session_id="a1", data={"user_id": "usr_a", "status": "active"}))
        await self.store.put(Session(session_id="a2", data={"user_id": "usr_b", "status": "active"}))
        only_a = await self.store.list_sessions(user_id="usr_a")
        self.assertEqual([s.session_id for s in only_a], ["a1"])

    async def test_list_sessions_applies_status_filter(self) -> None:
        await self.store.put(Session(session_id="p1", data={"user_id": "usr_a", "status": "active"}))
        await self.store.put(Session(session_id="p2", data={"user_id": "usr_a", "status": "completed"}))
        active_only = await self.store.list_sessions(status_filter="active", user_id="usr_a")
        self.assertEqual([s.session_id for s in active_only], ["p1"])

    async def test_delete_removes_session(self) -> None:
        await self.store.put(Session(session_id="d1", data={"user_id": "usr_a"}))
        await self.store.delete("d1")
        self.assertIsNone(await self.store.get("d1"))

    async def test_put_overwrites_existing_session_id(self) -> None:
        await self.store.put(Session(session_id="s1", data={"user_id": "usr_a", "n": 1}))
        await self.store.put(Session(session_id="s1", data={"user_id": "usr_a", "n": 2}))
        fetched = await self.store.get("s1")
        self.assertEqual(fetched.data["n"], 2)


class NormalizeTest(unittest.TestCase):
    def test_normalize_produces_unit_vector(self) -> None:
        vec = _normalize(np.array([3.0, 4.0]))
        self.assertAlmostEqual(vec[0], 0.6, places=5)
        self.assertAlmostEqual(vec[1], 0.8, places=5)
        norm = (vec[0] ** 2 + vec[1] ** 2) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_normalize_handles_zero_vector_without_dividing_by_zero(self) -> None:
        vec = _normalize(np.array([0.0, 0.0]))
        self.assertEqual(vec, [0.0, 0.0])


class PgVectorStoreTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.pool = _FakePgPool()

        class _FakeEmbedder:
            def embed_for_storage(self, texts):
                # Deterministic fake embedding: derive a 1024-d vector from
                # text length + a fixed seed component, so different texts get
                # measurably different (but reproducible) vectors.
                return [self._vec(t) for t in texts]

            def embed_for_query(self, text):
                return self._vec(text)

            @staticmethod
            def _vec(text: str) -> np.ndarray:
                base = np.zeros(1024, dtype=np.float32)
                base[0] = float(len(text))
                base[1] = 1.0
                return base

        patch(
            "core.storage.cloud.pgvector_store.get_pg_pool",
            new_callable=AsyncMock,
            return_value=self.pool,
        ).start()
        self.addCleanup(patch.stopall)
        with patch("rag.embedder.embedding_model.get_embedding_model", return_value=_FakeEmbedder()):
            self.store = PgVectorStore()

    async def test_upsert_then_search_finds_the_document(self) -> None:
        await self.store.upsert("usr_a", "doc1", "hello world", {"topic": "greeting"})
        hits = await self.store.search("usr_a", "hello world", k=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].doc_id, "doc1")
        self.assertEqual(hits[0].metadata["topic"], "greeting")

    async def test_search_scoped_to_user(self) -> None:
        await self.store.upsert("usr_a", "doc1", "hello world", {})
        hits = await self.store.search("usr_b", "hello world", k=5)
        self.assertEqual(hits, [])

    async def test_upsert_same_doc_id_replaces_not_duplicates(self) -> None:
        await self.store.upsert("usr_a", "doc1", "version one", {})
        await self.store.upsert("usr_a", "doc1", "version two", {})
        hits = await self.store.search("usr_a", "version two", k=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].text, "version two")


# ---------------------------------------------------------------------------
# Fake psycopg (sync) pool for PgChunkVectorStore.
# ---------------------------------------------------------------------------
class _FakeCursor:
    def __init__(self, rows: list, columns: list):
        self._rows = rows
        self.description = [type("Col", (), {"name": c}) for c in columns]

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def executemany(self, sql, rows):
        pass


class _FakePsycopgConn:
    def __init__(self, chunks: list):
        self._chunks = chunks

    def execute(self, sql: str, params=None):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("INSERT INTO vector_chunks"):
            user_id, chunk_id, session_id, content, extra_json, emb = params
            self._chunks.append({
                "user_id": user_id, "chunk_id": chunk_id, "session_id": session_id,
                "content": content, "extra": extra_json, "embedding": emb, "deleted": False,
            })
            return _FakeCursor([], [])
        if sql_norm.startswith("SELECT chunk_id, session_id, content, extra"):
            emb, user_id, _emb2, k = params
            query = np.array(emb)
            candidates = [c for c in self._chunks if c["user_id"] == user_id and not c["deleted"]]
            scored = sorted(
                candidates, key=lambda c: float(np.dot(query, np.array(c["embedding"]))), reverse=True
            )[:k]
            rows = [
                (c["chunk_id"], c["session_id"], c["content"], c["extra"],
                 float(np.dot(query, np.array(c["embedding"]))))
                for c in scored
            ]
            return _FakeCursor(rows, ["chunk_id", "session_id", "content", "extra", "score"])
        if sql_norm.startswith("SELECT COUNT(*)"):
            user_id = params[0]
            active = [c for c in self._chunks if c["user_id"] == user_id and not c["deleted"]]
            row = (len(active), 0, len({c["session_id"] for c in active}), len(active))
            return _FakeCursor(
                [row], ["active_chunks", "deleted_chunks", "total_sessions", "total_vectors"]
            )
        return _FakeCursor([], [])

    def cursor(self):
        return self

    def executemany(self, sql, rows):
        for row in rows:
            self.execute(sql, row)


class _FakePsycopgConnCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


class _FakePsycopgPool:
    def __init__(self):
        self._chunks: list = []

    def connection(self):
        return _FakePsycopgConnCtx(_FakePsycopgConn(self._chunks))


class PgChunkVectorStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = _FakePsycopgPool()
        patcher = patch(
            "core.storage.cloud.pgvector_store.get_pg_sync_pool", return_value=self.pool
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.store = PgChunkVectorStore(user_id="usr_a")

    def test_add_chunks_then_search_similar_round_trips(self) -> None:
        chunks = [{"session_id": "sess1", "content": "hi there", "chunk_id": "c1"}]
        embeddings = np.zeros((1, 1024), dtype=np.float32)
        embeddings[0, 0] = 1.0
        self.store.add_chunks(chunks, embeddings)

        query = np.zeros(1024, dtype=np.float32)
        query[0] = 1.0
        results = self.store.search_similar(query, top_k=5, threshold=0.5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "hi there")
        self.assertEqual(results[0]["session_id"], "sess1")
        self.assertGreaterEqual(results[0]["similarity_score"], 0.5)

    def test_search_similar_filters_below_threshold(self) -> None:
        chunks = [{"session_id": "sess1", "content": "unrelated"}]
        embeddings = np.ones((1, 1024), dtype=np.float32)
        self.store.add_chunks(chunks, embeddings)

        # Orthogonal query -> cosine similarity ~0, below any reasonable threshold.
        query = np.zeros(1024, dtype=np.float32)
        query[500] = 1.0
        results = self.store.search_similar(query, top_k=5, threshold=0.9)
        self.assertEqual(results, [])

    def test_add_chunks_rejects_mismatched_counts(self) -> None:
        with self.assertRaises(ValueError):
            self.store.add_chunks([{"session_id": "s", "content": "x"}], np.zeros((2, 1024)))

    def test_get_storage_stats_reflects_added_chunks(self) -> None:
        chunks = [
            {"session_id": "sess1", "content": "a"},
            {"session_id": "sess2", "content": "b"},
        ]
        embeddings = np.zeros((2, 1024), dtype=np.float32)
        self.store.add_chunks(chunks, embeddings)
        stats = self.store.get_storage_stats()
        self.assertEqual(stats["active_chunks"], 2)
        self.assertEqual(stats["total_sessions"], 2)
        self.assertEqual(stats["index_type"], "pgvector")

    def test_requires_user_id(self) -> None:
        with self.assertRaises(ValueError):
            PgChunkVectorStore(user_id="")


class StorageFactorySelectionTest(unittest.TestCase):
    """core/storage/factory.py must pick local vs. cloud purely off
    settings.is_cloud, and never import the cloud modules (or their optional
    deps) on the local path."""

    def test_local_mode_returns_sqlite_backend(self) -> None:
        with patch("core.storage.factory.settings") as fake_settings:
            fake_settings.is_cloud = False
            from core.storage.factory import get_session_store_backend
            from core.storage.local.sqlite_store import SQLiteSessionStore

            backend = get_session_store_backend()
            self.assertIsInstance(backend, SQLiteSessionStore)

    def test_cloud_mode_returns_postgres_backend(self) -> None:
        with patch("core.storage.factory.settings") as fake_settings:
            fake_settings.is_cloud = True
            from core.storage.factory import get_session_store_backend

            backend = get_session_store_backend()
            self.assertEqual(type(backend).__name__, "PostgresSessionStore")

    def test_local_mode_returns_faiss_vector_store(self) -> None:
        with patch("core.storage.factory.settings") as fake_settings:
            fake_settings.is_cloud = False
            from core.storage.factory import get_vector_store
            from core.storage.local.faiss_store import FAISSVectorStore

            store = get_vector_store()
            self.assertIsInstance(store, FAISSVectorStore)


class RedisUrlResolutionTest(unittest.TestCase):
    def test_redis_url_prefers_primary_alias(self) -> None:
        from core.config import TurtleSettings

        s = TurtleSettings(
            REDIS_URL="redis://primary", UPSTASH_REDIS_URL="redis://upstash"
        )
        self.assertEqual(s.redis_url, "redis://primary")

    def test_redis_url_falls_back_to_upstash_alias(self) -> None:
        from core.config import TurtleSettings

        s = TurtleSettings(UPSTASH_REDIS_URL="redis://upstash")
        self.assertEqual(s.redis_url, "redis://upstash")

    def test_redis_url_none_when_unset(self) -> None:
        from core.config import TurtleSettings

        s = TurtleSettings()
        self.assertIsNone(s.redis_url)


if __name__ == "__main__":
    unittest.main()
