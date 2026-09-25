"""
DDL-and-roundtrip test for core/storage/cloud/pgvector_store.py — both
classes it exports (PgVectorStore -> vector_docs, PgChunkVectorStore ->
vector_chunks) against a real Postgres with the pgvector extension.

PgVectorStore normally embeds text via the real Cohere API
(rag/embedder/embedding_model.py); that network call is not what this WP is
proving (it proves the store's own SQL executes against a real `vector`
column), so the embedder is swapped for a deterministic fake after
construction — every INSERT/UPDATE/SELECT statement below is still the
store's real, unmodified SQL.
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from test.cloud_integration.conftest import run_async

pytestmark = pytest.mark.cloud


class _FakeEmbedder:
    """Deterministic, offline stand-in for CohereEmbedding — same two methods
    PgVectorStore calls, no network."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def embed_for_storage(self, texts):
        return np.array([self._vector_for(t) for t in texts], dtype=np.float32)

    def embed_for_query(self, query: str):
        return self._vector_for(query)

    def _vector_for(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.random(self._dim, dtype=np.float32)


def test_pgvector_store_module_roundtrip() -> None:
    """Single DDL-and-roundtrip test for this module: exercises BOTH classes
    it defines (and therefore both tables, vector_docs and vector_chunks)."""
    from core.storage.cloud.pgvector_store import (
        PgChunkVectorStore,
        PgVectorStore,
        _VECTOR_DIM,
    )
    from core.storage.cloud import get_pg_sync_pool

    # --- PgVectorStore: vector_docs, upsert/search ---------------------
    store = PgVectorStore()
    store._embedder = _FakeEmbedder(_VECTOR_DIM)

    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"

    async def _run():
        await store.upsert(user_id, doc_id, "the quick brown fox", {"topic": "identity"})
        return await store.search(user_id, "the quick brown fox", k=5)

    hits = run_async(_run())
    assert len(hits) == 1
    assert hits[0].doc_id == doc_id
    assert hits[0].text == "the quick brown fox"
    assert hits[0].metadata == {"topic": "identity"}

    # --- PgChunkVectorStore: vector_chunks, add_chunks/search_similar/stats
    chunk_user_id = f"usr_{uuid.uuid4().hex[:12]}"
    chunk_store = PgChunkVectorStore(chunk_user_id)

    vec = np.random.default_rng(42).random((1, _VECTOR_DIM), dtype=np.float32)
    chunks = [{"chunk_id": "c1", "session_id": "sess_1", "content": "hello world"}]
    try:
        chunk_store.add_chunks(chunks, vec)

        # KNOWN PRE-EXISTING BUG (found by this test against a REAL Postgres —
        # exactly what S-9.5 says the SQL-prefix fakes in
        # test/cloud_storage_test.py cannot catch): PgChunkVectorStore's own
        # _normalize() hands search_similar's `embedding <=> %s` parameter a
        # plain Python list. pgvector.psycopg's register_vector() only
        # registers a Dumper for pgvector.Vector/numpy.ndarray (see
        # venv/Lib/site-packages/pgvector/psycopg/vector.py), not for a bare
        # list, so psycopg falls back to its built-in list dumper and sends a
        # `double precision[]` array literal. Postgres has no
        # `vector <=> double precision[]` operator overload, so the SELECT
        # raises UndefinedFunction. add_chunks' INSERT does NOT hit this: the
        # target column's type lets Postgres apply pgvector's own assignment
        # CAST (double precision[] -> vector), so the row above is written
        # successfully — it just can never be read back via search_similar.
        # This is a real defect in core/storage/cloud/pgvector_store.py,
        # which WP0.C does not own and must not edit; flagged as an open
        # question in the WP0.C report for the store's owner to fix (e.g. by
        # passing pgvector.Vector(emb) instead of emb.tolist()).
        results = chunk_store.search_similar(vec[0], top_k=5, threshold=0.0)
    except Exception as exc:  # noqa: BLE001 — see comment above
        if "vector" not in str(exc) or "<=>" not in str(exc):
            raise
        pytest.xfail(
            "core/storage/cloud/pgvector_store.py: PgChunkVectorStore.search_similar "
            f"cannot compare against a plain Python list embedding ({exc!r}); "
            "out of scope for WP0.C, see the WP0.C report's open questions."
        )
    else:
        assert len(results) == 1
        assert results[0]["content"] == "hello world"
        assert results[0]["session_id"] == "sess_1"
        assert results[0]["chunk_id"] == "c1"

        stats = chunk_store.get_storage_stats()
        assert stats["active_chunks"] >= 1
        assert stats["embedding_dimension"] == _VECTOR_DIM
    finally:
        pool = get_pg_sync_pool()
        with pool.connection() as conn:
            conn.execute("DELETE FROM vector_docs WHERE user_id = %s", (user_id,))
            conn.execute("DELETE FROM vector_chunks WHERE user_id = %s", (chunk_user_id,))
