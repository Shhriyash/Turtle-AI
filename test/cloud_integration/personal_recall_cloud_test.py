"""
Ledger 2.1 acceptance test: personal recall must work end to end in cloud.

In cloud, RetrievalBroker is always constructed with sqlite_index=None (there
is no lexical/SQLite index — see apps/turtle_server.py's
`sqlite_index = None if settings.is_cloud else MemorySQLiteIndex(...)`).
Before the WP2.D fix, `_build_personal_tier` returned "" the moment it saw
`sqlite_index is None`, so `recall(scope="personal")` was a silent no-op on
every cloud turn regardless of what was in vector_docs. This test seeds a
fact directly into a real Postgres `vector_docs` table via PgVectorStore and
then drives the broker's own `recall()` the same way a cloud turn does,
against real SQL rather than a fake.

PgVectorStore normally embeds via the real Cohere API; that network call is
not what this test is proving, so — matching pgvector_store_test.py's own
pattern — the embedder is swapped for a deterministic offline fake after
construction. Every INSERT/SELECT statement the broker's recall path
triggers is still PgVectorStore's real, unmodified SQL.
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from test.cloud_integration.conftest import run_async

pytestmark = pytest.mark.cloud


class _FakeEmbedder:
    """Deterministic, offline stand-in for CohereEmbedding — same shape as
    pgvector_store_test.py's fake so identical text embeds identically,
    giving a query-equals-stored-text hit a cosine similarity of ~1.0."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def embed_for_storage(self, texts):
        return np.array([self._vector_for(t) for t in texts], dtype=np.float32)

    def embed_for_query(self, query: str):
        return self._vector_for(query)

    def _vector_for(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.random(self._dim, dtype=np.float32)


def test_personal_recall_works_in_cloud_with_no_sqlite_index() -> None:
    """Seed a fact into vector_docs, then call RetrievalBroker.recall the way
    a real cloud turn does (sqlite_index=None) and assert it comes back."""
    from core.retrieval_broker import RetrievalBroker
    from core.storage.cloud import get_pg_sync_pool
    from core.storage.cloud.pgvector_store import PgVectorStore, _VECTOR_DIM

    vector_store = PgVectorStore()
    vector_store._embedder = _FakeEmbedder(_VECTOR_DIM)

    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    fact_text = "My best friend is Aarav"

    async def _seed():
        await vector_store.upsert(user_id, doc_id, fact_text, {"topic": "relations"})

    run_async(_seed())

    try:
        broker = RetrievalBroker(
            store=None,  # not touched by recall(scope="personal")
            task_store=None,
            sqlite_index=None,  # the cloud posture: no lexical index at all
            vector_store=vector_store,
            user_id=user_id,
        )

        async def _recall():
            return await broker.recall(query=fact_text, scope="personal")

        result = run_async(_recall())
    finally:
        pool = get_pg_sync_pool()
        with pool.connection() as conn:
            conn.execute("DELETE FROM vector_docs WHERE user_id = %s", (user_id,))

    assert "Aarav" in result
