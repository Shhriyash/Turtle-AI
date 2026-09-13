"""
core/storage/cloud/pgvector_store.py
-------------------------------------
Cloud (TURTLE_DEPLOY=cloud) replacement for BOTH FAISS stores in this repo,
using pgvector on the same Neon Postgres database as everything else (one
store instead of two — see the plan's Phase 1 table). Neither FAISS store
survives a serverless cold start: both persist an index file to local disk
(personal_memory_dir/vector/index.bin, rag/<uid>/vector/faiss_index.bin),
which does not exist on the next invocation.

Two classes here, matching the two DIFFERENT interfaces the FAISS stores
implement — kept separate rather than forced into one shape so each drop-in
replacement is a same-signature swap at its call site, not a refactor:

- PgVectorStore    -> replaces core/storage/local/faiss_store.FAISSVectorStore
                      (the VectorStore Protocol: upsert/search by doc_id).
                      Consumers: get_faiss_vector_store() call sites
                      (personal-memory RAG via RetrievalBroker).
- PgChunkVectorStore -> replaces rag/storage/vector_storage.VectorStorage
                      (the richer conversation-RAG surface: add_chunks,
                      search_similar, get_storage_stats). Consumer:
                      rag/system/complete_rag.py's RAGSystem, via
                      rag.storage.vector_storage.get_vector_storage(user_id).
                      SYNCHRONOUS on purpose (see class docstring): its one
                      call site invokes it directly on the event loop with no
                      await, so it is built on psycopg (sync), not asyncpg.

Both use pgvector's cosine distance operator (<=>) against a normalized
column, matching the FAISS stores' IndexFlatIP-on-normalized-vectors cosine
semantics: score = 1 - cosine_distance, so higher is still "more similar" and
existing threshold/ranking call sites need no sign-flip.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import numpy as np

from core.storage import Hit, VectorStore
from core.storage.cloud import get_pg_pool, get_pg_sync_pool

_VECTOR_DIM = 1024  # Cohere embed-english-v3.0 — matches both FAISS stores' default.


def _normalize(vec: np.ndarray) -> list[float]:
    """L2-normalize a single embedding vector, mirroring both FAISS stores'
    _normalize/_normalize_embeddings so cosine-via-dot-product semantics match."""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm == 0:
        norm = 1.0
    return (arr / norm).tolist()


# --- PgVectorStore: VectorStore Protocol (upsert/search by doc_id) ---------

_CREATE_VECTOR_DOCS_SQL = """
CREATE TABLE IF NOT EXISTS vector_docs (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(%(dim)s) NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
""" % {"dim": _VECTOR_DIM}
_CREATE_VECTOR_DOCS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_vector_docs_user ON vector_docs(user_id) "
    "WHERE NOT deleted"
)


class PgVectorStore(VectorStore):
    """Drop-in for FAISSVectorStore: implements upsert(user_id, doc_id, text,
    metadata) and search(user_id, query, k). Embeds with the same embedder
    (rag.embedder.embedding_model) so callers see identical vectors either way.
    """

    def __init__(self, embedding_dimension: int = _VECTOR_DIM) -> None:
        self.embedding_dimension = embedding_dimension
        self._initialized = False
        # Imported lazily (not at module load) — pulls in the Cohere client,
        # which requires COHERE_API_KEY; matching FAISSVectorStore's own lazy
        # embedder construction pattern (see core/background_tasks.py).
        from rag.embedder.embedding_model import get_embedding_model

        self._embedder = get_embedding_model()

    async def _ensure_init(self) -> Any:
        pool = await get_pg_pool()
        if not self._initialized:
            async with pool.acquire() as conn:
                await conn.execute(_CREATE_VECTOR_DOCS_SQL)
                await conn.execute(_CREATE_VECTOR_DOCS_INDEX_SQL)
            self._initialized = True
        return pool

    async def upsert(self, user_id: str, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        pool = await self._ensure_init()
        emb = _normalize(self._embedder.embed_for_storage([text])[0])
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Mirror FAISSVectorStore.upsert: soft-delete the old row for
                # this doc_id, then insert a fresh one — never mutate in place,
                # so a concurrent search never sees a half-written row.
                await conn.execute(
                    "UPDATE vector_docs SET deleted = true "
                    "WHERE user_id = $1 AND doc_id = $2 AND NOT deleted",
                    user_id, doc_id,
                )
                await conn.execute(
                    "INSERT INTO vector_docs (user_id, doc_id, text, metadata, embedding) "
                    "VALUES ($1, $2, $3, $4::jsonb, $5)",
                    user_id, doc_id, text, json.dumps(metadata), emb,
                )

    async def search(self, user_id: str, query: str, k: int) -> list[Hit]:
        pool = await self._ensure_init()
        emb = _normalize(self._embedder.embed_for_query(query))
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT doc_id, text, metadata, 1 - (embedding <=> $1) AS score "
                "FROM vector_docs WHERE user_id = $2 AND NOT deleted "
                "ORDER BY embedding <=> $1 LIMIT $3",
                emb, user_id, k,
            )
        hits: list[Hit] = []
        for row in rows:
            meta = row["metadata"]
            meta = meta if isinstance(meta, dict) else json.loads(meta or "{}")
            hits.append(Hit(doc_id=row["doc_id"], text=row["text"], score=float(row["score"]), metadata=meta))
        return hits


# --- PgChunkVectorStore: rag/storage/vector_storage.VectorStorage surface --

_CREATE_VECTOR_CHUNKS_SQL = """
CREATE TABLE IF NOT EXISTS vector_chunks (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    chunk_id TEXT,
    session_id TEXT NOT NULL DEFAULT 'unknown',
    content TEXT NOT NULL DEFAULT '',
    extra JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(%(dim)s) NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
""" % {"dim": _VECTOR_DIM}
_CREATE_VECTOR_CHUNKS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_vector_chunks_user ON vector_chunks(user_id) "
    "WHERE NOT deleted"
)


class PgChunkVectorStore:
    """Drop-in for rag.storage.vector_storage.VectorStorage: matches the
    subset of its API that rag/system/complete_rag.py::RAGSystem actually
    calls (add_chunks, search_similar, get_storage_stats). Extra per-chunk
    fields (timestamp, topics, turn_id_range, creation_time) round-trip
    through the `extra` JSONB column exactly as VectorStorage's metadata dict
    carried them, so callers reading those keys back see no shape change.

    SYNCHRONOUS, unlike every other class in this module: TurtleRAGSystem
    calls add_chunks/search_similar directly on the event loop with no await
    (apps/turtle_server.py:4443/4668) — an existing blocking-call pattern this
    class preserves rather than changes, so it is built on psycopg (sync), not
    asyncpg. It blocks the loop for the DB round-trip exactly as the local
    FAISS path already blocks it for disk I/O today; not a new regression.
    """

    def __init__(self, user_id: str, embedding_dimension: int = _VECTOR_DIM) -> None:
        if not user_id:
            raise ValueError("PgChunkVectorStore requires a user_id")
        self.user_id = user_id
        self.embedding_dimension = embedding_dimension
        self._initialized = False

    def _ensure_init(self) -> Any:
        pool = get_pg_sync_pool()
        if not self._initialized:
            with pool.connection() as conn:
                conn.execute(_CREATE_VECTOR_CHUNKS_SQL)
                conn.execute(_CREATE_VECTOR_CHUNKS_INDEX_SQL)
            self._initialized = True
        return pool

    _EXTRA_KEYS = ("timestamp", "topics", "turn_id_range", "creation_time", "chunk_id")

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match embeddings count ({embeddings.shape[0]})"
            )
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) must match expected "
                f"({self.embedding_dimension})"
            )
        pool = self._ensure_init()
        rows = []
        for chunk, emb in zip(chunks, embeddings):
            session_id = str(chunk.get("session_id", "unknown")).strip() or "unknown"
            content = chunk.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            extra = {k: chunk[k] for k in self._EXTRA_KEYS if k in chunk}
            rows.append((
                self.user_id,
                str(chunk.get("chunk_id", "")) or None,
                session_id,
                content,
                json.dumps(extra),
                _normalize(emb),
            ))
        with pool.connection() as conn:
            conn.cursor().executemany(
                "INSERT INTO vector_chunks (user_id, chunk_id, session_id, content, extra, embedding) "
                "VALUES (%s, %s, %s, %s, %s::jsonb, %s)",
                rows,
            )

    def search_similar(
        self, query_embedding: np.ndarray, top_k: int = 5, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        pool = self._ensure_init()
        emb = _normalize(query_embedding.reshape(-1) if query_embedding.ndim > 1 else query_embedding)
        with pool.connection() as conn:
            cur = conn.execute(
                "SELECT chunk_id, session_id, content, extra, "
                "1 - (embedding <=> %s) AS score "
                "FROM vector_chunks WHERE user_id = %s AND NOT deleted "
                "ORDER BY embedding <=> %s LIMIT %s",
                (emb, self.user_id, emb, top_k),
            )
            rows = cur.fetchall()
            columns = [c.name for c in cur.description]
        results: List[Dict[str, Any]] = []
        for raw_row in rows:
            row = dict(zip(columns, raw_row))
            score = float(row["score"])
            if score < threshold:
                continue
            extra = row["extra"]
            extra = extra if isinstance(extra, dict) else json.loads(extra or "{}")
            entry: Dict[str, Any] = {
                "session_id": row["session_id"],
                "content": row["content"],
                "similarity_score": score,
                **extra,
            }
            if row["chunk_id"]:
                entry["chunk_id"] = row["chunk_id"]
            results.append(entry)
        return results

    def get_storage_stats(self) -> Dict[str, Any]:
        pool = self._ensure_init()
        with pool.connection() as conn:
            cur = conn.execute(
                "SELECT "
                "COUNT(*) FILTER (WHERE NOT deleted) AS active_chunks, "
                "COUNT(*) FILTER (WHERE deleted) AS deleted_chunks, "
                "COUNT(DISTINCT session_id) FILTER (WHERE NOT deleted) AS total_sessions, "
                "COUNT(*) AS total_vectors "
                "FROM vector_chunks WHERE user_id = %s",
                (self.user_id,),
            )
            row = cur.fetchone()
            columns = [c.name for c in cur.description]
        stats = dict(zip(columns, row)) if row else {}
        return {
            "total_vectors": stats.get("total_vectors", 0),
            "active_chunks": stats.get("active_chunks", 0),
            "deleted_chunks": stats.get("deleted_chunks", 0),
            "total_sessions": stats.get("total_sessions", 0),
            "embedding_dimension": self.embedding_dimension,
            "index_type": "pgvector",
            "storage_size_mb": 0.0,  # Not meaningful for a shared managed DB.
        }
