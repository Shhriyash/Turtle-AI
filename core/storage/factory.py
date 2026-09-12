"""
core/storage/factory.py
------------------------
Single seam that selects local vs. cloud storage backends off
settings.is_cloud. Callers that used to import a concrete local class
directly (SQLiteSessionStore, get_faiss_vector_store) should import the
matching get_*() here instead, so TURTLE_DEPLOY=cloud repoints them without
touching call sites again in a future backend swap.
"""
from __future__ import annotations

from typing import Any, Dict

from core.config import settings
from core.storage import SessionStoreProtocol, VectorStore

# Process-wide singleton cache for the cloud vector store, mirroring
# core.storage.local.faiss_store's own singleton registry — avoids
# reconstructing the embedder client and re-issuing CREATE TABLE IF NOT
# EXISTS on every one of the 3 call sites' invocations.
_CLOUD_VECTOR_STORE_SINGLETONS: Dict[int, Any] = {}


def get_session_store_backend() -> SessionStoreProtocol:
    """SQLite locally, Postgres in cloud mode. See core/session_store.py's
    SessionStore, which takes this as its `backend`."""
    if settings.is_cloud:
        from core.storage.cloud.postgres_store import PostgresSessionStore

        return PostgresSessionStore()
    from core.storage.local.sqlite_store import SQLiteSessionStore

    return SQLiteSessionStore()


def get_vector_store(embedding_dimension: int = 1024) -> VectorStore:
    """FAISS (process-singleton) locally, pgvector (shared Postgres) in cloud
    mode. Replaces direct calls to
    core.storage.local.faiss_store.get_faiss_vector_store at all 3 call sites
    (apps/turtle_server.py x2, core/background_tasks.py) — same signature and
    return-type Protocol, so the migration to cloud storage doesn't add code
    at those call sites, just swaps the import.
    """
    if settings.is_cloud:
        store = _CLOUD_VECTOR_STORE_SINGLETONS.get(embedding_dimension)
        if store is None:
            from core.storage.cloud.pgvector_store import PgVectorStore

            store = PgVectorStore(embedding_dimension=embedding_dimension)
            _CLOUD_VECTOR_STORE_SINGLETONS[embedding_dimension] = store
        return store
    from core.storage.local.faiss_store import get_faiss_vector_store

    return get_faiss_vector_store(embedding_dimension=embedding_dimension)
