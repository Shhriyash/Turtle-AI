"""
core/storage/local/faiss_store.py
---------------------------------
Generic FAISS implementation of VectorStore protocol for multi-tenant usage.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np

from core.paths import personal_memory_dir
from core.storage import Hit, VectorStore
from rag.embedder.embedding_model import get_embedding_model

# Singleton registry for get_faiss_vector_store(); see its docstring.
_SINGLETONS: Dict[int, "FAISSVectorStore"] = {}
_SINGLETON_LOCK = threading.Lock()


def get_faiss_vector_store(embedding_dimension: int = 1024) -> "FAISSVectorStore":
    """Process-wide singleton FAISSVectorStore.

    Constructing one per WebSocket connection was a real leak, not a style
    problem. This class is ALREADY multi-tenant: ``_indices``/``_metadata`` are
    keyed by ``user_id`` and guarded by per-tenant locks. A fresh instance per
    socket therefore bought nothing and cost:

      * a duplicate in-RAM copy of each tenant's FAISS index, per connection;
      * a cold ``_load_tenant`` disk read on every connect;
      * per-tenant locks that no longer serialise anything, because two
        instances hold two different lock registries for the same tenant — the
        exact index.bin / metadata.json corruption ``_get_lock`` was written to
        prevent, reintroduced one level up.

    Reconnects are frequent (every page reload), so this was on the hot path.
    Keyed by dimension so a differently-sized embedder still gets its own store.
    """
    with _SINGLETON_LOCK:
        store = _SINGLETONS.get(embedding_dimension)
        if store is None:
            store = FAISSVectorStore(embedding_dimension)
            _SINGLETONS[embedding_dimension] = store
        return store


class FAISSVectorStore(VectorStore):
    def __init__(self, embedding_dimension: int = 1024):
        self.embedding_dimension = embedding_dimension
        self._indices: Dict[str, faiss.Index] = {}
        self._metadata: Dict[str, List[Dict[str, Any]]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        # Guards creation of entries in _locks (see _get_lock).
        self._locks_guard = threading.Lock()
        self._embedder = get_embedding_model()

    def _get_tenant_dir(self, user_id: str) -> Path:
        d = personal_memory_dir(user_id) / "vector"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _get_lock(self, user_id: str) -> threading.Lock:
        """Per-tenant lock, created under a registry guard.

        This was an unsynchronized check-then-insert. That was (barely) safe
        while every caller ran on the single event-loop thread, but moving
        search/upsert onto asyncio.to_thread made it genuinely multi-threaded:
        two first-time operations for the same tenant could each create and take
        a DIFFERENT lock, then concurrently mutate _indices/_metadata and
        overwrite index.bin / metadata.json — corrupting vector-to-metadata
        alignment. Guard the registry itself so the per-tenant lock is unique.
        """
        with self._locks_guard:
            lock = self._locks.get(user_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[user_id] = lock
            return lock

    def _load_tenant(self, user_id: str) -> None:
        if user_id in self._indices:
            return

        tdir = self._get_tenant_dir(user_id)
        index_path = tdir / "index.bin"
        meta_path = tdir / "metadata.json"

        if index_path.exists() and meta_path.exists():
            try:
                self._indices[user_id] = faiss.read_index(str(index_path))
                self._metadata[user_id] = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                self._indices[user_id] = faiss.IndexFlatIP(self.embedding_dimension)
                self._metadata[user_id] = []
        else:
            self._indices[user_id] = faiss.IndexFlatIP(self.embedding_dimension)
            self._metadata[user_id] = []

    def _save_tenant(self, user_id: str) -> None:
        tdir = self._get_tenant_dir(user_id)
        index_path = tdir / "index.bin"
        meta_path = tdir / "metadata.json"

        faiss.write_index(self._indices[user_id], str(index_path))
        meta_path.write_text(json.dumps(self._metadata[user_id], ensure_ascii=False))

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        if v.ndim == 1:
            v = v.reshape(1, -1)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (v / norms).astype(np.float32)

    async def upsert(self, user_id: str, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        """Upsert a document — offloaded so the event loop never blocks.

        The body acquires a threading.Lock and then does synchronous work
        (faiss disk I/O + a BLOCKING Cohere HTTP embed). Run inline on the event
        loop that froze every user's turn for the duration of the embed, and a
        second coroutine hitting the same tenant would block-acquire the lock on
        the sole loop thread — deadlocking until the HTTP call returned. The
        lock stays INSIDE the worker thread, which is what makes it safe.
        """
        await asyncio.to_thread(self._upsert_sync, user_id, doc_id, text, metadata)

    def _upsert_sync(self, user_id: str, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        with self._get_lock(user_id):
            self._load_tenant(user_id)
            
            # Mark old doc_id as deleted
            for meta in self._metadata[user_id]:
                if meta.get("doc_id") == doc_id:
                    meta["deleted"] = True

            emb = self._embedder.embed_for_storage([text])
            emb = self._normalize(emb)

            idx = self._indices[user_id].ntotal
            self._indices[user_id].add(emb)

            meta = {
                "vector_index": idx,
                "doc_id": doc_id,
                "text": text,
                "metadata": metadata,
                "deleted": False
            }
            self._metadata[user_id].append(meta)
            self._save_tenant(user_id)

    async def search(self, user_id: str, query: str, k: int) -> list[Hit]:
        """Search top-k documents — offloaded (see upsert for why)."""
        return await asyncio.to_thread(self._search_sync, user_id, query, k)

    def _search_sync(self, user_id: str, query: str, k: int) -> list[Hit]:
        with self._get_lock(user_id):
            self._load_tenant(user_id)
            if self._indices[user_id].ntotal == 0:
                return []

            emb = self._embedder.embed_for_query(query)
            emb = self._normalize(emb)
            
            search_k = min(k * 3, self._indices[user_id].ntotal)
            scores, indices = self._indices[user_id].search(emb, search_k)

            hits = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self._metadata[user_id]):
                    meta = self._metadata[user_id][idx]
                    if not meta.get("deleted", False):
                        hits.append(
                            Hit(
                                doc_id=meta["doc_id"],
                                text=meta["text"],
                                score=float(score),
                                metadata=meta.get("metadata", {})
                            )
                        )
                        if len(hits) >= k:
                            break
            return hits
