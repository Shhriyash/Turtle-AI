"""
Phase 3 — per-user RAG vector store isolation.

Embedding for user A and searching from user B must return zero hits.
The tenancy boundary is the on-disk directory (rag_vector_dir(user_id)),
not a metadata filter, so two VectorStorages with different storage_dirs
exercise the exact production isolation.
"""
from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

try:
    import numpy as np
    from rag.storage import vector_storage as vs_module
    from rag.storage.vector_storage import VectorStorage, get_vector_storage
    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover — faiss/numpy missing in some envs
    _IMPORT_ERROR = _e


def _unit_vector(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype("float32")
    vec /= np.linalg.norm(vec) + 1e-9
    return vec


def _chunks(seed: int) -> list[dict]:
    return [{
        "chunk_id": f"chunk_{seed}",
        "content": f"hello from user {seed}",
        "metadata": {"user_seed": seed},
    }]


@unittest.skipIf(_IMPORT_ERROR is not None, f"rag deps missing: {_IMPORT_ERROR}")
class RagTenancyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path("test") / "_tmp" / f"rag_tenancy_{uuid.uuid4().hex}"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.dim = 16
        # Use IndexFlatIP (not HNSW) so tiny vector counts behave deterministically.
        import os
        self._orig_index_mode = os.environ.get("RAG_FAISS_INDEX_TYPE")
        os.environ["RAG_FAISS_INDEX_TYPE"] = "flat"

    def tearDown(self) -> None:
        import os
        if self._orig_index_mode is None:
            os.environ.pop("RAG_FAISS_INDEX_TYPE", None)
        else:
            os.environ["RAG_FAISS_INDEX_TYPE"] = self._orig_index_mode
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_two_users_do_not_share_chunks(self) -> None:
        store_a = VectorStorage(storage_dir=str(self.tmp / "a"), embedding_dimension=self.dim)
        store_b = VectorStorage(storage_dir=str(self.tmp / "b"), embedding_dimension=self.dim)

        emb_a = _unit_vector(seed=1, dim=self.dim).reshape(1, -1)
        store_a.add_chunks(_chunks(1), emb_a)

        # User B searches with A's exact embedding. Must still get zero hits.
        results = store_b.search_similar(emb_a[0], top_k=5, threshold=0.0)
        self.assertEqual(results, [])

        # Sanity: A finds its own chunk.
        results_a = store_a.search_similar(emb_a[0], top_k=5, threshold=0.0)
        self.assertEqual(len(results_a), 1)
        self.assertEqual(results_a[0]["chunk_id"], "chunk_1")

    def test_get_vector_storage_cache_keys_by_user_id(self) -> None:
        # Two different user_ids never produce the same backing object.
        # We bypass the on-disk dir collision by clearing the cache and
        # asserting only on identity, not contents.
        # rag_vector_dir is redirected to the test tmp dir so this never
        # creates/writes index files under the real data/ tree.
        import unittest.mock as mock

        vs_module._vector_storage_by_user.clear()
        try:
            with mock.patch.object(
                vs_module,
                "rag_vector_dir",
                side_effect=lambda user_id: self.tmp / "cache" / user_id,
            ):
                a = get_vector_storage("usr_alice")
                b = get_vector_storage("usr_bob")
                self.assertIsNot(a, b)
                # Repeated calls for the same user_id return the cached store.
                self.assertIs(get_vector_storage("usr_alice"), a)
        finally:
            vs_module._vector_storage_by_user.clear()

    def test_get_vector_storage_requires_user_id(self) -> None:
        with self.assertRaises(ValueError):
            get_vector_storage("")


if __name__ == "__main__":
    unittest.main()
