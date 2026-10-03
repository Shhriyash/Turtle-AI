"""P6-A2: FAISS persistence must survive a crash and concurrent saves.

Local mode only (faiss is never used in cloud mode). These tests crash the
save at specific points and assert the on-disk state still loads and is
self-consistent, and that two concurrent saves cannot clobber each other.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

DIM = 8


class _FakeEmbedder:
    """Deterministic: identical text -> identical unit-ish vector."""

    @staticmethod
    def _vec(text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(DIM).astype(np.float32)

    def embed_for_storage(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_for_query(self, text):
        return self._vec(text).reshape(1, -1)


def _make_store(tmp_path: Path, monkeypatch):
    from core.storage.local import faiss_store

    monkeypatch.setattr(
        faiss_store, "personal_memory_dir", lambda user_id: tmp_path / user_id
    )
    monkeypatch.setattr(faiss_store, "get_embedding_model", lambda: _FakeEmbedder())
    return faiss_store.FAISSVectorStore(embedding_dimension=DIM)


def _fresh_store(tmp_path: Path, monkeypatch):
    """A second instance == what a restarted process sees (cold load from disk)."""
    return _make_store(tmp_path, monkeypatch)


def _doc_ids(hits):
    return [h.doc_id for h in hits]


def test_crash_between_index_and_metadata_write_recovers_last_committed_state(
    tmp_path, monkeypatch
):
    """The pair (index.bin, metadata.json) must not diverge into a state that
    misaligns later results. Crash the metadata commit of the 2nd upsert."""
    store = _make_store(tmp_path, monkeypatch)
    store._upsert_sync("u1", "doc-A", "alpha text", {})

    real_replace = os.replace
    real_write_text = Path.write_text

    def crashing_replace(src, dst, *a, **k):
        if Path(dst).name == "metadata.json":
            raise OSError("simulated crash before metadata commit")
        return real_replace(src, dst, *a, **k)

    def crashing_write_text(self, *a, **k):
        if self.name == "metadata.json":
            raise OSError("simulated crash before metadata commit")
        return real_write_text(self, *a, **k)

    monkeypatch.setattr(os, "replace", crashing_replace)
    monkeypatch.setattr(Path, "write_text", crashing_write_text)
    with pytest.raises(OSError):
        store._upsert_sync("u1", "doc-B", "bravo text", {})
    monkeypatch.undo()  # "process restarts": un-crash and drop the old store

    reloaded = _fresh_store(tmp_path, monkeypatch)
    reloaded._load_tenant("u1")
    assert reloaded._indices["u1"].ntotal == len(reloaded._metadata["u1"]), (
        "index and metadata diverged after crash"
    )
    # Recovered to the last committed state: doc-A present, doc-B never landed.
    hits = reloaded._search_sync("u1", "alpha text", 3)
    assert _doc_ids(hits) == ["doc-A"]

    # And the store is still usable AND aligned afterwards.
    reloaded._upsert_sync("u1", "doc-C", "charlie text", {})
    top = reloaded._search_sync("u1", "charlie text", 1)
    assert _doc_ids(top) == ["doc-C"]
    assert _doc_ids(reloaded._search_sync("u1", "alpha text", 1)) == ["doc-A"]


def test_torn_index_write_leaves_previous_index_intact(tmp_path, monkeypatch):
    """A crash MID-write of index.bin must not destroy the prior good index."""
    store = _make_store(tmp_path, monkeypatch)
    store._upsert_sync("u1", "doc-A", "alpha text", {})

    real_replace = os.replace
    real_write_index = faiss.write_index

    def torn_write_index(index, path, *a, **k):
        if not isinstance(path, (str, os.PathLike)):
            # faiss.serialize_index calls write_index with an in-memory writer.
            return real_write_index(index, path, *a, **k)
        # Half-written file, then the process dies.
        Path(path).write_bytes(b"\x00garbage-not-a-faiss-index")
        raise OSError("simulated crash mid-write")

    def crashing_replace(src, dst, *a, **k):
        if Path(dst).name == "index.bin":
            raise OSError("simulated crash before index commit")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(faiss, "write_index", torn_write_index)
    monkeypatch.setattr(os, "replace", crashing_replace)
    with pytest.raises(OSError):
        store._upsert_sync("u1", "doc-B", "bravo text", {})
    monkeypatch.undo()
    assert faiss.write_index is real_write_index

    reloaded = _fresh_store(tmp_path, monkeypatch)
    reloaded._load_tenant("u1")
    assert reloaded._indices["u1"].ntotal == 1
    assert _doc_ids(reloaded._search_sync("u1", "alpha text", 3)) == ["doc-A"]
    stray = [p.name for p in (tmp_path / "u1" / "vector").iterdir()
             if p.name not in ("index.bin", "metadata.json")]
    assert stray == [], f"temp files leaked: {stray}"


def test_vector_storage_concurrent_saves_do_not_collide(tmp_path, monkeypatch):
    """_save_index used ONE fixed temp name, so two concurrent saves wrote the
    same file and the second os.replace hit FileNotFoundError. Hold both saves
    inside os.replace at the same moment: both temp files must exist."""
    monkeypatch.setenv("RAG_FAISS_INDEX_TYPE", "flat")
    from rag.storage.vector_storage import VectorStorage

    vs = VectorStorage(storage_dir=str(tmp_path / "vs"), embedding_dimension=DIM)
    vs.faiss_index.add(np.ones((3, DIM), dtype=np.float32))

    real_replace = os.replace
    barrier = threading.Barrier(2)
    sources: list[str] = []
    errors: list[BaseException] = []

    def gated_replace(src, dst, *a, **k):
        if Path(dst).name == "faiss_index.bin":
            sources.append(Path(src).name)
            try:
                barrier.wait(timeout=1)
            except threading.BrokenBarrierError:
                pass
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", gated_replace)

    def run():
        try:
            vs._save_index()
        except BaseException as exc:  # noqa: BLE001 - assert on it below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    monkeypatch.undo()

    assert errors == [], f"concurrent save failed: {errors!r}"
    assert len(sources) == 2 and len(set(sources)) == 2, (
        f"temp names collided: {sources}"
    )
    assert faiss.read_index(str(vs.index_path)).ntotal == 3
    leftovers = [p.name for p in vs.storage_dir.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
