import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from rag.system.complete_rag import TurtleRAGSystem


class DummyEmbedder:
    def embed_for_storage(self, texts):
        return np.ones((len(texts), 1024), dtype=np.float32)

    def embed_for_query(self, query):
        return np.ones((1, 1024), dtype=np.float32)


class DummyVectorStore:
    def __init__(self):
        self.added_chunks = []

    def add_chunks(self, chunks, embeddings):
        self.added_chunks.extend(chunks)

    def search_similar(self, query_embedding, top_k=5, threshold=0.3):
        return []

    def get_storage_stats(self):
        return {
            "total_vectors": len(self.added_chunks),
            "total_sessions": len({c.get("session_id") for c in self.added_chunks}),
            "storage_size_mb": 0.0,
        }


class RAGFinalizationTests(unittest.IsolatedAsyncioTestCase):
    def _build_archive(self, base: Path, session_id: str) -> Path:
        archive_path = base / "archive" / session_id
        archive_path.mkdir(parents=True, exist_ok=True)

        messages = [
            ModelRequest(parts=[UserPromptPart(content="What did I say yesterday?")]),
            ModelResponse(parts=[TextPart(content="You asked about deployment logs.")]),
        ]
        (archive_path / "messages.json").write_bytes(ModelMessagesTypeAdapter.dump_json(messages))
        (archive_path / "session.json").write_text(
            json.dumps({"session_id": session_id, "created_at": "2026-04-01T10:00:00Z"}),
            encoding="utf-8",
        )
        return archive_path

    async def test_finalize_archived_session_indexes_turn_records(self):
        base = Path("test") / "_tmp" / f"rag_finalize_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        session_id = "s_finalize_ok"
        try:
            archive_path = self._build_archive(base, session_id)
            dummy_store = DummyVectorStore()

            with patch("rag.system.complete_rag.get_embedding_model", return_value=DummyEmbedder()), patch(
                "rag.system.complete_rag.get_vector_storage", return_value=dummy_store
            ):
                rag = TurtleRAGSystem(storage_dir=str(base / "rag"))

            ok = await rag.finalize_archived_session(session_id=session_id, archive_path=archive_path)
            self.assertTrue(ok)
            self.assertGreater(len(dummy_store.added_chunks), 0)
        finally:
            shutil.rmtree(base, ignore_errors=True)

    async def test_finalize_archived_session_uses_fallback_chunking_when_missing_method(self):
        base = Path("test") / "_tmp" / f"rag_finalize_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        session_id = "s_finalize_fallback"
        try:
            archive_path = self._build_archive(base, session_id)
            dummy_store = DummyVectorStore()

            with patch("rag.system.complete_rag.get_embedding_model", return_value=DummyEmbedder()), patch(
                "rag.system.complete_rag.get_vector_storage", return_value=dummy_store
            ):
                rag = TurtleRAGSystem(storage_dir=str(base / "rag"))

            class ChunkerWithoutTurnMethod:
                pass

            rag.chunker = ChunkerWithoutTurnMethod()
            ok = await rag.finalize_archived_session(session_id=session_id, archive_path=archive_path)
            self.assertTrue(ok)
            self.assertGreater(len(dummy_store.added_chunks), 0)
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
