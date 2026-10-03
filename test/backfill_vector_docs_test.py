"""Tests for scripts/backfill_vector_docs.py (ledger 2.2).

Covers the two properties WP2.D was told to be explicit about:
- dry run never touches the real (Cohere-backed) embedder, so it can't crash
  for an operator with no COHERE_API_KEY set;
- the doc_id scheme it re-derives matches core/background_tasks.py's own
  embed_personal_memory job exactly, so a later per-write embed for the same
  line overwrites this backfill's row instead of duplicating it.
"""
from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

from scripts.backfill_vector_docs import _docs_for_user, _line_to_text, _run


class LineToTextTests(unittest.TestCase):
    def test_strips_bullet_prefix(self) -> None:
        self.assertEqual(_line_to_text("- Best friend: Aarav"), "Best friend: Aarav")

    def test_strips_whitespace_only_bullet(self) -> None:
        self.assertEqual(_line_to_text("   "), "")

    def test_plain_line_unchanged(self) -> None:
        self.assertEqual(_line_to_text("Name: Shriyash"), "Name: Shriyash")


class DocsForUserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.user_id = f"usr_test_{uuid.uuid4().hex[:10]}"

    def tearDown(self) -> None:
        from core.paths import personal_memory_dir

        shutil.rmtree(personal_memory_dir(self.user_id), ignore_errors=True)

    def test_doc_id_matches_embed_personal_memory_scheme(self) -> None:
        # Same tenant guard core/background_tasks.py checks: a real usr_* id,
        # embed dispatch no-op'd so this stays offline (no Cohere call, no
        # queue job), same pattern test/retrieval_broker_test.py uses.
        with patch("core.worker.dispatch_embed_personal_memory_job"):
            from core.personal_memory_store import PersonalMemoryStore

            store = PersonalMemoryStore(self.user_id)
            store.write_topic(
                "relations", ["- Best friend: Aarav", "- Sister: Priya"], {"title": "Relations"}
            )

        docs = _docs_for_user(self.user_id)
        doc_ids = {doc_id for doc_id, _text, _meta in docs}
        self.assertIn("topic_relations_0", doc_ids)
        self.assertIn("topic_relations_1", doc_ids)
        texts = {text for _doc_id, text, _meta in docs}
        self.assertIn("Best friend: Aarav", texts)
        self.assertIn("Sister: Priya", texts)

    def test_empty_user_has_no_docs(self) -> None:
        self.assertEqual(_docs_for_user(self.user_id), [])


class DryRunNeverBuildsEmbedderTests(unittest.IsolatedAsyncioTestCase):
    """--dry-run (the default, no --confirm) must not import/construct the
    real vector store, since PgVectorStore/FAISSVectorStore's embedder is
    Cohere and raises without COHERE_API_KEY."""

    async def test_dry_run_does_not_call_get_vector_store(self) -> None:
        with patch("scripts.backfill_vector_docs._docs_for_user", return_value=[
            ("topic_relations_0", "Best friend: Aarav", {"topic": "relations", "line_index": 0}),
        ]):
            with patch("core.storage.factory.get_vector_store") as mock_get_store:
                result = await _run("usr_whatever", dry_run=True)
        self.assertEqual(result, 0)
        mock_get_store.assert_not_called()

    async def test_confirm_path_calls_upsert(self) -> None:
        mock_store = AsyncMock()
        with patch("scripts.backfill_vector_docs._docs_for_user", return_value=[
            ("topic_relations_0", "Best friend: Aarav", {"topic": "relations", "line_index": 0}),
        ]):
            with patch("core.storage.factory.get_vector_store", return_value=mock_store):
                result = await _run("usr_whatever", dry_run=False)
        self.assertEqual(result, 0)
        mock_store.upsert.assert_awaited_once_with(
            "usr_whatever", "topic_relations_0", "Best friend: Aarav",
            {"topic": "relations", "line_index": 0},
        )


if __name__ == "__main__":
    unittest.main()
