"""Tests for the SQLite FTS5 personal-memory index."""
import shutil
import unittest
import uuid
from pathlib import Path

from core.memory_journal import JournalStore, make_event
from core.memory_sqlite import MemorySQLiteIndex, _flatten_value


def _event(**overrides):
    base = dict(
        kind="fact",
        topic="relations",
        key="relations.best_friend",
        value={"best_friend": "Aarav"},
        confidence=1.0,
        source="explicit",
        extractor="deterministic",
        applied=True,
        session_id="s1",
        turn_id="t1",
        observed_at="2026-05-01T10:00:00Z",
    )
    base.update(overrides)
    return make_event(**base)


class MemorySQLiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"memory_sqlite_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.idx = MemorySQLiteIndex(db_path=self.base / "memory.sqlite")

    def tearDown(self) -> None:
        self.idx.close()
        shutil.rmtree(self.base, ignore_errors=True)

    def test_index_event_is_idempotent(self) -> None:
        event = _event()
        self.idx.index_event(event)
        self.idx.index_event(event)
        self.idx.index_event(event)
        self.assertEqual(self.idx.count(), 1)

    def test_search_finds_value_text(self) -> None:
        self.idx.index_event(_event())
        rows = self.idx.search("best friend")
        self.assertTrue(rows)
        self.assertEqual(rows[0].value["best_friend"], "Aarav")

    def test_porter_stemmer_matches_inflection(self) -> None:
        self.idx.index_event(
            _event(
                topic="workflow",
                key="workflow.habit",
                value={"habit": "meditation every morning"},
            )
        )
        rows = self.idx.search("meditating")
        self.assertTrue(rows)
        self.assertIn("meditation", rows[0].value_text)

    def test_topic_filter_excludes_other_topics(self) -> None:
        self.idx.index_event(
            _event(topic="preferences", key="preferences.tone", value={"tone": "concise"})
        )
        self.idx.index_event(
            _event(topic="identity", key="identity.tone_note", value={"note": "concise writer"})
        )
        rows = self.idx.search("concise", topic="preferences")
        self.assertTrue(rows)
        self.assertTrue(all(r.topic == "preferences" for r in rows))

    def test_applied_only_excludes_unapplied(self) -> None:
        self.idx.index_event(
            _event(key="relations.secret", value={"secret_friend": "Zephyr"}, applied=False)
        )
        self.assertEqual(self.idx.search("Zephyr"), [])
        self.assertTrue(self.idx.search("Zephyr", applied_only=False))

    def test_nested_value_is_findable(self) -> None:
        self.idx.index_event(
            _event(
                topic="projects",
                key="projects.current",
                value={"project": {"name": "Turtle", "stack": ["python", "faiss"]}},
            )
        )
        self.assertTrue(self.idx.search("Turtle"))
        self.assertTrue(self.idx.search("faiss"))

    def test_backfill_from_journal(self) -> None:
        journal = JournalStore(journal_dir=self.base / "journal")
        for i in range(50):
            journal.append(
                _event(
                    key=f"relations.friend_{i}",
                    value={"friend": f"Person{i}"},
                    observed_at=f"2026-05-01T10:{i:02d}:00Z",
                )
            )
        fresh = MemorySQLiteIndex(db_path=self.base / "backfill.sqlite")
        inserted = fresh.backfill_from_journal(journal)
        self.assertEqual(inserted, 50)
        self.assertEqual(fresh.count(), 50)
        # Re-run is a no-op.
        self.assertEqual(fresh.backfill_from_journal(journal), 0)
        fresh.close()

    def test_no_match_returns_empty(self) -> None:
        self.idx.index_event(_event())
        self.assertEqual(self.idx.search("nonexistent_token_xyz"), [])

    def test_flatten_value_bounds(self) -> None:
        self.assertEqual(_flatten_value({"a": "hello", "b": {"c": "world"}}), "hello world")
        self.assertEqual(_flatten_value({}), "")
        # Booleans are not indexed as text.
        self.assertEqual(_flatten_value({"flag": True, "name": "x"}), "x")


if __name__ == "__main__":
    unittest.main()
