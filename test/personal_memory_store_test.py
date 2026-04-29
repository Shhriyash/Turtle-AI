import shutil
import unittest
import uuid
from pathlib import Path

from core.personal_memory_store import PersonalMemoryStore


class PersonalMemoryStoreTests(unittest.TestCase):
    def _make_store(self, base: Path) -> PersonalMemoryStore:
        return PersonalMemoryStore(
            base_dir=base,
            index_path=base / "MEMORY.md",
            logs_dir=base / "logs",
            topic_paths={
                "identity": base / "identity.md",
                "preferences": base / "preferences.md",
                "workflow": base / "workflow.md",
                "contacts": base / "contacts.md",
                "projects": base / "projects.md",
            },
        )

    def test_write_topic_then_update_index(self) -> None:
        base = Path("test") / "_tmp" / f"personal_memory_store_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = self._make_store(base)
            topic = store.write_topic(
                "preferences",
                ["Prefers concise responses", "Keep tone direct"],
                {"confidence": "confirmed"},
            )
            self.assertEqual(topic.metadata["topic"], "preference")

            entries = store.update_index_entry("preferences", "Tone and verbosity defaults")
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].file_name, "preferences.md")
            self.assertEqual(entries[0].summary, "Tone and verbosity defaults")
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_append_daily_log_creates_nested_log_file(self) -> None:
        base = Path("test") / "_tmp" / f"personal_memory_log_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = self._make_store(base)
            log_path = store.append_daily_log(
                "Captured explicit preference",
                session_id="s-1",
                timestamp="2026-04-03T12:00:00Z",
            )
            self.assertTrue(log_path.exists())
            self.assertIn("[session:s-1]", log_path.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
