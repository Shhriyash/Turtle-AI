import shutil
import unittest
import uuid
from pathlib import Path

from core.personal_memory_store import PersonalMemoryStore


class PersonalMemorySnapshotTests(unittest.TestCase):
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

    def test_profile_snapshot_is_derived_from_markdown_topics(self) -> None:
        base = Path("test") / "_tmp" / f"personal_memory_snapshot_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = self._make_store(base)
            store.write_topic("identity", ["Name: Shriyash", "Primary email: shriyash@example.com"], {"confidence": "confirmed"})
            store.write_topic("preferences", ["Response style: concise", "Email tone: formal"], {"confidence": "confirmed"})
            store.write_topic("workflow", ["Prefers draft before send: true", "Preferred primary model: openrouter"], {"confidence": "confirmed"})
            store.write_topic("contacts", ["Frequent recipient: team@example.com (count: 3)"], {"confidence": "confirmed"})

            snapshot = store.load_profile_snapshot()

            self.assertEqual(snapshot["identity"]["name"], "Shriyash")
            self.assertEqual(snapshot["identity"]["emails"], ["shriyash@example.com"])
            self.assertEqual(snapshot["preferences"]["response_style"], "concise")
            self.assertEqual(snapshot["preferences"]["email_tone"], "formal")
            self.assertTrue(snapshot["workflow"]["prefers_draft_before_send"])
            self.assertEqual(snapshot["workflow"]["common_recipients"], ["team@example.com"])
            self.assertEqual(snapshot["tool_preferences"]["primary_llm"], "openrouter")
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
