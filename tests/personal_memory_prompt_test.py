import shutil
import unittest
import uuid
from pathlib import Path

from core.personal_memory_prompt import PersonalMemoryPromptBuilder, PersonalMemoryPromptConfig
from core.personal_memory_store import PersonalMemoryStore


class PersonalMemoryPromptTests(unittest.TestCase):
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

    def test_email_query_prefers_identity_and_contacts_topics(self) -> None:
        base = Path("tests") / "_tmp" / f"personal_memory_prompt_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = self._make_store(base)
            store.write_topic("identity", ["Primary email: user@example.com"], {"confidence": "confirmed"})
            store.write_topic("contacts", ["Frequent recipient: team@example.com"], {"confidence": "confirmed"})
            store.update_index_entry("identity", "Name, email, timezone")
            store.update_index_entry("contacts", "Frequent recipients and aliases")

            builder = PersonalMemoryPromptBuilder(
                store,
                config=PersonalMemoryPromptConfig(max_bytes=1024, max_topic_files=2),
            )
            block = builder.build_memory_block(task_type="email", query="send an email to the team")

            self.assertIn("user@example.com", block)
            self.assertIn("team@example.com", block)
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_prompt_block_respects_byte_cap(self) -> None:
        base = Path("tests") / "_tmp" / f"personal_memory_prompt_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = self._make_store(base)
            long_line = "x" * 400
            store.write_topic("preferences", [long_line, long_line], {"confidence": "confirmed"})
            store.update_index_entry("preferences", "Long preference content")

            builder = PersonalMemoryPromptBuilder(
                store,
                config=PersonalMemoryPromptConfig(max_bytes=120, max_topic_files=1),
            )
            block = builder.build_memory_block(task_type="general", query="what do I prefer")

            self.assertTrue(block)
            self.assertLessEqual(len(block.encode("utf-8")), 120)
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
