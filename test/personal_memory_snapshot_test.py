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
                "relations": base / "relations.md",
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

    def test_contacts_and_relations_reach_the_snapshot(self) -> None:
        """The email tool reads snapshot["contacts"]/["relations"] directly.

        Both keys were absent from the snapshot while that read was live, so
        the agent got an empty dict on every send. These are the shapes real
        stores on disk hold -- a plain `Label: value` line, which the old
        contacts parser skipped because it only matched "Frequent recipient:".
        """
        base = Path("test") / "_tmp" / f"personal_memory_contacts_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = self._make_store(base)
            store.write_topic(
                "contacts",
                [
                    "Frequent recipient: team@example.com (count: 3)",
                    "Github Profile: https://github.com/example",
                    "Shriyash Gmail Com: shriyash@example.com",
                ],
                {"confidence": "confirmed"},
            )
            store.write_topic("relations", ["Best Friend: Aarav", "Manager: Keshav"], {"confidence": "confirmed"})

            snapshot = store.load_profile_snapshot()

            contacts = snapshot["contacts"]
            # A value containing a colon must survive: only the FIRST colon splits.
            self.assertEqual(contacts["entries"]["github_profile"], "https://github.com/example")
            self.assertEqual(contacts["entries"]["shriyash_gmail_com"], "shriyash@example.com")
            self.assertEqual(contacts["frequent_recipients"], ["team@example.com"])
            self.assertEqual(sorted(contacts["emails"]), ["shriyash@example.com", "team@example.com"])
            # The "Frequent recipient:" line is a recipient, not a labelled entry.
            self.assertNotIn("frequent_recipient", contacts["entries"])

            self.assertEqual(snapshot["relations"], {"best_friend": "Aarav", "manager": "Keshav"})

            # The pre-existing surface must not regress.
            self.assertEqual(snapshot["workflow"]["common_recipients"], ["team@example.com"])
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_contacts_and_relations_are_empty_dicts_when_unwritten(self) -> None:
        """Consumers do `_snapshot.get("contacts") or {}` -- the key must exist
        and be dict-shaped even for a brand new user, never missing."""
        base = Path("test") / "_tmp" / f"personal_memory_empty_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            snapshot = self._make_store(base).load_profile_snapshot()
            self.assertEqual(
                snapshot["contacts"], {"entries": {}, "emails": [], "frequent_recipients": []}
            )
            self.assertEqual(snapshot["relations"], {})
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
