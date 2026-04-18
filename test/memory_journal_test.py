import shutil
import unittest
import uuid
from pathlib import Path

from core.memory_journal import JournalStore, make_event
from core.memory_migration import migrate_existing_topics
from core.memory_replayer import replay
from core.personal_memory_store import PersonalMemoryStore


def _make_store(base: Path) -> PersonalMemoryStore:
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
            "corrections": base / "corrections.md",
        },
    )


class MemoryJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"memory_journal_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.store = _make_store(self.base)
        self.journal = JournalStore(journal_dir=self.base / "journal")

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def test_append_and_iter_roundtrip(self) -> None:
        event = make_event(
            kind="fact",
            topic="identity",
            key="identity.name",
            value={"name": "Shriyash"},
            confidence=1.0,
            source="explicit",
            extractor="deterministic",
            applied=True,
            session_id="s1",
            turn_id="t1",
            observed_at="2026-04-13T10:00:00Z",
        )
        self.journal.append(event)
        loaded = self.journal.load_all()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].event_id, event.event_id)
        self.assertEqual(loaded[0].value["name"], "Shriyash")

    def test_append_is_idempotent_by_event_id(self) -> None:
        event = make_event(
            kind="fact",
            topic="identity",
            key="identity.name",
            value={"name": "A"},
            confidence=1.0,
            source="explicit",
            extractor="deterministic",
            applied=True,
            session_id="s1",
            turn_id="t1",
            observed_at="2026-04-13T10:00:00Z",
        )
        self.journal.append(event)
        self.journal.append(event)
        self.journal.append(event)
        self.assertEqual(len(self.journal.load_all()), 1)

    def test_replayer_produces_expected_topic_files(self) -> None:
        events = [
            make_event(
                kind="fact",
                topic="identity",
                key="identity.name",
                value={"name": "Shriyash"},
                confidence=1.0,
                source="explicit",
                extractor="deterministic",
                applied=True,
                session_id="s1",
                turn_id="t1",
                observed_at="2026-04-13T10:00:00Z",
            ),
            make_event(
                kind="fact",
                topic="identity",
                key="identity.home_city",
                value={"home_city": "Indore"},
                confidence=1.0,
                source="explicit",
                extractor="deterministic",
                applied=True,
                session_id="s1",
                turn_id="t1b",
                observed_at="2026-04-13T10:00:30Z",
            ),
            make_event(
                kind="fact",
                topic="identity",
                key="identity.primary_email",
                value={"primary_email": "user@example.com"},
                confidence=1.0,
                source="explicit",
                extractor="deterministic",
                applied=True,
                session_id="s1",
                turn_id="t2",
                observed_at="2026-04-13T10:01:00Z",
            ),
            make_event(
                kind="preference",
                topic="preferences",
                key="preferences.response_style",
                value={"response_style": "concise"},
                confidence=0.95,
                source="explicit",
                extractor="deterministic",
                applied=True,
                session_id="s1",
                turn_id="t3",
                observed_at="2026-04-13T10:02:00Z",
            ),
        ]
        self.journal.append_many(events)
        result = replay(self.journal.load_all(), store=self.store)
        self.assertIn("identity", result.written_topics)
        self.assertIn("preferences", result.written_topics)

        identity_text = (self.base / "identity.md").read_text(encoding="utf-8")
        self.assertIn("- Name: Shriyash", identity_text)
        self.assertIn("- Home city: Indore", identity_text)
        self.assertIn("- Primary email: user@example.com", identity_text)

        prefs_text = (self.base / "preferences.md").read_text(encoding="utf-8")
        self.assertIn("- Response style: concise", prefs_text)

        index_text = (self.base / "MEMORY.md").read_text(encoding="utf-8")
        self.assertIn("identity.md", index_text)
        self.assertIn("preferences.md", index_text)

    def test_replayer_is_deterministic(self) -> None:
        events = [
            make_event(
                kind="fact",
                topic="identity",
                key="identity.name",
                value={"name": "A"},
                confidence=1.0,
                source="explicit",
                extractor="deterministic",
                applied=True,
                session_id="s1",
                turn_id="t1",
                observed_at="2026-04-13T10:00:00Z",
                event_id="01AAAAAAAAAAAAAAAAAAAAAAA1",
            ),
        ]
        replay(events, store=self.store)
        first = (self.base / "identity.md").read_text(encoding="utf-8")
        replay(events, store=self.store)
        second = (self.base / "identity.md").read_text(encoding="utf-8")
        self.assertEqual(first, second)

    def test_replayer_latest_wins_per_key(self) -> None:
        events = [
            make_event(
                kind="fact",
                topic="identity",
                key="identity.name",
                value={"name": "Old"},
                confidence=1.0,
                source="explicit",
                extractor="deterministic",
                applied=True,
                session_id="s1",
                turn_id="t1",
                observed_at="2026-04-13T10:00:00Z",
            ),
            make_event(
                kind="fact",
                topic="identity",
                key="identity.name",
                value={"name": "New"},
                confidence=1.0,
                source="explicit",
                extractor="deterministic",
                applied=True,
                session_id="s1",
                turn_id="t2",
                observed_at="2026-04-13T11:00:00Z",
            ),
        ]
        replay(events, store=self.store)
        text = (self.base / "identity.md").read_text(encoding="utf-8")
        self.assertIn("- Name: New", text)
        self.assertNotIn("- Name: Old", text)

    def test_replayer_supersedes_removes_prior(self) -> None:
        old = make_event(
            kind="preference",
            topic="preferences",
            key="preferences.response_style",
            value={"response_style": "detailed"},
            confidence=0.95,
            source="explicit",
            extractor="deterministic",
            applied=True,
            session_id="s1",
            turn_id="t1",
            observed_at="2026-04-13T10:00:00Z",
        )
        correction = make_event(
            kind="correction",
            topic="preferences",
            key="preferences.response_style",
            value={"response_style": "concise"},
            confidence=1.0,
            source="explicit",
            extractor="deterministic",
            applied=True,
            session_id="s1",
            turn_id="t2",
            observed_at="2026-04-13T10:05:00Z",
            supersedes=old.event_id,
        )
        replay([old, correction], store=self.store)
        text = (self.base / "preferences.md").read_text(encoding="utf-8")
        self.assertIn("- Response style: concise", text)
        self.assertNotIn("- Response style: detailed", text)

    def test_replayer_skips_rejected_events(self) -> None:
        event = make_event(
            kind="preference",
            topic="preferences",
            key="preferences.humor_level",
            value={"humor_level": "low"},
            confidence=0.95,
            source="explicit",
            extractor="deterministic",
            applied=True,
            session_id="s1",
            turn_id="t1",
            observed_at="2026-04-13T10:00:00Z",
        )
        event.rejected = True
        replay([event], store=self.store)
        self.assertFalse((self.base / "preferences.md").exists())

    def test_migration_round_trip_preserves_topic_files(self) -> None:
        self.store.write_topic(
            "identity",
            ["- Name: Shriyash", "- Primary email: user@example.com", "- Timezone: UTC"],
            {"title": "Identity"},
        )
        self.store.update_index_entry("identity", "Name, email, timezone, preferred address")

        self.store.write_topic(
            "preferences",
            ["- Response style: concise", "- Humor level: low"],
            {"title": "Preferences"},
        )
        self.store.update_index_entry("preferences", "Tone, response style, and delivery defaults")

        before_identity = (self.base / "identity.md").read_text(encoding="utf-8")
        before_prefs = (self.base / "preferences.md").read_text(encoding="utf-8")

        before_identity_body = _strip_frontmatter(before_identity)
        before_prefs_body = _strip_frontmatter(before_prefs)

        result = migrate_existing_topics(store=self.store, journal=self.journal)
        self.assertGreater(result.emitted_event_count, 0)
        self.assertIn("identity", result.written_topics)
        self.assertIn("preferences", result.written_topics)

        after_identity = _strip_frontmatter(
            (self.base / "identity.md").read_text(encoding="utf-8")
        )
        after_prefs = _strip_frontmatter(
            (self.base / "preferences.md").read_text(encoding="utf-8")
        )
        self.assertEqual(before_identity_body, after_identity)
        self.assertEqual(before_prefs_body, after_prefs)

        self.assertGreater(len(self.journal.load_all()), 0)
        for event in self.journal.load_all():
            self.assertEqual(event.source, "migration")
            self.assertEqual(event.extractor, "migration")


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text.strip()
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        return text.strip()
    return parts[1].strip()


if __name__ == "__main__":
    unittest.main()
