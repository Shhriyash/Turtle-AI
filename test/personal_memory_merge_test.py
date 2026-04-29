import shutil
import unittest
import uuid
from pathlib import Path

from pydantic_ai.messages import ModelRequest, UserPromptPart

from core.personal_memory_extract import extract_memory_candidates_from_messages
from core.personal_memory_merge import merge_personal_memory_candidates
from core.personal_memory_store import PersonalMemoryStore


class PersonalMemoryMergeTests(unittest.TestCase):
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

    def test_merge_persists_explicit_facts_and_preferences(self) -> None:
        base = Path("test") / "_tmp" / f"personal_memory_merge_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = self._make_store(base)
            history = [
                ModelRequest(parts=[UserPromptPart(content="My name is Shriyash and my email is shriyash@example.com. Keep responses concise.")]),
            ]
            candidates = extract_memory_candidates_from_messages(
                message_history=history,
                session_id="session-1",
                profile=None,
            )
            result = merge_personal_memory_candidates(store=store, candidates=candidates)

            self.assertIn("identity", result.written_topics)
            self.assertIn("preferences", result.written_topics)
            identity_text = (base / "identity.md").read_text(encoding="utf-8")
            preferences_text = (base / "preferences.md").read_text(encoding="utf-8")
            self.assertIn("Name: Shriyash", identity_text)
            self.assertIn("Primary email: shriyash@example.com", identity_text)
            self.assertIn("Response style: concise", preferences_text)
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_merge_accepts_im_name_pattern(self) -> None:
        base = Path("test") / "_tmp" / f"personal_memory_merge_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = self._make_store(base)
            history = [
                ModelRequest(parts=[UserPromptPart(content="Hii, Im shriyash")]),
            ]
            candidates = extract_memory_candidates_from_messages(
                message_history=history,
                session_id="session-im",
                profile=None,
            )
            result = merge_personal_memory_candidates(store=store, candidates=candidates)

            self.assertIn("identity", result.written_topics)
            identity_text = (base / "identity.md").read_text(encoding="utf-8")
            self.assertIn("Name: shriyash", identity_text)
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_merge_extracts_location_from_wrapped_prompt(self) -> None:
        base = Path("test") / "_tmp" / f"personal_memory_merge_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = self._make_store(base)
            wrapped_prompt = (
                "Relevant user memory:\n"
                "[Identity]\n"
                "- Name: Shriyash\n\n"
                "User request:\n"
                "I am from Indore and I live in Bengaluru."
            )
            history = [
                ModelRequest(parts=[UserPromptPart(content=wrapped_prompt)]),
            ]

            candidates = extract_memory_candidates_from_messages(
                message_history=history,
                session_id="session-location",
                profile=None,
            )
            result = merge_personal_memory_candidates(store=store, candidates=candidates)

            self.assertIn("identity", result.written_topics)
            identity_text = (base / "identity.md").read_text(encoding="utf-8")
            self.assertIn("Home city: Indore", identity_text)
            self.assertIn("Current city: Bengaluru", identity_text)
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_weak_signals_are_not_persisted(self) -> None:
        base = Path("test") / "_tmp" / f"personal_memory_merge_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = self._make_store(base)
            history = [
                ModelRequest(parts=[UserPromptPart(content="Let's do an email task.")]),
            ]
            candidates = extract_memory_candidates_from_messages(
                message_history=history,
                session_id="session-2",
                profile=None,
            )
            result = merge_personal_memory_candidates(store=store, candidates=candidates)

            self.assertEqual(result.written_topics, [])
            self.assertFalse((base / "workflow.md").exists())
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
