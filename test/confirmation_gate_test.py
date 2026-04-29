import shutil
import unittest
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.confirmation_gate import ConfirmationGate
from core.memory_journal import JournalStore, make_event
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


def _candidate(
    *,
    key: str = "preferences.response_style",
    value: dict | None = None,
    observed_at: str = "2026-04-13T10:00:00Z",
    session_id: str = "s1",
    turn_id: str = "t1",
):
    return make_event(
        kind="preference",
        topic=key.split(".", 1)[0],
        key=key,
        value=value or {"response_style": "concise"},
        confidence=0.75,
        source="inferred",
        extractor="llm_turn",
        session_id=session_id,
        turn_id=turn_id,
        observed_at=observed_at,
        applied=False,
    )


class ConfirmationGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"confirm_gate_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.store = _make_store(self.base)
        self.journal = JournalStore(journal_dir=self.base / "journal")
        self.gate = ConfirmationGate(
            journal=self.journal,
            store=self.store,
            state_path=self.base / "confirmation_state.json",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def test_queue_then_peek_returns_prompt(self) -> None:
        event = _candidate()
        self.assertTrue(self.gate.queue_candidate(event))
        prompt = self.gate.next_prompt()
        self.assertIsNotNone(prompt)
        self.assertEqual(prompt.event_id, event.event_id)
        self.assertIn("concise", prompt.question)
        # peek does not remove
        self.assertEqual(self.gate.pending_count(), 1)

    def test_queue_rejects_explicit_events(self) -> None:
        explicit = make_event(
            kind="fact",
            topic="identity",
            key="identity.name",
            value={"name": "Shriyash"},
            confidence=1.0,
            source="explicit",
            extractor="deterministic",
            session_id="s1",
            turn_id="t1",
            applied=True,
        )
        self.assertFalse(self.gate.queue_candidate(explicit))
        self.assertEqual(self.gate.pending_count(), 0)

    def test_queue_rejects_already_applied_events(self) -> None:
        applied = make_event(
            kind="preference",
            topic="preferences",
            key="preferences.humor_level",
            value={"humor_level": "low"},
            confidence=0.9,
            source="inferred",
            extractor="llm_turn",
            session_id="s1",
            turn_id="t1",
            applied=True,
        )
        self.assertFalse(self.gate.queue_candidate(applied))

    def test_accept_writes_applied_event_and_updates_topic(self) -> None:
        event = _candidate()
        self.gate.queue_candidate(event)
        result = self.gate.record_response(event.event_id, accepted=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "explicit")
        self.assertEqual(result.confidence, 1.0)
        self.assertTrue(result.applied)
        self.assertEqual(result.supersedes, event.event_id)

        self.assertEqual(self.gate.pending_count(), 0)
        preferences = (self.base / "preferences.md").read_text(encoding="utf-8")
        self.assertIn("- Response style: concise", preferences)

    def test_reject_drops_candidate_and_starts_silence(self) -> None:
        event = _candidate()
        self.gate.queue_candidate(event)
        result = self.gate.record_response(event.event_id, accepted=False)
        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "contradiction")
        self.assertTrue(result.value.get("rejected"))
        self.assertEqual(result.supersedes, event.event_id)
        self.assertFalse(result.applied)

        self.assertEqual(self.gate.pending_count(), 0)
        self.assertFalse((self.base / "preferences.md").exists())
        self.assertTrue(self.gate.is_silenced("preferences", "preferences.response_style"))

    def test_silenced_key_is_dropped_on_requeue(self) -> None:
        first = _candidate(turn_id="t1")
        self.gate.queue_candidate(first)
        self.gate.record_response(first.event_id, accepted=False)

        second = _candidate(turn_id="t2", observed_at="2026-04-13T11:00:00Z")
        self.assertFalse(self.gate.queue_candidate(second))
        self.assertEqual(self.gate.pending_count(), 0)
        self.assertIsNone(self.gate.next_prompt())

    def test_silence_expires_after_window(self) -> None:
        expired_at = (
            (datetime.now(UTC) - timedelta(days=30))
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        stale_rejection = make_event(
            kind="contradiction",
            topic="preferences",
            key="preferences.response_style",
            value={"rejected": True},
            confidence=1.0,
            source="explicit",
            extractor="deterministic",
            session_id="s1",
            turn_id="t1",
            observed_at=expired_at,
            applied=False,
        )
        self.journal.append(stale_rejection)
        self.assertFalse(self.gate.is_silenced("preferences", "preferences.response_style"))

        fresh = _candidate(turn_id="t2", observed_at="2026-04-13T11:00:00Z")
        self.assertTrue(self.gate.queue_candidate(fresh))

    def test_state_persists_across_instances(self) -> None:
        event = _candidate()
        self.gate.queue_candidate(event)

        reopened = ConfirmationGate(
            journal=self.journal,
            store=self.store,
            state_path=self.base / "confirmation_state.json",
        )
        self.assertEqual(reopened.pending_count(), 1)
        prompt = reopened.next_prompt()
        self.assertIsNotNone(prompt)
        self.assertEqual(prompt.event_id, event.event_id)

    def test_record_response_with_unknown_event_id_is_noop(self) -> None:
        event = _candidate()
        self.gate.queue_candidate(event)
        self.assertIsNone(self.gate.record_response("does-not-exist", accepted=True))
        self.assertEqual(self.gate.pending_count(), 1)

    def test_candidate_not_rendered_until_confirmed(self) -> None:
        from core.memory_replayer import replay

        event = _candidate()
        self.gate.queue_candidate(event)
        replay(self.journal.load_all(), store=self.store)
        self.assertFalse((self.base / "preferences.md").exists())

        self.gate.record_response(event.event_id, accepted=True)
        self.assertTrue((self.base / "preferences.md").exists())

    def test_queue_is_fifo(self) -> None:
        first = _candidate(
            key="preferences.response_style",
            value={"response_style": "concise"},
            turn_id="t1",
            observed_at="2026-04-13T10:00:00Z",
        )
        second = _candidate(
            key="preferences.humor_level",
            value={"humor_level": "low"},
            turn_id="t2",
            observed_at="2026-04-13T10:05:00Z",
        )
        self.gate.queue_candidate(first)
        self.gate.queue_candidate(second)
        self.assertEqual(self.gate.next_prompt().event_id, first.event_id)
        self.gate.record_response(first.event_id, accepted=True)
        self.assertEqual(self.gate.next_prompt().event_id, second.event_id)


if __name__ == "__main__":
    unittest.main()
