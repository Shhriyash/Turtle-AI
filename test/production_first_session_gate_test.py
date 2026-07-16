"""
Phase 5 — confirmation gate first-session window.

A brand-new user who rejects a prompt should see follow-up prompts within
minutes, not weeks. After they cross the event threshold (or the account
ages past 24h), the gate must revert to the standard ``silence_days``
window. These tests pin both halves.
"""
from __future__ import annotations

import shutil
import unittest
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.confirmation_gate import ConfirmationGate
from core.memory_journal import JournalStore, make_event
from core.personal_memory_store import PersonalMemoryStore


def _store(base: Path) -> PersonalMemoryStore:
    return PersonalMemoryStore(
        base_dir=base,
        index_path=base / "MEMORY.md",
        logs_dir=base / "logs",
        topic_paths={
            "identity": base / "identity.md",
            "preferences": base / "preferences.md",
        },
    )


def _rejection(
    *,
    topic: str = "preferences",
    key: str = "preferences.response_style",
    observed_at: str,
) -> "object":
    return make_event(
        kind="contradiction",
        topic=topic,
        key=key,
        value={"rejected": True, "original_key": key},
        confidence=1.0,
        source="explicit",
        extractor="deterministic",
        session_id="s1",
        turn_id="t1",
        observed_at=observed_at,
        applied=False,
    )


class FirstSessionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"first_session_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.store = _store(self.base)
        self.journal = JournalStore(journal_dir=self.base / "journal")

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def _gate(self, **overrides: int) -> ConfirmationGate:
        kwargs: dict = dict(
            journal=self.journal,
            store=self.store,
            state_path=self.base / "confirmation_state.json",
            silence_days=14,
            first_session_window_minutes=5,
            first_session_event_threshold=20,
            first_session_account_age_hours=24,
        )
        kwargs.update(overrides)
        return ConfirmationGate(**kwargs)

    def test_first_session_user_silenced_only_briefly(self) -> None:
        gate = self._gate()
        # Sparse journal (well below threshold) + fresh journal_dir → first-session.
        now = datetime.now(UTC)
        rejected_4_min_ago = (now - timedelta(minutes=4)).isoformat().replace("+00:00", "Z")
        self.journal.append(_rejection(observed_at=rejected_4_min_ago))

        # Inside the 5-min first-session window → still silenced.
        self.assertTrue(gate.is_silenced("preferences", "preferences.response_style"))

        # Past the 5-min window → silence has lifted.
        rejected_10_min_ago = (now - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
        # Overwrite the journal with a single older rejection.
        shutil.rmtree(self.base / "journal", ignore_errors=True)
        self.journal = JournalStore(journal_dir=self.base / "journal")
        gate = self._gate()
        self.journal.append(_rejection(observed_at=rejected_10_min_ago))
        self.assertFalse(gate.is_silenced("preferences", "preferences.response_style"))

    def test_established_user_keeps_14_day_silence(self) -> None:
        # Pump the journal past the first-session threshold with throwaway events.
        now = datetime.now(UTC)
        for i in range(25):
            self.journal.append(make_event(
                kind="fact",
                topic="identity",
                key=f"identity.note_{i}",
                value={"i": i},
                confidence=1.0,
                source="explicit",
                extractor="deterministic",
                session_id="s_seed",
                turn_id=f"t_{i}",
                observed_at=(now - timedelta(days=30)).isoformat().replace("+00:00", "Z"),
                applied=True,
            ))
        # journal_dir.ctime is set at creation and not portably mutable, so we
        # disable the account-age branch of _is_first_session and rely purely
        # on the event-count branch (25 events > threshold of 20).
        gate = self._gate(first_session_account_age_hours=0)

        # A rejection 1 day ago should still silence (well inside 14-day window).
        one_day_ago = (now - timedelta(days=1)).isoformat().replace("+00:00", "Z")
        self.journal.append(_rejection(observed_at=one_day_ago))
        self.assertTrue(gate.is_silenced("preferences", "preferences.response_style"))

    def test_silence_disabled_when_silence_days_zero(self) -> None:
        gate = self._gate(silence_days=0)
        recent = (datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
        self.journal.append(_rejection(observed_at=recent))
        self.assertFalse(gate.is_silenced("preferences", "preferences.response_style"))


if __name__ == "__main__":
    unittest.main()
