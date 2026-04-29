"""Tests for Step 5 (DreamPass / Stage C) and Step 6 (decay in replayer)."""
import shutil
import unittest
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.confirmation_gate import ConfirmationGate
from core.dream_pass import (
    DreamPass,
    DreamPassResult,
    _measure_topic_line_counts,
    _parse_decisions,
    _restore_snapshot,
    _run_sanity_checks,
    _take_snapshot,
    _write_rollback_events,
)
from core.memory_journal import JournalStore, make_event
from core.memory_replayer import replay
from core.personal_memory_store import PersonalMemoryStore


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

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


def _make_gate(base: Path, journal: JournalStore, store: PersonalMemoryStore) -> ConfirmationGate:
    return ConfirmationGate(
        journal=journal,
        store=store,
        state_path=base / "confirmation_state.json",
    )


def _make_dream_pass(
    base: Path,
    journal: JournalStore,
    store: PersonalMemoryStore,
    gate: ConfirmationGate,
    **kwargs,
) -> DreamPass:
    return DreamPass(
        journal=journal,
        store=store,
        confirmation_gate=gate,
        state_path=base / "dream_pass_state.json",
        snapshots_dir=base / "snapshots",
        **kwargs,
    )


def _inferred_event(
    *,
    key: str,
    value: dict | None = None,
    observed_at: str = "2026-04-13T10:00:00Z",
    session_id: str = "s1",
    turn_id: str = "t1",
):
    topic = key.split(".", 1)[0]
    return make_event(
        kind="preference",
        topic=topic,
        key=key,
        value=value or {key.split(".")[-1]: "some_value"},
        confidence=0.75,
        source="inferred",
        extractor="llm_turn",
        session_id=session_id,
        turn_id=turn_id,
        observed_at=observed_at,
        applied=False,
    )


# ---------------------------------------------------------------------------
# Step 6: Decay in the replayer
# ---------------------------------------------------------------------------

class DecayReplayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"decay_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.store = _make_store(self.base)

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def _pref_event(self, observed_at: str, *, applied: bool = True):
        return make_event(
            kind="preference",
            topic="preferences",
            key="preferences.response_style",
            value={"response_style": "concise"},
            confidence=1.0,
            source="explicit",
            extractor="deterministic",
            session_id="s1",
            turn_id="t1",
            observed_at=observed_at,
            applied=applied,
        )

    def _identity_event(self, observed_at: str):
        return make_event(
            kind="fact",
            topic="identity",
            key="identity.name",
            value={"name": "Shriyash"},
            confidence=1.0,
            source="explicit",
            extractor="deterministic",
            session_id="s1",
            turn_id="t1",
            observed_at=observed_at,
            applied=True,
        )

    def _migration_event(self, observed_at: str):
        return make_event(
            kind="fact",
            topic="preferences",
            key="preferences.humor_level",
            value={"humor_level": "low"},
            confidence=1.0,
            source="migration",
            extractor="migration",
            session_id="migration",
            turn_id="migration-preferences-0",
            observed_at=observed_at,
            applied=True,
        )

    # --- stale events are excluded ---

    def test_fresh_event_renders(self) -> None:
        now = datetime.now(UTC)
        observed_at = (now - timedelta(days=1)).isoformat(timespec="seconds").replace("+00:00", "Z")
        event = self._pref_event(observed_at)
        result = replay([event], store=self.store, reference_time=now)
        self.assertIn("preferences", result.written_topics)
        text = (self.base / "preferences.md").read_text(encoding="utf-8")
        self.assertIn("- Response style: concise", text)

    def test_stale_event_excluded(self) -> None:
        now = datetime.now(UTC)
        observed_at = (now - timedelta(days=31)).isoformat(timespec="seconds").replace("+00:00", "Z")
        event = self._pref_event(observed_at)
        result = replay([event], store=self.store, reference_time=now)
        self.assertNotIn("preferences", result.written_topics)
        self.assertFalse((self.base / "preferences.md").exists())

    def test_29_days_not_decayed(self) -> None:
        """Events at 29 days old are still within the decay window and render."""
        now = datetime.now(UTC)
        observed_at = (now - timedelta(days=29)).isoformat(timespec="seconds").replace("+00:00", "Z")
        event = self._pref_event(observed_at)
        result = replay([event], store=self.store, reference_time=now)
        self.assertIn("preferences", result.written_topics)

    # --- exemptions ---

    def test_identity_event_exempt_from_decay(self) -> None:
        now = datetime.now(UTC)
        observed_at = (now - timedelta(days=365)).isoformat(timespec="seconds").replace("+00:00", "Z")
        event = self._identity_event(observed_at)
        result = replay([event], store=self.store, reference_time=now)
        self.assertIn("identity", result.written_topics)
        text = (self.base / "identity.md").read_text(encoding="utf-8")
        self.assertIn("- Name: Shriyash", text)

    def test_migration_event_exempt_from_decay(self) -> None:
        now = datetime.now(UTC)
        observed_at = (now - timedelta(days=365)).isoformat(timespec="seconds").replace("+00:00", "Z")
        event = self._migration_event(observed_at)
        result = replay([event], store=self.store, reference_time=now)
        self.assertIn("preferences", result.written_topics)

    # --- reinforcement: fresh event of same key keeps key alive ---

    def test_reinforcement_keeps_key_alive(self) -> None:
        """If a newer event for the same key exists, the old one is superseded anyway,
        and the newer one (within decay window) renders."""
        now = datetime.now(UTC)
        old_at = (now - timedelta(days=60)).isoformat(timespec="seconds").replace("+00:00", "Z")
        new_at = (now - timedelta(days=5)).isoformat(timespec="seconds").replace("+00:00", "Z")
        old_event = self._pref_event(old_at)
        new_event = make_event(
            kind="preference",
            topic="preferences",
            key="preferences.response_style",
            value={"response_style": "detailed"},
            confidence=1.0,
            source="explicit",
            extractor="deterministic",
            session_id="s2",
            turn_id="t2",
            observed_at=new_at,
            applied=True,
        )
        result = replay([old_event, new_event], store=self.store, reference_time=now)
        self.assertIn("preferences", result.written_topics)
        text = (self.base / "preferences.md").read_text(encoding="utf-8")
        self.assertIn("- Response style: detailed", text)
        self.assertNotIn("- Response style: concise", text)

    # --- unapplied events are not affected by decay (they never render anyway) ---

    def test_unapplied_event_never_renders_regardless_of_age(self) -> None:
        now = datetime.now(UTC)
        fresh_at = (now - timedelta(days=1)).isoformat(timespec="seconds").replace("+00:00", "Z")
        event = self._pref_event(fresh_at, applied=False)
        result = replay([event], store=self.store, reference_time=now)
        self.assertNotIn("preferences", result.written_topics)


# ---------------------------------------------------------------------------
# Step 5: DreamPass
# ---------------------------------------------------------------------------

class DreamPassShouldRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"dreamp_sr_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.store = _make_store(self.base)
        self.journal = JournalStore(journal_dir=self.base / "journal")
        self.gate = _make_gate(self.base, self.journal, self.store)
        self.dp = _make_dream_pass(self.base, self.journal, self.store, self.gate)

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def _queue_n_candidates(self, n: int) -> None:
        for i in range(n):
            ev = _inferred_event(
                key="preferences.response_style",
                value={"response_style": f"style_{i}"},
                turn_id=f"t{i}",
                observed_at=f"2026-04-13T10:0{i}:00Z",
            )
            self.gate.queue_candidate(ev)

    def test_false_when_no_candidates(self) -> None:
        self.assertFalse(self.dp.should_run())

    def test_false_when_few_candidates_and_no_history(self) -> None:
        self._queue_n_candidates(1)
        # No last_run_at in state -> time trigger doesn't fire without prior run.
        self.assertFalse(self.dp.should_run())

    def test_true_when_enough_candidates(self) -> None:
        self._queue_n_candidates(3)
        self.assertTrue(self.dp.should_run())

    def test_true_when_time_elapsed_since_last_pass(self) -> None:
        self._queue_n_candidates(1)
        # Simulate a last run 25 hours ago
        old_ts = (
            (datetime.now(UTC) - timedelta(hours=25))
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        self.dp._state["last_run_at"] = old_ts
        self.assertTrue(self.dp.should_run())

    def test_false_when_recent_pass_and_few_candidates(self) -> None:
        self._queue_n_candidates(1)
        recent_ts = (
            (datetime.now(UTC) - timedelta(hours=1))
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        self.dp._state["last_run_at"] = recent_ts
        self.assertFalse(self.dp.should_run())


class DreamPassRunTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"dreamp_run_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.store = _make_store(self.base)
        self.journal = JournalStore(journal_dir=self.base / "journal")
        self.gate = _make_gate(self.base, self.journal, self.store)
        self.dp = _make_dream_pass(self.base, self.journal, self.store, self.gate)

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def _queue_candidate(self, key: str, value: dict, turn_id: str = "t1") -> str:
        ev = _inferred_event(key=key, value=value, turn_id=turn_id)
        self.gate.queue_candidate(ev)
        return ev.event_id

    async def test_skipped_when_no_model_and_no_override(self) -> None:
        self._queue_candidate("preferences.response_style", {"response_style": "concise"})
        result = await self.dp.run(session_id="s1")
        self.assertEqual(result.skipped_reason, "no_model")

    async def test_skipped_when_no_pending_candidates(self) -> None:
        result = await self.dp.run(
            session_id="s1",
            _decisions_override=[],
        )
        self.assertEqual(result.skipped_reason, "no_pending_candidates")

    async def test_promote_writes_applied_event_and_updates_topic(self) -> None:
        eid = self._queue_candidate(
            "preferences.response_style", {"response_style": "concise"}
        )
        result = await self.dp.run(
            session_id="s1",
            _decisions_override=[{"event_id": eid, "decision": "promote"}],
        )
        self.assertEqual(result.promoted_count, 1)
        self.assertEqual(result.dropped_count, 0)
        self.assertFalse(result.rolled_back)
        self.assertEqual(result.skipped_reason, "")

        # Topic file written
        prefs = (self.base / "preferences.md").read_text(encoding="utf-8")
        self.assertIn("- Response style: concise", prefs)
        # Removed from gate pending
        self.assertEqual(self.gate.pending_count(), 0)

    async def test_drop_writes_contradiction_and_excludes_from_topic(self) -> None:
        eid = self._queue_candidate(
            "preferences.response_style", {"response_style": "verbose"}
        )
        result = await self.dp.run(
            session_id="s1",
            _decisions_override=[{"event_id": eid, "decision": "drop"}],
        )
        self.assertEqual(result.promoted_count, 0)
        self.assertEqual(result.dropped_count, 1)
        self.assertFalse(result.rolled_back)

        # Topic file should NOT have been written (candidate never applied)
        self.assertFalse((self.base / "preferences.md").exists())
        # Removed from gate pending
        self.assertEqual(self.gate.pending_count(), 0)
        # A contradiction event is in the journal
        all_events = self.journal.load_all()
        contradiction_events = [e for e in all_events if e.kind == "contradiction"]
        self.assertGreater(len(contradiction_events), 0)

    async def test_promote_multiple_candidates(self) -> None:
        eid1 = self._queue_candidate(
            "preferences.response_style",
            {"response_style": "concise"},
            turn_id="t1",
        )
        eid2 = self._queue_candidate(
            "preferences.humor_level",
            {"humor_level": "low"},
            turn_id="t2",
        )
        result = await self.dp.run(
            session_id="s1",
            _decisions_override=[
                {"event_id": eid1, "decision": "promote"},
                {"event_id": eid2, "decision": "promote"},
            ],
        )
        self.assertEqual(result.promoted_count, 2)
        self.assertEqual(self.gate.pending_count(), 0)
        prefs = (self.base / "preferences.md").read_text(encoding="utf-8")
        self.assertIn("- Response style: concise", prefs)
        self.assertIn("- Humor level: low", prefs)

    async def test_unknown_event_id_in_decisions_is_skipped(self) -> None:
        eid = self._queue_candidate(
            "preferences.response_style", {"response_style": "concise"}
        )
        result = await self.dp.run(
            session_id="s1",
            _decisions_override=[
                {"event_id": "does-not-exist", "decision": "promote"},
                {"event_id": eid, "decision": "promote"},
            ],
        )
        # Only the known candidate is counted
        self.assertEqual(result.promoted_count, 1)

    async def test_state_file_updated_after_successful_run(self) -> None:
        eid = self._queue_candidate(
            "preferences.response_style", {"response_style": "concise"}
        )
        await self.dp.run(
            session_id="s_state_test",
            _decisions_override=[{"event_id": eid, "decision": "promote"}],
        )
        state_path = self.base / "dream_pass_state.json"
        self.assertTrue(state_path.exists())
        import json
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertIn("last_run_at", state)
        self.assertEqual(state.get("last_session_id"), "s_state_test")
        self.assertEqual(state.get("pass_count"), 1)

    async def test_rollback_on_empty_decisions_leaves_state_unchanged(self) -> None:
        eid = self._queue_candidate(
            "preferences.response_style", {"response_style": "concise"}
        )
        result = await self.dp.run(
            session_id="s1",
            _decisions_override=[{"event_id": eid, "decision": "unknown_decision"}],
        )
        self.assertEqual(result.skipped_reason, "no_valid_decisions")
        self.assertFalse(result.rolled_back)

    async def test_promoted_event_uses_dream_pass_extractor(self) -> None:
        eid = self._queue_candidate(
            "preferences.response_style", {"response_style": "concise"}
        )
        await self.dp.run(
            session_id="s1",
            _decisions_override=[{"event_id": eid, "decision": "promote"}],
        )
        promoted = [
            e for e in self.journal.load_all()
            if e.extractor == "dream_pass" and e.applied
        ]
        self.assertGreater(len(promoted), 0)
        self.assertEqual(promoted[0].source, "synthesized")


class DreamPassSanityCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"dreamp_sanity_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.store = _make_store(self.base)

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def test_no_failures_on_clean_state(self) -> None:
        self.store.write_topic(
            "preferences",
            ["- Response style: concise", "- Humor level: low"],
            {"title": "Preferences"},
        )
        self.store.update_index_entry("preferences", "Prefs summary")
        before = _measure_topic_line_counts(self.store)
        failures = _run_sanity_checks(
            self.store,
            before_line_counts=before,
            before_index_count=1,
        )
        self.assertEqual(failures, [])

    def test_detects_excessive_shrinkage(self) -> None:
        # Write a topic with 8 lines, then remove it entirely
        lines = [f"- Line {i}: value_{i}" for i in range(8)]
        self.store.write_topic("preferences", lines, {"title": "Preferences"})
        self.store.update_index_entry("preferences", "Prefs summary")
        before = _measure_topic_line_counts(self.store)
        # Simulate shrinkage by overwriting with 1 line
        self.store.write_topic("preferences", ["- Response style: x"], {"title": "Preferences"})
        failures = _run_sanity_checks(
            self.store,
            before_line_counts=before,
            before_index_count=1,
        )
        self.assertGreater(len(failures), 0)
        self.assertTrue(any("preferences" in f for f in failures))

    def test_detects_index_count_drop(self) -> None:
        # Write 3 topics, then simulate MEMORY.md with only 1 entry
        for topic in ("identity", "preferences", "workflow"):
            self.store.write_topic(topic, ["- A: b"], {"title": topic.title()})
            self.store.update_index_entry(topic, f"{topic} summary")
        before = _measure_topic_line_counts(self.store)
        # Overwrite MEMORY.md with a single entry
        import json as _json
        (self.base / "MEMORY.md").write_text(
            "- [Identity](identity.md) - Identity summary\n", encoding="utf-8"
        )
        failures = _run_sanity_checks(
            self.store,
            before_line_counts=before,
            before_index_count=3,
        )
        self.assertTrue(any("index" in f.lower() or "MEMORY" in f for f in failures))


class DreamPassSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"dreamp_snap_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.store = _make_store(self.base)

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def test_snapshot_and_restore_round_trip(self) -> None:
        self.store.write_topic(
            "preferences",
            ["- Response style: concise"],
            {"title": "Preferences"},
        )
        self.store.update_index_entry("preferences", "Prefs summary")
        original_text = (self.base / "preferences.md").read_text(encoding="utf-8")

        snap_dir = _take_snapshot(self.store, self.base / "snapshots")

        # Mutate the file
        self.store.write_topic(
            "preferences",
            ["- Response style: verbose"],
            {"title": "Preferences"},
        )
        mutated = (self.base / "preferences.md").read_text(encoding="utf-8")
        self.assertNotEqual(original_text, mutated)

        # Restore
        _restore_snapshot(self.store, snap_dir)
        restored = (self.base / "preferences.md").read_text(encoding="utf-8")
        self.assertEqual(original_text, restored)

    def test_restore_deletes_files_not_in_snapshot(self) -> None:
        """Files that didn't exist at snapshot time are removed on restore."""
        snap_dir = _take_snapshot(self.store, self.base / "snapshots")

        # Create a new topic file after the snapshot
        self.store.write_topic(
            "preferences",
            ["- Response style: concise"],
            {"title": "Preferences"},
        )
        self.assertTrue((self.base / "preferences.md").exists())

        _restore_snapshot(self.store, snap_dir)
        self.assertFalse((self.base / "preferences.md").exists())


class ParseDecisionsTests(unittest.TestCase):
    def test_parses_clean_array(self) -> None:
        raw = '[{"event_id": "abc", "decision": "promote", "supersedes_existing": null}]'
        result = _parse_decisions(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["event_id"], "abc")
        self.assertEqual(result[0]["decision"], "promote")

    def test_strips_prose_around_json(self) -> None:
        raw = 'Sure, here you go:\n[{"event_id": "x", "decision": "drop"}]\nDone.'
        result = _parse_decisions(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["decision"], "drop")

    def test_drops_unknown_decisions(self) -> None:
        raw = '[{"event_id": "a", "decision": "merge"}, {"event_id": "b", "decision": "promote"}]'
        result = _parse_decisions(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["event_id"], "b")

    def test_empty_string_returns_empty(self) -> None:
        self.assertEqual(_parse_decisions(""), [])

    def test_invalid_json_returns_empty(self) -> None:
        self.assertEqual(_parse_decisions("not json at all"), [])


class WriteRollbackEventsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"rollback_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.journal = JournalStore(journal_dir=self.base / "journal")

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def test_writes_one_rollback_event_per_promoted(self) -> None:
        promoted = make_event(
            kind="preference",
            topic="preferences",
            key="preferences.response_style",
            value={"response_style": "concise"},
            confidence=0.9,
            source="synthesized",
            extractor="dream_pass",
            session_id="s1",
            turn_id="t1",
            applied=True,
        )
        self.journal.append(promoted)
        _write_rollback_events(
            journal=self.journal,
            promoted_events=[promoted],
            session_id="s1",
            sanity_failures=["test failure"],
        )
        all_events = self.journal.load_all()
        rollback_events = [e for e in all_events if e.event_id.startswith("rollbk_")]
        self.assertEqual(len(rollback_events), 1)
        rb = rollback_events[0]
        self.assertEqual(rb.kind, "contradiction")
        self.assertEqual(rb.supersedes, promoted.event_id)
        self.assertFalse(rb.applied)
        self.assertTrue(rb.value.get("rolled_back"))

    def test_rollback_events_honored_by_replayer(self) -> None:
        """Promoted event superseded by rollback event must not render in topic files."""
        store = PersonalMemoryStore(
            base_dir=self.base,
            index_path=self.base / "MEMORY.md",
            logs_dir=self.base / "logs",
            topic_paths={
                "identity": self.base / "identity.md",
                "preferences": self.base / "preferences.md",
                "workflow": self.base / "workflow.md",
                "contacts": self.base / "contacts.md",
                "projects": self.base / "projects.md",
                "corrections": self.base / "corrections.md",
            },
        )
        promoted = make_event(
            kind="preference",
            topic="preferences",
            key="preferences.response_style",
            value={"response_style": "concise"},
            confidence=0.9,
            source="synthesized",
            extractor="dream_pass",
            session_id="s1",
            turn_id="t1",
            applied=True,
        )
        self.journal.append(promoted)
        _write_rollback_events(
            journal=self.journal,
            promoted_events=[promoted],
            session_id="s1",
            sanity_failures=["test failure"],
        )
        result = replay(self.journal.load_all(), store=store)
        self.assertNotIn("preferences", result.written_topics)
        self.assertFalse((self.base / "preferences.md").exists())


if __name__ == "__main__":
    unittest.main()
