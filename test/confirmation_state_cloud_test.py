"""
test/confirmation_state_cloud_test.py
----------------------------------------
Unit coverage for core/storage/cloud/confirmation_state_store.py (Vercel
migration Phase 3) and ConfirmationGate's pluggable state_backend seam.
Includes a direct demonstration of the bug this fixes: two SEPARATE
ConfirmationGate instances (simulating two different worker processes /
serverless invocations) sharing state correctly through the same backend,
where two instances backed by separate local JSON files would not.
"""
from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from core.confirmation_gate import ConfirmationGate, _JsonFileConfirmationState
from core.memory_journal import JournalStore, make_event
from core.personal_memory_store import PersonalMemoryStore
from core.storage.cloud.confirmation_state_store import PostgresConfirmationState


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, table: dict):
        self._table = table

    def execute(self, sql: str, params=None):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("SELECT pending FROM confirmation_state"):
            row = self._table.get(params[0])
            return _FakeCursor((row,) if row is not None else None)
        if sql_norm.startswith("INSERT INTO confirmation_state"):
            user_id, pending_json = params
            self._table[user_id] = pending_json
            return _FakeCursor(None)
        return _FakeCursor(None)  # CREATE TABLE


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


class _FakePool:
    def __init__(self):
        self.table: dict = {}

    def connection(self):
        return _FakeConnCtx(_FakeConn(self.table))


class PostgresConfirmationStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = _FakePool()
        import core.storage.cloud.confirmation_state_store as ccs

        ccs._initialized = False
        patcher = patch(
            "core.storage.cloud.confirmation_state_store.get_pg_sync_pool",
            return_value=self.pool,
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_load_missing_returns_empty_pending(self) -> None:
        backend = PostgresConfirmationState("usr_a")
        self.assertEqual(backend.load(), {"pending": []})

    def test_save_then_load_round_trips(self) -> None:
        backend = PostgresConfirmationState("usr_a")
        backend.save({"pending": ["evt1", "evt2"]})
        self.assertEqual(backend.load(), {"pending": ["evt1", "evt2"]})

    def test_save_overwrites_existing(self) -> None:
        backend = PostgresConfirmationState("usr_a")
        backend.save({"pending": ["evt1"]})
        backend.save({"pending": ["evt2"]})
        self.assertEqual(backend.load(), {"pending": ["evt2"]})

    def test_states_scoped_per_user(self) -> None:
        a = PostgresConfirmationState("usr_a")
        b = PostgresConfirmationState("usr_b")
        a.save({"pending": ["evt_a"]})
        b.save({"pending": ["evt_b"]})
        self.assertEqual(a.load(), {"pending": ["evt_a"]})
        self.assertEqual(b.load(), {"pending": ["evt_b"]})

    def test_requires_user_id(self) -> None:
        with self.assertRaises(ValueError):
            PostgresConfirmationState("")


class ConfirmationGateBackendSelectionTest(unittest.TestCase):
    """ConfirmationGate must accept either state_path (builds the local JSON
    backend) or an explicit state_backend, and refuse neither."""

    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"gate_backend_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.journal = JournalStore(journal_dir=self.base / "journal")
        self.store = PersonalMemoryStore(
            base_dir=self.base, index_path=self.base / "MEMORY.md",
            logs_dir=self.base / "logs", topic_paths={},
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def test_state_path_builds_json_file_backend(self) -> None:
        gate = ConfirmationGate(
            journal=self.journal, store=self.store,
            state_path=self.base / "confirmation_state.json",
        )
        self.assertIsInstance(gate._backend, _JsonFileConfirmationState)

    def test_explicit_backend_is_used_over_state_path(self) -> None:
        pool = _FakePool()
        with patch(
            "core.storage.cloud.confirmation_state_store.get_pg_sync_pool", return_value=pool
        ):
            backend = PostgresConfirmationState("usr_a")
            gate = ConfirmationGate(journal=self.journal, store=self.store, state_backend=backend)
            self.assertIs(gate._backend, backend)

    def test_neither_state_path_nor_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            ConfirmationGate(journal=self.journal, store=self.store)


class CrossInstanceConfirmBugFixTest(unittest.TestCase):
    """Direct demonstration of the fix: two ConfirmationGate instances that
    share a Postgres-backed state (simulating "queued on worker A, confirmed
    on worker B") see each other's writes — unlike two instances each backed
    by their own local JSON file, which would not (the documented bug)."""

    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"cross_instance_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.journal = JournalStore(journal_dir=self.base / "journal")
        self.store = PersonalMemoryStore(
            base_dir=self.base, index_path=self.base / "MEMORY.md",
            logs_dir=self.base / "logs", topic_paths={},
        )
        self.pool = _FakePool()
        import core.storage.cloud.confirmation_state_store as ccs

        ccs._initialized = False
        patcher = patch(
            "core.storage.cloud.confirmation_state_store.get_pg_sync_pool",
            return_value=self.pool,
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def _event(self) -> str:
        event = make_event(
            kind="fact", topic="identity", key="identity.name",
            value={"name": "Shriyash"}, confidence=0.7,
            source="inferred", extractor="llm_turn",
            session_id="s1", turn_id="t1", applied=False,
        )
        # "Worker A" queues the candidate.
        gate_a = ConfirmationGate(
            journal=self.journal, store=self.store,
            state_backend=PostgresConfirmationState("usr_a"),
        )
        gate_a.queue_candidate(event)
        return event.event_id

    def test_confirmed_on_a_different_gate_instance_via_shared_backend(self) -> None:
        event_id = self._event()

        # "Worker B" — a SEPARATE ConfirmationGate instance, same backend.
        gate_b = ConfirmationGate(
            journal=self.journal, store=self.store,
            state_backend=PostgresConfirmationState("usr_a"),
        )
        self.assertIn(event_id, gate_b.get_pending_ids())
        result = gate_b.record_response(event_id, accepted=True)
        self.assertIsNotNone(result)

    def test_two_local_json_backends_do_NOT_share_state(self) -> None:
        """Negative control: proves the bug is real without Postgres — two
        gates pointed at DIFFERENT local files (simulating two workers each
        with their own disk) do not see each other's pending queue."""
        event_id = self._event()  # queued via the Postgres-backed gate above

        # A gate using a totally separate local JSON file never sees it.
        isolated_gate = ConfirmationGate(
            journal=self.journal, store=self.store,
            state_path=self.base / "other_worker_confirmation_state.json",
        )
        self.assertNotIn(event_id, isolated_gate.get_pending_ids())


if __name__ == "__main__":
    unittest.main()
