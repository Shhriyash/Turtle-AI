"""
test/wp3d_projection_transaction_test.py
-----------------------------------------
WP3.D (ledger 3.8, transaction half): core/memory_replayer.py::replay()
writes all ~11 topics in one backend transaction so a crash partway through
cannot leave some topics reflecting the new journal state and others stale.

No live Postgres is reachable in this environment, so this file proves the
behaviour two ways:

1. A hand-rolled fake psycopg pool/connection (`_FakePool`/`_FakeConn` below)
   that MODELS real commit/rollback semantics: `execute()` writes go into a
   per-connection staging dict, and are only merged into the shared
   "committed" store on clean `with conn:` exit, or discarded entirely if the
   block raises — exactly the "commit/rollback the transaction in case of
   success/error" contract `psycopg_pool.ConnectionPool.connection()`
   documents (verified by reading its source in this environment; see
   core/storage/cloud/personal_memory_store.py's own comments). This is a
   model of Postgres transaction behaviour, not Postgres itself.
2. `test_replay_is_atomic_against_real_postgres` under `test/cloud_integration/`
   (separate file, `@pytest.mark.cloud`) exercises the exact same scenario
   against a real Postgres connection and self-skips when DATABASE_URL is
   unset. It was NOT run in this sandbox (no live Postgres available here);
   only the fake-pool tests below were actually executed.

The first test class (`ReplayPartialWriteBugTest`) reproduces the ORIGINAL
defect directly against `PostgresPersonalMemoryBackend.write_topic`/
`delete_topic` bypassing `transaction()` (i.e. the shape every call had before
this ledger item): each topic write opens+commits its own connection, so an
injected failure on the Nth call leaves 1..N-1 committed. This documents the
defect and stays green forever as a regression guard on that documented
per-call behaviour — the FIX is not "every write is transactional", it is
"replay() now wraps its whole loop in one transaction", covered by the second
class.

The second class (`ReplayAtomicityFixTest`) drives `core.memory_replayer.replay()`
itself (its real per-topic loop) against the fake Postgres-shaped backend and
proves an injected failure on the 5th topic write leaves NO topic committed —
the ledger's own acceptance wording ("an injected failure on the fifth topic
write changes no topics").
"""
from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from core.memory_journal import MemoryEvent
from core.memory_replayer import ALL_TOPICS, replay
from core.personal_memory_store import PersonalMemoryStore
from core.storage.cloud.personal_memory_store import PostgresPersonalMemoryBackend

REFERENCE_TIME = datetime(2026, 9, 26, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fake psycopg pool/connection modeling real commit/rollback semantics.
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, row, rowcount=0):
        self._row = row
        self.rowcount = rowcount

    def fetchone(self):
        return self._row


class _FakeConn:
    """Models ONE checked-out psycopg connection's transaction. Writes land
    in a private staging dict (visible to further calls on this SAME
    connection, matching Postgres's own-transaction read-your-writes), and
    are only merged into the pool's shared "committed" store when this
    connection's `with conn:` block exits cleanly; an exception discards the
    staged writes entirely (rollback) — mirroring the psycopg_pool.
    ConnectionPool.connection() docstring's "commit/rollback the transaction
    in case of success/error" contract read from the installed package in
    this environment.
    """

    def __init__(self, committed_topics: dict, committed_logs: dict):
        self._committed_topics = committed_topics
        self._committed_logs = committed_logs
        # Start each transaction's local view from the committed state so
        # reads-before-write inside one transaction still see prior commits.
        self.topics = dict(committed_topics)
        self.logs = dict(committed_logs)

    def execute(self, sql: str, params=None):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("SELECT content FROM personal_memory_topics"):
            content = self.topics.get((params[0], params[1]))
            return _FakeCursor((content,) if content is not None else None)
        if sql_norm.startswith("INSERT INTO personal_memory_topics"):
            user_id, topic_name, content = params
            self.topics[(user_id, topic_name)] = content
            return _FakeCursor(None)
        if sql_norm.startswith("DELETE FROM personal_memory_topics"):
            existed = (params[0], params[1]) in self.topics
            self.topics.pop((params[0], params[1]), None)
            return _FakeCursor(None, rowcount=1 if existed else 0)
        if sql_norm.startswith("SELECT content FROM personal_memory_daily_logs"):
            content = self.logs.get((params[0], params[1]))
            return _FakeCursor((content,) if content is not None else None)
        if sql_norm.startswith("INSERT INTO personal_memory_daily_logs"):
            user_id, log_date, content = params
            self.logs[(user_id, log_date)] = content
            return _FakeCursor(None)
        if sql_norm.startswith("SELECT COALESCE(SUM(LENGTH(content)), 0) FROM personal_memory_topics"):
            total = sum(len(v) for (uid, _), v in self.topics.items() if uid == params[0])
            return _FakeCursor((total,))
        if sql_norm.startswith("SELECT COALESCE(SUM(LENGTH(content)), 0) FROM personal_memory_daily_logs"):
            total = sum(len(v) for (uid, _), v in self.logs.items() if uid == params[0])
            return _FakeCursor((total,))
        return _FakeCursor(None)  # CREATE TABLE

    # --- connection-context-manager semantics (commit / rollback) ---
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._committed_topics.clear()
            self._committed_topics.update(self.topics)
            self._committed_logs.clear()
            self._committed_logs.update(self.logs)
        # else: exception -> staged self.topics/self.logs are simply
        # discarded, nothing merged into the committed dicts (rollback).
        return False


class _FakeConnCtx:
    """Models `ConnectionPool.getconn()`/`putconn()` bracketing a `with
    conn:` block, per psycopg_pool.ConnectionPool.connection()'s own source
    (read in this environment):
        conn = self.getconn(...)
        try:
            with conn:
                yield conn
        finally:
            self.putconn(conn)
    """

    def __init__(self, conn: _FakeConn):
        self._conn = conn

    def __enter__(self):
        self._conn.__enter__()
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        return self._conn.__exit__(exc_type, exc, tb)


class _FakePool:
    def __init__(self):
        self.topics: dict = {}
        self.logs: dict = {}

    def connection(self):
        return _FakeConnCtx(_FakeConn(self.topics, self.logs))


def _patch_pool():
    import core.storage.cloud.personal_memory_store as pms

    pms._initialized = False
    return patch("core.storage.cloud.personal_memory_store.get_pg_sync_pool")


def _make_event(topic: str, key: str, statement: str, event_id: str) -> MemoryEvent:
    return MemoryEvent(
        event_id=event_id,
        session_id="s1",
        turn_id="t1",
        observed_at="2026-09-20T10:00:00Z",
        kind="fact",
        topic=topic,
        key=key,
        value={},
        confidence=0.9,
        source="explicit",
        extractor="llm_turn",
        applied=True,
        statement=statement,
    )


class ReplayPartialWriteBugTest(unittest.TestCase):
    """Reproduces the ORIGINAL defect: before this ledger item, every
    PostgresPersonalMemoryBackend.write_topic/delete_topic call opened and
    committed its OWN connection. This test drives the backend directly
    (bypassing `transaction()`) to document that shape stays true for any
    caller that doesn't opt into a transaction — replay() is the one caller
    ledger 3.8 fixes, not this backend method's own per-call contract."""

    def setUp(self) -> None:
        self.pool = _FakePool()
        patcher = _patch_pool()
        self.addCleanup(patcher.stop)
        mock_get = patcher.start()
        mock_get.return_value = self.pool

    def test_injected_failure_on_nth_call_leaves_earlier_calls_committed(self) -> None:
        backend = PostgresPersonalMemoryBackend("usr_repro")
        topics = ["identity", "preferences", "workflow", "contacts", "projects", "corrections"]

        real_write_topic = PostgresPersonalMemoryBackend.write_topic
        call_count = {"n": 0}

        def flaky_write_topic(self, topic_name, content):
            call_count["n"] += 1
            if call_count["n"] == 5:
                raise RuntimeError("simulated crash on the 5th topic write")
            return real_write_topic(self, topic_name, content)

        with patch.object(PostgresPersonalMemoryBackend, "write_topic", flaky_write_topic):
            for i, topic in enumerate(topics):
                if i == 4:  # 5th call (0-indexed) raises
                    with self.assertRaises(RuntimeError):
                        backend.write_topic(topic, f"content-{topic}")
                    break
                backend.write_topic(topic, f"content-{topic}")

        # THIS is the bug: topics 1-4 already landed in the "committed" store
        # even though the 5th call blew up and nothing after it ran.
        committed = {name for (_, name) in self.pool.topics}
        self.assertEqual(committed, {"identity", "preferences", "workflow", "contacts"})
        self.assertNotIn("projects", committed)
        self.assertNotIn("corrections", committed)


class ReplayAtomicityFixTest(unittest.TestCase):
    """Drives the FIXED replay() (its real per-topic loop, wrapped in
    `_topic_write_transaction`) against the fake Postgres-shaped backend and
    proves an injected failure on the 5th topic write leaves NO topic
    committed — the ledger's own acceptance criterion."""

    def setUp(self) -> None:
        self.pool = _FakePool()
        patcher = _patch_pool()
        self.addCleanup(patcher.stop)
        mock_get = patcher.start()
        mock_get.return_value = self.pool

    def _cloud_store(self) -> PersonalMemoryStore:
        with patch("core.personal_memory_store.settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.user_storage_cap_mb = 0  # disable cap for this test
            return PersonalMemoryStore(user_id="usr_atomic")

    def _events_for_five_plus_topics(self) -> list[MemoryEvent]:
        # ALL_TOPICS order (core.memory_schema.TOPICS): identity, preferences,
        # workflow, contacts, projects, corrections, relations, ... — give the
        # first 6 of those real content so "the 5th topic write" (projects)
        # is unambiguous and there is at least one topic after it (corrections)
        # that must ALSO never be written once the 5th call raises.
        assert list(ALL_TOPICS[:6]) == [
            "identity",
            "preferences",
            "workflow",
            "contacts",
            "projects",
            "corrections",
        ]
        return [
            _make_event("identity", "identity.name", "Name: Alice", "ev1"),
            _make_event("preferences", "preferences.response_style", "Response style: concise", "ev2"),
            _make_event("workflow", "workflow.prefers_draft_before_send", "Prefers draft before send: true", "ev3"),
            _make_event("contacts", "contacts.frequent_recipient.bob", "Frequent recipient: bob@example.com", "ev4"),
            _make_event("projects", "projects.turtle", "Project: Turtle", "ev5"),
            _make_event("corrections", "corrections.name_role", "Correction: role is engineer", "ev6"),
        ]

    def test_pre_fix_sanity_without_transaction_wrapper_would_partially_commit(self) -> None:
        """Control case: confirms the fake backend/pool actually behaves like
        the pre-fix world when nothing opens a `transaction()` — each
        `write_topic` call commits for itself. This isolates the fix's effect
        (the next test) from the fake's own plumbing."""
        store = self._cloud_store()
        real_write_topic = PostgresPersonalMemoryBackend.write_topic
        call_count = {"n": 0}

        def flaky_write_topic(self, topic_name, content):
            call_count["n"] += 1
            if call_count["n"] == 5:
                raise RuntimeError("simulated crash on the 5th topic write")
            return real_write_topic(self, topic_name, content)

        with patch.object(PostgresPersonalMemoryBackend, "write_topic", flaky_write_topic):
            with self.assertRaises(RuntimeError):
                for topic in ["identity", "preferences", "workflow", "contacts", "projects", "corrections"]:
                    store._backend._pg.write_topic(topic, f"content-{topic}")

        committed = {name for (_, name) in self.pool.topics}
        self.assertEqual(committed, {"identity", "preferences", "workflow", "contacts"})

    def test_injected_failure_on_fifth_topic_write_changes_no_topics(self) -> None:
        store = self._cloud_store()
        events = self._events_for_five_plus_topics()

        real_write_topic = PostgresPersonalMemoryBackend.write_topic
        call_count = {"n": 0}

        def flaky_write_topic(self, topic_name, content):
            call_count["n"] += 1
            if call_count["n"] == 5:
                raise RuntimeError("simulated crash on the 5th topic write")
            return real_write_topic(self, topic_name, content)

        with patch.object(PostgresPersonalMemoryBackend, "write_topic", flaky_write_topic):
            with self.assertRaises(RuntimeError):
                replay(events, store=store, reference_time=REFERENCE_TIME)

        # The fix's whole point: NOTHING committed, not even the first 4
        # topics that succeeded before the injected failure.
        self.assertEqual(self.pool.topics, {})
        self.assertEqual(call_count["n"], 5)

    def test_without_injected_failure_replay_still_commits_everything(self) -> None:
        """Sanity check that wrapping the loop in a transaction doesn't
        somehow suppress a normal, successful replay."""
        store = self._cloud_store()
        events = self._events_for_five_plus_topics()

        result = replay(events, store=store, reference_time=REFERENCE_TIME)

        committed = {name for (_, name) in self.pool.topics if name != "__index__"}
        self.assertEqual(
            committed, {"identity", "preferences", "workflow", "contacts", "projects", "corrections"}
        )
        self.assertEqual(
            set(result.written_topics),
            {"identity", "preferences", "workflow", "contacts", "projects", "corrections"},
        )


if __name__ == "__main__":
    unittest.main()
