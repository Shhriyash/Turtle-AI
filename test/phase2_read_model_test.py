"""Phase 2 W3 — the SQLite index consolidated into the read model.

Covers the schema migration (idempotent, survives old and new DBs), the derived
read-model columns (statement/status/superseded_by), the read-model queries
(event_exists/get_event/latest_for_key/events_for_key), and confirmation-gate
parity: the gate must behave bit-identically with and without the index wired.

Everything is tmp-isolated (db_path / journal_dir / store paths) so no test
touches data/.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from core.confirmation_gate import ConfirmationGate
from core.memory_journal import JournalStore, MemoryEvent
from core.memory_sqlite import MemorySQLiteIndex
from core.personal_memory_store import PersonalMemoryStore


# The events table exactly as it stood before Phase 2 W3 added the read-model
# columns. Used to prove a production DB predating the migration survives it.
_OLD_SCHEMA = """
CREATE TABLE events (
    event_id      TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    turn_id       TEXT NOT NULL,
    observed_at   TEXT NOT NULL,
    kind          TEXT NOT NULL,
    topic         TEXT NOT NULL,
    key           TEXT NOT NULL,
    value_json    TEXT NOT NULL,
    value_text    TEXT NOT NULL,
    confidence    REAL NOT NULL,
    source        TEXT NOT NULL,
    extractor     TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    evidence_text TEXT NOT NULL DEFAULT '',
    supersedes    TEXT,
    applied       INTEGER NOT NULL,
    rejected      INTEGER NOT NULL
);
"""


def _event(
    event_id: str,
    *,
    topic: str = "preferences",
    key: str = "preferences.response_style",
    value: dict | None = None,
    kind: str = "preference",
    source: str = "inferred",
    extractor: str = "llm_turn",
    confidence: float = 0.75,
    observed_at: str = "2026-07-17T09:00:00Z",
    session_id: str = "s1",
    turn_id: str = "t1",
    supersedes: str | None = None,
    applied: bool = False,
    rejected: bool = False,
    statement: str = "",
) -> MemoryEvent:
    return MemoryEvent(
        event_id=event_id,
        session_id=session_id,
        turn_id=turn_id,
        observed_at=observed_at,
        kind=kind,
        topic=topic,
        key=key,
        value=value if value is not None else {"response_style": "concise"},
        confidence=confidence,
        source=source,
        extractor=extractor,
        evidence={},
        supersedes=supersedes,
        applied=applied,
        rejected=rejected,
        statement=statement,
    )


def _make_store(base) -> PersonalMemoryStore:
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


# --------------------------------------------------------------------------- #
# (a) migration idempotence                                                   #
# --------------------------------------------------------------------------- #

def test_migration_adds_columns_over_old_schema_and_is_idempotent(tmp_path):
    db_path = tmp_path / "legacy.sqlite"

    # Stand up a pre-migration DB with one row and no read-model columns.
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_OLD_SCHEMA)
    conn.execute(
        """
        INSERT INTO events (
            event_id, session_id, turn_id, observed_at, kind, topic, key,
            value_json, value_text, confidence, source, extractor,
            evidence_json, evidence_text, supersedes, applied, rejected
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy-1", "s0", "t0", "2026-01-01T00:00:00Z", "fact",
            "identity", "identity.name", json.dumps({"name": "Shriyash"}),
            "Shriyash", 1.0, "explicit", "deterministic", "{}", "", None, 1, 0,
        ),
    )
    conn.commit()
    conn.close()

    # First open migrates the columns in.
    idx = MemorySQLiteIndex(db_path=db_path)
    columns = {row["name"] for row in idx._conn.execute("PRAGMA table_info(events)").fetchall()}
    assert {"statement", "status", "superseded_by"} <= columns

    # Existing row is still readable; the new columns backfilled to ''.
    legacy = idx.get_event("legacy-1")
    assert legacy is not None
    assert legacy.key == "identity.name"
    assert legacy.value == {"name": "Shriyash"}
    assert legacy.statement == ""
    assert legacy.status == ""
    assert legacy.superseded_by == ""
    assert idx.count() == 1
    idx.close()

    # Second instantiation over the already-migrated file is a no-op.
    idx2 = MemorySQLiteIndex(db_path=db_path)
    columns2 = {row["name"] for row in idx2._conn.execute("PRAGMA table_info(events)").fetchall()}
    assert {"statement", "status", "superseded_by"} <= columns2
    assert idx2.count() == 1
    assert idx2.get_event("legacy-1") is not None
    idx2.close()


# --------------------------------------------------------------------------- #
# (b) status / statement populated on index_event                             #
# --------------------------------------------------------------------------- #

def test_index_event_populates_statement_and_status(tmp_path):
    idx = MemorySQLiteIndex(db_path=tmp_path / "m.sqlite")

    idx.index_event(_event("applied-1", applied=True, statement="Response style: concise"))
    idx.index_event(_event("pending-1", applied=False))
    idx.index_event(_event("rejected-1", applied=False, rejected=True))

    applied = idx.get_event("applied-1")
    assert applied.status == "applied"
    assert applied.statement == "Response style: concise"

    pending = idx.get_event("pending-1")
    assert pending.status == "pending"
    assert pending.statement == ""

    rejected = idx.get_event("rejected-1")
    assert rejected.status == "rejected"

    assert idx.event_exists("applied-1") is True
    assert idx.event_exists("nope") is False
    idx.close()


# --------------------------------------------------------------------------- #
# (c) superseded_by set via backfill — tombstones and supersedes chains       #
# --------------------------------------------------------------------------- #

def test_backfill_sets_superseded_by_for_tombstones(tmp_path):
    journal = JournalStore(journal_dir=tmp_path / "j")
    original = journal.append(_event("orig-1", applied=True))
    tombstone = journal.append_rejection(original)

    idx = MemorySQLiteIndex(db_path=tmp_path / "m.sqlite")
    idx.backfill_from_journal(journal)

    row = idx.get_event("orig-1")
    assert row.superseded_by == tombstone.event_id
    assert row.status == "rejected"
    idx.close()


def test_backfill_sets_superseded_by_for_supersedes_chain(tmp_path):
    journal = JournalStore(journal_dir=tmp_path / "j")
    journal.append(_event("a-1", value={"response_style": "verbose"}, applied=True,
                          observed_at="2026-07-17T09:00:00Z"))
    journal.append(_event("b-1", value={"response_style": "concise"}, applied=True,
                          observed_at="2026-07-17T10:00:00Z", supersedes="a-1"))

    idx = MemorySQLiteIndex(db_path=tmp_path / "m.sqlite")
    idx.backfill_from_journal(journal)

    assert idx.get_event("a-1").superseded_by == "b-1"
    assert idx.get_event("b-1").superseded_by == ""
    idx.close()


# --------------------------------------------------------------------------- #
# (d) latest_for_key returns newest applied non-superseded row                 #
# --------------------------------------------------------------------------- #

def test_latest_for_key_returns_newest_non_superseded(tmp_path):
    journal = JournalStore(journal_dir=tmp_path / "j")
    journal.append(_event("a-1", value={"response_style": "verbose"}, applied=True,
                          observed_at="2026-07-17T09:00:00Z"))
    journal.append(_event("b-1", value={"response_style": "concise"}, applied=True,
                          observed_at="2026-07-17T10:00:00Z", supersedes="a-1"))

    idx = MemorySQLiteIndex(db_path=tmp_path / "m.sqlite")
    idx.backfill_from_journal(journal)

    latest = idx.latest_for_key("preferences", "preferences.response_style")
    assert latest is not None
    assert latest.event_id == "b-1"
    assert latest.value == {"response_style": "concise"}

    # A rejected latest is excluded even without a superseding row.
    journal2 = JournalStore(journal_dir=tmp_path / "j2")
    only = journal2.append(_event("only-1", applied=True))
    journal2.append_rejection(only)
    idx2 = MemorySQLiteIndex(db_path=tmp_path / "m2.sqlite")
    idx2.backfill_from_journal(journal2)
    assert idx2.latest_for_key("preferences", "preferences.response_style") is None

    idx.close()
    idx2.close()


def test_events_for_key_scopes_to_topic_key(tmp_path):
    idx = MemorySQLiteIndex(db_path=tmp_path / "m.sqlite")
    idx.index_event(_event("k1", key="preferences.response_style", applied=True,
                           observed_at="2026-07-17T09:00:00Z"))
    idx.index_event(_event("k2", key="preferences.response_style",
                           kind="contradiction", value={"rejected": True},
                           observed_at="2026-07-17T10:00:00Z"))
    idx.index_event(_event("other", key="preferences.humor_level",
                           value={"humor_level": "low"}, applied=True))

    rows = idx.events_for_key("preferences", "preferences.response_style")
    assert [r.event_id for r in rows] == ["k1", "k2"]  # oldest-first

    # limit keeps the most-recent rows but still hands them back oldest-first.
    limited = idx.events_for_key("preferences", "preferences.response_style", limit=1)
    assert [r.event_id for r in limited] == ["k2"]
    idx.close()


# --------------------------------------------------------------------------- #
# (e) gate parity — identical outcomes and journal contents with/without index #
# --------------------------------------------------------------------------- #

def _run_representative_flow(gate: ConfirmationGate) -> list:
    """A silence/queue/accept flow exercising every routed lookup.

    Returns a list of deterministic outcome tuples so two gates can be compared.
    """
    outcomes: list = []

    c1 = _event("cand-1", key="preferences.response_style",
                value={"response_style": "concise"}, turn_id="t1")
    outcomes.append(("queue_c1", gate.queue_candidate(c1)))

    prompt = gate.next_prompt()
    outcomes.append(("prompt", prompt.event_id, prompt.question, prompt.topic,
                     prompt.key, prompt.all_event_ids))

    rejected = gate.record_response("cand-1", accepted=False)
    outcomes.append(("reject", rejected.kind, rejected.value.get("rejected"),
                     rejected.supersedes, rejected.applied))

    outcomes.append(("silenced", gate.is_silenced("preferences", "preferences.response_style")))

    # Same key, later — must be dropped by the fresh silence window.
    c2 = _event("cand-2", key="preferences.response_style",
                value={"response_style": "concise"}, turn_id="t2",
                observed_at="2026-07-17T09:30:00Z")
    outcomes.append(("requeue_silenced", gate.queue_candidate(c2)))

    # Different key — queues and accepts cleanly.
    c3 = _event("cand-3", key="preferences.humor_level",
                value={"humor_level": "low"}, turn_id="t3")
    outcomes.append(("queue_c3", gate.queue_candidate(c3)))

    prompt3 = gate.next_prompt()
    outcomes.append(("prompt3", prompt3.event_id, prompt3.topic, prompt3.key))

    accepted = gate.record_response("cand-3", accepted=True)
    outcomes.append(("accept", accepted.source, accepted.applied,
                     accepted.supersedes, accepted.value))

    outcomes.append(("pending", gate.pending_count()))
    return outcomes


def _journal_fingerprint(journal: JournalStore) -> list[str]:
    """Order-independent content projection, minus the non-deterministic
    event_id / observed_at that make_event stamps on response events."""
    rows = []
    for event in journal.iter_events():
        payload = event.to_payload()
        payload.pop("event_id", None)
        payload.pop("observed_at", None)
        rows.append(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    return sorted(rows)


def test_gate_parity_with_and_without_index(tmp_path):
    # Without the index: pure journal-scan behavior.
    base_no = tmp_path / "no_index"
    base_no.mkdir()
    journal_no = JournalStore(journal_dir=base_no / "journal")
    gate_no = ConfirmationGate(
        journal=journal_no,
        store=_make_store(base_no),
        state_path=base_no / "state.json",
    )
    outcomes_no = _run_representative_flow(gate_no)

    # With the index wired through the journal's write-through hook.
    base_idx = tmp_path / "with_index"
    base_idx.mkdir()
    index = MemorySQLiteIndex(db_path=base_idx / "memory.sqlite")
    journal_idx = JournalStore(journal_dir=base_idx / "journal", on_append=index.index_event)
    gate_idx = ConfirmationGate(
        journal=journal_idx,
        store=_make_store(base_idx),
        state_path=base_idx / "state.json",
        sqlite_index=index,
    )
    outcomes_idx = _run_representative_flow(gate_idx)

    # Bit-identical outcomes and journal contents either way.
    assert outcomes_no == outcomes_idx
    assert _journal_fingerprint(journal_no) == _journal_fingerprint(journal_idx)

    # And the flow actually exercised the interesting paths (guards against both
    # sides being wrong in the same way).
    outcomes = dict((o[0], o) for o in outcomes_no)
    assert outcomes["queue_c1"][1] is True
    assert outcomes["reject"][1:] == ("contradiction", True, "cand-1", False)
    assert outcomes["silenced"][1] is True
    assert outcomes["requeue_silenced"][1] is False
    assert outcomes["queue_c3"][1] is True
    assert outcomes["accept"][1:] == ("explicit", True, "cand-3", {"humor_level": "low"})
    assert outcomes["pending"][1] == 0

    index.close()


@pytest.mark.parametrize("with_index", [False, True])
def test_load_event_roundtrips_through_index(tmp_path, with_index):
    """_load_event via the index must reconstruct the same fields the gate reads."""
    base = tmp_path / ("idx" if with_index else "plain")
    base.mkdir()
    index = MemorySQLiteIndex(db_path=base / "memory.sqlite") if with_index else None
    on_append = index.index_event if index is not None else None
    journal = JournalStore(journal_dir=base / "journal", on_append=on_append)
    gate = ConfirmationGate(
        journal=journal,
        store=_make_store(base),
        state_path=base / "state.json",
        sqlite_index=index,
    )

    candidate = _event("cand-x", key="preferences.response_style",
                       value={"response_style": "concise"}, turn_id="t7")
    gate.queue_candidate(candidate)

    loaded = gate._load_event("cand-x")
    assert loaded is not None
    assert loaded.event_id == "cand-x"
    assert loaded.topic == "preferences"
    assert loaded.key == "preferences.response_style"
    assert loaded.value == {"response_style": "concise"}
    assert loaded.session_id == "s1"
    assert loaded.turn_id == "t7"
    assert loaded.kind == "preference"
    assert gate._load_event("missing") is None

    if index is not None:
        index.close()
