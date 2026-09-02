import asyncio

import pytest

import core.paths as core_paths
from core.memory_journal import JournalStore, MemoryEvent
from core.memory_sqlite import MemorySQLiteIndex, _escape_fts_query
from core.personal_memory_store import PersonalMemoryStore
from core.retrieval_broker import RetrievalBroker
from core.task_history import TaskHistoryStore


@pytest.fixture()
def pm_root(tmp_path, monkeypatch):
    root = tmp_path / "pm"
    root.mkdir()
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snapshots")
    return root


def _event(
    event_id: str,
    *,
    observed_at: str = "2026-01-01T10:00:00Z",
    topic: str = "identity",
    key: str = "name",
    value: dict | None = None,
    evidence_text: str = "",
    applied: bool = True,
    kind: str = "fact",
    supersedes: str | None = None,
) -> MemoryEvent:
    return MemoryEvent(
        event_id=event_id,
        session_id="s1",
        turn_id="t1",
        observed_at=observed_at,
        kind=kind,
        topic=topic,
        key=key,
        value=value or {},
        confidence=1.0,
        source="explicit",
        extractor="deterministic",
        evidence={"text": evidence_text} if evidence_text else {},
        supersedes=supersedes,
        applied=applied,
    )


def test_escape_fts_query_filters_stopwords_and_supports_fallback():
    expr = _escape_fts_query("what editor do I use", operator="AND")
    assert '"editor"' in expr
    assert '"use"' in expr
    assert '"i"' not in expr
    assert '"do"' not in expr
    assert '"what"' not in expr
    assert " AND " in expr

    or_expr = _escape_fts_query("what editor do I use", operator="OR")
    assert " OR " in or_expr

    assert _escape_fts_query("what is my", operator="AND")


def test_search_honors_rejection_tombstones_and_latest_per_key(tmp_path):
    ev1 = _event(
        "ev1",
        observed_at="2026-01-01T10:00:00Z",
        value={"name": "an AI engineer"},
        evidence_text="a national level player",
    )
    ev2 = _event(
        "ev2",
        observed_at="2026-01-01T11:00:00Z",
        value={"name": "Shriyash"},
        evidence_text="my name is Shriyash",
    )

    journal = JournalStore(user_id="default", journal_dir=tmp_path / "j")
    journal.append(ev1)
    journal.append(ev2)
    journal.append_rejection(ev1)

    idx = MemorySQLiteIndex(db_path=tmp_path / "m.sqlite")
    idx.backfill_from_journal(journal)

    rows = idx.search("name", topic="identity", limit=10)
    assert len(rows) == 1
    assert rows[0].topic == "identity"
    assert rows[0].key == "name"
    assert "Shriyash" in rows[0].value_text


def test_weak_lexical_path_has_relevance_floor(tmp_path, pm_root):
    idx = MemorySQLiteIndex(db_path=tmp_path / "m.sqlite")
    sport = _event(
        "sport1",
        topic="preferences",
        key="sport",
        value={"sport": "taekwondo", "level": "national"},
        evidence_text="i like to play taekwondo",
    )
    idx.index_event(sport)

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text("", encoding="utf-8")
    broker = RetrievalBroker(
        store=PersonalMemoryStore(),
        task_store=TaskHistoryStore(task_file),
        sqlite_index=idx,
        vector_store=None,
        session_store=None,
        rag_system=None,
    )

    assert asyncio.run(broker.recall(query="what editor do I use", scope="personal")) == ""
    sport_recall = asyncio.run(broker.recall(query="what sport do I play", scope="personal"))
    assert "taekwondo" in sport_recall.lower()


def test_mark_rejected_excludes_event_from_search(tmp_path):
    idx = MemorySQLiteIndex(db_path=tmp_path / "m.sqlite")
    ev = _event(
        "ev1",
        value={"name": "poisoned"},
        evidence_text="my name is poisoned",
    )
    idx.index_event(ev)

    assert idx.search("poisoned", topic="identity")
    idx.mark_rejected("ev1")
    assert idx.search("poisoned", topic="identity") == []
