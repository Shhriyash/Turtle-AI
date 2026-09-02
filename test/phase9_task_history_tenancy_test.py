"""
Phase 9 — task history must be tenant-scoped.

PRIVACY DEFECT: data/tasks/history.sqlite is ONE global FTS5 index shared by
every user, and TaskHistoryIndex.search() ran an unfiltered MATCH over it.
RetrievalBroker._build_task_tier (tier 4) feeds that result straight into the
prompt, so one user's task text could be spliced into another user's memory
context. TaskHistoryRecord had no user_id field at all.

Fix: user_id column on the FTS table (+ rebuild migration for pre-tenancy
indexes), carried on the record, and REQUIRED by search() — which fails closed
(returns nothing) rather than searching everyone when no owner is supplied.
"""
from __future__ import annotations

import pytest

from core.task_history import TaskHistoryStore


@pytest.fixture()
def history_path(tmp_path):
    return tmp_path / "history.jsonl"


def _seed(store: TaskHistoryStore, *, user_id: str, query: str, outcome: str):
    return store.record(
        session_id=f"s_{user_id}",
        turn_id=f"t_{user_id}",
        task_type="email",
        status="ok",
        query=query,
        tool_used="send_email_assistant",
        outcome=outcome,
        user_id=user_id,
    )


def test_search_does_not_leak_across_users(history_path):
    """The core defect: user B must never see user A's task text."""
    store_a = TaskHistoryStore(history_path, user_id="usr_alice")
    _seed(store_a, user_id="usr_alice", query="quarterly acquisition memo",
          outcome="emailed the acquisition memo to the board")

    store_b = TaskHistoryStore(history_path, user_id="usr_bob")
    hits = store_b.search("acquisition memo", max_results=5)
    assert hits == [], f"cross-tenant leak: user B saw {hits}"

    # ...and the owner still finds their own record.
    own = store_a.search("acquisition memo", max_results=5)
    assert own and own[0]["user_id"] == "usr_alice"


def test_format_search_results_is_scoped(history_path):
    """The broker calls this one — it must be scoped too, not just search()."""
    store_a = TaskHistoryStore(history_path, user_id="usr_alice")
    _seed(store_a, user_id="usr_alice", query="secret project falcon",
          outcome="ran the falcon report")

    store_b = TaskHistoryStore(history_path, user_id="usr_bob")
    assert store_b.format_search_results("falcon", max_results=1) == ""
    assert "falcon" in store_a.format_search_results("falcon", max_results=1).lower()


def test_search_fails_closed_without_owner(history_path):
    """No owner => no results. Never fall back to searching every tenant."""
    store_a = TaskHistoryStore(history_path, user_id="usr_alice")
    _seed(store_a, user_id="usr_alice", query="widget pricing",
          outcome="sent widget pricing")

    anon = TaskHistoryStore(history_path, user_id="")
    assert anon.search("widget pricing", max_results=5) == []
    assert store_a.search("widget pricing", max_results=5, user_id=None) != []


def test_list_by_session_is_scoped(history_path):
    store_a = TaskHistoryStore(history_path, user_id="usr_alice")
    _seed(store_a, user_id="usr_alice", query="alpha", outcome="did alpha")

    store_b = TaskHistoryStore(history_path, user_id="usr_bob")
    assert store_b.list_by_session("s_usr_alice") == []
    assert len(store_a.list_by_session("s_usr_alice")) == 1


def test_record_carries_owner_into_jsonl(history_path):
    """JSONL is the source of truth for index rebuilds — the owner must persist
    there, or a rebuild would silently produce unattributed (unsearchable) rows."""
    store = TaskHistoryStore(history_path, user_id="usr_alice")
    _seed(store, user_id="usr_alice", query="beta", outcome="did beta")
    rows = store.load_records()
    assert rows and rows[0]["user_id"] == "usr_alice"


def test_pre_tenancy_index_is_rebuilt(tmp_path):
    """An index created before tenancy has no user_id column. FTS5 can't ALTER
    ADD COLUMN, so it must be dropped and recreated — and must not crash."""
    import sqlite3

    sqlite_path = tmp_path / "history.sqlite"
    old = sqlite3.connect(str(sqlite_path))
    old.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS task_history_fts USING fts5(
            session_id UNINDEXED, turn_id UNINDEXED, task_type, status UNINDEXED,
            timestamp UNINDEXED, query, tool_used, outcome, payload_json
        );
        """
    )
    old.commit()
    old.close()

    from core.task_history_index import TaskHistoryIndex

    idx = TaskHistoryIndex(sqlite_path)
    cols = {r[1] for r in idx._connection.execute("PRAGMA table_info(task_history_fts)")}
    assert "user_id" in cols
    idx.close()
