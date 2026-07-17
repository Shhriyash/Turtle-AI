"""Phase 2 W2 — query-aware always-on memory injection + cross-session seeding.

Two read-path fixes from the 2026-07-16 production autopsy:

1. RetrievalBroker.build_context injected only the static identity/preferences
   card and ignored *query* entirely, so any stored fact outside those two
   topics was invisible unless the model separately called the recall tool. The
   new [Relevant Memory] tier runs the Phase 1 search layer against the query
   (with an injection-grade relevance floor — an always-on tier prefers empty
   over wrong).

2. The [Recent Summary] tier is session-scoped and a fresh session's summary is
   empty, so a new conversation started with zero continuity.
   get_summary_tail_with_carryover seeds it from the previous completed session.
"""
import asyncio

import pytest

import core.paths as core_paths
from core.memory_journal import make_event
from core.memory_sqlite import MemorySQLiteIndex
from core.personal_memory_store import PersonalMemoryStore
from core.retrieval_broker import DEFAULT_BUDGET, RetrievalBroker, _estimate_tokens
from core.session_store import SessionStore
from core.storage import Session
from core.storage.local.sqlite_store import SQLiteSessionStore
from core.task_history import TaskHistoryStore


@pytest.fixture()
def pm_root(tmp_path, monkeypatch):
    root = tmp_path / "pm"
    root.mkdir()
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snapshots")
    return root


def _broker(store, tmp_path, index, *, session_store=None) -> RetrievalBroker:
    return RetrievalBroker(
        store=store,
        task_store=TaskHistoryStore(tmp_path / "tasks" / "history.jsonl"),
        sqlite_index=index,
        session_store=session_store,
        vector_store=None,
        rag_system=None,
    )


# ---------------------------------------------------------------------------
# (a) The canonical autopsy failure, fixed at the injection layer.
# ---------------------------------------------------------------------------

def test_relevant_memory_tier_injects_best_friend(pm_root, tmp_path):
    store = PersonalMemoryStore()
    index = MemorySQLiteIndex(db_path=tmp_path / "memory.sqlite")
    index.index_event(
        make_event(
            kind="fact", topic="relations", key="relations.best_friend",
            value={"name": "Elvin"}, confidence=1.0, source="explicit",
            extractor="deterministic", applied=True,
            session_id="s1", turn_id="t1", observed_at="2026-05-01T10:00:00Z",
            evidence={"text": "my best friend is Elvin"},
        )
    )
    broker = _broker(store, tmp_path, index)
    result = asyncio.run(
        broker.build_context(task_type="general", query="who is my best friend")
    )
    index.close()

    assert "[Relevant Memory]" in result
    # The stored fact is now injected on the always-on turn, no recall call.
    relevant_section = result.split("[Relevant Memory]", 1)[1]
    assert "Elvin" in relevant_section


# ---------------------------------------------------------------------------
# (b) The injection-grade floor holds — an unrelated query injects nothing.
# ---------------------------------------------------------------------------

def test_unrelated_query_injects_no_relevant_memory(pm_root, tmp_path):
    store = PersonalMemoryStore()
    index = MemorySQLiteIndex(db_path=tmp_path / "memory.sqlite")
    index.index_event(
        make_event(
            kind="fact", topic="relations", key="relations.best_friend",
            value={"name": "Elvin"}, confidence=1.0, source="explicit",
            extractor="deterministic", applied=True,
            session_id="s1", turn_id="t1", observed_at="2026-05-01T10:00:00Z",
            evidence={"text": "my best friend is Elvin"},
        )
    )
    broker = _broker(store, tmp_path, index)
    result = asyncio.run(
        broker.build_context(task_type="general", query="what's the weather")
    )
    index.close()

    assert "[Relevant Memory]" not in result


# ---------------------------------------------------------------------------
# (c) A hit whose value already appears in the Tier-1 body is not duplicated.
# ---------------------------------------------------------------------------

def test_relevant_hit_already_in_identity_is_not_duplicated(pm_root, tmp_path):
    store = PersonalMemoryStore()
    store.write_topic("identity", ["- Name: Shriyash"], {"title": "Identity"})
    store.update_index_entry("identity", "Name and email details")
    index = MemorySQLiteIndex(db_path=tmp_path / "memory.sqlite")
    index.index_event(
        make_event(
            kind="fact", topic="identity", key="identity.name",
            value={"name": "Shriyash"}, confidence=1.0, source="explicit",
            extractor="deterministic", applied=True,
            session_id="s1", turn_id="t1", observed_at="2026-05-01T10:00:00Z",
            evidence={"text": "my name is Shriyash"},
        )
    )
    broker = _broker(store, tmp_path, index)
    result = asyncio.run(
        broker.build_context(task_type="general", query="what is my name")
    )
    index.close()

    assert "- Name: Shriyash" in result           # present in the [Identity] block
    assert result.count("Shriyash") == 1          # not re-injected by the relevant tier


# ---------------------------------------------------------------------------
# (d) The assembled block stays under the total token budget.
# ---------------------------------------------------------------------------

def test_full_block_respects_total_token_budget(pm_root, tmp_path):
    store = PersonalMemoryStore()
    # Long Tier-1 bodies.
    store.write_topic(
        "identity", [f"- Detail {i}: {'x' * 40}" for i in range(40)], {"title": "Identity"}
    )
    store.update_index_entry("identity", "identity details " * 10)
    store.write_topic(
        "preferences", [f"- Pref {i}: {'y' * 40}" for i in range(40)], {"title": "Preferences"}
    )
    store.update_index_entry("preferences", "preference details " * 10)

    # A large corpus that matches the query so the relevant tier fills up.
    index = MemorySQLiteIndex(db_path=tmp_path / "memory.sqlite")
    for i in range(40):
        index.index_event(
            make_event(
                kind="fact", topic="projects", key=f"projects.item_{i}",
                value={"detail": f"project alpha milestone {i} " + "context " * 10},
                confidence=1.0, source="explicit", extractor="deterministic",
                applied=True, session_id="s1", turn_id=f"t{i}",
                observed_at="2026-05-01T10:00:00Z",
                evidence={"text": f"project alpha milestone {i}"},
            )
        )

    # A session store carrying a long rolling summary so the summary tier fills.
    session = SessionStore(SQLiteSessionStore(db_path=tmp_path / "s.sqlite"), user_id="usr_a")
    session.rolling_summary = [
        {
            "timestamp": "2026-05-01T10:00:00Z",
            "turn_id_range": [i, i + 1],
            "bullets": ["User is building the Turtle memory system " + "with detail " * 6],
        }
        for i in range(8)
    ]

    broker = _broker(store, tmp_path, index, session_store=session)
    result = asyncio.run(
        broker.build_context(task_type="general", query="project alpha milestone")
    )
    index.close()

    assert _estimate_tokens(result) <= DEFAULT_BUDGET.total_tokens


# ---------------------------------------------------------------------------
# (e) Cross-session carryover seeds from the prior completed session, tenant-scoped.
# ---------------------------------------------------------------------------

def test_carryover_seeds_from_prior_completed_same_user(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await backend.init_db()

        # usr_a's completed session, carrying a summary.
        await backend.put(
            Session(
                session_id="a_done",
                data={
                    "status": "completed",
                    "user_id": "usr_a",
                    "messages": [],
                    "summary": [
                        {
                            "timestamp": "2026-05-01T10:00:00Z",
                            "turn_id_range": [1, 2],
                            "bullets": ["User is building Turtle"],
                        }
                    ],
                    "updated_at": "2026-05-01T10:05:00Z",
                },
            )
        )
        # A different tenant's completed session — must NOT leak across users.
        await backend.put(
            Session(
                session_id="b_done",
                data={
                    "status": "completed",
                    "user_id": "usr_b",
                    "messages": [],
                    "summary": [
                        {
                            "timestamp": "2026-05-01T09:00:00Z",
                            "turn_id_range": [1, 1],
                            "bullets": ["Other user secret"],
                        }
                    ],
                    "updated_at": "2026-05-01T09:05:00Z",
                },
            )
        )

        # A brand-new session for usr_a: its own rolling summary is empty.
        store = SessionStore(SQLiteSessionStore(db_path=db_path), user_id="usr_a")
        await store.start_or_restore("strict_new")
        return await store.get_summary_tail_with_carryover(max_entries=6)

    tail = asyncio.run(scenario())

    assert len(tail) == 1
    assert tail[0]["bullets"] == ["User is building Turtle"]
    joined = " ".join(bullet for entry in tail for bullet in entry["bullets"])
    assert "Other user secret" not in joined
