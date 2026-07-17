"""Phase 2 W5 — the store -> restart -> recall end-to-end guarantee.

The 2026-07-16 production autopsy's headline failure: facts the user disclosed
survived neither a process restart nor the retrieval path. The journal is the
source of truth and every derived projection (the FTS index, the topic markdown)
must be rebuildable from it — so a fact written in one process must resurface
after a *fresh* set of stores is constructed over the same directories and the
index is rebuilt from the journal alone.

This module pins that guarantee end to end, fully offline (no LLM, no network)
and tmp-isolated (``core.paths.PERSONAL_MEMORY_DIR`` is monkeypatched, stores are
constructed directly, and nothing is written under the repo's ``data/`` tree).

Four flows:
  (a) FULL LIFECYCLE  — an explicit evidence-grounded fact fed through the real
      server write funnel (``_journal_and_queue_candidates``) survives a
      simulated restart and is served by BOTH prompt injection
      (``build_context``) and the recall tool.
  (b) UPDATE FLOW     — the latest applied value per (topic, key) wins after a
      rebuild; the superseded value is not served.
  (c) REJECTION FLOW  — a journal rejection tombstone survives an index rebuild;
      the rejected fact is served by neither injection nor recall.
  (d) CORRECTION-VS-HISTORY — a corrected fact whose event ``supersedes`` the
      poisoned one is the only value served after rebuild.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import core.paths as core_paths
from core.memory_journal import JournalStore, make_event
from core.memory_sqlite import MemorySQLiteIndex
from core.personal_memory_extract import PersonalMemoryCandidate
from core.personal_memory_store import PersonalMemoryStore
from core.retrieval_broker import RetrievalBroker
from core.task_history import TaskHistoryStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def pm_root(tmp_path, monkeypatch):
    """Redirect all personal-memory paths under tmp so nothing touches data/."""
    root = tmp_path / "pm"
    root.mkdir()
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snapshots")
    return root


class _StubGate:
    """Minimal stand-in for ConfirmationGate.

    ``_journal_and_queue_candidates`` only calls ``queue_candidate`` (for the
    pending, non-auto-applied candidates). The real gate is owned by a
    concurrent workstream, so the E2E path uses this stub.
    """

    def __init__(self) -> None:
        self.queued: list = []

    def queue_candidate(self, event) -> bool:
        self.queued.append(event)
        return True

    def next_prompt(self):  # pragma: no cover - defensive; not exercised here
        return None


def _make_broker(store, tmp_path, index, *, journal_store=None) -> RetrievalBroker:
    """A broker wired exactly like the server's, minus network tiers.

    ``session_store``/``rag_system``/``vector_store`` are None, so the summary,
    episodic, and vector tiers no-op — the test exercises only the journal-backed
    FTS path (injection Tier 1.5 and recall scope=personal).
    """
    return RetrievalBroker(
        store=store,
        task_store=TaskHistoryStore(tmp_path / "tasks" / "history.jsonl"),
        journal_store=journal_store,
        sqlite_index=index,
        session_store=None,
        rag_system=None,
        vector_store=None,
    )


def _fresh_index_from_journal(journal_dir, db_path) -> tuple[JournalStore, MemorySQLiteIndex]:
    """Simulate a process restart: a brand-new JournalStore + a brand-new
    MemorySQLiteIndex at a *fresh* db path, rebuilt from the journal alone.

    Using a fresh db path (rather than reopening the live one) is deliberate: it
    forces ``backfill_from_journal`` to reconstruct the entire served state from
    the append-only journal, which is the exact durability property the autopsy
    found broken.
    """
    journal = JournalStore(user_id="default", journal_dir=journal_dir)
    index = MemorySQLiteIndex(db_path=db_path)
    index.backfill_from_journal(journal)
    return journal, index


# ---------------------------------------------------------------------------
# (a) FULL LIFECYCLE — the single most important regression in the repo.
# ---------------------------------------------------------------------------

def test_full_lifecycle_explicit_fact_survives_restart(pm_root, tmp_path):
    from apps.turtle_server import _journal_and_queue_candidates

    journal_dir = tmp_path / "journal"

    # --- Phase 1: a live process writes the fact through the real funnel ---
    live_index = MemorySQLiteIndex(db_path=tmp_path / "memory.sqlite")
    journal = JournalStore(
        user_id="default", journal_dir=journal_dir, on_append=live_index.index_event
    )
    store = PersonalMemoryStore()
    gate = _StubGate()
    state = SimpleNamespace(
        journal_store=journal,
        personal_memory_store=store,
        confirmation_gate=gate,
        user_id="default",
    )

    candidate = PersonalMemoryCandidate(
        topic="relations",
        key="best_friend",
        value="Elvin",
        line="- Best Friend: Elvin",
        overwrite_policy="replace",
        confidence="confirmed",
        sensitivity="normal",
        source_session_id="s_e2e",
        evidence="my best friend is Elvin",
        source="explicit",
        extraction_source="deterministic",
    )

    _journal_and_queue_candidates(state, [candidate], session_id="s_e2e")

    # The fact was journaled (the durable source of truth), not merely queued.
    assert any(ev.key == "relations.best_friend" for ev in journal.load_all())
    live_index.close()

    # --- Phase 2: SIMULATE RESTART — fresh stores over the same journal dir ---
    journal2, index2 = _fresh_index_from_journal(journal_dir, tmp_path / "memory_restart.sqlite")
    store2 = PersonalMemoryStore()
    broker = _make_broker(store2, tmp_path, index2, journal_store=journal2)

    injected = asyncio.run(
        broker.build_context(task_type="general", query="who is my best friend")
    )
    recalled = asyncio.run(
        broker.recall(query="who is my best friend", scope="personal")
    )
    index2.close()

    # Prompt-time injection surfaces the fact under the [Relevant Memory] header.
    assert "[Relevant Memory]" in injected
    assert "Elvin" in injected.split("[Relevant Memory]", 1)[1]
    # The recall tool independently surfaces it too.
    assert "Elvin" in recalled


# ---------------------------------------------------------------------------
# (b) UPDATE FLOW — latest-per-key wins after a rebuild.
# ---------------------------------------------------------------------------

def test_update_flow_latest_value_wins_after_restart(pm_root, tmp_path):
    journal_dir = tmp_path / "journal"
    journal = JournalStore(user_id="default", journal_dir=journal_dir)

    journal.append(
        make_event(
            kind="preference", topic="preferences", key="preferences.favourite_editor",
            value={"value": "VS Code"}, confidence=1.0, source="explicit",
            extractor="deterministic", session_id="s1", turn_id="t1",
            observed_at="2026-05-01T10:00:00Z", applied=True,
            evidence={"text": "my favourite editor is VS Code"},
        )
    )
    journal.append(
        make_event(
            kind="preference", topic="preferences", key="preferences.favourite_editor",
            value={"value": "Cursor"}, confidence=1.0, source="explicit",
            extractor="deterministic", session_id="s2", turn_id="t2",
            observed_at="2026-05-02T10:00:00Z", applied=True,
            evidence={"text": "actually my favourite editor is Cursor now"},
        )
    )

    journal2, index2 = _fresh_index_from_journal(journal_dir, tmp_path / "memory.sqlite")
    broker = _make_broker(PersonalMemoryStore(), tmp_path, index2, journal_store=journal2)

    injected = asyncio.run(
        broker.build_context(task_type="general", query="what is my favourite editor")
    )
    recalled = asyncio.run(
        broker.recall(query="what is my favourite editor", scope="personal")
    )
    index2.close()

    assert "Cursor" in recalled and "VS Code" not in recalled
    assert "Cursor" in injected and "VS Code" not in injected


# ---------------------------------------------------------------------------
# (c) REJECTION FLOW — a tombstone survives an index rebuild.
# ---------------------------------------------------------------------------

def test_rejection_tombstone_survives_rebuild(pm_root, tmp_path):
    journal_dir = tmp_path / "journal"
    journal = JournalStore(user_id="default", journal_dir=journal_dir)

    event = make_event(
        kind="fact", topic="projects", key="projects.project.zephyr",
        value={"name": "Zephyr"}, confidence=1.0, source="explicit",
        extractor="deterministic", session_id="s1", turn_id="t1",
        observed_at="2026-05-01T10:00:00Z", applied=True,
        evidence={"text": "i am working on project Zephyr"},
    )
    journal.append(event)
    journal.append_rejection(event)

    journal2, index2 = _fresh_index_from_journal(journal_dir, tmp_path / "memory.sqlite")
    broker = _make_broker(PersonalMemoryStore(), tmp_path, index2, journal_store=journal2)

    injected = asyncio.run(
        broker.build_context(task_type="general", query="what project am i working on")
    )
    recalled = asyncio.run(
        broker.recall(query="what project am i working on", scope="personal")
    )
    index2.close()

    # The rejected fact must be served by neither path after the rebuild.
    assert "Zephyr" not in injected
    assert "Zephyr" not in recalled
    assert recalled == ""


# ---------------------------------------------------------------------------
# (d) CORRECTION-VS-HISTORY — the poisoned-name scenario.
# ---------------------------------------------------------------------------

def test_correction_supersedes_poisoned_name_after_rebuild(pm_root, tmp_path):
    journal_dir = tmp_path / "journal"
    journal = JournalStore(user_id="default", journal_dir=journal_dir)

    poisoned = make_event(
        kind="fact", topic="identity", key="identity.name",
        value={"name": "Robert"}, confidence=1.0, source="explicit",
        extractor="deterministic", session_id="s1", turn_id="t1",
        observed_at="2026-05-01T10:00:00Z", applied=True,
        evidence={"text": "misheard name Robert"},
    )
    journal.append(poisoned)
    journal.append(
        make_event(
            kind="correction", topic="identity", key="identity.name",
            value={"name": "Shriyash"}, confidence=1.0, source="explicit",
            extractor="deterministic", session_id="s2", turn_id="t2",
            observed_at="2026-05-02T10:00:00Z", applied=True,
            supersedes=poisoned.event_id,
            evidence={"text": "no, my name is Shriyash"},
        )
    )

    journal2, index2 = _fresh_index_from_journal(journal_dir, tmp_path / "memory.sqlite")
    broker = _make_broker(PersonalMemoryStore(), tmp_path, index2, journal_store=journal2)

    injected = asyncio.run(
        broker.build_context(task_type="general", query="what is my name")
    )
    recalled = asyncio.run(
        broker.recall(query="what is my name", scope="personal")
    )
    index2.close()

    assert "Shriyash" in recalled and "Robert" not in recalled
    assert "Shriyash" in injected and "Robert" not in injected
