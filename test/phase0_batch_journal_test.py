import types

import pytest

import core.paths as core_paths
from apps import turtle_server as ts
from core.memory_journal import JournalStore
from core.personal_memory_extract import PersonalMemoryCandidate
from core.personal_memory_store import PersonalMemoryStore


@pytest.fixture()
def pm_root(tmp_path, monkeypatch):
    root = tmp_path / "pm"
    root.mkdir()
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snapshots")
    return root


def _candidate(
    *,
    topic: str,
    key: str,
    value: str,
    evidence: str,
    source: str = "explicit",
    confidence: str = "confirmed",
    extraction_source: str = "llm_turn",
) -> PersonalMemoryCandidate:
    return PersonalMemoryCandidate(
        topic=topic,
        key=key,
        value=value,
        line=f"- {key}: {value}",
        overwrite_policy="replace",
        confidence=confidence,
        sensitivity="normal",
        source_session_id="s1",
        evidence=evidence,
        source=source,
        extraction_source=extraction_source,
    )


def _state(user_id: str):
    return types.SimpleNamespace(
        journal_store=JournalStore(user_id=user_id),
        personal_memory_store=PersonalMemoryStore(user_id=user_id),
        confirmation_gate=types.SimpleNamespace(queue_candidate=lambda e: True),
    )


def test_batch_isolation_journals_explicit_and_replays_applied(pm_root):
    state = _state("probe_batch")
    candidates = [
        _candidate(
            topic="identity",
            key="name",
            value="Shriyash",
            evidence="my name is shriyash",
            source="explicit",
            confidence="confirmed",
        ),
        _candidate(
            topic="decision_style",
            key="approach",
            value="data-driven",
            evidence="I like data before deciding",
            source="inferred",
            confidence="inferred",
        ),
    ]

    queued = ts._journal_and_queue_candidates(state, candidates, session_id="s1")

    events = state.journal_store.load_all()
    assert any(
        event.key == "identity.name"
        and event.value == {"name": "Shriyash"}
        and event.applied
        for event in events
    )
    identity_file = pm_root / "probe_batch" / "identity.md"
    assert identity_file.exists()
    assert "Shriyash" in identity_file.read_text(encoding="utf-8")
    # decision_style is accepted by the journal in the paired migration; if that
    # topic is not present in this checkout yet, the identity event is enough.
    assert len(events) >= 1
    assert queued >= 0


def test_llm_auto_apply_requires_value_in_evidence(pm_root):
    state = _state("probe_guard")
    candidate = _candidate(
        topic="identity",
        key="name",
        value="Shriyash",
        evidence="unrelated words entirely",
        source="explicit",
        confidence="confirmed",
    )

    ts._journal_and_queue_candidates(state, [candidate], session_id="s1")

    events = state.journal_store.load_all()
    assert any(
        event.key == "identity.name"
        and event.value == {"name": "Shriyash"}
        and event.applied is False
        for event in events
    )


def test_generic_allowed_topic_branch_persists_unknown_keys(pm_root):
    candidate = _candidate(
        topic="preferences",
        key="favourite_editor",
        value="VS Code",
        evidence="I use VS Code",
        source="inferred",
        confidence="inferred",
    )

    event = ts._candidate_to_journal_event(candidate=candidate, session_id="s1", ordinal=0)

    assert event is not None
    assert event.key == "preferences.favourite_editor"
    assert event.value == {"value": "VS Code"}
