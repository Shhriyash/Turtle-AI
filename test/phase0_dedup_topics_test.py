import pytest

import core.paths as core_paths
from core.memory_journal import JournalStore, MemoryEvent, validate_event


@pytest.fixture()
def pm_root(tmp_path, monkeypatch):
    root = tmp_path / "pm"
    root.mkdir()
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snapshots")
    return root


def _event(**overrides) -> MemoryEvent:
    base = dict(
        event_id="ev_base",
        session_id="s",
        turn_id="t1",
        observed_at="2026-07-16T10:00:00Z",
        kind="fact",
        topic="identity",
        key="identity.name",
        value={"name": "Shriyash"},
        confidence=0.8,
        source="inferred",
        extractor="llm_turn",
        applied=False,
    )
    base.update(overrides)
    return MemoryEvent(**base)


def test_applied_restatement_survives_pending_candidate(pm_root):
    store = JournalStore(user_id="probe_dedup")
    store.append_many([_event(event_id="ev_pending")])

    out = store.append_many(
        [
            _event(
                event_id="ev_explicit",
                turn_id="t2",
                observed_at="2026-07-16T10:00:01Z",
                confidence=1.0,
                source="explicit",
                extractor="deterministic",
                applied=True,
            )
        ]
    )

    assert len(out) == 1
    events = store.load_all()
    assert len(events) == 2
    assert sum(1 for ev in events if ev.applied) == 1


def test_identical_duplicates_still_dedup(pm_root):
    store = JournalStore(user_id="probe_dedup_dup")
    store.append_many([_event(event_id="ev_one")])

    out = store.append_many([_event(event_id="ev_two")])

    assert out == []
    assert len(store.load_all()) == 1


@pytest.mark.parametrize(
    "topic",
    ["working_style", "communication_style", "tool_preferences", "decision_style"],
)
def test_new_topics_validate(pm_root, topic):
    validate_event(
        _event(
            event_id=f"ev_{topic}",
            kind="preference",
            topic=topic,
            key=f"{topic}.probe",
            value={"value": "x"},
            confidence=0.9,
        )
    )
