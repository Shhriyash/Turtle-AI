from datetime import UTC, datetime

import pytest

import core.paths as core_paths
from core.memory_journal import MemoryEvent
from core.memory_replayer import replay

REFERENCE_TIME = datetime(2026, 7, 16, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    root = tmp_path / "pm"
    root.mkdir()
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snapshots")
    from core.personal_memory_store import PersonalMemoryStore

    return PersonalMemoryStore(user_id="probe_render")


def _event(**overrides) -> MemoryEvent:
    base = dict(
        event_id="ev_base",
        session_id="s",
        turn_id="t1",
        observed_at="2026-07-11T10:00:00Z",
        kind="preference",
        topic="preferences",
        key="preferences.response_style",
        value={"response_style": "concise"},
        confidence=0.9,
        source="explicit",
        extractor="llm_turn",
        applied=True,
    )
    base.update(overrides)
    return MemoryEvent(**base)


def _topic_text(store, topic: str) -> str:
    path = store.get_topic_path(topic)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def test_explicit_facts_never_decay(store):
    replay(
        [_event(event_id="ev_old_explicit", observed_at="2026-05-17T10:00:00Z")],
        store=store,
        reference_time=REFERENCE_TIME,
    )
    assert "concise" in _topic_text(store, "preferences")


def test_inferred_facts_still_decay(store):
    replay(
        [
            _event(
                event_id="ev_old_inferred",
                observed_at="2026-05-17T10:00:00Z",
                key="preferences.humor_level",
                value={"humor_level": "light"},
                source="inferred",
            )
        ],
        store=store,
        reference_time=REFERENCE_TIME,
    )
    assert "light" not in _topic_text(store, "preferences")


def test_generic_renderer_projects_unknown_keys(store):
    replay(
        [
            _event(
                event_id="ev_sport",
                key="preferences.sport",
                value={"sport": "taekwondo", "level": "national"},
                source="inferred",
            )
        ],
        store=store,
        reference_time=REFERENCE_TIME,
    )
    assert "Sport: taekwondo, national" in _topic_text(store, "preferences")


def test_corrections_with_structured_value_render(store):
    replay(
        [
            _event(
                event_id="ev_correction",
                kind="correction",
                topic="corrections",
                key="corrections.name_role",
                value={"name": "Shriyash", "role": "AI engineer"},
            )
        ],
        store=store,
        reference_time=REFERENCE_TIME,
    )
    text = _topic_text(store, "corrections")
    assert "Shriyash" in text
    assert "AI engineer" in text


def test_style_topics_are_projectable(store):
    replay(
        [
            _event(
                event_id="ev_editor",
                topic="tool_preferences",
                key="tool_preferences.editor",
                value={"tool": "VS Code"},
            )
        ],
        store=store,
        reference_time=REFERENCE_TIME,
    )
    assert "VS Code" in _topic_text(store, "tool_preferences")
