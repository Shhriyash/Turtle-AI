"""Phase 2 / W1: the memory schema registry.

Pins the three things W1 consolidates into core.memory_schema:
  - render_statement covers every topic (and honors a pre-rendered statement),
  - decide_write_policy's full truth table (applied / pending / rejected),
  - MemoryEvent.statement round-trips (and old payloads without it still load),
  - replay() render parity for the canonical fixtures (byte-level line checks).
"""

from datetime import UTC, datetime

import pytest

import core.paths as core_paths
from core.memory_journal import MemoryEvent
from core.memory_replayer import replay
from core.memory_schema import (
    TOPICS,
    decide_write_policy,
    render_statement,
    statement_for,
)

REFERENCE_TIME = datetime(2026, 7, 16, tzinfo=UTC)


def _event(*, key: str, value: dict, topic: str | None = None, **overrides) -> MemoryEvent:
    base = dict(
        event_id="ev_base",
        session_id="s",
        turn_id="t1",
        observed_at="2026-07-11T10:00:00Z",
        kind="preference",
        topic=topic or key.split(".", 1)[0],
        key=key,
        value=value,
        confidence=0.9,
        source="explicit",
        extractor="llm_turn",
        applied=True,
    )
    base.update(overrides)
    return MemoryEvent(**base)


# Representative (key, value) per topic — real key shapes exercised by the
# renderer's whitelist and generic fallback. preferences appears twice
# (whitelisted response_style + generic sport) so all 11 topics are covered.
_REPRESENTATIVE = [
    ("identity.name", {"name": "Shriyash"}),
    ("preferences.response_style", {"response_style": "concise"}),
    ("preferences.sport", {"sport": "taekwondo", "level": "national"}),
    ("corrections.name_role", {"name": "Shriyash", "role": "AI engineer"}),
    ("tool_preferences.editor", {"tool": "VS Code"}),
    ("relations.best_friend", {"role": "best_friend", "name": "Alex"}),
    ("workflow.prefers_draft_before_send", {"prefers_draft_before_send": True}),
    ("contacts.frequent_recipient.a@b.com", {"email": "a@b.com"}),
    ("projects.project.x", {"name": "X"}),
    ("working_style.note", {"note": "iterative"}),
    ("communication_style.note", {"note": "direct"}),
    ("decision_style.note", {"note": "data-driven"}),
]


# --- (a) every topic renders a non-empty statement --------------------------

@pytest.mark.parametrize("key,value", _REPRESENTATIVE, ids=[k for k, _ in _REPRESENTATIVE])
def test_every_topic_renders_non_empty_statement(key, value):
    statement = render_statement(_event(key=key, value=value))
    assert statement, f"empty statement for {key}"
    assert not statement.startswith("- "), "statement must not carry the bullet prefix"


def test_representative_events_cover_all_registry_topics():
    covered = {key.split(".", 1)[0] for key, _ in _REPRESENTATIVE}
    assert covered == set(TOPICS)


def test_statement_for_matches_render_statement():
    # The extractor-side helper must agree with the replayer-side renderer.
    for key, value in _REPRESENTATIVE:
        topic = key.split(".", 1)[0]
        assert statement_for(topic, key, value) == render_statement(
            _event(key=key, value=value, topic=topic)
        )


# --- (b) a pre-rendered statement renders verbatim --------------------------

def test_explicit_statement_field_renders_verbatim():
    event = _event(
        key="identity.name",
        value={"name": "WouldRenderDifferently"},
        statement="Name: Verbatim Snapshot",
    )
    assert render_statement(event) == "Name: Verbatim Snapshot"


def test_blank_statement_falls_through_to_key_template():
    event = _event(key="identity.name", value={"name": "Shriyash"}, statement="   ")
    assert render_statement(event) == "Name: Shriyash"


def test_generic_fallback_never_hides_a_renderable_value():
    # Unknown key, non-empty value -> must still project (the corrections.name_role
    # / preferences.sport regression that vanished facts in production).
    statement = render_statement(
        _event(key="preferences.sport", value={"sport": "taekwondo", "level": "national"})
    )
    assert statement == "Sport: taekwondo, national"


def test_genuinely_empty_value_renders_empty():
    assert render_statement(_event(key="identity.name", value={"name": ""})) == ""


# --- (c) decide_write_policy truth table ------------------------------------

@pytest.mark.parametrize(
    "source,topic,confidence,evidence,expected",
    [
        # explicit + grounded + high confidence -> applied, ANY topic
        ("explicit", "identity", 0.95, True, "applied"),
        ("explicit", "corrections", 0.90, True, "applied"),
        # explicit but not grounded -> pending
        ("explicit", "identity", 0.99, False, "pending"),
        # explicit grounded but below the 0.9 bar -> pending
        ("explicit", "preferences", 0.85, True, "pending"),
        # inferred/synthesized on a low-risk topic at >=0.85 -> applied
        ("inferred", "preferences", 0.85, False, "applied"),
        ("inferred", "workflow", 0.90, True, "applied"),
        ("synthesized", "projects", 0.86, False, "applied"),
        # inferred/synthesized below 0.85 -> pending
        ("inferred", "preferences", 0.84, True, "pending"),
        # inferred/synthesized on a non-auto-apply topic -> pending
        ("inferred", "identity", 0.99, True, "pending"),
        ("synthesized", "decision_style", 0.99, True, "pending"),
        # rejected: unknown source, unknown topic, confidence out of range
        ("migration", "preferences", 0.90, True, "rejected"),
        ("bogus", "preferences", 0.90, True, "rejected"),
        ("explicit", "not_a_topic", 0.95, True, "rejected"),
        ("explicit", "preferences", 1.5, True, "rejected"),
        ("inferred", "preferences", -0.1, True, "rejected"),
    ],
)
def test_decide_write_policy_truth_table(source, topic, confidence, evidence, expected):
    assert (
        decide_write_policy(
            source=source, topic=topic, confidence=confidence, evidence_supported=evidence
        )
        == expected
    )


# --- (d) statement round-trips through the payload --------------------------

def test_statement_round_trips_through_payload():
    event = _event(key="identity.name", value={"name": "Shriyash"}, statement="Name: Shriyash")
    payload = event.to_payload()
    assert payload["statement"] == "Name: Shriyash"
    restored = MemoryEvent.from_payload(payload)
    assert restored.statement == "Name: Shriyash"


def test_old_payload_without_statement_still_loads():
    payload = {
        "event_id": "ev_old",
        "session_id": "s",
        "turn_id": "t1",
        "observed_at": "2026-07-11T10:00:00Z",
        "kind": "fact",
        "topic": "identity",
        "key": "identity.name",
        "value": {"name": "Shriyash"},
        "confidence": 1.0,
        "source": "explicit",
        "extractor": "deterministic",
        "applied": True,
    }
    restored = MemoryEvent.from_payload(payload)
    assert restored.statement == ""
    # And it still renders via the key template, unchanged.
    assert render_statement(restored) == "Name: Shriyash"


# --- (e) replay() render parity for the canonical fixtures ------------------

@pytest.fixture()
def store(tmp_path, monkeypatch):
    root = tmp_path / "pm"
    root.mkdir()
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snapshots")
    from core.personal_memory_store import PersonalMemoryStore

    return PersonalMemoryStore(user_id="probe_phase2")


def _topic_text(store, topic: str) -> str:
    path = store.get_topic_path(topic)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def test_replay_render_parity_whitelisted_and_generic(store):
    replay(
        [
            _event(event_id="ev_name", topic="identity", key="identity.name", value={"name": "Shriyash"}),
            _event(event_id="ev_style", key="preferences.response_style", value={"response_style": "concise"}),
            _event(
                event_id="ev_sport",
                key="preferences.sport",
                value={"sport": "taekwondo", "level": "national"},
                source="inferred",
            ),
            _event(
                event_id="ev_corr",
                kind="correction",
                topic="corrections",
                key="corrections.name_role",
                value={"name": "Shriyash", "role": "AI engineer"},
            ),
        ],
        store=store,
        reference_time=REFERENCE_TIME,
    )
    assert "- Name: Shriyash" in _topic_text(store, "identity")
    prefs = _topic_text(store, "preferences")
    assert "- Response style: concise" in prefs
    assert "- Sport: taekwondo, national" in prefs
    assert "- Name Role: Shriyash, AI engineer" in _topic_text(store, "corrections")


def test_replay_honors_snapshotted_statement(store):
    # An event carrying a statement snapshot renders it verbatim through replay.
    replay(
        [
            _event(
                event_id="ev_snap",
                key="preferences.response_style",
                value={"response_style": "concise"},
                statement="Response style: snapshot-wins",
            )
        ],
        store=store,
        reference_time=REFERENCE_TIME,
    )
    assert "- Response style: snapshot-wins" in _topic_text(store, "preferences")
