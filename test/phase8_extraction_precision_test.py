"""
Phase 8 — memory-extraction precision bugs (found via the live 'what Turtle
remembers' viewer against a real persona's data).

Bug A — a transient, one-off request ("Give me one quick high-protein breakfast
idea") was mis-modeled by the Stage-B session extractor as a durable preference
(preferences.meals.quick_high_protein_breakfast = {"requested": true}) and
journaled as a candidate. A request is not a standing preference. Guarded by
core.personal_memory_extract._is_request_shaped + a Stage-B loop drop + a prompt
rule.

Bug B — the `remember` tool fired twice for one fact under two keys
(projects.codename_atlas = "I'm working on a project codenamed Atlas." AND
projects.project_codename = "Atlas"), leaving the same fact stored twice. Guarded
by the tool contract (remember.md) plus a length-gated, same-topic/same-session
containment backstop in apps.turtle_server._store_remembered_fact.

All offline — Bug A tests the pure guard directly; Bug B drives the real store
helper against a JournalStore under a tmp root.
"""
from __future__ import annotations

import types

import pytest

import core.paths as core_paths
from apps import turtle_server as ts
from core.personal_memory_extract import _is_request_shaped
from core.personal_memory_store import PersonalMemoryStore


# ── Bug A: request-shaped guard ──────────────────────────────────────────────

def test_bare_request_flag_is_request_shaped():
    # The exact real-world shape that polluted preferences.md.
    assert _is_request_shaped({"requested": True}) is True
    assert _is_request_shaped({"asked": True}) is True
    assert _is_request_shaped({"one_off": True, "wants": True}) is True


def test_legitimate_boolean_pref_is_not_request_shaped():
    # workflow.prefers_draft_before_send legitimately stores a bare boolean —
    # the guard must key on the request-marker set, NOT on "value is boolean".
    assert _is_request_shaped({"prefers_draft_before_send": True}) is False


def test_substantive_values_are_not_request_shaped():
    # A routine / real preference carries descriptive content — keep it.
    assert _is_request_shaped(
        {"routine": "morning briefing", "cadence": "daily", "time": "08:00"}
    ) is False
    # A dict with a marker key but also a real field is not a pure request flag.
    assert _is_request_shaped({"requested": True, "meal": "high-protein"}) is False
    # A marker key carrying descriptive text may be a real fact — don't drop it
    # here (the prompt rule handles textual one-off asks upstream).
    assert _is_request_shaped({"request": "eggs every morning"}) is False


def test_non_dict_and_empty_are_not_request_shaped():
    assert _is_request_shaped({}) is False
    assert _is_request_shaped(None) is False
    assert _is_request_shaped("give me a breakfast idea") is False
    assert _is_request_shaped(["requested"]) is False


# ── Bug B: remember-tool double-store collapse ───────────────────────────────

@pytest.fixture()
def pm_root(tmp_path, monkeypatch):
    root = tmp_path / "pm"
    root.mkdir()
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snapshots")
    monkeypatch.setattr(ts, "PERSONAL_MEMORY_ENABLED", True)
    return root


def _real_state(user_id: str, session_id: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        journal_store=ts.JournalStore(user_id=user_id),
        personal_memory_store=PersonalMemoryStore(user_id=user_id),
        session_store=types.SimpleNamespace(session_id=session_id),
        user_id=user_id,
    )


def _applied_topic_events(state, topic: str):
    return [e for e in state.journal_store.load_all() if e.topic == topic and e.applied]


def test_restated_fact_under_second_key_collapses(pm_root):
    """The Atlas bug: same fact, two keys — second call is absorbed, not stored."""
    state = _real_state("usr_atlas", "sess_atlas")

    first = ts._store_remembered_fact(
        state,
        topic="projects",
        key_slug="codename_atlas",
        value_text="I'm working on a project codenamed Atlas.",
    )
    assert first.startswith("Stored:")

    second = ts._store_remembered_fact(
        state, topic="projects", key_slug="project_codename", value_text="Atlas"
    )
    # "Atlas" is contained in the sentence already stored this session → collapse.
    assert "Already remembered" in second
    assert "Stored:" not in second

    # Exactly ONE projects fact ended up in the journal, not two.
    assert len(_applied_topic_events(state, "projects")) == 1


def test_distinct_facts_same_topic_do_not_collapse(pm_root):
    """Two genuinely different projects must both persist (no over-dedup)."""
    state = _real_state("usr_two", "sess_two")

    a = ts._store_remembered_fact(
        state, topic="projects", key_slug="project_a", value_text="Atlas"
    )
    b = ts._store_remembered_fact(
        state, topic="projects", key_slug="project_b", value_text="Borealis"
    )
    assert a.startswith("Stored:")
    assert b.startswith("Stored:")
    assert len(_applied_topic_events(state, "projects")) == 2


def test_short_substring_facts_are_length_gated(pm_root):
    """'Sam' vs 'Sam Smith': the <4-char value must not trigger a false collapse."""
    state = _real_state("usr_sam", "sess_sam")

    a = ts._store_remembered_fact(
        state, topic="identity", key_slug="name", value_text="Sam"
    )
    b = ts._store_remembered_fact(
        state, topic="identity", key_slug="full_name", value_text="Sam Smith"
    )
    assert a.startswith("Stored:")
    assert b.startswith("Stored:")
    assert len(_applied_topic_events(state, "identity")) == 2


def test_contained_value_different_session_does_not_collapse(pm_root):
    """The containment backstop is session-scoped: it only catches the model's
    redundant double-call within ONE session. A contained restatement in a LATER
    session (different key, so journal-level H1 dedup doesn't apply either) is a
    fresh statement and must persist, not be silently absorbed."""
    first_state = _real_state("usr_xsess", "sess_one")
    r1 = ts._store_remembered_fact(
        first_state,
        topic="projects",
        key_slug="codename_atlas",
        value_text="I'm working on a project codenamed Atlas.",
    )
    assert r1.startswith("Stored:")

    # New session, same user/journal, contained value under a DIFFERENT key —
    # in-session this would collapse, but across sessions it must NOT.
    second_state = _real_state("usr_xsess", "sess_two")
    r2 = ts._store_remembered_fact(
        second_state, topic="projects", key_slug="project_codename", value_text="Atlas"
    )
    assert r2.startswith("Stored:")
    assert len(_applied_topic_events(second_state, "projects")) == 2
