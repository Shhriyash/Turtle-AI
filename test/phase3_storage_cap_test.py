"""
Phase 3 — storage-cap user notification (W4).

When a memory write hits the per-user storage cap, StorageCapExceededError used
to vanish into a broad `except Exception` and the user never learned their
writes were failing. These tests verify the new plumbing:

  * _notify_storage_cap fires at most once per session, always LOGs, and stashes
    a WS "notice" frame for the handler to deliver;
  * the three write funnels — _apply_explicit_facts_from_turn,
    _journal_and_queue_candidates, and the remember tool's store helper
    (_store_remembered_fact) — catch the cap error, notify, and never let it
    escape; the remember helper returns an honest failure string instead of a
    fabricated "Stored".

All offline: the cap breach is injected by a journal_store double whose
append_many raises, so no real disk cap or large writes are needed.
"""
from __future__ import annotations

import types

import pytest

import core.paths as core_paths
from apps import turtle_server as ts
from core.guardrails import StorageCapExceededError
from core.personal_memory_extract import PersonalMemoryCandidate
from core.personal_memory_store import PersonalMemoryStore


# ── fixtures / doubles ───────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_notice_registries():
    """Isolate the module-level once-guard + pending registry between tests."""
    ts._STORAGE_CAP_NOTIFIED.clear()
    ts._PENDING_STORAGE_CAP_NOTICES.clear()
    yield
    ts._STORAGE_CAP_NOTIFIED.clear()
    ts._PENDING_STORAGE_CAP_NOTICES.clear()


@pytest.fixture()
def pm_root(tmp_path, monkeypatch):
    root = tmp_path / "pm"
    root.mkdir()
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snapshots")
    monkeypatch.setattr(ts, "PERSONAL_MEMORY_ENABLED", True)
    return root


class _RaisingJournal:
    """Journal double whose write path always trips the storage cap."""

    def __init__(self, user_id: str = "usr_cap") -> None:
        self.user_id = user_id

    def append_many(self, events):
        raise StorageCapExceededError(self.user_id, used_bytes=100, cap_bytes=50)

    def load_all(self):
        return []


def _capped_state(user_id: str, session_id: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        journal_store=_RaisingJournal(user_id),
        personal_memory_store=PersonalMemoryStore(user_id=user_id),
        confirmation_gate=types.SimpleNamespace(queue_candidate=lambda e: True),
        session_store=types.SimpleNamespace(session_id=session_id),
        user_id=user_id,
    )


def _candidate(*, topic="identity", key="name", value="Shriyash", evidence="my name is shriyash"):
    return PersonalMemoryCandidate(
        topic=topic,
        key=key,
        value=value,
        line=f"- {key}: {value}",
        overwrite_policy="replace",
        confidence="confirmed",
        sensitivity="normal",
        source_session_id="s1",
        evidence=evidence,
        source="explicit",
        extraction_source="llm_turn",
    )


# ── _notify_storage_cap semantics ────────────────────────────────────────

def test_notify_fires_once_per_session(pm_root, capsys):
    state = _capped_state("usr_a", "sess_a")

    first = ts._notify_storage_cap(state)
    second = ts._notify_storage_cap(state)

    assert first is True    # first breach this session → fires
    assert second is False  # subsequent breaches suppressed

    out = capsys.readouterr().out
    assert out.count("storage cap reached") == 1  # LOG printed exactly once

    # A WS "notice" frame is stashed for the handler to deliver.
    notice = ts.pop_pending_storage_cap_notice("sess_a")
    assert notice == {
        "type": "notice",
        "code": "storage_cap",
        "message": ts._STORAGE_CAP_NOTICE_MESSAGE,
    }


def test_notify_distinct_sessions_each_fire(pm_root):
    assert ts._notify_storage_cap(_capped_state("u", "sess_1")) is True
    assert ts._notify_storage_cap(_capped_state("u", "sess_2")) is True


# ── _store_remembered_fact (remember tool path) ──────────────────────────

def test_remember_helper_returns_honest_failure_on_cap(pm_root, capsys):
    state = _capped_state("usr_rem", "sess_rem")

    result = ts._store_remembered_fact(
        state, topic="identity", key_slug="nickname", value_text="Shri"
    )

    # Honest failure the model relays — NOT a fabricated "Stored".
    assert result.startswith("I couldn't save that")
    assert "cap" in result
    assert "Stored:" not in result
    assert capsys.readouterr().out.count("storage cap reached") == 1
    assert ts.pop_pending_storage_cap_notice("sess_rem") is not None


def test_remember_helper_success_path_still_stores(pm_root):
    # A real journal_store under the tmp root must still succeed normally.
    state = types.SimpleNamespace(
        journal_store=ts.JournalStore(user_id="usr_ok"),
        personal_memory_store=PersonalMemoryStore(user_id="usr_ok"),
        session_store=types.SimpleNamespace(session_id="sess_ok"),
        user_id="usr_ok",
    )
    result = ts._store_remembered_fact(
        state, topic="identity", key_slug="nickname", value_text="Shri"
    )
    assert result.startswith("Stored:")
    assert "identity.nickname" in result


# ── _journal_and_queue_candidates funnel ─────────────────────────────────

def test_journal_and_queue_notifies_and_swallows_cap(pm_root, capsys):
    state = _capped_state("usr_jq", "sess_jq")

    # Must not raise — the cap error is caught inside the funnel.
    queued = ts._journal_and_queue_candidates(state, [_candidate()], session_id="sess_jq")

    assert queued == 0  # nothing persisted, nothing queued
    assert capsys.readouterr().out.count("storage cap reached") == 1
    assert ts.pop_pending_storage_cap_notice("sess_jq") is not None


# ── _apply_explicit_facts_from_turn funnel ───────────────────────────────

def test_apply_explicit_facts_notifies_and_swallows_cap(pm_root, capsys):
    state = _capped_state("usr_ap", "sess_ap")

    # "my email is ..." deterministically extracts an explicit identity fact,
    # so the funnel reaches the (capped) journal write. Must not raise.
    ts._apply_explicit_facts_from_turn(
        state,
        session_id="sess_ap",
        turn_id="t1",
        user_text="my email is bob@example.com",
        task_type="general",
    )

    assert capsys.readouterr().out.count("storage cap reached") == 1
    assert ts.pop_pending_storage_cap_notice("sess_ap") is not None


def test_apply_explicit_facts_notifies_once_across_turns(pm_root, capsys):
    state = _capped_state("usr_ap2", "sess_ap2")

    for _ in range(2):
        ts._apply_explicit_facts_from_turn(
            state,
            session_id="sess_ap2",
            turn_id="t",
            user_text="my email is bob@example.com",
            task_type="general",
        )

    # Same session across two failing turns → user notified exactly once.
    assert capsys.readouterr().out.count("storage cap reached") == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
