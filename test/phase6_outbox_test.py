"""
Phase 6 (W1) — durable routine-delivery outbox.

core.routine_outbox persists each user's pending routine frames under their
personal-memory dir so a process restart between a routine fire and its
delivery no longer drops the notice. apps.turtle_server keeps the in-memory
dict as the hot cache and write-throughs to disk; pop_pending_routine_notices
merges disk + memory (deduped) so a restarted process still surfaces queued
frames on the user's next connect.

Journal/data isolation mirrors phase3/phase5: PERSONAL_MEMORY_DIR is redirected
into a tmp dir so nothing touches the production data/ tree.
"""
from __future__ import annotations

import json

import pytest

import core.paths as core_paths
import core.routine_outbox as outbox
from core.guardrails import StorageCapExceededError


# ── fixtures / helpers ───────────────────────────────────────────────────

@pytest.fixture()
def pm_root(tmp_path, monkeypatch):
    """Redirect per-user memory writes into a tmp dir (see phase3/phase5)."""
    root = tmp_path / "pm"
    root.mkdir()
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root)
    return root


def _frame(key="workflow.morning_routine",
           fired="2026-07-20T07:30:00+00:00", msg="hi"):
    return {
        "type": "routine",
        "code": "routine_fire",
        "message": msg,
        "routine_key": key,
        "fired_at": fired,
    }


def _outbox_file(pm_root, user_id):
    return pm_root / user_id / "routine_outbox.json"


# ── core.routine_outbox: save / load round-trip ──────────────────────────

def test_save_creates_file_and_load_roundtrips(pm_root):
    f = _frame()
    outbox.save_outbox("u1", [f])

    path = _outbox_file(pm_root, "u1")
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == [f]
    assert outbox.load_outbox("u1") == [f]


def test_load_missing_file_returns_empty(pm_root):
    assert outbox.load_outbox("nobody") == []


# ── (e) empty save deletes the file ──────────────────────────────────────

def test_empty_save_deletes_file(pm_root):
    outbox.save_outbox("u1", [_frame()])
    path = _outbox_file(pm_root, "u1")
    assert path.exists()

    outbox.save_outbox("u1", [])  # empty → residue removed
    assert not path.exists()
    assert outbox.load_outbox("u1") == []


# ── (d) corrupt / non-list file → logged, treated empty, overwritten ─────

def test_corrupt_file_treated_empty_and_overwritten(pm_root, capsys):
    path = _outbox_file(pm_root, "u1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not ]json[", encoding="utf-8")

    assert outbox.load_outbox("u1") == []  # corrupt → definitively empty
    # Corrupt files get their own message (distinct from transient "load
    # failed", which returns None so pop won't clear an unread file).
    assert "routine_outbox corrupt" in capsys.readouterr().out

    # Next save overwrites the corrupt file cleanly.
    f = _frame(msg="clean")
    outbox.save_outbox("u1", [f])
    assert outbox.load_outbox("u1") == [f]


def test_non_list_json_treated_empty(pm_root, capsys):
    path = _outbox_file(pm_root, "u1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "routine"}), encoding="utf-8")

    assert outbox.load_outbox("u1") == []
    assert "corrupt (non-list)" in capsys.readouterr().out


# ── (f) per-user cap honored on disk ─────────────────────────────────────

def test_disk_cap_honored(pm_root):
    frames = [_frame(key=f"k{i}", fired=f"t{i}", msg=f"m{i}") for i in range(8)]
    outbox.save_outbox("u1", frames)

    loaded = outbox.load_outbox("u1")
    assert len(loaded) == outbox._MAX_FRAMES
    assert loaded[0]["message"] == "m3"   # most-recent _MAX_FRAMES kept
    assert loaded[-1]["message"] == "m7"


# ── (g) I/O and storage-cap failures on save are swallowed ───────────────

def test_save_io_failure_is_swallowed(pm_root, monkeypatch, capsys):
    def _boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(outbox, "atomic_write_json", _boom)
    outbox.save_outbox("u1", [_frame()])  # must NOT raise

    assert "routine_outbox save failed" in capsys.readouterr().out
    assert not _outbox_file(pm_root, "u1").exists()


def test_save_storage_cap_error_is_swallowed(pm_root, monkeypatch):
    def _cap(*a, **k):
        raise StorageCapExceededError("u1", 9_999, 1)

    monkeypatch.setattr(outbox, "atomic_write_json", _cap)
    outbox.save_outbox("u1", [_frame()])  # must NOT raise
    assert not _outbox_file(pm_root, "u1").exists()


# ── (a) turtle_server write-through: stash persists to disk ──────────────

def test_stash_write_through_persists(pm_root, monkeypatch):
    import apps.turtle_server as ts

    monkeypatch.setattr(ts, "_PENDING_ROUTINE_NOTICES", {})
    f = _frame(msg="stashed")
    ts._stash_pending_routine_notice("u1", f)

    # Memory hot cache holds it...
    assert ts._PENDING_ROUTINE_NOTICES["u1"] == [f]
    # ...and it was written through to disk.
    path = _outbox_file(pm_root, "u1")
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == [f]


def test_stash_write_through_caps_disk(pm_root, monkeypatch):
    import apps.turtle_server as ts

    monkeypatch.setattr(ts, "_PENDING_ROUTINE_NOTICES", {})
    for i in range(8):
        ts._stash_pending_routine_notice(
            "u1", _frame(key=f"k{i}", fired=f"t{i}", msg=f"m{i}")
        )

    disk = json.loads(_outbox_file(pm_root, "u1").read_text(encoding="utf-8"))
    assert len(disk) == ts._PENDING_ROUTINE_MAX_PER_USER
    assert disk[0]["message"] == "m3"
    assert disk[-1]["message"] == "m7"


# ── (b) pop merges disk + memory, dedupes, clears both ───────────────────

def test_pop_merges_disk_and_memory_dedupes_and_clears(pm_root, monkeypatch):
    import apps.turtle_server as ts

    fA = _frame(key="kA", fired="tA", msg="A")
    fB = _frame(key="kB", fired="tB", msg="B")
    # Memory holds A (the write-through mirror); disk holds A + an extra B.
    monkeypatch.setattr(ts, "_PENDING_ROUTINE_NOTICES", {"u1": [fA]})
    outbox.save_outbox("u1", [fA, fB])

    drained = ts.pop_pending_routine_notices("u1")

    # disk-first order, A deduped against the memory mirror, B surfaced.
    assert [(fr["routine_key"], fr["fired_at"]) for fr in drained] == [
        ("kA", "tA"), ("kB", "tB"),
    ]
    # Both stores cleared.
    assert ts._PENDING_ROUTINE_NOTICES.get("u1", []) == []
    assert not _outbox_file(pm_root, "u1").exists()
    # Second pop is empty.
    assert ts.pop_pending_routine_notices("u1") == []


# ── (c) restart simulation: memory dies, disk survives ───────────────────

def test_restart_simulation_pop_returns_disk_frames(pm_root, monkeypatch):
    import apps.turtle_server as ts

    monkeypatch.setattr(ts, "_PENDING_ROUTINE_NOTICES", {})
    f = _frame(msg="survivor")
    ts._stash_pending_routine_notice("u1", f)  # persisted to disk
    assert _outbox_file(pm_root, "u1").exists()

    # Simulate process death: the in-memory dict is empty on the new process.
    monkeypatch.setattr(ts, "_PENDING_ROUTINE_NOTICES", {})

    drained = ts.pop_pending_routine_notices("u1")
    assert drained == [f]                         # recovered from disk
    assert not _outbox_file(pm_root, "u1").exists()  # cleared after drain


# ── (g') a failed write-through leaves the in-memory queue working ───────

def test_stash_save_failure_keeps_memory_queue(pm_root, monkeypatch, capsys):
    import apps.turtle_server as ts

    monkeypatch.setattr(ts, "_PENDING_ROUTINE_NOTICES", {})

    def _boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(outbox, "atomic_write_json", _boom)

    f = _frame(msg="mem-only")
    ts._stash_pending_routine_notice("u1", f)  # must NOT raise

    # The persist failed (logged), but the memory queue is intact and usable.
    assert "routine_outbox save failed" in capsys.readouterr().out
    assert ts._PENDING_ROUTINE_NOTICES["u1"] == [f]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
