"""
Sandbox — the guest-side filesystem helper (core/sandbox/guest/fsops.py).

fsops.py normally runs INSIDE the container, bind-mounted read-only. It is
stdlib-only and takes its root from a module constant, so it can be driven
directly here against a tmp dir with no Docker involved.

What is being verified is the CONTRACT the runtime depends on:
  * exactly one JSON object on stdout, always parseable, even on failure;
  * exit code 0 regardless of outcome (the JSON body carries the result, so a
    non-zero exit would be ambiguous between "bad path" and "container died");
  * containment rejections produce clean, actionable errors.

Note what these tests do NOT prove. The containment checks in fsops.py are
belt-and-braces for good error messages — the actual jail is the mount
namespace. A symlink pointing at /etc/passwd resolves to the CONTAINER's
/etc/passwd, which is why the helper runs in there and not host-side.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "turtle_fsops_under_test",
    Path(__file__).resolve().parents[1] / "core" / "sandbox" / "guest" / "fsops.py",
)
fsops = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(fsops)


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setattr(fsops, "WORKSPACE", str(ws))
    return ws


def _run(monkeypatch, capsys, argv, stdin: bytes = b"") -> dict:
    """Invoke main() and parse the single JSON object it emits."""
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(stdin)))
    with pytest.raises(SystemExit) as exc:
        fsops.main(["fsops.py", *argv])
        raise SystemExit(0)
    assert exc.value.code == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def _run_ok(monkeypatch, capsys, argv, stdin: bytes = b"") -> dict:
    """Same, for the success path where main() returns instead of exiting."""
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(stdin)))
    fsops.main(["fsops.py", *argv])
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


# ── read ─────────────────────────────────────────────────────────────────

def test_read_returns_content(workspace, monkeypatch, capsys):
    (workspace / "a.txt").write_text("hello", encoding="utf-8")
    out = _run_ok(monkeypatch, capsys, ["read", "a.txt", "1000"])
    assert out == {
        "ok": True, "path": "a.txt", "size_bytes": 5,
        "truncated": False, "content": "hello",
    }


def test_read_truncates_and_says_so(workspace, monkeypatch, capsys):
    (workspace / "big.txt").write_text("y" * 500, encoding="utf-8")
    out = _run_ok(monkeypatch, capsys, ["read", "big.txt", "100"])
    assert out["truncated"] is True
    assert len(out["content"]) == 100
    assert out["size_bytes"] == 500


def test_read_of_partial_utf8_does_not_crash(workspace, monkeypatch, capsys):
    """A byte-capped read routinely lands mid-sequence; a decode error would turn
    'your file is long' into 'the tool crashed'."""
    (workspace / "u.txt").write_bytes("héllo wörld".encode("utf-8"))
    out = _run_ok(monkeypatch, capsys, ["read", "u.txt", "2"])
    assert out["ok"] is True
    assert isinstance(out["content"], str)


def test_read_missing_file(workspace, monkeypatch, capsys):
    out = _run(monkeypatch, capsys, ["read", "nope.txt", "100"])
    assert out["ok"] is False and out["code"] == "not_found"


def test_read_of_a_directory_is_a_clean_error(workspace, monkeypatch, capsys):
    (workspace / "sub").mkdir()
    out = _run(monkeypatch, capsys, ["read", "sub", "100"])
    assert out["code"] == "is_dir"


# ── write ────────────────────────────────────────────────────────────────

def test_write_creates_parent_directories(workspace, monkeypatch, capsys):
    out = _run_ok(monkeypatch, capsys, ["write", "deep/nested/x.txt", "1000"], b"data")
    assert out["ok"] is True and out["bytes_written"] == 4
    assert (workspace / "deep" / "nested" / "x.txt").read_bytes() == b"data"


def test_write_is_atomic_and_leaves_no_temp_file(workspace, monkeypatch, capsys):
    """A half-written file after a timeout is worse than no write: the model
    reads it back and reasons about truncated content as if it were real."""
    _run_ok(monkeypatch, capsys, ["write", "x.txt", "1000"], b"complete")
    assert (workspace / "x.txt").read_bytes() == b"complete"
    assert list(workspace.glob("*.turtle-tmp")) == []


def test_write_reports_overwrite(workspace, monkeypatch, capsys):
    (workspace / "x.txt").write_text("old", encoding="utf-8")
    out = _run_ok(monkeypatch, capsys, ["write", "x.txt", "1000"], b"new")
    assert out["overwrote"] is True
    assert (workspace / "x.txt").read_text(encoding="utf-8") == "new"


def test_write_refuses_oversized_payload(workspace, monkeypatch, capsys):
    out = _run(monkeypatch, capsys, ["write", "x.txt", "10"], b"A" * 50)
    assert out["code"] == "too_large"
    assert not (workspace / "x.txt").exists()


# ── list ─────────────────────────────────────────────────────────────────

def test_list_reports_types_and_sizes(workspace, monkeypatch, capsys):
    (workspace / "a.txt").write_text("xyz", encoding="utf-8")
    (workspace / "sub").mkdir()
    out = _run_ok(monkeypatch, capsys, ["list", "."])
    by_name = {e["name"]: e for e in out["entries"]}
    assert by_name["a.txt"]["type"] == "file" and by_name["a.txt"]["size_bytes"] == 3
    assert by_name["sub"]["type"] == "dir" and by_name["sub"]["size_bytes"] is None


def test_list_caps_entry_count(workspace, monkeypatch, capsys):
    for i in range(fsops.MAX_LIST_ENTRIES + 25):
        (workspace / f"f{i:04d}.txt").write_text("x", encoding="utf-8")
    out = _run_ok(monkeypatch, capsys, ["list", "."])
    assert out["truncated"] is True
    assert len(out["entries"]) == fsops.MAX_LIST_ENTRIES


def test_list_of_a_file_is_a_clean_error(workspace, monkeypatch, capsys):
    (workspace / "a.txt").write_text("x", encoding="utf-8")
    out = _run(monkeypatch, capsys, ["list", "a.txt"])
    assert out["code"] == "not_dir"


# ── containment (belt-and-braces, NOT the boundary) ──────────────────────

@pytest.mark.parametrize(
    "bad", ["/etc/passwd", "../secret", "a/../../secret", "..\\windows\\x", "/"]
)
def test_absolute_and_traversal_paths_are_refused(workspace, monkeypatch, capsys, bad):
    out = _run(monkeypatch, capsys, ["read", bad, "100"])
    assert out["ok"] is False and out["code"] == "bad_path"


def test_symlink_out_of_the_workspace_is_refused(workspace, monkeypatch, capsys, tmp_path):
    """This is the case v1's host-side Path.resolve() check would have FOLLOWED.
    Here the realpath check catches it; in production the mount namespace means
    there is nothing outside to point at in the first place."""
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (workspace / "link.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation requires privileges on this platform")
    out = _run(monkeypatch, capsys, ["read", "link.txt", "100"])
    assert out["ok"] is False and out["code"] == "bad_path"


# ── contract ─────────────────────────────────────────────────────────────

def test_every_outcome_emits_exactly_one_json_object(workspace, monkeypatch, capsys):
    """The runtime parses the last stdout line as JSON. Anything else means the
    container itself misbehaved, which must stay distinguishable from a clean
    helper-reported error."""
    (workspace / "a.txt").write_text("x", encoding="utf-8")
    for argv, runner in (
        (["read", "a.txt", "100"], _run_ok),
        (["read", "missing", "100"], _run),
        (["list", "."], _run_ok),
        (["bogus-op", "."], _run),
        ([], _run),
    ):
        payload = runner(monkeypatch, capsys, argv)
        assert isinstance(payload, dict) and "ok" in payload


def test_helper_is_stdlib_only():
    """It runs against whatever interpreter the sandbox image ships, not
    Turtle's, so a `core.*` or third-party import would break at exec time in a
    way nothing here would catch."""
    source = (
        Path(__file__).resolve().parents[1] / "core" / "sandbox" / "guest" / "fsops.py"
    ).read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            module = stripped.split()[1].split(".")[0]
            assert module in {"json", "os", "sys"}, f"non-stdlib import: {stripped}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
