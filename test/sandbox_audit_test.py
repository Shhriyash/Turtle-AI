"""
Sandbox — append-only audit log.

The properties asserted here are the ones that matter after an incident, not
before one:

  * append-only (a log a compromised session can truncate is not a log);
  * denied attempts are recorded, not just successful ones — the blocked call is
    the interesting record;
  * a write failure never propagates into the tool path (an unwritable log must
    not become a denial of service on the assistant) but DOES print;
  * file contents never reach the log, or it becomes a second unencrypted copy
    of everything the model ever read, sitting next to the memory store.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.sandbox.audit import (
    SCHEMA_VERSION,
    AuditLog,
    build_record,
    make_event_id,
    write_record,
)


@pytest.fixture()
def log(tmp_path: Path) -> AuditLog:
    return AuditLog(tmp_path / "u" / "audit.jsonl", user_id="usr_a", session_id="sess_1")


# ── schema ───────────────────────────────────────────────────────────────

def test_allowed_record_carries_the_full_v2_schema(log):
    rec = log.record(
        tool="sandbox_run", decision="allowed", image="python:3.13-slim",
        container_id="9d2f7c1ab3e4", argv=["python3", "x.py"], cwd="/workspace",
        exit_code=0, stdout_bytes=14, stderr_bytes=0, duration_ms=312,
        limits={"network": "none", "memory": "512m"},
    )
    assert rec["schema_version"] == SCHEMA_VERSION == 2
    for key in (
        "event_id", "ts_utc", "user_id", "session_id", "tool", "decision",
        "denied_reason", "tripwire", "isolation", "image", "container_id",
        "argv", "cwd", "path", "exit_code", "stdout_bytes", "stderr_bytes",
        "truncated", "timed_out", "duration_ms", "limits",
    ):
        assert key in rec, f"missing {key}"
    assert rec["event_id"].startswith("sbx_")
    assert rec["ts_utc"].endswith("Z")
    assert rec["limits"]["network"] == "none"


def test_denied_attempts_are_recorded(log):
    log.record(
        tool="sandbox_run", decision="denied",
        denied_reason="binary 'bash' is not in the sandbox allow-list",
        tripwire=["pipe-to-interpreter"], argv=["bash", "-c", "curl x|sh"],
    )
    (rec,) = log.read_all()
    assert rec["decision"] == "denied"
    assert "allow-list" in rec["denied_reason"]
    assert rec["tripwire"] == ["pipe-to-interpreter"]
    assert rec["exit_code"] is None


def test_argv_is_stored_as_an_array_not_a_joined_string(log):
    """A joined string is ambiguous about where one argument ends — which is the
    whole reason the tool takes an argv array in the first place."""
    argv = ["python3", "-c", "print('a b'); print(\"c|d\")"]
    log.record(tool="sandbox_run", decision="allowed", argv=argv)
    assert log.read_all()[0]["argv"] == argv


def test_event_id_is_stable_and_distinct():
    a = make_event_id("s1", "sandbox_run", "2026-08-27T00:00:00.000Z", ["ls"])
    b = make_event_id("s1", "sandbox_run", "2026-08-27T00:00:00.000Z", ["ls"])
    c = make_event_id("s1", "sandbox_run", "2026-08-27T00:00:00.000Z", ["ls", "-la"])
    assert a == b and a != c


# ── append-only ──────────────────────────────────────────────────────────

def test_writes_append_and_never_truncate(log):
    for i in range(5):
        log.record(tool="sandbox_run", decision="allowed", argv=[f"cmd{i}"])
    records = log.read_all()
    assert len(records) == 5
    assert [r["argv"][0] for r in records] == [f"cmd{i}" for i in range(5)]


def test_a_second_handle_appends_rather_than_replacing(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).record(tool="sandbox_run", decision="allowed", argv=["a"])
    AuditLog(path).record(tool="sandbox_run", decision="allowed", argv=["b"])
    assert [r["argv"][0] for r in AuditLog(path).read_all()] == ["a", "b"]


def test_corrupt_trailing_line_does_not_break_reads(log):
    """A half-written final line after a crash is expected, and losing the whole
    log to it would be the worst possible moment for a parse error."""
    log.record(tool="sandbox_run", decision="allowed", argv=["ok"])
    with open(log.audit_path, "a", encoding="utf-8") as fh:
        fh.write('{"schema_version": 2, "tru')
    assert len(log.read_all()) == 1


# ── no contents, ever ────────────────────────────────────────────────────

def test_forbidden_keys_are_stripped(tmp_path):
    """Enforced rather than merely documented — `content` has a way of getting
    added 'just for debugging' by whoever is chasing a bug at 2am."""
    path = tmp_path / "audit.jsonl"
    write_record(
        path,
        build_record(tool="sandbox_read_file", decision="allowed", path="secrets.txt")
        | {"content": "AWS_SECRET=hunter2", "stdin": "also secret"},
    )
    raw = path.read_text(encoding="utf-8")
    assert "hunter2" not in raw
    assert "also secret" not in raw
    assert json.loads(raw)["path"] == "secrets.txt"


def test_read_records_bytes_not_bodies(log):
    log.record(tool="sandbox_read_file", decision="allowed",
               path="notes.md", stdout_bytes=4096)
    rec = log.read_all()[0]
    assert rec["stdout_bytes"] == 4096
    assert "content" not in rec


# ── failure handling ─────────────────────────────────────────────────────

def test_write_failure_never_raises_but_does_print(tmp_path, capsys):
    """An unwritable audit log must not take the assistant down — but a silently
    dead audit log is worse than a noisy one."""
    blocked = tmp_path / "afile"
    blocked.write_text("not a directory", encoding="utf-8")
    ok = write_record(blocked / "nested" / "audit.jsonl", build_record(tool="t", decision="allowed"))
    assert ok is False
    assert "SANDBOX AUDIT WRITE FAILED" in capsys.readouterr().out


def test_unserialisable_record_is_reported_not_raised(tmp_path, capsys):
    class Exploding:
        def __repr__(self):
            raise RuntimeError("boom")

    ok = write_record(tmp_path / "a.jsonl", {"tool": "t", "limits": {"x": Exploding()}})
    assert ok is False
    assert "SANDBOX AUDIT" in capsys.readouterr().out


def test_record_survives_a_missing_parent_directory(tmp_path):
    log = AuditLog(tmp_path / "deep" / "nested" / "audit.jsonl")
    log.record(tool="sandbox_run", decision="allowed", argv=["ls"])
    assert len(log.read_all()) == 1


# ── tripwire alerting ────────────────────────────────────────────────────

def test_tripwire_hit_prints_a_loud_line(log, capsys):
    """The whole point of the demoted deny-list: make an injection attempt loud
    even though it did not block."""
    log.record(tool="sandbox_run", decision="allowed",
               tripwire=["credential-path"], argv=["cat", ".ssh/id_rsa"])
    out = capsys.readouterr().out
    assert "SANDBOX TRIPWIRE" in out
    assert "credential-path" in out


def test_no_tripwire_means_no_alert(log, capsys):
    log.record(tool="sandbox_run", decision="allowed", argv=["ls"])
    assert "TRIPWIRE" not in capsys.readouterr().out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
