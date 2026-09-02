"""
core/sandbox/audit.py
---------------------
Append-only JSONL audit log. One record per sandbox tool invocation — allowed,
denied, or errored.

Design rules, each of which exists because the obvious alternative is wrong:

* **Append-only.** Opened "a", never "w". A log a compromised session can
  truncate is not a log. (Rotation is deferred; when it lands it must copy-then-
  start-a-new-file, never truncate in place.)

* **Written BEFORE the result returns to the model.** If the process dies
  mid-call, the attempt is still on disk. An audit log that only records
  successful completions loses exactly the records you want after an incident.

* **Never raises into the tool path.** An unwritable audit log must not become a
  denial of service on the assistant. But it DOES print loudly — a silently dead
  audit log is worse than a noisy one, and "we thought it was logging" is the
  standard post-incident sentence.

* **Never records file contents.** Paths and byte counts only. Otherwise the log
  becomes a second, unencrypted copy of everything the model ever read, sitting
  next to the memory store. Same reason argv is recorded but stdin is not.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 2

# Serialise appends across threads in one process. JSONL + O_APPEND is atomic
# enough for small records on POSIX, but Windows has no such guarantee and the
# routine scheduler already fires work off the app loop onto a worker thread —
# so two sandbox calls really can race here.
_WRITE_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def make_event_id(session_id: str, tool: str, ts: str, argv: Sequence[str]) -> str:
    """Stable id for cross-referencing a transcript turn with its audit record."""
    material = f"{session_id}|{tool}|{ts}|{json.dumps(list(argv), sort_keys=True)}"
    return "sbx_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def build_record(
    *,
    tool: str,
    decision: str,
    user_id: str = "",
    session_id: str = "",
    denied_reason: str | None = None,
    tripwire: Sequence[str] = (),
    isolation: str = "docker",
    image: str = "",
    container_id: str = "",
    argv: Sequence[str] = (),
    cwd: str = "",
    path: str | None = None,
    exit_code: int | None = None,
    stdout_bytes: int | None = None,
    stderr_bytes: int | None = None,
    truncated: bool = False,
    timed_out: bool = False,
    duration_ms: int = 0,
    limits: Mapping[str, Any] | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    """Shape one record. Pure — no I/O — so the schema is testable without disk."""
    stamp = ts or _utc_now_iso()
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": make_event_id(session_id, tool, stamp, argv),
        "ts_utc": stamp,
        "user_id": user_id,
        "session_id": session_id,
        "tool": tool,
        "decision": decision,             # allowed | denied | error
        "denied_reason": denied_reason,
        "tripwire": list(tripwire),
        "isolation": isolation,
        "image": image,
        "container_id": container_id,
        "argv": list(argv),
        "cwd": cwd,
        "path": path,
        "exit_code": exit_code,
        "stdout_bytes": stdout_bytes,
        "stderr_bytes": stderr_bytes,
        "truncated": truncated,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "limits": dict(limits or {}),
    }


# Keys that must never appear in a record, enforced rather than merely
# documented. `content`/`stdin` have a way of getting added "just for debugging"
# by whoever is chasing a bug at 2am.
_FORBIDDEN_KEYS = frozenset({"content", "stdin", "body", "text", "data", "secret"})


def _strip_forbidden(record: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in _FORBIDDEN_KEYS}


def write_record(audit_path: Path, record: Mapping[str, Any]) -> bool:
    """Append one record. Returns True on success. NEVER raises."""
    try:
        clean = _strip_forbidden(record)
        line = json.dumps(clean, ensure_ascii=False, default=str, separators=(",", ":"))
    except Exception as exc:  # a record we cannot serialise is still worth noting
        print(f"LOG: SANDBOX AUDIT SERIALISE FAILED — {exc}")
        return False

    try:
        with _WRITE_LOCK:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            # O_APPEND on the fd (not just mode "a" semantics) so concurrent
            # writers cannot interleave partial lines on POSIX.
            fd = os.open(
                audit_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
            try:
                os.write(fd, (line + "\n").encode("utf-8"))
            finally:
                os.close(fd)
        return True
    except Exception as exc:
        print(f"LOG: SANDBOX AUDIT WRITE FAILED ({audit_path}) — {exc}")
        return False


class AuditLog:
    """Thin per-user handle. Holds the path, nothing else — no open fd, so a log
    dir that disappears mid-session degrades to a warning per write instead of a
    stale descriptor writing into a deleted inode."""

    def __init__(self, audit_path: Path, *, user_id: str = "", session_id: str = "") -> None:
        self.audit_path = Path(audit_path)
        self.user_id = user_id
        self.session_id = session_id

    def record(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("user_id", self.user_id)
        kwargs.setdefault("session_id", self.session_id)
        rec = build_record(**kwargs)
        write_record(self.audit_path, rec)
        if rec.get("tripwire"):
            # The whole point of the demoted deny-list: make an injection attempt
            # loud even though it did not block.
            print(
                f"LOG: SANDBOX TRIPWIRE {rec['tripwire']} tool={rec['tool']} "
                f"decision={rec['decision']} session={rec['session_id']} "
                f"argv={rec['argv']}"
            )
        return rec

    def read_all(self) -> list[dict[str, Any]]:
        """Parse the log back. Test/forensics helper — skips corrupt lines rather
        than failing, since a half-written final line after a crash is expected."""
        if not self.audit_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.audit_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
