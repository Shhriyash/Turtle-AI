"""
core/sandbox/models.py
----------------------
Pydantic result types returned by the sandbox tools.

Pydantic (not dataclasses, as v1 had it) for one concrete reason: these get
serialised straight into a tool return string with .model_dump_json(), and v1's
dataclass version needed a `default=lambda o: o.__dict__` hack in json.dumps for
nested DirEntry lists — which silently produces garbage the moment a field is
anything other than a plain scalar. model_dump_json() is correct by construction.

Every result carries `denied` + `denied_reason` rather than raising, because a
raised exception inside an @agent.tool body becomes an opaque retry loop: the
model gets "tool failed", tries the same thing again, and burns the request
budget. A structured denial it can read tells it to stop and explain.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

TRUNCATION_SENTINEL = "[TRUNCATED: capped at {cap} bytes; {total} bytes produced]"


class ShellResult(BaseModel):
    """Outcome of one sandboxed exec."""

    denied: bool = False
    denied_reason: Optional[str] = None

    argv: list[str] = Field(default_factory=list)
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""

    timed_out: bool = False
    truncated: bool = False
    duration_ms: int = 0

    # Which boundary actually ran this. Always "docker" today; present so that a
    # future gVisor/Firecracker runtime is visible in the transcript and the
    # audit log rather than being an invisible swap.
    isolation: str = "docker"
    # Tripwire rule ids that matched. Informational — a hit does NOT deny by
    # default (see docs/sandbox_design_v2.md §2 Layer 4).
    tripwire: list[str] = Field(default_factory=list)

    def to_agent_string(self) -> str:
        if self.denied:
            return f"[sandbox denied] {self.denied_reason}"
        head = f"exit_code={self.exit_code}"
        if self.timed_out:
            head += " (timed out)"
        if self.truncated:
            head += " (output truncated)"
        parts = [head]
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        if not self.stdout and not self.stderr:
            parts.append("(no output)")
        return "\n".join(parts)


class FileResult(BaseModel):
    """Outcome of a read or write."""

    denied: bool = False
    denied_reason: Optional[str] = None

    path: str = ""
    ok: bool = False
    content: Optional[str] = None      # reads only
    bytes_written: Optional[int] = None  # writes only
    truncated: bool = False
    snapshot: Optional[str] = None     # snapshot filename taken before overwrite
    tripwire: list[str] = Field(default_factory=list)

    def to_agent_string(self) -> str:
        if self.denied:
            return f"[sandbox denied] {self.denied_reason}"
        if not self.ok:
            return f"[sandbox error] {self.denied_reason or 'operation failed'}"
        if self.content is not None:
            suffix = " (truncated)" if self.truncated else ""
            return f"{self.path}{suffix}:\n{self.content}"
        return f"Wrote {self.bytes_written} bytes to {self.path}"


class DirEntry(BaseModel):
    name: str
    type: Literal["file", "dir", "symlink", "other"] = "file"
    size_bytes: Optional[int] = None
    modified_utc: Optional[str] = None


class DirResult(BaseModel):
    denied: bool = False
    denied_reason: Optional[str] = None

    path: str = ""
    ok: bool = False
    entries: list[DirEntry] = Field(default_factory=list)
    truncated: bool = False

    def to_agent_string(self) -> str:
        if self.denied:
            return f"[sandbox denied] {self.denied_reason}"
        if not self.ok:
            return f"[sandbox error] {self.denied_reason or 'listing failed'}"
        if not self.entries:
            return f"{self.path}: (empty)"
        lines = [f"{self.path}:"]
        for e in self.entries:
            size = "" if e.size_bytes is None else f"  {e.size_bytes}B"
            marker = "/" if e.type == "dir" else ("@" if e.type == "symlink" else "")
            lines.append(f"  {e.name}{marker}{size}")
        if self.truncated:
            lines.append("  … (listing truncated)")
        return "\n".join(lines)


def truncate_output(raw: bytes, cap: int) -> tuple[str, bool]:
    """Decode with a visible sentinel when capped.

    errors="replace" rather than "strict": a capped read almost always lands
    mid-UTF-8-sequence, and a UnicodeDecodeError there would turn "your output
    was long" into "the tool crashed"."""
    if len(raw) <= cap:
        return raw.decode("utf-8", errors="replace"), False
    text = raw[:cap].decode("utf-8", errors="replace")
    return text + "\n" + TRUNCATION_SENTINEL.format(cap=cap, total=len(raw)), True


def render_stream(kept: bytes, total: int, cap: int) -> tuple[str, bool]:
    """Render a stream the RUNNER may already have capped.

    The subtle failure this exists for: default_runner caps while streaming, so
    by the time the tool sees the bytes they are already <= cap and a naive
    truncate_output() finds nothing to truncate — the model then reads a
    silently-clipped result as if it were the complete output and reasons on it.
    `total` is what was actually produced, so the sentinel is driven by that."""
    text, truncated = truncate_output(kept, cap)
    if not truncated and total > len(kept):
        text += "\n" + TRUNCATION_SENTINEL.format(cap=cap, total=total)
        truncated = True
    return text, truncated
