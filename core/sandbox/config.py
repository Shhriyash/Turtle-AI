"""
core/sandbox/config.py
----------------------
Frozen sandbox policy objects, plus the fail-closed gate that decides whether
they can be constructed at all.

Why a separate BaseSettings class instead of ~20 new fields on TurtleSettings:
apps/turtle_server.py and core/config.py are being refactored concurrently, and
this lands without touching either. Same pydantic-settings machinery, same .env
file, so behaviour and precedence are identical. Fold SandboxSettings into
TurtleSettings once that refactor is done (tracked in docs/sandbox_design_v2.md
§10).

Everything the runtime consumes is a FROZEN dataclass. That is deliberate: the
policy is read on every tool call from inside closures registered on several
Agent objects, and a mutable policy would let one turn's mutation leak into the
next. Freeze it once at agent-build time and never think about it again.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _PROJECT_ROOT / ".env"

Distribution = Literal["web", "chatbot", "desktop"]


# ---------------------------------------------------------------------------
# Default binary allow-list
# ---------------------------------------------------------------------------
# name -> absolute path INSIDE the container (python:3.13-slim layout).
#
# Absolute paths, not bare names, because a bare name is resolved by whatever
# PATH the container happens to have, and PATH is attacker-influenceable the
# moment anything writes into an early PATH entry. An absolute path is not.
#
# Deliberately ABSENT and not an oversight:
#   sh / bash  — an in-container shell destroys the argv-level audit trail and
#                re-opens every quoting/metacharacter bug the argv array closes.
#                Opt-in via TURTLE_SANDBOX_ALLOW_SHELL_COMPOSITION (off).
#   curl/wget  — useless under --network none, and a loaded gun the day someone
#                flips the network flag for a pip install.
#   chmod/chown/apt/su/sudo/nc/ssh — no legitimate use inside a throwaway box.
_DEFAULT_BINARIES: dict[str, str] = {
    "python3": "/usr/local/bin/python3",
    "python": "/usr/local/bin/python3",
    "pip": "/usr/local/bin/pip3",
    "pip3": "/usr/local/bin/pip3",
    "pytest": "/usr/local/bin/pytest",
    "ls": "/bin/ls",
    "cat": "/bin/cat",
    "echo": "/bin/echo",
    "grep": "/bin/grep",
    "sed": "/bin/sed",
    "head": "/usr/bin/head",
    "tail": "/usr/bin/tail",
    "wc": "/usr/bin/wc",
    "sort": "/usr/bin/sort",
    "uniq": "/usr/bin/uniq",
    "cut": "/usr/bin/cut",
    "tr": "/usr/bin/tr",
    "find": "/usr/bin/find",
    "awk": "/usr/bin/awk",
    "diff": "/usr/bin/diff",
    "git": "/usr/bin/git",
}

# Runtime-internal absolute paths. These are NEVER reachable from a model-authored
# argv (the allow-list check runs on argv[0] before the wrapper is applied); the
# runtime composes them itself.
GUEST_TIMEOUT_BIN = "/usr/bin/timeout"
GUEST_TEST_BIN = "/usr/bin/test"
GUEST_PYTHON_BIN = "/usr/local/bin/python3"
GUEST_FSOPS_PATH = "/opt/turtle/fsops.py"
GUEST_WORKSPACE = "/workspace"
GUEST_RO_MOUNT_BASE = "/mnt/ro"


class SandboxSettings(BaseSettings):
    """Raw env/.env surface. Nothing here is consumed directly — it is validated
    and frozen into SandboxConfig/FsPolicy/ShellPolicy below."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        # Fields are aliased to their env var names. Without populate_by_name a
        # kwarg using the FIELD name is silently ignored and the default wins —
        # which in tests means every "sandbox enabled" case quietly asserts
        # against a disabled sandbox and passes for the wrong reason.
        populate_by_name=True,
    )

    distribution: str = Field(default="web", alias="TURTLE_DISTRIBUTION")
    # OFF by default on every distribution, desktop included. Handing a model a
    # shell is a decision the user makes knowingly, not one they inherit from a
    # version bump.
    enabled: bool = Field(default=False, alias="TURTLE_SANDBOX_ENABLED")

    image: str = Field(default="python:3.13-slim", alias="TURTLE_SANDBOX_IMAGE")
    docker_bin: str = Field(default="docker", alias="TURTLE_SANDBOX_DOCKER_BIN")
    workspace_root: str = Field(default="", alias="TURTLE_SANDBOX_WORKSPACE_ROOT")

    network: str = Field(default="none", alias="TURTLE_SANDBOX_NETWORK")
    memory: str = Field(default="512m", alias="TURTLE_SANDBOX_MEMORY")
    cpus: str = Field(default="1.0", alias="TURTLE_SANDBOX_CPUS")
    pids_limit: int = Field(default=128, alias="TURTLE_SANDBOX_PIDS_LIMIT")
    run_as: str = Field(default="", alias="TURTLE_SANDBOX_RUN_AS")

    timeout_seconds: int = Field(default=30, alias="TURTLE_SANDBOX_TIMEOUT")
    max_output_bytes: int = Field(default=65536, alias="TURTLE_SANDBOX_MAX_OUTPUT_BYTES")
    max_read_bytes: int = Field(default=262144, alias="TURTLE_SANDBOX_MAX_READ_BYTES")
    max_write_bytes: int = Field(default=10485760, alias="TURTLE_SANDBOX_MAX_WRITE_BYTES")
    container_ttl_seconds: int = Field(default=900, alias="TURTLE_SANDBOX_CONTAINER_TTL")

    allowed_binaries: str = Field(default="", alias="TURTLE_SANDBOX_ALLOWED_BINARIES")
    extra_binaries: str = Field(default="", alias="TURTLE_SANDBOX_EXTRA_BINARIES")
    ro_mounts: str = Field(default="", alias="TURTLE_SANDBOX_RO_MOUNTS")

    tripwire_blocks: bool = Field(default=False, alias="TURTLE_SANDBOX_TRIPWIRE_BLOCKS")
    allow_shell_composition: bool = Field(
        default=False, alias="TURTLE_SANDBOX_ALLOW_SHELL_COMPOSITION"
    )
    snapshot_before_write: bool = Field(
        default=True, alias="TURTLE_SANDBOX_SNAPSHOT_BEFORE_WRITE"
    )
    snapshot_keep: int = Field(default=10, alias="TURTLE_SANDBOX_SNAPSHOT_KEEP")
    snapshot_max_bytes: int = Field(
        default=64 * 1024 * 1024, alias="TURTLE_SANDBOX_SNAPSHOT_MAX_BYTES"
    )


# ---------------------------------------------------------------------------
# Frozen policy objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadOnlyMount:
    """A host path exposed to the container read-only at /mnt/ro/<name>."""

    host_path: str
    name: str

    @property
    def container_path(self) -> str:
        return f"{GUEST_RO_MOUNT_BASE}/{self.name}"


@dataclass(frozen=True)
class FsPolicy:
    """Filesystem capability. `None` in Capabilities means the fs tools are not
    registered at all — see core/capabilities.py."""

    workspace_root: Path
    max_read_bytes: int = 262_144
    max_write_bytes: int = 10_485_760
    readonly_mounts: tuple[ReadOnlyMount, ...] = ()
    snapshot_before_write: bool = True
    snapshot_keep: int = 10
    snapshot_max_bytes: int = 64 * 1024 * 1024

    def user_root(self, user_id: str) -> Path:
        """Per-user sandbox root. Workspaces are per-USER and persistent (so
        "carry on with my project tomorrow" works); containers are per-SESSION
        and ephemeral. See design v2 §13 Q2 — this is the one place the two
        lifetimes are reconciled."""
        safe = _safe_id(user_id) or "anonymous"
        return self.workspace_root / safe

    def workspace_for(self, user_id: str) -> Path:
        return self.user_root(user_id) / "workspace"

    def audit_path_for(self, user_id: str) -> Path:
        return self.user_root(user_id) / "audit.jsonl"

    def snapshots_for(self, user_id: str) -> Path:
        return self.user_root(user_id) / "snapshots"


@dataclass(frozen=True)
class ShellPolicy:
    """Shell capability. Constructible ONLY when the Docker boundary was proven
    available — see build_capabilities(). `None` in Capabilities means the shell
    tool is not registered at all."""

    image: str
    docker_bin: str
    network: str = "none"
    memory: str = "512m"
    cpus: str = "1.0"
    pids_limit: int = 128
    run_as: str = "1000:1000"
    timeout_seconds: int = 30
    max_output_bytes: int = 65_536
    container_ttl_seconds: int = 900
    allowed_binaries: Mapping[str, str] = field(default_factory=dict)
    tripwire_blocks: bool = False
    allow_shell_composition: bool = False

    def resolve_binary(self, name: str) -> str | None:
        """argv[0] -> absolute in-container path, or None if not allow-listed.

        Rejects anything with a path separator up front: the allow-list is keyed
        by BARE NAME, so `./python3` or `/usr/local/bin/python3` supplied by the
        model must miss rather than be normalised into a hit. Normalising here is
        how allow-lists turn into deny-lists by accident."""
        candidate = (name or "").strip()
        if not candidate or "/" in candidate or "\\" in candidate:
            return None
        return self.allowed_binaries.get(candidate)


@dataclass(frozen=True)
class SandboxConfig:
    """Everything the runtime needs, in one frozen object."""

    distribution: str
    enabled: bool
    fs: FsPolicy
    shell: ShellPolicy
    # Populated by docker_probe(); "" when the boundary is available.
    unavailable_reason: str = ""

    @property
    def available(self) -> bool:
        return not self.unavailable_reason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_id(raw: str) -> str:
    """Filesystem-safe slug for a user_id/session_id used as a path component.

    user_id comes from the identity store and session_id is server-minted, so
    neither is model-controlled today — but they end up as *path components*, and
    the cost of being wrong about that assumption once is a directory traversal
    in the audit-log path. Cheap insurance.

    Leading dots are stripped separately from the charset filter: filtering alone
    turns "../../etc" into "....etc", which is contained but reads like a
    traversal in logs, and a bare "." or ".." would survive as a real one."""
    cleaned = "".join(c for c in (raw or "") if c.isalnum() or c in "._-")
    cleaned = cleaned.lstrip(".")[:64]
    return "" if set(cleaned) <= {".", "_", "-"} else cleaned


def _resolve_run_as(explicit: str) -> str:
    """Pick the container uid:gid.

    The trap this exists for: a hard-coded non-root uid (65534/nobody) breaks
    bind-mount writes on Linux. The host workspace dir is owned by the user's uid
    (typically 1000); a container running as 65534 gets EACCES on every write and
    the sandbox looks broken for reasons unrelated to security. Matching the host
    uid fixes it and is still non-root — provided Turtle itself isn't running as
    root, which we check.

    Docker Desktop on Windows/macOS translates ownership through its VM, so any
    non-root uid works there; os.getuid() doesn't exist on Windows anyway."""
    explicit = (explicit or "").strip()
    if explicit:
        uid = explicit.split(":", 1)[0]
        if uid in {"0", "root"}:
            # Refusing rather than silently correcting: someone setting this to
            # root has a mental model we need to break, not paper over.
            raise ValueError(
                "TURTLE_SANDBOX_RUN_AS must not be root — a root-uid container "
                "gives --cap-drop ALL far less to drop and makes bind-mounted "
                "host files writable as root."
            )
        return explicit

    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:  # Windows
        return "1000:1000"
    uid_i, gid_i = getuid(), getgid()
    if uid_i == 0:
        print(
            "LOG: SANDBOX WARNING — Turtle is running as root; sandbox containers "
            "will run as nobody (65534:65534). Bind-mount writes may fail."
        )
        return "65534:65534"
    return f"{uid_i}:{gid_i}"


def _parse_binaries(narrow_csv: str, extra_csv: str) -> dict[str, str]:
    """Build the effective name -> abs-path map.

    ALLOWED_BINARIES narrows (intersect with the known map). EXTRA_BINARIES
    widens and requires an explicit absolute path per entry. Widening being more
    awkward than narrowing is the intended ergonomic: the easy direction should
    be the safe one."""
    narrow = {n.strip() for n in narrow_csv.split(",") if n.strip()}
    result = (
        {k: v for k, v in _DEFAULT_BINARIES.items() if k in narrow}
        if narrow
        else dict(_DEFAULT_BINARIES)
    )
    for item in extra_csv.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        name, _, path = item.partition("=")
        name, path = name.strip(), path.strip()
        # An operator-supplied relative path would be PATH-resolved inside the
        # container, which is exactly what the absolute-path rule exists to
        # avoid. Drop it rather than half-honour it.
        if name and path.startswith("/"):
            result[name] = path
    return result


def _parse_ro_mounts(raw: str) -> tuple[ReadOnlyMount, ...]:
    """Parse os.pathsep-separated host paths into ro mount specs.

    Non-existent paths are dropped: `docker run` fails hard on a missing bind
    source on Linux and silently *creates a root-owned directory* on some Docker
    Desktop versions, so validating here is the difference between a clear log
    line and a mystery."""
    mounts: list[ReadOnlyMount] = []
    seen: set[str] = set()
    for chunk in raw.split(os.pathsep):
        chunk = chunk.strip().strip('"')
        if not chunk:
            continue
        host = Path(chunk).expanduser()
        if not host.exists():
            print(f"LOG: SANDBOX ro-mount skipped (does not exist): {host}")
            continue
        name = _safe_id(host.name) or "mount"
        base, n = name, 2
        while name in seen:
            name, n = f"{base}{n}", n + 1
        seen.add(name)
        mounts.append(ReadOnlyMount(host_path=str(host.resolve()), name=name))
    return tuple(mounts)


# ---------------------------------------------------------------------------
# Fail-closed Docker probe
# ---------------------------------------------------------------------------

def docker_probe(docker_bin: str, *, timeout: float = 8.0) -> str:
    """Return "" if the Docker boundary is usable, else a human reason string.

    Two checks, both needed: the CLI can be on PATH while the daemon is down —
    which is the exact state of the machine this was written on (Docker 29.6.2
    installed, dockerDesktopLinuxEngine pipe missing). `docker --version` would
    have happily reported success and we'd have failed at first container start,
    i.e. mid-conversation instead of at boot.

    Note this returns a REASON rather than raising or returning a bool: that
    string is what the user sees when they ask why Turtle can't run anything,
    and "Docker is not available" without the cause is unactionable."""
    exe = shutil.which(docker_bin) or (docker_bin if os.path.isabs(docker_bin) else "")
    if not exe:
        return f"Docker CLI {docker_bin!r} is not on PATH."
    try:
        proc = subprocess.run(
            [exe, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"`{docker_bin} info` timed out after {timeout}s (daemon hung?)."
    except OSError as exc:
        return f"Could not execute {docker_bin!r}: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "no detail"
        return f"Docker daemon is not reachable: {tail}"
    return ""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_sandbox_config(
    settings: SandboxSettings | None = None,
    *,
    probe: bool = True,
) -> SandboxConfig:
    """Build the frozen config. Never raises for a *disabled* sandbox — that is
    the normal path for web/chatbot — but records why it is unavailable.

    `probe=False` skips the Docker round-trip; used by tests and by callers that
    already know the answer. It does NOT make an unavailable sandbox available:
    the runtime still fails closed at container start."""
    s = settings or SandboxSettings()

    root = Path(s.workspace_root).expanduser() if s.workspace_root else None
    if root is None:
        from core.paths import DATA_DIR

        root = DATA_DIR / "sandbox"
    if not root.is_absolute():
        # A relative workspace root is CWD-relative, and launching the server from
        # another directory would silently strand every workspace somewhere new —
        # the same hazard core/config.py's data_dir validator exists to fix.
        root = _PROJECT_ROOT / root

    fs = FsPolicy(
        workspace_root=root,
        max_read_bytes=max(1024, int(s.max_read_bytes)),
        max_write_bytes=max(1024, int(s.max_write_bytes)),
        readonly_mounts=_parse_ro_mounts(s.ro_mounts),
        snapshot_before_write=bool(s.snapshot_before_write),
        snapshot_keep=max(0, int(s.snapshot_keep)),
        snapshot_max_bytes=max(0, int(s.snapshot_max_bytes)),
    )

    reason = ""
    try:
        run_as = _resolve_run_as(s.run_as)
    except ValueError as exc:
        run_as = "1000:1000"
        reason = str(exc)

    shell = ShellPolicy(
        image=s.image,
        docker_bin=s.docker_bin,
        network=s.network or "none",
        memory=s.memory,
        cpus=s.cpus,
        pids_limit=max(1, int(s.pids_limit)),
        run_as=run_as,
        # Clamp: the tool exposes a `timeout` arg, and a model that has been
        # talked into running a crypto miner will ask for 86400. The policy
        # ceiling wins over the argument, always.
        timeout_seconds=max(1, int(s.timeout_seconds)),
        max_output_bytes=max(1024, int(s.max_output_bytes)),
        container_ttl_seconds=max(30, int(s.container_ttl_seconds)),
        allowed_binaries=_parse_binaries(s.allowed_binaries, s.extra_binaries),
        tripwire_blocks=bool(s.tripwire_blocks),
        allow_shell_composition=bool(s.allow_shell_composition),
    )

    distribution = (s.distribution or "web").strip().lower()
    enabled = bool(s.enabled)

    if not reason:
        if distribution != "desktop":
            reason = (
                f"distribution is {distribution!r}; filesystem and shell are "
                "desktop-only capabilities."
            )
        elif not enabled:
            reason = "TURTLE_SANDBOX_ENABLED is off."
        elif probe:
            reason = docker_probe(s.docker_bin)

    return SandboxConfig(
        distribution=distribution,
        enabled=enabled,
        fs=fs,
        shell=shell,
        unavailable_reason=reason,
    )
