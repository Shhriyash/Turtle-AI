"""
core/sandbox/docker_runtime.py
------------------------------
Container lifecycle + command execution. This module IS the security boundary
(docs/sandbox_design_v2.md §3); everything else in the package is bookkeeping
around it.

Two rules that must survive every future edit:

  1. **No `sh -c` on a model-authored string, ever, host-side.** Every subprocess
     launched from here is an argv array with shell=False semantics
     (`asyncio.create_subprocess_exec`). v1's
     `create_subprocess_exec("/bin/sh", "-c", command)` line was the RCE; there
     is no version of it that is safe with a deny-list in front.

  2. **argv[0] is resolved through ShellPolicy.allowed_binaries to an absolute
     in-container path.** Bare names are PATH-resolved, and PATH is influenceable
     by anything that can write to an early PATH entry — which, in a box where
     the model has a shell, is the model.

We shell out to the Docker CLI rather than using the docker-py SDK: no new
dependency, argv arrays are the CLI's native interface, it works identically with
podman, and it is trivially mockable (see `runner`), which is what lets the flag
set itself be unit-tested — the flags ARE the security property, so they get
asserted like one.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional, Sequence

from core.sandbox.config import (
    GUEST_FSOPS_PATH,
    GUEST_PYTHON_BIN,
    GUEST_TEST_BIN,
    GUEST_TIMEOUT_BIN,
    GUEST_WORKSPACE,
    FsPolicy,
    ShellPolicy,
)
from core.sandbox.guest import FSOPS_HOST_PATH

# How much output past the cap we keep draining before killing the process.
# Reading exactly `cap` and killing immediately would fail a program that emits
# cap+1 bytes and then exits cleanly; draining forever lets `yes` stream
# gigabytes through the docker socket until the timeout. A bounded overrun gets
# both: graceful for near-misses, hard-stopped for runaways.
_OVERFLOW_DRAIN_BYTES = 4 * 1024 * 1024
_CHUNK = 64 * 1024
# Grace on top of the in-container `timeout` wrapper. The wrapper is the real
# enforcement (see _build_exec_argv); this only catches a hung docker daemon.
_HOST_TIMEOUT_GRACE_S = 5.0


class SandboxRuntimeError(RuntimeError):
    """Container lifecycle failed. Surfaces to the model as a denial, not a crash."""


@dataclass
class CliResult:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""
    stdout_total: int = 0
    stderr_total: int = 0
    timed_out: bool = False
    truncated: bool = False
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    def err_text(self) -> str:
        return self.stderr.decode("utf-8", errors="replace")


Runner = Callable[..., Awaitable[CliResult]]


async def _read_capped(
    stream: asyncio.StreamReader | None, cap: int
) -> tuple[bytes, int, bool]:
    """Read a pipe with a memory cap. Returns (kept, total_seen, overflowed).

    Capping happens WHILE streaming, not after. `communicate()` buffers the whole
    stream first, so a runaway producer OOMs the host Python process long before
    any post-hoc truncation runs — the cap has to be applied at the read, or it
    isn't a cap."""
    if stream is None:
        return b"", 0, False
    kept = bytearray()
    total = 0
    overflowed = False
    while True:
        chunk = await stream.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if len(kept) < cap:
            kept.extend(chunk[: cap - len(kept)])
        if total > cap + _OVERFLOW_DRAIN_BYTES:
            overflowed = True
            break
    return bytes(kept), total, overflowed


async def default_runner(
    argv: Sequence[str],
    *,
    stdin: bytes | None = None,
    timeout: float | None = None,
    output_cap: int = 65_536,
) -> CliResult:
    """Run an argv array with shell=False, capped output, and a hard timeout."""
    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return CliResult(
            returncode=127,
            stderr=f"could not exec {argv[0]!r}: {exc}".encode(),
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def _feed() -> None:
        if stdin is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(stdin)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # The child exited before consuming stdin (bad path, size refusal).
            # Its stderr already explains why; a raised BrokenPipe here would
            # replace that explanation with a traceback.
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    timed_out = False
    try:
        feeder = asyncio.ensure_future(_feed())
        out_task = asyncio.ensure_future(_read_capped(proc.stdout, output_cap))
        err_task = asyncio.ensure_future(_read_capped(proc.stderr, output_cap))
        done = asyncio.gather(feeder, out_task, err_task)
        await asyncio.wait_for(done, timeout=timeout)
        rc = await asyncio.wait_for(proc.wait(), timeout=5.0)
    except (asyncio.TimeoutError, TimeoutError):
        timed_out = True
        for task in (out_task, err_task, feeder):
            task.cancel()
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        rc = -1

    def _result_of(task: "asyncio.Future") -> tuple[bytes, int, bool]:
        if task.done() and not task.cancelled() and task.exception() is None:
            return task.result()
        return b"", 0, False

    out, out_total, out_over = _result_of(out_task)
    err, err_total, err_over = _result_of(err_task)

    if (out_over or err_over) and not timed_out:
        # Producer ran past the drain budget — stop it rather than let it keep
        # streaming into a discard.
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass

    return CliResult(
        returncode=rc,
        stdout=out,
        stderr=err,
        stdout_total=out_total,
        stderr_total=err_total,
        timed_out=timed_out,
        truncated=out_total > len(out) or err_total > len(err),
        duration_ms=int((time.monotonic() - start) * 1000),
    )


@dataclass
class ContainerHandle:
    key: tuple[str, str]
    name: str
    container_id: str
    workspace: Path
    created_at: float
    last_used: float
    # Allow-listed binaries whose absolute path was PROBED and found executable
    # in this image. A swapped image degrades to "that binary isn't available"
    # instead of a confusing exec failure on every call.
    verified_binaries: dict[str, str] = field(default_factory=dict)


def _hash8(raw: str) -> str:
    return hashlib.sha256((raw or "").encode("utf-8")).hexdigest()[:8]


class DockerRuntime:
    """Owns the containers. One instance per process; created at agent-build time.

    `runner` is injectable so tests can assert the exact argv handed to Docker
    without a daemon. That is not a testing convenience bolted on afterwards —
    the flag set is the security property, so it has to be assertable.
    """

    def __init__(
        self,
        policy: ShellPolicy,
        fs: FsPolicy,
        *,
        runner: Runner | None = None,
    ) -> None:
        self.policy = policy
        self.fs = fs
        self._runner: Runner = runner or default_runner
        self._containers: dict[tuple[str, str], ContainerHandle] = {}
        # One lock per session key: two tool calls in the same turn (models do
        # emit parallel tool calls — see the parallel-tool-call work) would
        # otherwise both miss the cache and race two `docker run`s for one
        # session, and the loser leaks a 512MB container.
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    # ------------------------------------------------------------------ #
    # argv construction — pure, so it can be asserted in tests
    # ------------------------------------------------------------------ #

    def container_name(self, user_id: str, session_id: str) -> str:
        return f"turtle-sbx-{_hash8(user_id)}-{_hash8(session_id)}"

    def build_run_argv(self, *, name: str, workspace: Path) -> list[str]:
        """The `docker run` line. Every flag here is load-bearing — see design
        v2 §3.1 for why each one exists and what breaks without it."""
        p = self.policy
        argv: list[str] = [
            p.docker_bin, "run", "--detach",
            "--name", name,
            "--label", "turtle.sandbox=1",
            # No egress by default. Network access is what turns "read a file"
            # into "exfiltrate a file", which is the whole payload of a prompt
            # injection.
            "--network", p.network,
            # Never uid 0: --cap-drop has far less to drop against root, and
            # root in the container writes root-owned files through the bind
            # mount onto the user's real disk.
            "--user", p.run_as,
            # Immutable root fs: no persistence inside the box, nothing to
            # implant that survives a restart.
            "--read-only",
            # The only writable non-workspace surface, and it is noexec — a
            # downloaded binary in /tmp cannot be run. /workspace stays exec-able
            # on purpose; running the user's own code is the point.
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--cap-drop", "ALL",
            # Stops a setuid binary in the image handing capabilities back after
            # --cap-drop took them away.
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(p.pids_limit),
            "--memory", p.memory,
            # MUST equal --memory. Omitting it lets the container swap to twice
            # --memory, so the RAM cap you think you set is silently doubled and
            # host swap becomes the DoS target.
            "--memory-swap", p.memory,
            "--cpus", p.cpus,
            "--ulimit", "nofile=256:512",
            "--ulimit", f"fsize={self.fs.max_write_bytes}",
            # tini as PID 1. Without it `sleep` is PID 1 and never reaps
            # children, so an abandoned process pool exhausts --pids-limit and
            # the container wedges — presenting as "the sandbox randomly stopped
            # working".
            "--init",
            "--workdir", GUEST_WORKSPACE,
            "--mount", f"type=bind,source={workspace},target={GUEST_WORKSPACE}",
            # Read-only so the model cannot rewrite the helper to lie about what
            # it read or wrote.
            "--mount",
            f"type=bind,source={FSOPS_HOST_PATH},target={GUEST_FSOPS_PATH},readonly",
        ]
        for mount in self.fs.readonly_mounts:
            argv += [
                "--mount",
                f"type=bind,source={mount.host_path},target={mount.container_path},readonly",
            ]
        # The container is a namespace holder; all work arrives via docker exec.
        # Idling costs one sleep process and saves the ~300-800ms container start
        # on every tool call after the first.
        argv += ["--entrypoint", "/bin/sleep", p.image, "infinity"]
        return argv

    def build_exec_argv(
        self,
        container_id: str,
        argv: Sequence[str],
        *,
        workdir: str = GUEST_WORKSPACE,
        timeout_s: int,
        with_stdin: bool = False,
    ) -> list[str]:
        """The `docker exec` line, with the in-container timeout wrapper.

        The wrapper is not decoration. Killing the host-side `docker exec` client
        DETACHES from the process rather than killing it — a host-timed-out
        `python3 spin.py` keeps a core pinned for the rest of the session.
        `timeout --signal=KILL` runs inside the namespace and reaps the real
        process. The host-side wait_for is only a net for a hung daemon."""
        out = [self.policy.docker_bin, "exec", "--workdir", workdir]
        if with_stdin:
            out.append("--interactive")
        out += [
            container_id,
            GUEST_TIMEOUT_BIN, "--signal=KILL", f"{max(1, int(timeout_s))}s",
            *[str(a) for a in argv],
        ]
        return out

    def clamp_timeout(self, requested: int | None) -> int:
        """Policy ceiling always wins over the tool argument.

        The tool exposes a `timeout` parameter, and a model that has been talked
        into running something expensive will ask for 86400."""
        ceiling = self.policy.timeout_seconds
        if requested is None:
            return ceiling
        try:
            return max(1, min(int(requested), ceiling))
        except (TypeError, ValueError):
            return ceiling

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _lock_for(self, key: tuple[str, str]) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def ensure_container(self, user_id: str, session_id: str) -> ContainerHandle:
        key = (user_id, session_id)
        async with self._lock_for(key):
            existing = self._containers.get(key)
            if existing is not None and await self._is_running(existing):
                existing.last_used = time.monotonic()
                return existing
            if existing is not None:
                # Died under us (OOM kill, daemon restart, user `docker rm`).
                # Drop it and rebuild rather than surfacing a confusing exec
                # error on every subsequent call.
                self._containers.pop(key, None)

            workspace = self.fs.workspace_for(user_id)
            workspace.mkdir(parents=True, exist_ok=True)
            name = self.container_name(user_id, session_id)

            # A same-named container can survive a hard crash. Remove before
            # create so a leftover doesn't make the whole sandbox permanently
            # unusable for that session.
            await self._runner(
                [self.policy.docker_bin, "rm", "--force", name], timeout=20
            )

            result = await self._runner(
                self.build_run_argv(name=name, workspace=workspace), timeout=120
            )
            if not result.ok:
                raise SandboxRuntimeError(
                    f"could not start sandbox container: {result.err_text().strip() or result.text().strip()}"
                )
            container_id = result.text().strip().splitlines()[-1][:12] if result.text().strip() else name

            handle = ContainerHandle(
                key=key,
                name=name,
                container_id=container_id or name,
                workspace=workspace,
                created_at=time.monotonic(),
                last_used=time.monotonic(),
            )
            handle.verified_binaries = await self._probe_binaries(handle)
            self._containers[key] = handle
            print(
                f"LOG: SANDBOX container up name={name} id={handle.container_id} "
                f"binaries={len(handle.verified_binaries)}/{len(self.policy.allowed_binaries)}"
            )
            return handle

    async def _is_running(self, handle: ContainerHandle) -> bool:
        result = await self._runner(
            [
                self.policy.docker_bin, "inspect",
                "--format", "{{.State.Running}}", handle.container_id,
            ],
            timeout=15,
        )
        return result.ok and result.text().strip().lower().startswith("true")

    async def _probe_binaries(self, handle: ContainerHandle) -> dict[str, str]:
        """Verify each allow-listed absolute path is actually executable here.

        Runs `test -x <path>` per binary, concurrently — one round trip each,
        once per container. This is an internal, constant argv; it is not
        model-authored and does not go through the allow-list check (it IS the
        allow-list check)."""
        names = list(self.policy.allowed_binaries.items())
        if not names:
            return {}

        async def probe(name: str, path: str) -> tuple[str, str] | None:
            res = await self._runner(
                [
                    self.policy.docker_bin, "exec", handle.container_id,
                    GUEST_TEST_BIN, "-x", path,
                ],
                timeout=15,
            )
            return (name, path) if res.ok else None

        results = await asyncio.gather(
            *(probe(n, p) for n, p in names), return_exceptions=True
        )
        verified: dict[str, str] = {}
        for item in results:
            if isinstance(item, tuple):
                verified[item[0]] = item[1]
        missing = sorted(set(self.policy.allowed_binaries) - set(verified))
        if missing:
            print(
                f"LOG: SANDBOX binaries not present in image {self.policy.image}: "
                f"{', '.join(missing)}"
            )
        return verified

    async def stop_session(self, user_id: str, session_id: str) -> None:
        handle = self._containers.pop((user_id, session_id), None)
        self._locks.pop((user_id, session_id), None)
        if handle is None:
            return
        await self._runner(
            [self.policy.docker_bin, "rm", "--force", handle.container_id], timeout=30
        )

    async def reap_idle(self, *, now: float | None = None) -> int:
        """Remove containers idle past the TTL. Each one holds `--memory` worth
        of the user's RAM, so a long-lived process that never reaps is a slow
        memory leak measured in hundreds of MB."""
        stamp = now if now is not None else time.monotonic()
        ttl = self.policy.container_ttl_seconds
        stale = [
            key for key, h in self._containers.items() if stamp - h.last_used > ttl
        ]
        for key in stale:
            await self.stop_session(*key)
        return len(stale)

    async def sweep_orphans(self) -> int:
        """Remove any turtle.sandbox container this process does not claim.

        Belt-and-braces for a hard crash: without it, every ungraceful shutdown
        leaks a container forever."""
        listing = await self._runner(
            [
                self.policy.docker_bin, "ps", "--all", "--quiet",
                "--filter", "label=turtle.sandbox=1",
            ],
            timeout=30,
        )
        if not listing.ok:
            return 0
        claimed = {h.container_id for h in self._containers.values()}
        removed = 0
        for cid in listing.text().split():
            if cid[:12] in claimed:
                continue
            await self._runner(
                [self.policy.docker_bin, "rm", "--force", cid], timeout=30
            )
            removed += 1
        if removed:
            print(f"LOG: SANDBOX swept {removed} orphaned container(s)")
        return removed

    async def shutdown(self) -> None:
        for key in list(self._containers):
            await self.stop_session(*key)

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def resolve_argv(
        self, handle: ContainerHandle, argv: Sequence[str]
    ) -> tuple[list[str] | None, str | None]:
        """Allow-list check on argv[0]. Returns (resolved_argv, denial_reason).

        Checks against the container's VERIFIED map, not the configured one, so a
        binary the image doesn't actually have produces "not available in this
        sandbox image" rather than an opaque exec failure."""
        items = [str(a) for a in (argv or [])]
        if not items:
            return None, "argv must not be empty."
        name = items[0]
        path = self.policy.resolve_binary(name)
        if path is None:
            allowed = ", ".join(sorted(self.policy.allowed_binaries))
            return None, (
                f"binary {name!r} is not in the sandbox allow-list. "
                f"Available: {allowed}."
            )
        if name not in handle.verified_binaries:
            return None, (
                f"binary {name!r} is allow-listed but not present in the sandbox "
                f"image ({self.policy.image})."
            )
        return [handle.verified_binaries[name], *items[1:]], None

    async def exec_in(
        self,
        handle: ContainerHandle,
        argv: Sequence[str],
        *,
        timeout_s: int,
        stdin: bytes | None = None,
        workdir: str = GUEST_WORKSPACE,
        output_cap: int | None = None,
    ) -> CliResult:
        handle.last_used = time.monotonic()
        cli = self.build_exec_argv(
            handle.container_id,
            argv,
            workdir=workdir,
            timeout_s=timeout_s,
            with_stdin=stdin is not None,
        )
        return await self._runner(
            cli,
            stdin=stdin,
            timeout=timeout_s + _HOST_TIMEOUT_GRACE_S,
            output_cap=output_cap if output_cap is not None else self.policy.max_output_bytes,
        )

    async def fsop(
        self,
        handle: ContainerHandle,
        op: str,
        rel_path: str,
        *,
        limit: int,
        stdin: bytes | None = None,
    ) -> CliResult:
        """Invoke the bind-mounted guest helper. argv is runtime-composed; the
        model contributes only `rel_path` and (for writes) the stdin blob."""
        return await self.exec_in(
            handle,
            [GUEST_PYTHON_BIN, GUEST_FSOPS_PATH, op, rel_path, str(limit)],
            timeout_s=self.clamp_timeout(None),
            stdin=stdin,
            # +8KB headroom for the JSON envelope around a `limit`-sized payload.
            output_cap=max(limit + 8192, self.policy.max_output_bytes),
        )

    # ------------------------------------------------------------------ #
    # Snapshots (recoverability, NOT a security control)
    # ------------------------------------------------------------------ #

    def snapshot_workspace(self, user_id: str, *, reason: str) -> Optional[str]:
        """Tar the workspace before a destructive op. Best-effort: a failed
        snapshot must never block the operation, or a full disk turns into "the
        assistant stopped working"."""
        if not self.fs.snapshot_before_write:
            return None
        workspace = self.fs.workspace_for(user_id)
        if not workspace.exists():
            return None
        try:
            total = sum(f.stat().st_size for f in workspace.rglob("*") if f.is_file())
            if self.fs.snapshot_max_bytes and total > self.fs.snapshot_max_bytes:
                print(
                    f"LOG: SANDBOX snapshot skipped — workspace {total}B exceeds "
                    f"cap {self.fs.snapshot_max_bytes}B"
                )
                return None
            snap_dir = self.fs.snapshots_for(user_id)
            snap_dir.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in reason)[:40]
            # Millisecond precision, not seconds. A rapid write sequence (the
            # model rewriting one file three times in a turn) collides on a
            # second-resolution name and each snapshot silently overwrites the
            # previous one — losing exactly the version the user wanted back.
            now = time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(now)) + f"_{int(now * 1000) % 1000:03d}"
            name = f"{stamp}_{safe}.tar.gz"
            with tarfile.open(snap_dir / name, "w:gz") as tar:
                tar.add(workspace, arcname="workspace")
            self._prune_snapshots(snap_dir)
            return name
        except Exception as exc:
            print(f"LOG: SANDBOX snapshot failed — {exc}")
            return None

    def _prune_snapshots(self, snap_dir: Path) -> None:
        keep = self.fs.snapshot_keep
        if keep <= 0:
            return
        try:
            snaps = sorted(
                snap_dir.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            for stale in snaps[keep:]:
                stale.unlink(missing_ok=True)
        except OSError:
            pass


def docker_binary_present(docker_bin: str) -> bool:
    return bool(shutil.which(docker_bin) or (os.path.isabs(docker_bin) and os.path.exists(docker_bin)))
