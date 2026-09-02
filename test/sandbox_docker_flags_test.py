"""
Sandbox — the Docker flag set IS the security boundary, so it is asserted like
one.

docs/sandbox_design_v2.md §3.1 lists each flag and what breaks without it. Every
assertion below maps to a line in that table. If one of these tests starts
failing, the correct response is to fix the runtime, not to relax the test —
each flag was chosen because its absence is a known escape or DoS:

  --network none          egress is the payload of every real prompt injection
  --user <non-root>       root defeats --cap-drop and writes root-owned files
                          through the bind mount onto the user's real disk
  --read-only             no persistence inside the box
  --tmpfs noexec          a downloaded binary in /tmp cannot be executed
  --cap-drop ALL          removes what the process starts with
  --security-opt no-new-privileges   stops setuid handing capabilities back
  --pids-limit            fork bombs stop here
  --memory + --memory-swap equal     omitting swap SILENTLY DOUBLES the RAM cap
  --init                  PID 1 reaps zombies, or --pids-limit is exhausted

Fully offline: DockerRuntime takes an injected runner, so no daemon is needed
(and none is running on the dev machine this was written on).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.sandbox.config import (
    GUEST_FSOPS_PATH,
    GUEST_TIMEOUT_BIN,
    GUEST_WORKSPACE,
    FsPolicy,
    ReadOnlyMount,
    ShellPolicy,
)
from core.sandbox.docker_runtime import DockerRuntime


@pytest.fixture()
def policy() -> ShellPolicy:
    return ShellPolicy(
        image="python:3.13-slim",
        docker_bin="docker",
        network="none",
        memory="512m",
        cpus="1.0",
        pids_limit=128,
        run_as="1000:1000",
        timeout_seconds=30,
        max_output_bytes=65536,
        allowed_binaries={"python3": "/usr/local/bin/python3", "ls": "/bin/ls"},
    )


@pytest.fixture()
def fs(tmp_path: Path) -> FsPolicy:
    return FsPolicy(workspace_root=tmp_path / "sandbox", max_write_bytes=10_485_760)


def _pairs(argv: list[str]) -> list[tuple[str, str]]:
    """Flag/value pairs, so assertions do not depend on argument ORDER."""
    return [(argv[i], argv[i + 1]) for i in range(len(argv) - 1)]


# ── docker run ───────────────────────────────────────────────────────────

def test_run_argv_carries_every_isolation_flag(policy, fs):
    rt = DockerRuntime(policy, fs)
    argv = rt.build_run_argv(name="turtle-sbx-a-b", workspace=fs.workspace_root / "u")
    pairs = _pairs(argv)

    assert argv[:3] == ["docker", "run", "--detach"]
    assert ("--network", "none") in pairs
    assert ("--user", "1000:1000") in pairs
    assert ("--cap-drop", "ALL") in pairs
    assert ("--security-opt", "no-new-privileges") in pairs
    assert ("--pids-limit", "128") in pairs
    assert ("--memory", "512m") in pairs
    assert ("--cpus", "1.0") in pairs
    assert "--read-only" in argv
    assert "--init" in argv
    assert ("--workdir", GUEST_WORKSPACE) in pairs
    assert ("--label", "turtle.sandbox=1") in pairs


def test_memory_swap_equals_memory(policy, fs):
    """Omitting --memory-swap lets the container swap to 2x --memory, so the RAM
    cap you think you set is silently doubled. This must never drift."""
    rt = DockerRuntime(policy, fs)
    pairs = _pairs(rt.build_run_argv(name="n", workspace=fs.workspace_root))
    assert ("--memory-swap", "512m") in pairs
    assert dict(pairs)["--memory-swap"] == dict(pairs)["--memory"]


def test_tmpfs_is_noexec(policy, fs):
    rt = DockerRuntime(policy, fs)
    tmpfs = dict(_pairs(rt.build_run_argv(name="n", workspace=fs.workspace_root)))["--tmpfs"]
    assert tmpfs.startswith("/tmp:")
    for opt in ("noexec", "nosuid", "nodev"):
        assert opt in tmpfs


def test_never_runs_as_root(policy, fs):
    rt = DockerRuntime(policy, fs)
    user = dict(_pairs(rt.build_run_argv(name="n", workspace=fs.workspace_root)))["--user"]
    assert user.split(":")[0] not in {"0", "root"}


def test_workspace_bind_is_the_only_writable_mount(policy, fs):
    """The workspace bind is rw; every other bind must carry `readonly`."""
    ws = fs.workspace_root / "usr_x" / "workspace"
    rt = DockerRuntime(policy, fs)
    mounts = [v for k, v in _pairs(rt.build_run_argv(name="n", workspace=ws)) if k == "--mount"]

    writable = [m for m in mounts if "readonly" not in m]
    assert len(writable) == 1
    assert f"target={GUEST_WORKSPACE}" in writable[0]
    assert f"source={ws}" in writable[0]


def test_guest_helper_is_mounted_read_only(policy, fs):
    """The model must not be able to rewrite fsops.py to lie about what it read."""
    rt = DockerRuntime(policy, fs)
    mounts = [v for k, v in _pairs(rt.build_run_argv(name="n", workspace=fs.workspace_root)) if k == "--mount"]
    helper = [m for m in mounts if f"target={GUEST_FSOPS_PATH}" in m]
    assert len(helper) == 1
    assert helper[0].endswith(",readonly")


def test_extra_readonly_mounts_are_readonly(fs):
    p = ShellPolicy(image="i", docker_bin="docker")
    fs2 = FsPolicy(
        workspace_root=fs.workspace_root,
        readonly_mounts=(ReadOnlyMount(host_path="/host/projects", name="projects"),),
    )
    rt = DockerRuntime(p, fs2)
    mounts = [v for k, v in _pairs(rt.build_run_argv(name="n", workspace=fs.workspace_root)) if k == "--mount"]
    extra = [m for m in mounts if "target=/mnt/ro/projects" in m]
    assert len(extra) == 1
    assert extra[0].endswith(",readonly")


def test_fsize_ulimit_tracks_write_cap(policy, fs):
    """The bind mount is the one place container writes reach host storage, so
    --ulimit fsize is the only disk quota available without a dedicated volume."""
    rt = DockerRuntime(policy, fs)
    ulimits = [v for k, v in _pairs(rt.build_run_argv(name="n", workspace=fs.workspace_root)) if k == "--ulimit"]
    assert f"fsize={fs.max_write_bytes}" in ulimits


def test_container_holds_namespace_via_sleep(policy, fs):
    """Entrypoint is a plain sleep — the container is a namespace holder and all
    work arrives via `docker exec`, so container start is paid once per session."""
    argv = DockerRuntime(policy, fs).build_run_argv(name="n", workspace=fs.workspace_root)
    assert argv[-4:] == ["--entrypoint", "/bin/sleep", "python:3.13-slim", "infinity"]


# ── docker exec ──────────────────────────────────────────────────────────

def test_exec_wraps_in_container_timeout(policy, fs):
    """Killing the host-side `docker exec` client DETACHES from the process
    instead of killing it, so a host-timed-out `python3 spin.py` would keep a
    core pinned for the rest of the session. The in-container wrapper reaps it."""
    rt = DockerRuntime(policy, fs)
    argv = rt.build_exec_argv("cid123", ["/usr/local/bin/python3", "x.py"], timeout_s=17)

    assert argv[0:2] == ["docker", "exec"]
    assert ("--workdir", GUEST_WORKSPACE) in _pairs(argv)
    idx = argv.index("cid123")
    assert argv[idx + 1] == GUEST_TIMEOUT_BIN
    assert argv[idx + 2] == "--signal=KILL"
    assert argv[idx + 3] == "17s"
    assert argv[idx + 4:] == ["/usr/local/bin/python3", "x.py"]


def test_exec_only_opens_stdin_when_needed(policy, fs):
    rt = DockerRuntime(policy, fs)
    assert "--interactive" not in rt.build_exec_argv("c", ["/bin/ls"], timeout_s=5)
    assert "--interactive" in rt.build_exec_argv("c", ["/bin/ls"], timeout_s=5, with_stdin=True)


def test_no_shell_anywhere_in_any_argv(policy, fs):
    """The v1 RCE was `create_subprocess_exec("/bin/sh", "-c", command)`. Nothing
    the runtime composes may reintroduce a `-c` shell invocation."""
    rt = DockerRuntime(policy, fs)
    for argv in (
        rt.build_run_argv(name="n", workspace=fs.workspace_root),
        rt.build_exec_argv("c", ["/bin/ls", "-la"], timeout_s=5),
    ):
        joined = " ".join(argv)
        assert "/bin/sh" not in joined
        assert "/bin/bash" not in joined
        assert " -c " not in f" {joined} "


# ── timeout clamping ─────────────────────────────────────────────────────

def test_timeout_ceiling_beats_the_tool_argument(policy, fs):
    """The tool exposes a `timeout` arg, and a model talked into running
    something expensive will ask for 86400. Policy wins."""
    rt = DockerRuntime(policy, fs)
    assert rt.clamp_timeout(86400) == 30
    assert rt.clamp_timeout(5) == 5
    assert rt.clamp_timeout(None) == 30
    assert rt.clamp_timeout(0) == 1
    assert rt.clamp_timeout("nonsense") == 30  # type: ignore[arg-type]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
