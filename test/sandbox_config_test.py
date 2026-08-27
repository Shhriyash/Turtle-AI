"""
Sandbox — capability policy construction and the fail-closed matrix.

The load-bearing assertion in this file is `test_fail_closed_matrix`: filesystem
and shell must be None unless ALL of distribution=desktop, sandbox enabled, and
a reachable Docker daemon hold. `None` means the tool is never registered, which
means no prompt injection can reach it — see core/capabilities.py for why that
is a stronger claim than a runtime check.

Fully offline: `probe=False` skips the Docker round-trip, and the daemon-down
case is simulated by monkeypatching docker_probe.
"""
from __future__ import annotations

import pytest

from core.capabilities import build_capabilities
from core.sandbox import config as cfg_mod
from core.sandbox.config import (
    FsPolicy,
    SandboxSettings,
    ShellPolicy,
    _parse_binaries,
    _parse_ro_mounts,
    _resolve_run_as,
    _safe_id,
    load_sandbox_config,
)


def _settings(**overrides) -> SandboxSettings:
    """Build settings without reading the developer's real .env.

    _env_file=None matters: a dev with TURTLE_SANDBOX_ENABLED=1 in .env would
    otherwise flip half of these assertions depending on whose machine runs the
    suite."""
    base = {"distribution": "desktop", "enabled": True}
    base.update(overrides)
    return SandboxSettings(_env_file=None, **base)


# ── fail-closed matrix ───────────────────────────────────────────────────

@pytest.mark.parametrize(
    "distribution,enabled,docker_ok,expect_sandbox",
    [
        ("desktop", True, True, True),      # only row that gets a shell
        ("desktop", True, False, False),    # daemon down → fail closed
        ("desktop", False, True, False),    # master switch off
        ("web", True, True, False),         # web never gets a shell
        ("chatbot", True, True, False),     # chat surfaces never get a shell
    ],
)
def test_fail_closed_matrix(monkeypatch, distribution, enabled, docker_ok, expect_sandbox):
    monkeypatch.setattr(
        cfg_mod, "docker_probe", lambda _bin, **_kw: "" if docker_ok else "daemon down"
    )
    cfg = load_sandbox_config(
        _settings(distribution=distribution, enabled=enabled), probe=True
    )
    caps = build_capabilities(sandbox_config=cfg)

    assert cfg.available is expect_sandbox
    assert (caps.shell is not None) is expect_sandbox
    assert (caps.filesystem is not None) is expect_sandbox
    assert caps.has_sandbox is expect_sandbox
    if not expect_sandbox:
        # An unexplained missing capability is a support ticket; the reason is
        # what turns it into a one-line answer.
        assert caps.sandbox_unavailable_reason


def test_disabled_default_is_off_everywhere():
    """Handing a model a shell must be a decision the user makes, never one they
    inherit from a version bump."""
    s = SandboxSettings(_env_file=None)
    assert s.enabled is False
    assert s.distribution == "web"


def test_no_degraded_fallback_when_docker_missing(monkeypatch):
    """There is deliberately no in-process execution path. If there were, the
    moment the boundary disappeared would be the moment we started running
    model-authored commands directly on the user's machine."""
    monkeypatch.setattr(cfg_mod, "docker_probe", lambda _b, **_k: "Docker CLI not on PATH.")
    caps = build_capabilities(settings=_settings(), probe=True)
    assert caps.shell is None and caps.filesystem is None
    assert "PATH" in caps.sandbox_unavailable_reason
    # web_search/email are unaffected — the sandbox failing must not take the
    # rest of the assistant down with it.
    assert caps.web_search and caps.email


def test_web_capabilities_survive_sandbox_being_off():
    caps = build_capabilities(settings=_settings(distribution="web"), probe=False)
    assert caps.web_search is True and caps.email is True


# ── policy objects are frozen ────────────────────────────────────────────

def test_policies_are_frozen(tmp_path):
    """Read from closures on several Agent objects; a mutable policy would let
    one turn quietly change what a later turn can do."""
    fs = FsPolicy(workspace_root=tmp_path)
    shell = ShellPolicy(image="i", docker_bin="docker")
    with pytest.raises(Exception):
        fs.max_write_bytes = 1  # type: ignore[misc]
    with pytest.raises(Exception):
        shell.network = "bridge"  # type: ignore[misc]


# ── allow-list ───────────────────────────────────────────────────────────

def test_allow_list_narrows_by_name():
    cfg = load_sandbox_config(_settings(allowed_binaries="python3,ls"), probe=False)
    assert set(cfg.shell.allowed_binaries) == {"python3", "ls"}


def test_allow_list_default_excludes_shells_and_network_tools():
    """sh/bash destroy the argv-level audit trail; curl/wget are a loaded gun the
    day someone flips --network for a pip install."""
    binaries = load_sandbox_config(_settings(), probe=False).shell.allowed_binaries
    for banned in ("sh", "bash", "zsh", "curl", "wget", "sudo", "su", "chmod", "apt", "nc", "ssh"):
        assert banned not in binaries


def test_extra_binaries_require_an_absolute_path():
    """A relative path would be PATH-resolved inside the container, which is
    exactly what the absolute-path rule exists to prevent."""
    got = _parse_binaries("", "node=/usr/bin/node,bad=node,worse=")
    assert got["node"] == "/usr/bin/node"
    assert "bad" not in got and "worse" not in got


def test_resolve_binary_rejects_paths_not_just_unknown_names():
    """The allow-list is keyed by BARE NAME. Normalising './python3' or an
    absolute path into a hit is how allow-lists turn into deny-lists."""
    p = ShellPolicy(image="i", docker_bin="d", allowed_binaries={"python3": "/usr/local/bin/python3"})
    assert p.resolve_binary("python3") == "/usr/local/bin/python3"
    assert p.resolve_binary("./python3") is None
    assert p.resolve_binary("/usr/local/bin/python3") is None
    assert p.resolve_binary("..\\python3") is None
    assert p.resolve_binary("bash") is None
    assert p.resolve_binary("") is None


# ── uid resolution ───────────────────────────────────────────────────────

def test_explicit_root_uid_is_refused():
    """Refusing rather than silently correcting: someone setting this to root has
    a mental model that needs breaking, not papering over."""
    with pytest.raises(ValueError):
        _resolve_run_as("0:0")
    with pytest.raises(ValueError):
        _resolve_run_as("root:root")


def test_root_run_as_makes_the_sandbox_unavailable():
    cfg = load_sandbox_config(_settings(run_as="0:0"), probe=False)
    assert not cfg.available
    assert "root" in cfg.unavailable_reason


def test_resolved_uid_is_never_zero(monkeypatch):
    monkeypatch.setattr(cfg_mod.os, "getuid", lambda: 0, raising=False)
    monkeypatch.setattr(cfg_mod.os, "getgid", lambda: 0, raising=False)
    assert _resolve_run_as("").split(":")[0] not in {"0", "root"}


def test_explicit_non_root_uid_passes_through():
    assert _resolve_run_as("1234:5678") == "1234:5678"


# ── paths ────────────────────────────────────────────────────────────────

def test_workspace_root_is_anchored_not_cwd_relative():
    """A CWD-relative root would strand every workspace somewhere new when the
    server is launched from another directory — the same hazard core/config.py's
    data_dir validator exists to fix."""
    cfg = load_sandbox_config(_settings(workspace_root="sbx"), probe=False)
    assert cfg.fs.workspace_root.is_absolute()
    assert cfg.fs.workspace_root.name == "sbx"


def test_per_user_paths_are_separated(tmp_path):
    fs = FsPolicy(workspace_root=tmp_path)
    assert fs.workspace_for("usr_a") != fs.workspace_for("usr_b")
    assert fs.audit_path_for("usr_a").name == "audit.jsonl"
    assert fs.snapshots_for("usr_a").parent == fs.user_root("usr_a")


def test_user_id_cannot_traverse_out_of_the_sandbox_root(tmp_path):
    """user_id is server-minted today, but it becomes a PATH COMPONENT here and
    the cost of that assumption being wrong once is a traversal in the audit path."""
    fs = FsPolicy(workspace_root=tmp_path)
    hostile = fs.workspace_for("../../etc/passwd")
    assert hostile.resolve().is_relative_to(tmp_path.resolve())
    assert _safe_id("../../x") == "..x".lstrip(".")     # separators + leading dots gone
    assert _safe_id("..") == ""                          # a pure traversal collapses
    assert _safe_id("/etc/passwd") == "etcpasswd"


def test_ro_mounts_drop_nonexistent_paths(tmp_path):
    """docker run fails hard on a missing bind source on Linux and silently
    creates a root-owned dir on some Docker Desktop versions — validating here is
    the difference between a clear log line and a mystery."""
    real = tmp_path / "projects"
    real.mkdir()
    mounts = _parse_ro_mounts(f"{real}{__import__('os').pathsep}{tmp_path / 'missing'}")
    assert len(mounts) == 1
    assert mounts[0].name == "projects"
    assert mounts[0].container_path == "/mnt/ro/projects"


# ── clamps ───────────────────────────────────────────────────────────────

def test_limits_have_floors():
    cfg = load_sandbox_config(
        _settings(timeout_seconds=0, max_output_bytes=1, pids_limit=0, container_ttl_seconds=1),
        probe=False,
    )
    assert cfg.shell.timeout_seconds >= 1
    assert cfg.shell.max_output_bytes >= 1024
    assert cfg.shell.pids_limit >= 1
    assert cfg.shell.container_ttl_seconds >= 30


def test_network_defaults_to_none():
    assert load_sandbox_config(_settings(), probe=False).shell.network == "none"


def test_tripwire_blocking_is_off_by_default():
    """It buys approximately nothing (see test/sandbox_tripwire_test.py) and
    shipping it on would imply the deny-list is a control."""
    assert load_sandbox_config(_settings(), probe=False).shell.tripwire_blocks is False


def test_shell_composition_is_off_by_default():
    assert load_sandbox_config(_settings(), probe=False).shell.allow_shell_composition is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
