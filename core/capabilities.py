"""
core/capabilities.py
--------------------
The frozen capability policy that decides which tools are REGISTERED on the
agent — not which tools are allowed to run.

That distinction is the entire security argument for the desktop sandbox, so it
is worth stating flatly:

    Not registered  ≠  registered but denied.

A tool absent from the JSON schema handed to the model cannot be reached by any
prompt injection. There is no string an attacker can plant on a web page that
makes a model emit a call to a function it was never told exists — the provider's
tool-calling layer rejects unknown names before Turtle sees them. That is a
STRUCTURAL guarantee. A runtime `if not allowed: return "denied"` is not: it
depends on our code being correct on every path, forever, including the paths
added next year by someone who hasn't read this file.

So: no `if platform == "desktop"` inside any tool body, and no
registered-then-denied stubs. The web and chat-bot distributions construct
Capabilities(filesystem=None, shell=None) and their models never learn a shell
exists.

Turtle is one kernel with three distributions:
  * web      — browser UI. web_search + email. No fs, no shell.
  * chatbot  — Discord/Slack/etc. Same as web. NEVER gets a shell: the "user"
               on those surfaces can be anyone in a channel.
  * desktop  — the only distribution that can get filesystem + shell, and only
               when Docker is actually available (fail closed, see
               core/sandbox/config.docker_probe).
"""
from __future__ import annotations

from dataclasses import dataclass

from core.sandbox.config import (
    FsPolicy,
    SandboxConfig,
    SandboxSettings,
    ShellPolicy,
    load_sandbox_config,
)

__all__ = ["Capabilities", "build_capabilities"]


@dataclass(frozen=True)
class Capabilities:
    """What this distribution's agent is allowed to be TOLD about.

    Frozen because it is read from closures registered on several Agent objects
    (main + every fallback rung) and is expected to be identical across all of
    them for the whole process lifetime. A mutable policy would let one turn
    quietly change what a later turn can do.
    """

    web_search: bool = True
    email: bool = True
    filesystem: FsPolicy | None = None   # None => tool never registered
    shell: ShellPolicy | None = None     # None => tool never registered

    # Why the capability is off, for logs and for answering the user's "why
    # can't you run that?". Empty when the sandbox is on.
    sandbox_unavailable_reason: str = ""

    @property
    def has_sandbox(self) -> bool:
        return self.filesystem is not None or self.shell is not None

    def describe(self) -> str:
        """One-line summary for the startup log."""
        bits = []
        if self.web_search:
            bits.append("web_search")
        if self.email:
            bits.append("email")
        if self.filesystem is not None:
            bits.append("filesystem")
        if self.shell is not None:
            bits.append("shell")
        listed = ", ".join(bits) or "none"
        if self.sandbox_unavailable_reason:
            return f"{listed} (sandbox off: {self.sandbox_unavailable_reason})"
        return listed


def build_capabilities(
    *,
    settings: SandboxSettings | None = None,
    sandbox_config: SandboxConfig | None = None,
    web_search: bool = True,
    email: bool = True,
    probe: bool = True,
) -> Capabilities:
    """Resolve the policy for this process. Called once, at agent-build time.

    Fail-closed by construction: filesystem/shell are populated ONLY when
    load_sandbox_config() reports no unavailable_reason, which requires all of
    distribution == "desktop", TURTLE_SANDBOX_ENABLED truthy, the docker CLI on
    PATH, and a reachable daemon.

    There is deliberately no in-process fallback when Docker is missing. A
    fallback would mean the moment the isolation boundary disappears is exactly
    the moment we start running model-authored commands directly on the user's
    machine — the failure mode is maximally dangerous precisely when the safety
    net is gone.
    """
    cfg = sandbox_config or load_sandbox_config(settings, probe=probe)

    if not cfg.available:
        # Log once at build time. A silently-absent shell on a desktop install is
        # a support ticket; a logged reason is a one-line answer.
        if cfg.distribution == "desktop" and cfg.enabled:
            print(f"LOG: SANDBOX unavailable — {cfg.unavailable_reason}")
        return Capabilities(
            web_search=web_search,
            email=email,
            filesystem=None,
            shell=None,
            sandbox_unavailable_reason=cfg.unavailable_reason,
        )

    print(
        f"LOG: SANDBOX enabled — image={cfg.shell.image} network={cfg.shell.network} "
        f"mem={cfg.shell.memory} cpus={cfg.shell.cpus} pids={cfg.shell.pids_limit} "
        f"user={cfg.shell.run_as} binaries={len(cfg.shell.allowed_binaries)} "
        f"ro_mounts={len(cfg.fs.readonly_mounts)}"
    )
    return Capabilities(
        web_search=web_search,
        email=email,
        filesystem=cfg.fs,
        shell=cfg.shell,
        sandbox_unavailable_reason="",
    )
