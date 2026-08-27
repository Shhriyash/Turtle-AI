"""
core/sandbox
------------
Desktop-only filesystem + shell capability for Turtle.

Read `docs/sandbox_design_v2.md` before changing anything in this package.
`docs/sandbox_architecture.md` (v1) is SUPERSEDED — its regex deny-list is not a
security boundary and must not be reintroduced as one.

The one-paragraph version of the security model:

    Docker container isolation is the boundary. One container per session, with
    --network none, --cap-drop ALL, --read-only, non-root, and cgroup limits.
    Model-authored commands are argv arrays whose argv[0] must be in a binary
    allow-list resolved to an absolute path inside the container; we never run
    `sh -c` on a model-authored string host-side. The path jail is the mount
    namespace, not a Path.resolve() string check. The deny-list survives only as
    a logging tripwire for detecting prompt injection — it is never the control.

And the capability rule, which is the other half of the argument:

    A tool that was never registered cannot be reached by prompt injection. The
    web and chat-bot distributions build Capabilities(filesystem=None,
    shell=None) and their models are never shown a shell in the tool schema.
    There is no `if platform == "desktop"` inside any tool body.
"""
from __future__ import annotations

from core.sandbox.config import (
    FsPolicy,
    SandboxConfig,
    ShellPolicy,
    docker_probe,
    load_sandbox_config,
)
from core.sandbox.models import DirEntry, DirResult, FileResult, ShellResult

__all__ = [
    "FsPolicy",
    "ShellPolicy",
    "SandboxConfig",
    "load_sandbox_config",
    "docker_probe",
    "ShellResult",
    "FileResult",
    "DirResult",
    "DirEntry",
    "SandboxUnavailable",
]


class SandboxUnavailable(RuntimeError):
    """The isolation boundary is not available, so no sandbox tool may run.

    Raised only from the *construction* path (policy building / container start),
    never from inside a registered tool — by the time a tool exists, the boundary
    was proven available. Carries a user-facing `reason` because "the sandbox is
    off" with no explanation is the kind of thing that gets debugged by disabling
    the safety check.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)
