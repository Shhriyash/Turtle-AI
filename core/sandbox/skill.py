"""
core/sandbox/skill.py
---------------------
Forward-compat hook for Phase-2 coding skills. Carried over from
docs/sandbox_architecture.md §7 — this part of v1 was right.

Phase 0 ships ZERO concrete implementations. The interface exists now so that
adding `UnderstandCodebaseSkill` / `TestRunnerSkill` later is a new file plus one
`SANDBOX_SKILLS.append(...)`, with no changes to config.py, docker_runtime.py,
tools.py, or turtle_server.py.

One clause is STRENGTHENED relative to v1. v1 said a skill "must not bypass
sandbox_security checks" and "must not perform I/O outside ctx.workspace_root" —
both of which were host-side honour-system rules. v2's version: a skill executes
by composing DockerRuntime calls. It gets a runtime handle and a session key, not
a host path it could open() directly. A skill that reaches for `open()` on the
host workspace has stepped outside the mount namespace and re-created the exact
symlink escape the container exists to prevent.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid a runtime import cycle: runtime does not import skills
    from core.sandbox.docker_runtime import DockerRuntime


@dataclass
class SkillContext:
    """What a skill is given. Deliberately NOT a host filesystem path."""

    user_id: str
    session_id: str
    task_description: str
    runtime: "DockerRuntime"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    success: bool
    output: str                                  # LLM-consumable summary
    artifacts: list[str] = field(default_factory=list)   # workspace-relative paths
    metadata: dict[str, Any] = field(default_factory=dict)


# Module-level registry. Skills append themselves at import time.
SANDBOX_SKILLS: list["SandboxSkill"] = []


class SandboxSkill(abc.ABC):
    """A reusable multi-step capability that operates inside the sandbox.

    Skills are called as Python from inside a tool body — they are NOT exposed to
    the model as tools. That is intentional: it keeps the model's tool schema
    small (every entry is attack surface, per core/capabilities.py) while still
    allowing composite behaviour.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique kebab-case identifier, e.g. 'understand-task'."""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """One sentence, shown in agent diagnostics."""

    @abc.abstractmethod
    async def run(self, ctx: SkillContext) -> SkillResult:
        """Execute. Must be idempotent where possible. Must reach the filesystem
        only through ctx.runtime — never host-side open()."""

    def is_applicable(self, ctx: SkillContext) -> bool:
        """Optional pre-check. Return False to skip."""
        return True


def register_skill(skill: SandboxSkill) -> SandboxSkill:
    """Idempotent registration — module reimport under --reload would otherwise
    stack duplicates and run each skill twice per turn."""
    if not any(s.name == skill.name for s in SANDBOX_SKILLS):
        SANDBOX_SKILLS.append(skill)
    return skill
