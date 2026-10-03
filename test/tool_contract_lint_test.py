"""
Contract-lint for core/system_prompts/tools/*.md (ledger 1b.6, item 2).

_load_tool_contract() in apps/turtle_server.py falls back to a bare
"Tool: {name}" description when a contract file is missing, silently and
with no log line. core/sandbox/tools.py documents that failure having
actually happened in production once: a `**/*.md` gitignore rule untracked
every sandbox contract and degraded all of them to that one-liner, undetected
until someone went looking.

This module guards both directions of the same bug class:

1. Every tool currently registered on the live agent
   (apps.turtle_server's agents_mgr) has a contract file on disk, named
   `{tool_name}.md` under core/system_prompts/tools/. A tool added to the
   registry without a contract silently degrades to "Tool: {name}" and this
   fails loudly instead.

2. Every tool name *mentioned* inside a contract file's prose (e.g. "use
   `calendar_confirm` instead") names something real, i.e. either a
   currently-registered tool or a tool that genuinely exists in the codebase
   but is not (yet) wired into the live registry. A prompt that tells the
   model to reach for a capability that does not exist anywhere (e.g. the
   old "use search_web or history_tool instead" line) fails this.

Extraction approach and its limits
-----------------------------------
Tool names in these files are always snake_case with an underscore (every
real tool name in this repo is a verb_noun compound; no single-word tool
exists). Prose mentions of a tool consistently follow one of two shapes,
verified against the current corpus:

    call `foo_bar`          / call foo_bar
    use `foo_bar`           / use foo_bar
    invoke `foo_bar`        / invoke foo_bar
    ... or `foo_bar` instead / ... or foo_bar instead

so this module extracts identifiers matching those two regexes rather than
every backtick-wrapped or snake_case token in the file. A naive "every
snake_case token" scan would false-positive constantly on parameter names
that are also snake_case (`place_id`, `attendee_emails`, `time_min_iso`,
`travel_mode`, ...) and on error-code-shaped strings (`credentials_missing`,
`invalid_place_id`); restricting to the call/use/invoke/"or ... instead"
verb context avoids that (checked: zero false positives against every
contract file in the repo as of this test's authoring).

Known limits (false negatives this approach will NOT catch):
- A tool name mentioned in prose without any of those four surrounding verb
  shapes, e.g. a bare list like "options are foo_bar, baz_qux" with no
  call/use/invoke/instead nearby.
- A tool name that has no underscore (none exist today; if one is ever
  added, this heuristic needs a companion signal, e.g. an explicit
  allowlist of single-word tool names).
This is a lint, not a full parser; it is designed to catch exactly the
class of bug the ledger flagged (an invented tool name in "use X instead"
phrasing) while staying quiet on the params/error-codes that make up the
bulk of the prose.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

CONTRACTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "core" / "system_prompts" / "tools"
)

# core/sandbox/tools.py implements these four and registers them via
# register_sandbox_tools() — but the live server (apps/turtle_server.py)
# never calls that registrar, so they are real, contracted tools that are
# simply not part of the main agent's registry yet. A parallel WP may wire
# them up later (ledger note). Keep this list to exactly those four: it is
# a deliberate, named allowlist, not a general escape hatch — anything
# added here should correspond to an actual implementation elsewhere in the
# codebase, not a wish.
KNOWN_UNREGISTERED_BUT_REAL = frozenset({
    "sandbox_run",
    "sandbox_read_file",
    "sandbox_write_file",
    "sandbox_list_dir",
})

# Prose shapes that mean "this identifier names a tool", not just any
# snake_case word. See module docstring for why this beats a blanket scan.
_MENTION_PATTERNS = [
    re.compile(r"\b(?:call|use|invoke)\s+`?([a-z][a-z0-9_]*_[a-z0-9_]*)`?"),
    re.compile(r"\bor\s+`?([a-z][a-z0-9_]*_[a-z0-9_]*)`?\s+instead\b"),
]


def _registered_tool_names() -> set[str]:
    """The tool names actually wired onto the live main agent.

    Imports apps.turtle_server, which (as of this WP) requires dummy
    GROQ_API_KEY / COHERE_API_KEY at module-import time — the same env the
    rest of the suite already assumes (see test/main_assistant_prompt_test.py
    and the tests.yml CI job's dummy keys).
    """
    from apps import turtle_server

    toolset = turtle_server.agents_mgr.main_assistant._function_toolset
    return set(toolset.tools.keys())


def _contract_files() -> list[Path]:
    return sorted(CONTRACTS_DIR.glob("*.md"))


def _mentions_in(text: str) -> set[str]:
    found: set[str] = set()
    for pattern in _MENTION_PATTERNS:
        found.update(pattern.findall(text))
    return found


def test_every_registered_tool_has_a_contract_file():
    """A tool wired into the live registry with no contract file silently
    degrades to "Tool: {name}" (_load_tool_contract's fallback) — this
    catches that before it ships, including a hypothetical 13th tool."""
    registered = _registered_tool_names()
    missing = sorted(
        name for name in registered
        if not (CONTRACTS_DIR / f"{name}.md").exists()
    )
    assert not missing, (
        f"registered tool(s) with no contract file (degrades to a bare "
        f"'Tool: {{name}}' description): {missing}"
    )


def test_contract_prose_never_references_a_nonexistent_tool():
    """Every tool name *mentioned* inside contract prose must be either a
    currently-registered tool or one of the known-real-but-unregistered
    sandbox tools. Anything else is an invented capability (the
    'history_tool' bug this test was written to catch)."""
    universe = _registered_tool_names() | KNOWN_UNREGISTERED_BUT_REAL
    problems = []
    for path in _contract_files():
        text = path.read_text(encoding="utf-8")
        for mention in sorted(_mentions_in(text)):
            if mention not in universe:
                problems.append(f"{path.name}: references unknown tool `{mention}`")
    assert not problems, "\n".join(problems)


def test_contract_filenames_match_a_real_tool():
    """A contract file whose stem names neither a registered tool nor a
    known-unregistered-but-real one is orphaned/mis-named and should not
    exist under this directory."""
    universe = _registered_tool_names() | KNOWN_UNREGISTERED_BUT_REAL
    stray = sorted(
        path.name for path in _contract_files()
        if path.stem not in universe
    )
    assert not stray, f"contract file(s) with no matching real tool: {stray}"


@pytest.mark.parametrize("name", sorted(KNOWN_UNREGISTERED_BUT_REAL))
def test_known_unregistered_allowlist_entries_are_not_actually_registered(name):
    """Guards the allowlist itself from going stale: if one of the sandbox
    four ever gets wired into the live registry, it should be removed from
    KNOWN_UNREGISTERED_BUT_REAL (dead entries here would silently mask a
    real future 'unknown tool' bug for anything reusing this name)."""
    assert name not in _registered_tool_names(), (
        f"{name} is now registered on the live agent — remove it from "
        "KNOWN_UNREGISTERED_BUT_REAL in this test file"
    )
