"""Guard: every consumer must read keys that load_profile_snapshot() produces.

The email tool read `snapshot["contacts"]` and `snapshot["relations"]` while
`load_profile_snapshot()` returned neither key, so the email agent was handed
`{"contacts": {}, "relations": {}}` on every send -- silently, because the
read sat behind a bare `except Exception`. Nothing failed, nothing logged, and
the tool looked like it was supplying known-contact context.

Fixing that one call site leaves the class of bug wide open: any future
consumer can invent a key, get `None`, and fail the same invisible way. So
this test re-derives the contract on every run instead of hardcoding it:

1. ask a real (empty) store what top-level keys the snapshot actually has;
2. statically sweep the repo for every key any consumer reads off a snapshot;
3. assert (2) is a subset of (1).

Step 2 is AST-based, not a regex, and it is scoped per function: a name only
counts if it was assigned from a `*.load_profile_snapshot()` call in the same
scope, so an unrelated local called `profile` cannot produce a false positive.
"""

from __future__ import annotations

import ast
import shutil
import unittest
import uuid
from pathlib import Path

from core.memory_schema import TOPICS
from core.personal_memory_store import PersonalMemoryStore

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories that are not ours to police.
SKIP_DIR_NAMES = {".git", ".venv", "venv", "__pycache__", "node_modules", ".claude"}

SNAPSHOT_CALL = "load_profile_snapshot"


def _snapshot_top_level_keys() -> set[str]:
    """The authoritative shape, taken from the function itself.

    Explicit base_dir/index_path/logs_dir force the local backend, so this
    holds in cloud mode too (the isolation rule the other store tests use).
    `topic_paths` has to be passed as well: DEFAULT_TOPICS is built from
    `user_id`, NOT from `base_dir`, so a store given only `base_dir` still
    reads the real `default` user's topic files.
    """
    base = Path(__file__).resolve().parent / "_tmp" / f"snapshot_contract_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        store = PersonalMemoryStore(
            base_dir=base,
            index_path=base / "MEMORY.md",
            logs_dir=base / "logs",
            topic_paths={topic: base / f"{topic}.md" for topic in TOPICS},
        )
        return set(store.load_profile_snapshot().keys())
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _is_snapshot_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == SNAPSHOT_CALL
    )


def _string_key(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


class _ScopeScanner(ast.NodeVisitor):
    """Collect (key, lineno) for every key read off a snapshot in one module.

    A new scope starts at each module/function/lambda/comprehension boundary;
    names bound to a snapshot in an enclosing scope stay visible in nested
    ones, matching how Python itself resolves them.
    """

    def __init__(self) -> None:
        self.reads: list[tuple[str, int]] = []
        self._scopes: list[set[str]] = [set()]

    # -- scope bookkeeping -------------------------------------------------
    def _push_scope(self) -> None:
        self._scopes.append(set(self._scopes[-1]))

    def _pop_scope(self) -> None:
        self._scopes.pop()

    def _visit_scoped(self, node: ast.AST) -> None:
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()

    visit_FunctionDef = _visit_scoped
    visit_AsyncFunctionDef = _visit_scoped
    visit_Lambda = _visit_scoped

    # Class bodies do not create a name scope for nested functions, but
    # treating them like one is harmless here and keeps the walk uniform.
    visit_ClassDef = _visit_scoped

    def _bind(self, name: str) -> None:
        self._scopes[-1].add(name)

    def _is_bound(self, name: str) -> bool:
        return name in self._scopes[-1]

    # -- bindings ----------------------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_snapshot_call(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._bind(target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and _is_snapshot_call(node.value) and isinstance(node.target, ast.Name):
            self._bind(node.target.id)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if _is_snapshot_call(node.value) and isinstance(node.target, ast.Name):
            self._bind(node.target.id)
        self.generic_visit(node)

    # -- reads -------------------------------------------------------------
    def _reads_snapshot(self, node: ast.AST) -> bool:
        """True when `node` is a snapshot: a bound name, or the call itself."""
        if isinstance(node, ast.Name):
            return self._is_bound(node.id)
        return _is_snapshot_call(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # snapshot["contacts"]
        key = _string_key(node.slice)
        if key is not None and self._reads_snapshot(node.value):
            self.reads.append((key, node.lineno))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # snapshot.get("contacts") / snapshot.get("contacts", {})
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and node.args
            and self._reads_snapshot(func.value)
        ):
            key = _string_key(node.args[0])
            if key is not None:
                self.reads.append((key, node.lineno))
        self.generic_visit(node)


def _python_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in SKIP_DIR_NAMES for part in path.relative_to(REPO_ROOT).parts):
            continue
        files.append(path)
    return files


def _collect_snapshot_key_reads() -> dict[str, list[str]]:
    """Map every key read off a snapshot to the `path:line` sites reading it."""
    sites: dict[str, list[str]] = {}
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        scanner = _ScopeScanner()
        scanner.visit(tree)
        if not scanner.reads:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for key, lineno in scanner.reads:
            sites.setdefault(key, []).append(f"{rel}:{lineno}")
    return sites


class ProfileSnapshotContractTests(unittest.TestCase):
    def test_every_consumer_reads_a_key_the_snapshot_produces(self) -> None:
        produced = _snapshot_top_level_keys()
        self.assertTrue(produced, "load_profile_snapshot() returned no keys at all")

        read_sites = _collect_snapshot_key_reads()
        self.assertTrue(
            read_sites,
            "the AST sweep found no snapshot key reads -- the scanner has "
            "stopped matching the code it is supposed to police",
        )

        phantom = {key: sites for key, sites in sorted(read_sites.items()) if key not in produced}
        if phantom:
            detail = "\n".join(f"  {key!r} read at {', '.join(sites)}" for key, sites in phantom.items())
            self.fail(
                "These consumers read top-level keys that load_profile_snapshot() "
                "does not produce, so they silently receive None:\n"
                f"{detail}\n"
                f"Keys the snapshot actually produces: {sorted(produced)}\n"
                "Either add the key to load_profile_snapshot() or read the data "
                "from the store directly."
            )

    def test_scanner_catches_a_phantom_key(self) -> None:
        """The sweep is only a guard if it can still fail. Prove it does.

        Without this, a scanner that silently stopped matching (an AST change,
        a refactor to a different call shape) would leave the suite green and
        the class of bug unguarded again.
        """
        source = (
            "def handler(store):\n"
            "    snapshot = store.load_profile_snapshot()\n"
            "    a = snapshot.get('identity')\n"
            "    b = snapshot['definitely_not_a_real_key']\n"
            "    c = snapshot.get('another_phantom', {})\n"
            "    return a, b, c\n"
        )
        scanner = _ScopeScanner()
        scanner.visit(ast.parse(source))
        found = {key for key, _ in scanner.reads}
        self.assertEqual(found, {"identity", "definitely_not_a_real_key", "another_phantom"})

    def test_scanner_ignores_unrelated_locals(self) -> None:
        """A dict that never came from the snapshot must not be policed."""
        source = (
            "def handler(payload):\n"
            "    profile = {'anything': 1}\n"
            "    return profile.get('anything'), payload['whatever']\n"
        )
        scanner = _ScopeScanner()
        scanner.visit(ast.parse(source))
        self.assertEqual(scanner.reads, [])

    def test_scanner_scope_does_not_leak_between_functions(self) -> None:
        source = (
            "def one(store):\n"
            "    profile = store.load_profile_snapshot()\n"
            "    return profile.get('identity')\n"
            "def two(profile):\n"
            "    return profile.get('not_policed_here')\n"
        )
        scanner = _ScopeScanner()
        scanner.visit(ast.parse(source))
        self.assertEqual({key for key, _ in scanner.reads}, {"identity"})

    def test_known_consumer_keys_are_covered_by_the_sweep(self) -> None:
        """Anchor the sweep to the real call sites it must keep watching."""
        read_sites = _collect_snapshot_key_reads()
        for key in ("identity", "preferences", "workflow", "contacts", "relations"):
            self.assertIn(
                key,
                read_sites,
                f"no consumer reads {key!r} any more -- if that is intended, drop it "
                "from this list, but check the sweep is still finding call sites",
            )


if __name__ == "__main__":
    unittest.main()
