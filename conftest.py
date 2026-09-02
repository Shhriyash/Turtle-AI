from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
import shutil

# The suite must be deterministic and fully offline. Disable the personal-memory
# embed background job so that write_topic under a running loop never fires a
# live Cohere embed or a real data/ vector write (the exact tripwire documented
# in test/retrieval_broker_test.py:134-136). setdefault so an explicit override
# in the environment still wins.
os.environ.setdefault("TURTLE_PERSONAL_EMBED_ENABLED", "0")

# ---------------------------------------------------------------------------
# Test data isolation (ISSUE-010)
#
# Every persistent write lands under TURTLE_DATA_DIR. Point it at a throwaway
# directory here, before pytest imports any test module and therefore before
# anything imports core.paths, which resolves DATA_DIR once at import time.
#
# This is assigned, not setdefault'd. A stray TURTLE_DATA_DIR in the shell
# pointing at real memory is exactly the accident this guards against, and no
# test should want production state. A test that truly does can set it itself.
#
# Without this the suite minted synthetic tenants (usr_alice, usr_bob,
# usr_h1a..d, default, and a MagicMock that had been stringified into a path)
# straight into data/memory/personal alongside real accounts.
# ---------------------------------------------------------------------------
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="turtle-test-data-")
os.environ["TURTLE_DATA_DIR"] = _TEST_DATA_DIR


def _safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except Exception:
        # Some stale Windows temp dirs can be ACL-locked.
        # Ignore and continue; pytest will still run.
        pass


# ---------------------------------------------------------------------------
# data/ write tripwire
#
# Tests must never write into the repo's production data/ tree — they should
# use tmp dirs (test/_tmp, tmp_path, ...). We snapshot every file under data/
# at session start and compare at session finish; any created or modified file
# now FAILS the run.
#
# It warned rather than failed while the suite was known dirty, which was
# right: a hard failure would have been red on every run and quickly tuned
# out. Now that conftest redirects TURTLE_DATA_DIR the tree stays clean, so
# a hit means a regression rather than the status quo.
# ---------------------------------------------------------------------------

_DATA_SNAPSHOT: dict[str, tuple[float, int]] = {}


def _snapshot_data_tree(root: Path) -> dict[str, tuple[float, int]]:
    data_dir = root / "data"
    snapshot: dict[str, tuple[float, int]] = {}
    if not data_dir.exists():
        return snapshot
    try:
        for path in data_dir.rglob("*"):
            try:
                if path.is_file():
                    stat = path.stat()
                    snapshot[str(path)] = (stat.st_mtime, stat.st_size)
            except OSError:
                # File vanished mid-walk or is locked — ignore.
                continue
    except OSError:
        pass
    return snapshot


def pytest_sessionstart() -> None:
    root = Path(__file__).resolve().parent
    _safe_rmtree(root / ".pytest_cache")

    for stale in root.glob("pytest-cache-files-*"):
        _safe_rmtree(stale)

    global _DATA_SNAPSHOT
    _DATA_SNAPSHOT = _snapshot_data_tree(root)


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ARG001
    root = Path(__file__).resolve().parent
    _safe_rmtree(Path(_TEST_DATA_DIR))
    after = _snapshot_data_tree(root)
    dirty = sorted(
        path for path, sig in after.items() if _DATA_SNAPSHOT.get(path) != sig
    )
    if dirty:
        listing = "\n  ".join(dirty)
        message = (
            f"tests wrote into production data/: {len(dirty)} file(s) changed:"
            f"\n  {listing}"
            "\n\nconftest points TURTLE_DATA_DIR at a temp dir, so this means"
            "\nsomething resolved a data path before that redirect, or"
            "\nhardcoded a repo path."
        )
        warnings.warn(UserWarning(message), stacklevel=1)
        # Fail the run. The note above the snapshot helpers explains why this
        # escalated from a warning.
        session.exitstatus = 1