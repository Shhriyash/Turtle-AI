from __future__ import annotations

import os
import warnings
from pathlib import Path
import shutil

# The suite must be deterministic and fully offline. Disable the personal-memory
# embed background job so that write_topic under a running loop never fires a
# live Cohere embed or a real data/ vector write (the exact tripwire documented
# in test/retrieval_broker_test.py:134-136). setdefault so an explicit override
# in the environment still wins.
os.environ.setdefault("TURTLE_PERSONAL_EMBED_ENABLED", "0")


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
# raises a UserWarning (warning, not failure, so the suite still reports its
# real result).
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
    after = _snapshot_data_tree(root)
    dirty = sorted(
        path for path, sig in after.items() if _DATA_SNAPSHOT.get(path) != sig
    )
    if dirty:
        listing = "\n  ".join(dirty)
        warnings.warn(
            UserWarning(
                f"tests wrote into production data/: {len(dirty)} file(s) created/changed:\n  {listing}"
            ),
            stacklevel=1,
        )