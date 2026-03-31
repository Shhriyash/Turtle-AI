from __future__ import annotations

from pathlib import Path
import shutil


def _safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except Exception:
        # Some stale Windows temp dirs can be ACL-locked.
        # Ignore and continue; pytest will still run.
        pass


def pytest_sessionstart() -> None:
    root = Path(__file__).resolve().parent
    _safe_rmtree(root / ".pytest_cache")

    for stale in root.glob("pytest-cache-files-*"):
        _safe_rmtree(stale)