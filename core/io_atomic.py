from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _fsync_parent(path: Path) -> None:
    try:
        parent_fd = os.open(str(path.parent), os.O_RDONLY)
    except Exception:
        return
    try:
        os.fsync(parent_fd)
    except Exception:
        pass
    finally:
        os.close(parent_fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        _fsync_parent(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, data.encode(encoding))


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2, ensure_ascii: bool = False) -> None:
    atomic_write_text(path, json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii))
