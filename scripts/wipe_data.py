from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.paths import DATA_DIR


def _delete_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Wipe Turtle data directories before first boot.")
    parser.add_argument("--confirm", action="store_true", help="Delete the data paths listed below.")
    args = parser.parse_args()

    targets = [
        DATA_DIR / "rag",
        DATA_DIR / "sessions" / "active",
        DATA_DIR / "sessions" / "archive",
        DATA_DIR / "memory",
        DATA_DIR / "tasks" / "history.jsonl",
        DATA_DIR / "tasks" / "history.sqlite",
    ]

    print("Planned deletions:")
    for target in targets:
        print(f" - {target}")

    if not args.confirm:
        print("\nDry run only. Re-run with --confirm to delete.")
        return 1

    for target in targets:
        _delete_path(target)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
