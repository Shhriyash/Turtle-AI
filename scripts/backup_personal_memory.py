"""
scripts/backup_personal_memory.py
---------------------------------
Phase 7 — daily snapshot of irreplaceable per-user memory.

Creates a single tar.gz containing data/memory/personal/ at the moment of
invocation, names it by UTC date, and prunes older snapshots beyond the
retention window. Designed to be run from cron / Task Scheduler.

Usage:
    python -m scripts.backup_personal_memory \
        [--out path/to/snapshots] \
        [--retention-days 30]

The destination directory defaults to ``data/backups/personal/``. When you're
ready to ship offsite (S3 / B2), wrap this with a sync step in the same cron
job — keep the local snapshot for fast restore, push the copy for durability.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import DATA_DIR, PERSONAL_MEMORY_DIR  # noqa: E402


def _utc_date_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def create_snapshot(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not PERSONAL_MEMORY_DIR.exists():
        raise SystemExit(f"Nothing to back up: {PERSONAL_MEMORY_DIR} does not exist.")

    archive_path = out_dir / f"personal_memory_{_utc_date_stamp()}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(PERSONAL_MEMORY_DIR, arcname="personal")
    return archive_path


def prune_old(out_dir: Path, retention_days: int) -> list[Path]:
    if retention_days <= 0:
        return []
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    removed: list[Path] = []
    for entry in out_dir.glob("personal_memory_*.tar.gz"):
        try:
            mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=UTC)
        except OSError:
            continue
        if mtime < cutoff:
            try:
                entry.unlink()
                removed.append(entry)
            except OSError:
                pass
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "backups" / "personal",
        help="Snapshot output directory (default: data/backups/personal)",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=30,
        help="Delete snapshots older than N days (0 to keep all)",
    )
    args = parser.parse_args()

    archive = create_snapshot(args.out)
    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"Snapshot created: {archive} ({size_mb:.2f} MB)")

    pruned = prune_old(args.out, args.retention_days)
    for p in pruned:
        print(f"Pruned old snapshot: {p}")


if __name__ == "__main__":
    main()
