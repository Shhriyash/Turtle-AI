"""
core/storage/cloud/routine_outbox_store.py
---------------------------------------------
Cloud (TURTLE_DEPLOY=cloud) backend for core/routine_outbox.py: the durable
write-through queue of routine-delivery notices for a user with no live
socket open. The local JSON file under personal_memory_dir(user_id) does not
survive a serverless cold start, which would silently drop every queued
routine notice on the very next invocation.

SYNCHRONOUS (psycopg), matching core/routine_outbox.py's own plain-sync,
never-raising call sites (apps/turtle_server.py calls load_outbox/save_outbox
directly with no await).
"""
from __future__ import annotations

import json
from typing import Any, Optional

from core.storage.cloud import get_pg_sync_pool

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS routine_outbox (
    user_id TEXT PRIMARY KEY,
    frames JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

_initialized = False

# Ledger 3.6(e): both load/save below already degrade gracefully (never
# raise -- see each docstring), but the failure was only ever a `print()`,
# with nothing counting how often it happens. This is an in-process counter
# only; there is no metrics module yet (core/telemetry.py's emit_once is a
# funnel-event dedup helper, not a general counter), so it does not reach a
# dashboard until Phase 4 wires real metrics export. It still gives a single
# process visibility into "how degraded am I right now" and is cheap to
# assert against in tests.
_METRICS = {"load_failures": 0, "save_failures": 0}


def get_outbox_failure_counts() -> dict[str, int]:
    """Snapshot of this process's outbox degrade counters (see _METRICS)."""
    return dict(_METRICS)


def _ensure_init() -> Any:
    global _initialized
    pool = get_pg_sync_pool()
    if not _initialized:
        with pool.connection() as conn:
            conn.execute(_CREATE_TABLE_SQL)
        _initialized = True
    return pool


def load_outbox_pg(user_id: str) -> Optional[list[dict[str, Any]]]:
    """Drop-in for core.routine_outbox.load_outbox. Never raises — an error
    here signals "unknown, don't overwrite" via None, matching the local
    version's transient-read-failure contract."""
    try:
        pool = _ensure_init()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT frames FROM routine_outbox WHERE user_id = %s", (user_id,)
            ).fetchone()
        if row is None:
            return []
        frames = row[0]
        frames = frames if isinstance(frames, list) else json.loads(frames or "[]")
        return [f for f in frames if isinstance(f, dict)]
    except Exception as e:
        _METRICS["load_failures"] += 1
        print(f"LOG: routine_outbox (postgres) load failed user={user_id}: {e}")
        return None


def save_outbox_pg(user_id: str, frames: list[dict[str, Any]], *, max_frames: int) -> None:
    """Drop-in for core.routine_outbox.save_outbox. Never raises."""
    try:
        pool = _ensure_init()
        capped = list(frames)[-max_frames:] if frames else []
        with pool.connection() as conn:
            if not capped:
                conn.execute("DELETE FROM routine_outbox WHERE user_id = %s", (user_id,))
                return
            conn.execute(
                "INSERT INTO routine_outbox (user_id, frames, updated_at) "
                "VALUES (%s, %s::jsonb, now()) "
                "ON CONFLICT (user_id) DO UPDATE SET frames = EXCLUDED.frames, "
                "updated_at = now()",
                (user_id, json.dumps(capped)),
            )
    except Exception as e:
        _METRICS["save_failures"] += 1
        print(f"LOG: routine_outbox (postgres) save failed user={user_id}: {e}")
