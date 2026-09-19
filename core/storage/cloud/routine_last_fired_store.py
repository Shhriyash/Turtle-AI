"""
core/storage/cloud/routine_last_fired_store.py
--------------------------------------------------
Exactly-once dedup for the cron-tick endpoint (apps/cron_tick_routes.py):
claims a (user_id, routine_key, fire_bucket) tuple before firing so a routine
fires exactly once per SCHEDULED occurrence even if two ticks cover
overlapping ranges (a retry, a tick that died before advancing its cursor, a
rewound cursor) or one trigger run overlaps the previous one. fire_bucket
comes from core.routine_cron_tick.compute_due_occurrences and identifies the
scheduled time, not the tick that observed it.

SYNCHRONOUS (psycopg), called via asyncio.to_thread from the async cron-tick
route — matching this migration's established per-call-site driver choice.
"""
from __future__ import annotations

from typing import Any

from core.storage.cloud import get_pg_sync_pool

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS routine_last_fired (
    user_id TEXT NOT NULL,
    routine_key TEXT NOT NULL,
    fire_bucket TEXT NOT NULL,
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, routine_key, fire_bucket)
)
"""
# Old buckets accumulate forever otherwise (one row per routine per scheduled
# fire) — bounded by an index on claimed_at so a periodic prune stays cheap.
_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_routine_last_fired_claimed_at "
    "ON routine_last_fired(claimed_at)"
)

_initialized = False


def _ensure_init() -> Any:
    global _initialized
    pool = get_pg_sync_pool()
    if not _initialized:
        with pool.connection() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
        _initialized = True
    return pool


def try_claim_fire(user_id: str, routine_key: str, fire_bucket: str) -> bool:
    """Atomically claim this (user, routine, scheduled-occurrence) tuple.

    Returns True the FIRST time it's called for a given tuple (proceed to
    fire) and False every subsequent time (already fired — skip). The
    ON CONFLICT DO NOTHING + rowcount check is the atomic part: two
    concurrent cron-tick invocations racing on the same tuple can only ever
    have one winner, unlike a SELECT-then-INSERT check.
    """
    pool = _ensure_init()
    with pool.connection() as conn:
        cur = conn.execute(
            "INSERT INTO routine_last_fired (user_id, routine_key, fire_bucket) "
            "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            (user_id, routine_key, fire_bucket),
        )
        return cur.rowcount == 1


def prune_older_than(cutoff_iso: str) -> int:
    """Delete claim rows older than cutoff_iso (an ISO-8601 timestamp).
    Best-effort housekeeping; not called automatically — invoke from an
    admin/maintenance task. Returns the number of rows deleted."""
    pool = _ensure_init()
    with pool.connection() as conn:
        cur = conn.execute(
            "DELETE FROM routine_last_fired WHERE claimed_at < %s", (cutoff_iso,)
        )
        return cur.rowcount
