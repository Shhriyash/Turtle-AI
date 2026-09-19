"""
core/storage/cloud/cron_tick_cursor_store.py
---------------------------------------------
The "when did the cron tick last run?" cursor for apps/cron_tick_routes.py.

The tick fires whatever came due in (last_tick, now], so it needs last_tick
to survive between invocations — and serverless has no process to keep it in,
which is the same constraint that replaced the in-process scheduler with an
external trigger in the first place. One row in Postgres, single-row by
construction (id is pinned to 1 by a CHECK), read at the start of a tick and
advanced at the end.

This is the cursor, not the dedupe: routine_last_fired_store.py still claims
each (user, routine, occurrence) before firing, so even a cursor that rewinds
— a restore, a clock skew, two ticks racing — cannot double-fire a routine.

SYNCHRONOUS (psycopg), called via asyncio.to_thread from the async cron-tick
route — matching this migration's established per-call-site driver choice.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from core.storage.cloud import get_pg_sync_pool

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cron_tick_cursor (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    last_tick_at TIMESTAMPTZ NOT NULL
)
"""

_initialized = False


def _ensure_init() -> Any:
    global _initialized
    pool = get_pg_sync_pool()
    if not _initialized:
        with pool.connection() as conn:
            conn.execute(_CREATE_TABLE_SQL)
        _initialized = True
    return pool


def read_last_tick() -> Optional[datetime]:
    """The last advanced tick time (UTC-aware), or None before the first tick
    ever completes."""
    pool = _ensure_init()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT last_tick_at FROM cron_tick_cursor WHERE id = 1"
        ).fetchone()
    if not row or row[0] is None:
        return None
    stamp = row[0]
    # A TIMESTAMPTZ comes back aware, but a column written by an older driver
    # (or a fake pool in tests) can hand back a naive value; the caller does
    # arithmetic against an aware now_utc, so normalize rather than crash.
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def write_last_tick(ticked_at: datetime) -> None:
    """Advance the cursor. Called after a tick has finished scanning, so a
    tick that dies partway leaves the cursor where it was and the next one
    re-covers the same range (the bucket dedupe absorbs the overlap)."""
    pool = _ensure_init()
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO cron_tick_cursor (id, last_tick_at) VALUES (1, %s) "
            "ON CONFLICT (id) DO UPDATE SET last_tick_at = EXCLUDED.last_tick_at",
            (ticked_at,),
        )
