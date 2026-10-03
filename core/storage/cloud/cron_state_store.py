"""
core/storage/cloud/cron_state_store.py
----------------------------------------
Global (not per-user) scheduler state for the cron-tick endpoint
(apps/cron_tick_routes.py): a single row holding `last_tick_at`, the upper
edge of the window the previous tick enumerated. The next tick enumerates
every routine occurrence in (last_tick_at, now], which is what lets a tick
that GitHub delivered hours late still find the occurrences it stepped over.

cron_state has no user_id, so it is deliberately NOT part of
core/tenant_purge.py.

Concurrency: `locked_tick_state()` reads the row with `FOR UPDATE SKIP
LOCKED` inside a transaction that stays open for the whole tick. A second,
overlapping tick finds the row locked and gets None (it should skip) rather
than enumerating the same window or queueing behind the first. The lock is
released when the first tick commits (advancing last_tick_at) or rolls back
(on an exception, leaving last_tick_at untouched so the window is retried).

Pool note: the open transaction pins one connection of the shared sync pool
(core.storage.cloud.get_pg_sync_pool, max_size=5) for the duration of the
tick, while the tick's own work borrows connections one at a time, so a tick
needs at most 2 of 5 at once. SKIP LOCKED (rather than blocking) is chosen so
overlapping ticks never pile up holding connections while they wait.

SYNCHRONOUS (psycopg), called from the tick's worker thread.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Optional

from core.storage.cloud import get_pg_sync_pool

# One row, id fixed at 1. last_tick_at is NULL until the first tick commits.
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cron_state (
    id SMALLINT PRIMARY KEY CHECK (id = 1),
    last_tick_at TIMESTAMPTZ
)
"""
_SEED_ROW_SQL = (
    "INSERT INTO cron_state (id, last_tick_at) VALUES (1, NULL) ON CONFLICT DO NOTHING"
)

_initialized = False


def _ensure_init() -> Any:
    global _initialized
    pool = get_pg_sync_pool()
    if not _initialized:
        with pool.connection() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_SEED_ROW_SQL)
        _initialized = True
    return pool


class TickState:
    """Handle yielded by locked_tick_state(); valid only inside the block."""

    def __init__(self, conn: Any, last_tick_at: Optional[datetime]) -> None:
        self._conn = conn
        self.last_tick_at = last_tick_at

    def advance(self, tick_at: datetime) -> None:
        """Set last_tick_at; takes effect when the block exits cleanly."""
        self._conn.execute(
            "UPDATE cron_state SET last_tick_at = %s WHERE id = 1", (tick_at,)
        )
        self.last_tick_at = tick_at


@contextmanager
def locked_tick_state() -> Iterator[Optional[TickState]]:
    """Hold the global tick lock for the duration of the block.

    Yields a TickState (with .last_tick_at, None on the very first tick ever),
    or None when another tick currently holds the lock. Commits on a clean
    exit, rolls back if the block raises.
    """
    pool = _ensure_init()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT last_tick_at FROM cron_state WHERE id = 1 FOR UPDATE SKIP LOCKED"
        ).fetchone()
        if row is None:
            yield None
            return
        yield TickState(conn, row[0])


def get_last_tick_at() -> Optional[datetime]:
    """Non-locking read of last_tick_at (None before the first tick)."""
    pool = _ensure_init()
    with pool.connection() as conn:
        row = conn.execute("SELECT last_tick_at FROM cron_state WHERE id = 1").fetchone()
    return row[0] if row else None
