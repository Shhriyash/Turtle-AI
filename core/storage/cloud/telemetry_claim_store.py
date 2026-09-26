"""
core/storage/cloud/telemetry_claim_store.py
--------------------------------------------
Exactly-once dedup for core/telemetry.py's emit_once() in cloud mode (WP2.C /
ledger 2.6): claims a (user_id, event) pair before emitting so a funnel event
fires exactly once per user, not once per cold start.

Mirrors core/storage/cloud/routine_last_fired_store.py's try_claim_fire
exactly -- same atomic-claim pattern (INSERT ... ON CONFLICT DO NOTHING plus
a rowcount check), same module-level _initialized flag, same lazy
CREATE TABLE IF NOT EXISTS. There is no migrations framework yet; every
cloud store declares its own DDL inline (Phase 3 changes that).

SYNCHRONOUS (psycopg), called via asyncio.to_thread from async call sites --
matching this migration's established per-call-site driver choice.
"""
from __future__ import annotations

from typing import Any

from core.storage.cloud import get_pg_sync_pool

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS telemetry_once (
    user_id TEXT NOT NULL,
    event TEXT NOT NULL,
    first_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, event)
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


def try_claim_once(user_id: str, event: str) -> bool:
    """Atomically claim this (user_id, event) pair.

    Returns True the FIRST time it's called for a given pair (proceed to
    emit) and False every subsequent time (already emitted -- skip). The
    ON CONFLICT DO NOTHING + rowcount check is the atomic part: two
    concurrent callers racing on the same pair (e.g. two cold starts) can
    only ever have one winner, unlike a SELECT-then-INSERT check.
    """
    pool = _ensure_init()
    with pool.connection() as conn:
        cur = conn.execute(
            "INSERT INTO telemetry_once (user_id, event) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING",
            (user_id, event),
        )
        return cur.rowcount == 1
