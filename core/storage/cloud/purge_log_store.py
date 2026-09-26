"""
core/storage/cloud/purge_log_store.py
--------------------------------------
Ledger item 2.4 (S-4.2 follow-up): a content-free audit trail that a purge
happened, without keeping who it was.

Deliberately does NOT store the plaintext user_id or email — only a SHA-256
hash of the user_id. The row's job is to let an operator (or a future audit)
prove "an erasure request for this user was received and completed" without
that proof itself becoming a second place personal data lingers after
/forget-me supposedly deleted it. counts is the same per-table dict
core/tenant_purge.py::purge_user returns, so the row also records *what* was
deleted (row counts only — never content).

Async (asyncpg), matching core/tenant_purge.py's own async call site (it
already holds the async pg pool open for the purge transaction itself).
Lazy-init CREATE TABLE IF NOT EXISTS behind a module-level flag, following
every other cloud store in this package — there is no migrations framework
yet (that's Phase 3).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

from core.storage.cloud import get_pg_pool

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS purge_log (
    id BIGSERIAL PRIMARY KEY,
    user_id_sha256 TEXT NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL,
    counts JSONB NOT NULL
)
"""
_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_purge_log_user_id_sha256 "
    "ON purge_log(user_id_sha256)"
)

_initialized = False


def hash_user_id(user_id: str) -> str:
    """The ONE place a user_id is turned into its purge_log identifier —
    every write must route through this so the plaintext id never reaches
    the table."""
    return hashlib.sha256((user_id or "").encode("utf-8")).hexdigest()


async def _ensure_init() -> Any:
    global _initialized
    pool = await get_pg_pool()
    if not _initialized:
        async with pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)
            await conn.execute(_CREATE_INDEX_SQL)
        _initialized = True
    return pool


async def write_purge_log(
    user_id: str,
    *,
    requested_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    counts: dict[str, Any],
) -> None:
    """Insert one content-free purge_log row. counts is JSON-serialized as-is
    (per-table row counts + redis_keys + revoked — see
    core/tenant_purge.py::purge_user's return shape)."""
    import json

    pool = await _ensure_init()
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO purge_log (user_id_sha256, requested_at, completed_at, counts) "
            "VALUES ($1, $2, $3, $4::jsonb)",
            hash_user_id(user_id),
            requested_at or now,
            completed_at or now,
            json.dumps(counts),
        )
