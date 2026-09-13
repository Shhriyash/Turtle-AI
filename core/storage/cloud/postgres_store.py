"""
core/storage/cloud/postgres_store.py
-------------------------------------
Cloud (TURTLE_DEPLOY=cloud) counterpart to
core/storage/local/sqlite_store.py::SQLiteSessionStore, backed by Neon
Postgres instead of a per-process SQLite file.

Mirrors SQLiteSessionStore's public surface exactly (get/put/init_db/
list_sessions/delete) so SessionStore (core/session_store.py) works unchanged
regardless of which backend it holds — see core/storage/__init__.py's
SessionStoreProtocol docstring for the duck-typed extended surface this
implements in full.

Why Postgres instead of keeping SQLite: a serverless function instance has no
persistent local disk, so a SQLite file written by one invocation is gone by
the next. Postgres (Neon) is the durable, shared store every invocation talks
to identically, which is also what makes the /api/memory/confirm 404
(ISSUE class: _ACTIVE_STATES_BY_USER, fixed in Phase 3) impossible here.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Optional

from core.storage import Session, SessionStoreProtocol
from core.storage.cloud import get_pg_pool

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    data JSONB NOT NULL,
    user_id TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""
_CREATE_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)"


class PostgresSessionStore(SessionStoreProtocol):
    def __init__(self) -> None:
        # Guards against calling any query method before init_db() has run in
        # this process — cheap to check, and a much clearer failure than a
        # driver-level "relation sessions does not exist".
        self._initialized = False

    async def init_db(self) -> None:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)
            await conn.execute(_CREATE_INDEX_SQL)
        self._initialized = True

    async def _ensure_init(self) -> Any:
        pool = await get_pg_pool()
        if not self._initialized:
            await self.init_db()
        return pool

    async def get(self, session_id: str) -> Optional[Session]:
        pool = await self._ensure_init()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM sessions WHERE session_id = $1", session_id
            )
        if row is None:
            return None
        try:
            # asyncpg returns JSONB as a str unless a codec is registered; decode
            # defensively so a raw dict (future codec) also passes through.
            raw = row["data"]
            data = raw if isinstance(raw, dict) else json.loads(raw)
            return Session(session_id=session_id, data=data)
        except Exception:
            return None

    async def put(self, session: Session) -> None:
        pool = await self._ensure_init()
        user_id = session.data.get("user_id", "") or ""
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (session_id, data, user_id, updated_at)
                VALUES ($1, $2::jsonb, $3, now())
                ON CONFLICT (session_id)
                DO UPDATE SET data = EXCLUDED.data, user_id = EXCLUDED.user_id,
                              updated_at = now()
                """,
                session.session_id,
                json.dumps(session.data),
                user_id,
            )

    async def list_sessions(
        self, status_filter: str | None = None, user_id: str | None = None
    ) -> list[Session]:
        pool = await self._ensure_init()
        sessions: list[Session] = []
        async with pool.acquire() as conn:
            if user_id is not None:
                rows = await conn.fetch(
                    "SELECT session_id, data FROM sessions WHERE user_id = $1", user_id
                )
            else:
                rows = await conn.fetch("SELECT session_id, data FROM sessions")
        for row in rows:
            try:
                raw = row["data"]
                data = raw if isinstance(raw, dict) else json.loads(raw)
                if status_filter and data.get("status") != status_filter:
                    continue
                sessions.append(Session(session_id=row["session_id"], data=data))
            except Exception:
                pass
        return sessions

    async def delete(self, session_id: str) -> None:
        pool = await self._ensure_init()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM sessions WHERE session_id = $1", session_id)
