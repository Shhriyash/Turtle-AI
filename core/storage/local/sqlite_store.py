"""
core/storage/local/sqlite_store.py
----------------------------------
G1: SQLite-backed implementations of the core storage protocols for local mode.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from core.config import settings
from core.storage import Session, SessionStoreProtocol


class SQLiteSessionStore(SessionStoreProtocol):
    # Additive tenancy migration: (column, type/def) pairs an older DB may lack.
    # Fresh DBs already carry these from the CREATE below, so the ALTER is a
    # no-op there; a pre-existing production DB gains them via ALTER TABLE.
    _MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
        ("user_id", "TEXT NOT NULL DEFAULT ''"),
    )

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (settings.data_dir / "sessions.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                '''CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT '',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )'''
            )
            # Bring older DBs (session_id/data/updated_at only) up to schema
            # before the index and backfill touch the user_id column.
            await self._migrate_columns(db)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)"
            )
            # Backfill legacy rows whose tenant still lives only in the data JSON
            # (SQLite JSON1 is built in). Idempotent: only touches rows still ''.
            try:
                await db.execute(
                    "UPDATE sessions SET user_id = COALESCE(json_extract(data, '$.user_id'), '') "
                    "WHERE user_id = ''"
                )
            except Exception:
                # json_extract aborts the whole UPDATE on a JSON1-less SQLite
                # build or a single malformed legacy data blob — and a failed
                # backfill must not brick session storage at init. Fall back to
                # per-row Python parsing, skipping only the unparseable rows.
                async with db.execute(
                    "SELECT session_id, data FROM sessions WHERE user_id = ''"
                ) as cursor:
                    rows = [row async for row in cursor]
                for session_id, raw in rows:
                    try:
                        parsed_user = str(json.loads(raw).get("user_id", "") or "")
                    except Exception:
                        continue
                    if parsed_user:
                        await db.execute(
                            "UPDATE sessions SET user_id = ? WHERE session_id = ?",
                            (parsed_user, session_id),
                        )
            await db.commit()

    async def _migrate_columns(self, db: aiosqlite.Connection) -> None:
        """Add the tenancy column(s) if an older DB predates them.

        SQLite has no ``ADD COLUMN IF NOT EXISTS``, so probe ``PRAGMA
        table_info`` and only ALTER for columns that are actually missing. Only
        the duplicate-column race (concurrent open) is swallowed; any other
        failure (locked DB, corrupt file) must surface here, at init, instead of
        as a confusing ``no such column`` from a later query. A final probe
        verifies every required column exists. ``DEFAULT ''`` backfills existing
        rows so the ``NOT NULL`` column is well-defined on old data.
        """
        async def _columns() -> set[str]:
            async with db.execute("PRAGMA table_info(sessions)") as cursor:
                return {row[1] async for row in cursor}

        existing = await _columns()
        for column, definition in self._MIGRATION_COLUMNS:
            if column in existing:
                continue
            try:
                await db.execute(f"ALTER TABLE sessions ADD COLUMN {column} {definition}")
            except sqlite3.OperationalError as exc:
                # Column already present (concurrent open) — migration is additive.
                if "duplicate column" not in str(exc).lower():
                    raise
        await db.commit()
        present = await _columns()
        missing = [c for c, _ in self._MIGRATION_COLUMNS if c not in present]
        if missing:
            raise sqlite3.OperationalError(
                f"sessions.sqlite tenancy migration incomplete; missing columns: {missing}"
            )

    async def get(self, session_id: str) -> Optional[Session]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT data FROM sessions WHERE session_id = ?", (session_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    try:
                        data = json.loads(row[0])
                        return Session(session_id=session_id, data=data)
                    except Exception:
                        pass
        return None

    async def put(self, session: Session) -> None:
        # Mirror the tenant into its own indexed column (kept in sync with the
        # data JSON) and refresh the column-level updated_at on every write.
        user_id = session.data.get("user_id", "") or ""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO sessions (session_id, data, user_id, updated_at) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (session.session_id, json.dumps(session.data), user_id)
            )
            await db.commit()

    async def list_sessions(
        self, status_filter: str | None = None, user_id: str | None = None
    ) -> list[Session]:
        sessions = []
        # Tenancy is a real indexed column now, so scope in SQL. Status still
        # lives inside the data JSON, so that filter stays in Python.
        if user_id is not None:
            query = "SELECT session_id, data FROM sessions WHERE user_id = ?"
            params: tuple[Any, ...] = (user_id,)
        else:
            query = "SELECT session_id, data FROM sessions"
            params = ()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    try:
                        data = json.loads(row[1])
                        if status_filter and data.get("status") != status_filter:
                            continue
                        sessions.append(Session(session_id=row[0], data=data))
                    except Exception:
                        pass
        return sessions

    async def delete(self, session_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            await db.commit()
