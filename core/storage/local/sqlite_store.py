"""
core/storage/local/sqlite_store.py
----------------------------------
G1: SQLite-backed implementations of the core storage protocols for local mode.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from core.config import settings
from core.storage import Session, SessionStoreProtocol


class SQLiteSessionStore(SessionStoreProtocol):
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (settings.data_dir / "sessions.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                '''CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )'''
            )
            await db.commit()

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
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO sessions (session_id, data) VALUES (?, ?)",
                (session.session_id, json.dumps(session.data))
            )
            await db.commit()

    async def list_sessions(
        self, status_filter: str | None = None, user_id: str | None = None
    ) -> list[Session]:
        sessions = []
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT session_id, data FROM sessions") as cursor:
                async for row in cursor:
                    try:
                        data = json.loads(row[1])
                        if status_filter and data.get("status") != status_filter:
                            continue
                        # user_id lives inside the data JSON; no schema migration.
                        # Legacy rows have no user_id and only match user_id="".
                        if user_id is not None and data.get("user_id", "") != user_id:
                            continue
                        sessions.append(Session(session_id=row[0], data=data))
                    except Exception:
                        pass
        return sessions

    async def delete(self, session_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            await db.commit()
