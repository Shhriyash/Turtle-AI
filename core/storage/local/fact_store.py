"""
core/storage/local/fact_store.py
---------------------------------
G1: SQLite-backed FactStore for local mode.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import aiosqlite

from core.config import settings
from core.storage import Fact, FactStore


class SQLiteFactStore(FactStore):
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (settings.data_dir / "facts.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """CREATE TABLE IF NOT EXISTS facts (
                    id       TEXT PRIMARY KEY,
                    user_id  TEXT NOT NULL,
                    topic    TEXT NOT NULL,
                    key      TEXT NOT NULL,
                    value    TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    UNIQUE (user_id, topic, key)
                )"""
            )
            await db.execute("CREATE INDEX IF NOT EXISTS idx_facts_user_topic ON facts(user_id, topic)")
            await db.commit()

    async def upsert_fact(self, user_id: str, topic: str, key: str, value: str, metadata: dict | None = None) -> None:
        meta_json = json.dumps(metadata or {})
        fact_id = f"fact_{uuid.uuid4().hex[:12]}"
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO facts (id, user_id, topic, key, value, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, topic, key) DO UPDATE SET
                       value    = excluded.value,
                       metadata = excluded.metadata""",
                (fact_id, user_id, topic, key, value, meta_json),
            )
            await db.commit()

    async def get_facts(self, user_id: str, topic: str) -> list[Fact]:
        facts: list[Fact] = []
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id, topic, key, value, metadata FROM facts WHERE user_id = ? AND topic = ?",
                (user_id, topic),
            ) as cursor:
                async for row in cursor:
                    try:
                        meta = json.loads(row[4]) if row[4] else {}
                    except Exception:
                        meta = {}
                    facts.append(Fact(id=row[0], topic=row[1], key=row[2], value=row[3], metadata=meta))
        return facts
