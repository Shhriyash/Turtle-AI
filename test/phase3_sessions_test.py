"""
test/phase3_sessions_test.py
----------------------------
Phase 3 W1: the sessions table gains a real, indexed ``user_id`` column so
tenancy is enforced in SQL rather than only inside the serialized data JSON.

Covers the additive migration (legacy DB -> column added + backfilled + index),
the SQL-side tenant filter, that put() mirrors the tenant into the column, and
that mark_finalized refuses cross-tenant sessions while legacy unowned rows stay
sweepable.
"""
import asyncio
import json

import aiosqlite

from core.session_store import SessionStore
from core.storage import Session
from core.storage.local.sqlite_store import SQLiteSessionStore


def run(coro):
    return asyncio.run(coro)


async def _table_columns(db_path) -> set[str]:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("PRAGMA table_info(sessions)") as cursor:
            return {row[1] async for row in cursor}


async def _index_names(db_path) -> set[str]:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ) as cursor:
            return {row[0] async for row in cursor}


async def _column_user_id(db_path, session_id: str) -> str | None:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


def test_migration_adds_column_backfills_and_indexes(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"

        # Build a LEGACY DB by hand: the old schema had no user_id column and the
        # tenant lived only inside the data JSON.
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                '''CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )'''
            )
            await db.execute(
                "INSERT INTO sessions (session_id, data) VALUES (?, ?)",
                ("owned", json.dumps({"status": "active", "user_id": "usr_a"})),
            )
            await db.execute(
                "INSERT INTO sessions (session_id, data) VALUES (?, ?)",
                ("legacy", json.dumps({"status": "completed"})),  # no user_id key
            )
            await db.commit()

        assert "user_id" not in await _table_columns(db_path)

        # Opening the store runs the additive migration.
        store = SQLiteSessionStore(db_path=db_path)
        await store.init_db()

        assert "user_id" in await _table_columns(db_path)
        assert "idx_sessions_user" in await _index_names(db_path)
        # Backfill: JSON tenant copied into the column; missing -> ''.
        assert await _column_user_id(db_path, "owned") == "usr_a"
        assert await _column_user_id(db_path, "legacy") == ""

        # Re-running init_db is idempotent (no duplicate-column crash).
        await store.init_db()
        assert await _column_user_id(db_path, "owned") == "usr_a"

    run(scenario())


def test_list_sessions_filters_by_column(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await backend.init_db()
        await backend.put(Session(session_id="a1", data={"user_id": "usr_a"}))
        await backend.put(Session(session_id="a2", data={"user_id": "usr_a"}))
        await backend.put(Session(session_id="b1", data={"user_id": "usr_b"}))

        only_a = await backend.list_sessions(user_id="usr_a")
        assert sorted(s.session_id for s in only_a) == ["a1", "a2"]

        only_b = await backend.list_sessions(user_id="usr_b")
        assert [s.session_id for s in only_b] == ["b1"]

        # No user_id kwarg -> unscoped.
        assert len(await backend.list_sessions()) == 3

    run(scenario())


def test_put_writes_the_user_id_column(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await backend.init_db()
        await backend.put(Session(session_id="c1", data={"user_id": "usr_c"}))

        # The column (not just the JSON) carries the tenant.
        assert await _column_user_id(db_path, "c1") == "usr_c"
        # A row with no user_id in data lands as '' in the column.
        await backend.put(Session(session_id="c2", data={"status": "active"}))
        assert await _column_user_id(db_path, "c2") == ""

    run(scenario())


def test_mark_finalized_refuses_cross_tenant(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await backend.init_db()
        await backend.put(
            Session(
                session_id="owned_by_b",
                data={
                    "status": "pending_finalization",
                    "user_id": "usr_b",
                    "messages": [],
                },
            )
        )

        # usr_a must not be able to finalize usr_b's session.
        store = SessionStore(backend, user_id="usr_a")
        await store.mark_finalized("owned_by_b")

        untouched = await backend.get("owned_by_b")
        assert untouched is not None
        assert untouched.data["status"] == "pending_finalization"

        # The rightful owner still can.
        owner_store = SessionStore(backend, user_id="usr_b")
        await owner_store.mark_finalized("owned_by_b")
        finalized = await backend.get("owned_by_b")
        assert finalized is not None
        assert finalized.data["status"] == "completed"

    run(scenario())


def test_legacy_unowned_rows_still_swept(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await backend.init_db()
        # Legacy unowned pending row (no user_id).
        await backend.put(
            Session(
                session_id="legacy",
                data={
                    "status": "pending_finalization",
                    "messages": [],
                    "pending_email": {},
                    "summary": [],
                },
            )
        )
        # Another tenant's pending row.
        await backend.put(
            Session(
                session_id="other",
                data={
                    "status": "pending_finalization",
                    "user_id": "usr_b",
                    "messages": [],
                    "pending_email": {},
                    "summary": [],
                },
            )
        )

        store = SessionStore(backend, user_id="usr_a")
        await store.init_backend()

        pending_ids = [sid for sid, _ in await store.list_pending_finalization_archives()]
        assert pending_ids == ["legacy"]

    run(scenario())
