"""P6-A2: local SQLite stores run in WAL mode; MemorySQLiteIndex is thread-safe.

Local mode only. journal_mode=WAL is persisted in the database file, so each
test inspects the file with a FRESH raw sqlite3 connection after the store has
initialised it -- that is what every later connection will inherit.

busy_timeout is deliberately not tested: Python's sqlite3/aiosqlite default
``timeout=5.0`` already sets PRAGMA busy_timeout=5000 (see
test_python_default_already_sets_busy_timeout, which pins that assumption).
"""
from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path

from core.memory_journal import make_event
from core.memory_sqlite import MemorySQLiteIndex


def _mode(db_path: Path) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    finally:
        conn.close()


def test_python_default_already_sets_busy_timeout():
    """Pins the premise for NOT adding a busy_timeout pragma (ledger clause is a no-op)."""
    conn = sqlite3.connect(":memory:")
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()

    async def aio() -> int:
        import aiosqlite

        async with aiosqlite.connect(":memory:") as db:
            return (await (await db.execute("PRAGMA busy_timeout")).fetchone())[0]

    assert asyncio.run(aio()) == 5000


def test_session_store_uses_wal(tmp_path):
    from core.storage.local.sqlite_store import SQLiteSessionStore

    db = tmp_path / "sessions.sqlite"
    asyncio.run(SQLiteSessionStore(db_path=db).init_db())
    assert _mode(db) == "wal"


def test_identity_manager_uses_wal(tmp_path):
    from core.identity import IdentityManager

    db = tmp_path / "users.sqlite"
    asyncio.run(IdentityManager(db_path=db).init_db())
    assert _mode(db) == "wal"


def test_link_code_store_uses_wal(tmp_path):
    from core.account_linking import LinkCodeStore

    db = tmp_path / "links.sqlite"
    LinkCodeStore(db)
    assert _mode(db) == "wal"


def test_task_history_index_uses_wal(tmp_path):
    from core.task_history_index import TaskHistoryIndex

    db = tmp_path / "history.sqlite"
    idx = TaskHistoryIndex(db)
    try:
        assert _mode(db) == "wal"
    finally:
        idx.close()


def test_idempotency_db_uses_wal(tmp_path, monkeypatch):
    import tools.idempotency as idem

    db = tmp_path / "tool_invocations.db"
    monkeypatch.setattr(idem, "_DB_PATH", db)
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)
    idem._ensure_db().close()
    assert _mode(db) == "wal"


def test_memory_index_still_wal(tmp_path):
    db = tmp_path / "memory.sqlite"
    idx = MemorySQLiteIndex(db_path=db)
    try:
        assert _mode(db) == "wal"
    finally:
        idx.close()


# ---------------------------------------------------------------------------
# Cross-thread use of MemorySQLiteIndex
# ---------------------------------------------------------------------------

def _event(n: int):
    return make_event(
        kind="fact",
        topic="relations",
        key=f"relations.friend_{n}",
        value={"friend": f"Person{n}"},
        confidence=1.0,
        source="explicit",
        extractor="deterministic",
        applied=True,
        session_id="s1",
        turn_id=f"t{n}",
        observed_at="2026-05-01T10:00:00Z",
    )


def test_memory_index_usable_from_another_thread(tmp_path):
    """Created on the main thread, written + read via asyncio.to_thread.

    The 6.9 write funnel will move index_event into a worker thread; before the
    fix this raised sqlite3.ProgrammingError ("SQLite objects created in a
    thread can only be used in that same thread").
    """
    idx = MemorySQLiteIndex(db_path=tmp_path / "memory.sqlite")
    try:
        ev = _event(1)

        async def go():
            await asyncio.to_thread(idx.index_event, ev)
            exists = await asyncio.to_thread(idx.event_exists, ev.event_id)
            n = await asyncio.to_thread(idx.count)
            found = await asyncio.to_thread(idx.search, "Person1")
            return exists, n, found

        exists, n, found = asyncio.run(go())
        assert exists is True and n == 1
        assert [r.event_id for r in found] == [ev.event_id]
        assert idx.is_stale is False
    finally:
        idx.close()


def test_memory_index_concurrent_threads_are_serialized(tmp_path):
    """Sharing one connection across threads is only safe if access is
    serialized: hammer it with writers and readers and require zero errors and
    every row present."""
    idx = MemorySQLiteIndex(db_path=tmp_path / "memory.sqlite")
    errors: list[BaseException] = []
    writers, per_writer = 4, 25

    def writer(w: int):
        try:
            for i in range(per_writer):
                idx.index_event(_event(w * 1000 + i))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            for _ in range(60):
                idx.count()
                idx.search("Person")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(writers)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    try:
        assert errors == [], f"thread errors: {errors[:3]!r}"
        assert idx.count() == writers * per_writer
        assert idx.is_stale is False
    finally:
        idx.close()
