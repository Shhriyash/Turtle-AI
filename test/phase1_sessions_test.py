import asyncio
from datetime import datetime, timedelta, timezone

from pydantic_ai.messages import ModelRequest, UserPromptPart

from core.session_store import SessionStore
from core.storage import Session
from core.storage.local.sqlite_store import SQLiteSessionStore


def run(coro):
    return asyncio.run(coro)


def old_iso(hours: int = 0, days: int = 0) -> str:
    value = datetime.now(timezone.utc) - timedelta(hours=hours, days=days)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


async def init_backend(backend: SQLiteSessionStore) -> None:
    if hasattr(backend, "init_db"):
        await backend.init_db()


def test_resume_if_active_is_tenant_scoped(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"

        store_a = SessionStore(SQLiteSessionStore(db_path=db_path), user_id="usr_a")
        await store_a.start_or_restore("strict_new")
        await store_a.replace_messages(
            [ModelRequest(parts=[UserPromptPart(content="hello from a")])]
        )
        session_a = store_a.session_id

        store_b = SessionStore(SQLiteSessionStore(db_path=db_path), user_id="usr_b")
        result_b = await store_b.start_or_restore("resume_if_active")

        assert result_b.restored is False
        assert store_b.session_id is not None
        assert store_b.session_id != session_a

    run(scenario())


def test_resume_if_active_demotes_stale_active_session(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await init_backend(backend)
        await backend.put(
            Session(
                session_id="old",
                data={
                    "status": "active",
                    "user_id": "usr_a",
                    "messages": [],
                    "pending_email": {},
                    "summary": [],
                    "updated_at": "2026-05-30T00:00:00Z",
                },
            )
        )

        store = SessionStore(SQLiteSessionStore(db_path=db_path), user_id="usr_a")
        result = await store.start_or_restore("resume_if_active")

        old = await backend.get("old")
        assert result.restored is False
        assert store.session_id != "old"
        assert old is not None
        assert old.data["status"] == "pending_finalization"

    run(scenario())


def test_mark_finalized_removes_session_from_pending_sweep(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await init_backend(backend)
        await backend.put(
            Session(
                session_id="pending",
                data={
                    "status": "pending_finalization",
                    "user_id": "usr_a",
                    "messages": [],
                    "pending_email": {},
                    "summary": [],
                    "updated_at": old_iso(),
                },
            )
        )

        store = SessionStore(SQLiteSessionStore(db_path=db_path), user_id="usr_a")
        await store.init_backend()
        assert [sid for sid, _ in await store.list_pending_finalization_archives()] == ["pending"]

        await store.mark_finalized("pending")

        assert await store.list_pending_finalization_archives() == []

    run(scenario())


def test_pending_sweep_includes_legacy_but_excludes_other_tenants(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await init_backend(backend)
        await backend.put(
            Session(
                session_id="legacy",
                data={
                    "status": "pending_finalization",
                    "messages": [],
                    "pending_email": {},
                    "summary": [],
                    "updated_at": old_iso(),
                },
            )
        )
        await backend.put(
            Session(
                session_id="other",
                data={
                    "status": "pending_finalization",
                    "user_id": "usr_b",
                    "messages": [],
                    "pending_email": {},
                    "summary": [],
                    "updated_at": old_iso(),
                },
            )
        )

        store = SessionStore(SQLiteSessionStore(db_path=db_path), user_id="usr_a")
        await store.init_backend()

        pending_ids = [sid for sid, _ in await store.list_pending_finalization_archives()]
        assert pending_ids == ["legacy"]

    run(scenario())


def test_pending_email_ttl_clears_stale_draft(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        store = SessionStore(SQLiteSessionStore(db_path=db_path), user_id="usr_a")
        await store.start_or_restore("strict_new")
        await store.set_pending_email(recipients=["a@b.com"])

        store._pending_email_updated_at = old_iso(hours=2)

        assert store.get_pending_email() == store._default_pending_email()

    run(scenario())
