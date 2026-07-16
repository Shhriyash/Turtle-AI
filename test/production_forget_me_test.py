"""
Phase 7 — /forget-me purge.

The endpoint is the GDPR backstop. The test pins the deletion semantics:
when ``_purge_user`` runs, the user's memory dir, RAG dir, and SQLite rows
must all be gone.
"""
from __future__ import annotations

import asyncio
import shutil
import unittest
import uuid
from pathlib import Path

try:
    import aiosqlite  # noqa: F401
    from apps import admin_routes
    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover — missing optional deps in test env
    _IMPORT_ERROR = _e
    admin_routes = None  # type: ignore[assignment]

from core import paths as paths_mod
from core.identity import identity_manager


@unittest.skipIf(_IMPORT_ERROR is not None, f"admin_routes deps missing: {_IMPORT_ERROR}")
class ForgetMePurgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path("test") / "_tmp" / f"forget_me_{uuid.uuid4().hex}"
        self.tmp.mkdir(parents=True, exist_ok=True)
        # Redirect every on-disk location the purge touches.
        self._orig_personal = paths_mod.PERSONAL_MEMORY_DIR
        self._orig_rag = paths_mod.RAG_DATA_DIR
        self._orig_admin_personal = admin_routes.PERSONAL_MEMORY_DIR
        self._orig_admin_rag = admin_routes.RAG_DATA_DIR
        self._orig_db = identity_manager.db_path

        paths_mod.PERSONAL_MEMORY_DIR = self.tmp / "personal"
        paths_mod.RAG_DATA_DIR = self.tmp / "rag"
        admin_routes.PERSONAL_MEMORY_DIR = paths_mod.PERSONAL_MEMORY_DIR
        admin_routes.RAG_DATA_DIR = paths_mod.RAG_DATA_DIR
        identity_manager.db_path = self.tmp / "users.sqlite"

    def tearDown(self) -> None:
        paths_mod.PERSONAL_MEMORY_DIR = self._orig_personal
        paths_mod.RAG_DATA_DIR = self._orig_rag
        admin_routes.PERSONAL_MEMORY_DIR = self._orig_admin_personal
        admin_routes.RAG_DATA_DIR = self._orig_admin_rag
        identity_manager.db_path = self._orig_db
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def _seed_user(self, user_id: str) -> None:
        await identity_manager.init_db()
        async with aiosqlite.connect(identity_manager.db_path) as db:
            await db.execute("INSERT INTO users (user_id) VALUES (?)", (user_id,))
            await db.execute(
                "INSERT INTO channel_mappings (channel, channel_user_id, user_id) VALUES (?, ?, ?)",
                ("web_email", f"{user_id}@example.com", user_id),
            )
            await db.commit()
        memory = paths_mod.PERSONAL_MEMORY_DIR / user_id
        memory.mkdir(parents=True, exist_ok=True)
        (memory / "identity.md").write_text("- Name: " + user_id, encoding="utf-8")
        rag = paths_mod.RAG_DATA_DIR / user_id / "vector"
        rag.mkdir(parents=True, exist_ok=True)
        (rag / "faiss_index.bin").write_bytes(b"\x00\x01")

    async def _row_count(self, user_id: str) -> int:
        async with aiosqlite.connect(identity_manager.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM users WHERE user_id = ?", (user_id,)
            ) as cursor:
                u = (await cursor.fetchone())[0]
            async with db.execute(
                "SELECT COUNT(*) FROM channel_mappings WHERE user_id = ?", (user_id,)
            ) as cursor:
                c = (await cursor.fetchone())[0]
        return u + c

    def test_purge_user_removes_memory_rag_and_rows(self) -> None:
        async def scenario() -> None:
            await self._seed_user("usr_alice")
            await self._seed_user("usr_bob")  # control: must survive
            self.assertEqual(await self._row_count("usr_alice"), 2)

            removed = await admin_routes._purge_user("usr_alice")

            self.assertTrue(removed["memory"])
            self.assertTrue(removed["rag"])
            self.assertEqual(removed["rows"], 2)
            self.assertFalse((paths_mod.PERSONAL_MEMORY_DIR / "usr_alice").exists())
            self.assertFalse((paths_mod.RAG_DATA_DIR / "usr_alice").exists())
            self.assertEqual(await self._row_count("usr_alice"), 0)

            # Bob is untouched.
            self.assertTrue((paths_mod.PERSONAL_MEMORY_DIR / "usr_bob").exists())
            self.assertEqual(await self._row_count("usr_bob"), 2)

        asyncio.run(scenario())

    def test_purge_user_is_idempotent(self) -> None:
        async def scenario() -> None:
            await self._seed_user("usr_alice")
            await admin_routes._purge_user("usr_alice")
            # A second purge of the same user must not raise.
            removed = await admin_routes._purge_user("usr_alice")
            self.assertFalse(removed["memory"])
            self.assertFalse(removed["rag"])
            self.assertEqual(removed["rows"], 0)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
