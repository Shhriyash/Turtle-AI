"""
Phase 3 (W2) — identity stability across resets.

Pins the self-healing rebind contract in core.identity:

  * A users.sqlite reset that leaves data/memory/personal/<id>/ intact must
    rebind a verified email to its ORIGINAL user_id (no fresh mint, no orphaned
    memory) — the "3 user_ids burned" bug.
  * Cloud mode requires a VERIFIED marker to rebind; dev mode accepts unverified.
  * A /forget-me purge removes the marker with the dir, so a purged email mints a
    FRESH id (no silent resurrection).
  * resolve_user populates users.primary_email for web_email.
  * Channel isolation still holds (same handle on another channel != same user).

Tests redirect identity DB (per-instance db_path) + core.paths.PERSONAL_MEMORY_DIR
into a tmp tree, exactly like production_onboarding_test.py, so nothing touches
the real data/ tree (conftest tripwire).
"""
from __future__ import annotations

import asyncio
import shutil
import unittest
import uuid
from pathlib import Path

try:
    import aiosqlite
    from core import paths as paths_mod
    from core.config import settings
    from core.identity import IdentityManager, normalize_email, write_account_marker
    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover — missing optional deps in test env
    _IMPORT_ERROR = _e


@unittest.skipIf(_IMPORT_ERROR is not None, f"identity deps missing: {_IMPORT_ERROR}")
class IdentityResetStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path("test") / "_tmp" / f"phase3_identity_{uuid.uuid4().hex}"
        self.tmp.mkdir(parents=True, exist_ok=True)
        # Redirect the personal-memory tree that write_account_marker() and the
        # rebind scan read/write, so markers land in tmp, not data/.
        self._orig_personal = paths_mod.PERSONAL_MEMORY_DIR
        paths_mod.PERSONAL_MEMORY_DIR = self.tmp / "personal"
        paths_mod.PERSONAL_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        # Each IdentityManager gets an explicit db_path under tmp.
        self.db_path = self.tmp / "users.sqlite"
        # is_cloud is derived from deploy_mode; snapshot so cloud/dev toggling
        # in a test can't leak into the next.
        self._orig_deploy = settings.deploy_mode
        settings.deploy_mode = "local"

    def tearDown(self) -> None:
        paths_mod.PERSONAL_MEMORY_DIR = self._orig_personal
        settings.deploy_mode = self._orig_deploy
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- helpers -----------------------------------------------------------

    async def _fresh_manager(self) -> IdentityManager:
        mgr = IdentityManager(db_path=self.db_path)
        await mgr.init_db()
        return mgr

    async def _primary_email(self, user_id: str) -> str | None:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT primary_email FROM users WHERE user_id = ?", (user_id,)
            ) as cur:
                row = await cur.fetchone()
        return row[0] if row else None

    async def _mapped_user(self, email: str) -> str | None:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT user_id FROM channel_mappings WHERE channel = ? AND channel_user_id = ?",
                ("web_email", email),
            ) as cur:
                row = await cur.fetchone()
        return row[0] if row else None

    # -- (a) reset survival ------------------------------------------------

    def test_reset_survival_rebinds_same_user_id(self) -> None:
        async def scenario() -> None:
            mgr = await self._fresh_manager()
            uid1 = await mgr.resolve_user("web_email", "alice@example.com")
            write_account_marker(uid1, "alice@example.com", verified=True)

            # Simulate a manual reset: blow away users.sqlite entirely while the
            # memory dir (with account.json) survives.
            self.db_path.unlink()
            self.assertFalse(self.db_path.exists())

            mgr2 = await self._fresh_manager()
            uid2 = await mgr2.resolve_user("web_email", "alice@example.com")

            self.assertEqual(uid2, uid1, "reset must rebind to the original id, not mint")
            # DB was repopulated by the rebind.
            self.assertEqual(await self._primary_email(uid1), "alice@example.com")
            self.assertEqual(await self._mapped_user("alice@example.com"), uid1)

        asyncio.run(scenario())

    # -- (b) verified-marker gating ---------------------------------------

    def test_unverified_marker_rejected_in_cloud(self) -> None:
        async def scenario() -> None:
            mgr = await self._fresh_manager()
            uid1 = await mgr.resolve_user("web_email", "bob@example.com")
            write_account_marker(uid1, "bob@example.com", verified=False)

            self.db_path.unlink()
            settings.deploy_mode = "cloud"  # require proof of ownership

            mgr2 = await self._fresh_manager()
            uid2 = await mgr2.resolve_user("web_email", "bob@example.com")

            self.assertNotEqual(
                uid2, uid1, "cloud must NOT rebind from an unverified marker"
            )

        asyncio.run(scenario())

    def test_unverified_marker_accepted_in_dev(self) -> None:
        async def scenario() -> None:
            mgr = await self._fresh_manager()
            uid1 = await mgr.resolve_user("web_email", "bob@example.com")
            write_account_marker(uid1, "bob@example.com", verified=False)

            self.db_path.unlink()
            settings.deploy_mode = "local"  # dev accepts unverified

            mgr2 = await self._fresh_manager()
            uid2 = await mgr2.resolve_user("web_email", "bob@example.com")

            self.assertEqual(uid2, uid1, "dev must rebind even from an unverified marker")

        asyncio.run(scenario())

    # -- (c) purge prevents resurrection ----------------------------------

    def test_purge_prevents_resurrection(self) -> None:
        async def scenario() -> None:
            mgr = await self._fresh_manager()
            uid1 = await mgr.resolve_user("web_email", "carol@example.com")
            write_account_marker(uid1, "carol@example.com", verified=True)

            # Purge = the memory dir (and its account.json) is gone AND the id
            # rows are gone. The marker must NOT outlive the dir.
            shutil.rmtree(paths_mod.PERSONAL_MEMORY_DIR / uid1, ignore_errors=True)
            self.db_path.unlink()

            mgr2 = await self._fresh_manager()
            uid2 = await mgr2.resolve_user("web_email", "carol@example.com")

            self.assertNotEqual(
                uid2, uid1, "a purged user must not be resurrected via marker rebind"
            )

        asyncio.run(scenario())

    # -- (d) primary_email populated --------------------------------------

    def test_primary_email_populated_on_resolve(self) -> None:
        async def scenario() -> None:
            mgr = await self._fresh_manager()
            uid = await mgr.resolve_user("web_email", "dave@example.com")
            self.assertEqual(await self._primary_email(uid), "dave@example.com")

        asyncio.run(scenario())

    # -- (e) channel isolation --------------------------------------------

    def test_channel_isolation_still_holds(self) -> None:
        async def scenario() -> None:
            mgr = await self._fresh_manager()
            uid_email = await mgr.resolve_user("web_email", "erin@example.com")
            uid_wa = await mgr.resolve_user("whatsapp", "erin@example.com")
            self.assertNotEqual(
                uid_email, uid_wa, "same handle on a different channel must be a different user"
            )

        asyncio.run(scenario())

    # -- bonus: normalization dedup ---------------------------------------

    def test_email_casing_and_whitespace_dedup(self) -> None:
        async def scenario() -> None:
            mgr = await self._fresh_manager()
            uid1 = await mgr.resolve_user("web_email", "  Frank@Example.com ")
            uid2 = await mgr.resolve_user("web_email", "frank@example.com")
            self.assertEqual(uid1, uid2, "casing/whitespace must not mint a duplicate id")
            self.assertEqual(normalize_email("  Frank@Example.com "), "frank@example.com")

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
