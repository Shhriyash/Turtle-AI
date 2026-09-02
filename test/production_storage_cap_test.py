"""
Phase 6 — per-user storage cap.

Verifies the guardrail at both layers:
  * the helper raises StorageCapExceededError when usage exceeds the cap
  * PersonalMemoryStore.write_topic refuses to write past the cap
  * the cap is a no-op when set to 0 (escape hatch for tests / dev)
"""
from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from core.config import settings
from core.guardrails import StorageCapExceededError, enforce_storage_cap
from core.personal_memory_store import PersonalMemoryStore


class StorageCapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"storage_cap_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self._orig_cap = settings.user_storage_cap_mb

    def tearDown(self) -> None:
        settings.user_storage_cap_mb = self._orig_cap
        shutil.rmtree(self.base, ignore_errors=True)

    def test_enforce_passes_when_under_cap(self) -> None:
        settings.user_storage_cap_mb = 50
        # Empty dir, tiny incoming write → must not raise.
        enforce_storage_cap("usr_alice", self.base, incoming_bytes=1024)

    def test_enforce_raises_when_incoming_exceeds_cap(self) -> None:
        settings.user_storage_cap_mb = 1  # 1 MiB cap
        with self.assertRaises(StorageCapExceededError) as ctx:
            enforce_storage_cap("usr_alice", self.base, incoming_bytes=2 * 1024 * 1024)
        # Error carries enough info for the UI to render a useful message.
        self.assertEqual(ctx.exception.user_id, "usr_alice")
        self.assertGreater(ctx.exception.used_bytes, ctx.exception.cap_bytes)

    def test_enforce_counts_existing_files(self) -> None:
        settings.user_storage_cap_mb = 1
        # Pre-seed the directory with ~900 KiB of junk.
        (self.base / "blob.bin").write_bytes(b"x" * 900 * 1024)
        # A 200 KiB incoming write pushes total over 1 MiB.
        with self.assertRaises(StorageCapExceededError):
            enforce_storage_cap("usr_alice", self.base, incoming_bytes=200 * 1024)

    def test_cap_zero_disables_enforcement(self) -> None:
        settings.user_storage_cap_mb = 0
        # No matter how large the incoming write, must not raise.
        enforce_storage_cap("usr_alice", self.base, incoming_bytes=10**9)

    def test_personal_memory_store_blocks_oversize_topic_write(self) -> None:
        settings.user_storage_cap_mb = 1
        store = PersonalMemoryStore(
            user_id="usr_alice",
            base_dir=self.base,
            index_path=self.base / "MEMORY.md",
            logs_dir=self.base / "logs",
            topic_paths={"identity": self.base / "identity.md"},
        )
        # Single-line, but the line is huge.
        oversized_line = "- " + ("a" * (2 * 1024 * 1024))
        with self.assertRaises(StorageCapExceededError):
            store.write_topic("identity", [oversized_line])
        # The topic file must NOT be persisted on failure.
        self.assertFalse((self.base / "identity.md").exists())

    def test_personal_memory_store_allows_normal_write(self) -> None:
        settings.user_storage_cap_mb = 1
        store = PersonalMemoryStore(
            user_id="usr_alice",
            base_dir=self.base,
            index_path=self.base / "MEMORY.md",
            logs_dir=self.base / "logs",
            topic_paths={"identity": self.base / "identity.md"},
        )
        store.write_topic("identity", ["- Name: Alice"])
        self.assertTrue((self.base / "identity.md").exists())


if __name__ == "__main__":
    unittest.main()
