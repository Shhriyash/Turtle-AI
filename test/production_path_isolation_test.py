"""
Phase 2b + Phase 3 — multi-tenant path isolation.

These tests pin down the single most load-bearing change in the migration:
``personal_memory_dir(user_id)`` and ``rag_vector_dir(user_id)`` must nest
under the parent dir, never collapse to the same path for different users.
A regression here is the entire multi-tenancy bug.
"""
from __future__ import annotations

import unittest

from core.paths import (
    PERSONAL_MEMORY_DIR,
    RAG_DATA_DIR,
    personal_journal_dir,
    personal_memory_dir,
    personal_memory_file,
    rag_vector_dir,
)


class PathIsolationTests(unittest.TestCase):
    def test_personal_memory_dir_nests_per_user(self) -> None:
        a = personal_memory_dir("usr_alice")
        b = personal_memory_dir("usr_bob")
        self.assertNotEqual(a, b)
        self.assertEqual(a.parent, PERSONAL_MEMORY_DIR)
        self.assertEqual(b.parent, PERSONAL_MEMORY_DIR)
        self.assertEqual(a.name, "usr_alice")

    def test_personal_memory_dir_rejects_empty_user(self) -> None:
        with self.assertRaises(ValueError):
            personal_memory_dir("")

    def test_personal_memory_file_lives_inside_user_dir(self) -> None:
        path = personal_memory_file("usr_alice", "identity.md")
        self.assertEqual(path.parent, personal_memory_dir("usr_alice"))
        self.assertEqual(path.name, "identity.md")

    def test_personal_journal_dir_per_user(self) -> None:
        a = personal_journal_dir("usr_alice")
        b = personal_journal_dir("usr_bob")
        self.assertNotEqual(a, b)
        self.assertTrue(a.is_relative_to(personal_memory_dir("usr_alice")))
        self.assertTrue(b.is_relative_to(personal_memory_dir("usr_bob")))

    def test_rag_vector_dir_per_user(self) -> None:
        a = rag_vector_dir("usr_alice")
        b = rag_vector_dir("usr_bob")
        self.assertNotEqual(a, b)
        # Both must live under data/rag/<user_id>/vector/
        self.assertEqual(a.parent.parent, RAG_DATA_DIR)
        self.assertEqual(b.parent.parent, RAG_DATA_DIR)


if __name__ == "__main__":
    unittest.main()
