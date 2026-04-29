import shutil
import unittest
import uuid
from pathlib import Path

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from core.session_store import SessionStore


class SessionStoreLayoutTests(unittest.TestCase):
    def test_default_layout_namespaces_active_session_by_session_id(self) -> None:
        base = Path("test") / "_tmp" / f"session_store_{uuid.uuid4().hex}"
        active_dir = base / "active"
        archive_dir = base / "archive"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = SessionStore(active_dir=active_dir, archive_dir=archive_dir)
            first = store.start_or_restore(mode="strict_new")

            self.assertFalse(first.restored)
            self.assertTrue(first.session_id)
            session_dir = active_dir / first.session_id
            self.assertTrue((session_dir / "session.json").exists())
            self.assertTrue((session_dir / "messages.json").exists())

            history = [
                ModelRequest(parts=[UserPromptPart(content="hello")]),
                ModelResponse(parts=[TextPart(content="world")]),
            ]
            store.replace_messages(history)

            restored_store = SessionStore(active_dir=active_dir, archive_dir=archive_dir)
            restored = restored_store.start_or_restore(mode="resume_if_active")
            self.assertTrue(restored.restored)
            self.assertEqual(restored.session_id, first.session_id)
            self.assertEqual(restored.message_count, len(history))

            second_store = SessionStore(active_dir=active_dir, archive_dir=archive_dir)
            second = second_store.start_or_restore(mode="strict_new")
            self.assertNotEqual(second.session_id, first.session_id)
            archived_manifest = archive_dir / first.session_id / "session.json"
            archived_messages = archive_dir / first.session_id / "messages.json"
            self.assertTrue(archived_manifest.exists())
            self.assertTrue(archived_messages.exists())

            archived_payload = archived_manifest.read_text(encoding="utf-8")
            self.assertIn('"status": "pending_finalization"', archived_payload)

            restored_messages = ModelMessagesTypeAdapter.validate_json(archived_messages.read_bytes())
            self.assertEqual(len(restored_messages), len(history))
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_append_only_updates_use_delta_log(self) -> None:
        base = Path("test") / "_tmp" / f"session_store_{uuid.uuid4().hex}"
        active_dir = base / "active"
        archive_dir = base / "archive"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = SessionStore(active_dir=active_dir, archive_dir=archive_dir)
            started = store.start_or_restore(mode="strict_new")
            session_dir = active_dir / started.session_id

            history1 = [
                ModelRequest(parts=[UserPromptPart(content="hello")]),
                ModelResponse(parts=[TextPart(content="world")]),
            ]
            store.replace_messages(history1)
            snapshot_path = session_dir / "messages.json"
            delta_path = session_dir / "messages.delta.jsonl"
            snapshot_before = snapshot_path.read_bytes()

            history2 = history1 + [
                ModelRequest(parts=[UserPromptPart(content="next")]),
                ModelResponse(parts=[TextPart(content="turn")]),
            ]
            store.replace_messages(history2)

            snapshot_after = snapshot_path.read_bytes()
            self.assertEqual(snapshot_before, snapshot_after)
            self.assertTrue(delta_path.exists())
            delta_lines = [line for line in delta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(delta_lines), 2)

            restored_store = SessionStore(active_dir=active_dir, archive_dir=archive_dir)
            restored = restored_store.start_or_restore(mode="resume_if_active")
            self.assertTrue(restored.restored)
            self.assertEqual(restored.message_count, len(history2))
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
