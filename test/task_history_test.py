import shutil
import unittest
import uuid
from pathlib import Path

from core.task_history import TaskHistoryStore


class TaskHistoryStoreTests(unittest.TestCase):
    def test_record_and_filter_by_session(self) -> None:
        base = Path("test") / "_tmp" / f"task_history_{uuid.uuid4().hex}"
        history_path = base / "history.jsonl"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = TaskHistoryStore(history_path)
            store.record(
                session_id="s1",
                turn_id="t1",
                task_type="email",
                status="completed",
                query="send an email",
                tool_used="send_email_now",
                outcome="sent",
                payload={"recipients": ["team@example.com"]},
            )
            store.record(
                session_id="s2",
                turn_id="t2",
                task_type="web",
                status="completed",
                query="search news",
                tool_used="search_duckduckgo",
                outcome="ok",
            )

            records = store.load_records()
            self.assertEqual(len(records), 2)
            session_records = store.list_by_session("s1")
            self.assertEqual(len(session_records), 1)
            self.assertEqual(session_records[0]["task_type"], "email")
            self.assertEqual(session_records[0]["payload"]["recipients"], ["team@example.com"])
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_search_and_format_results(self) -> None:
        base = Path("test") / "_tmp" / f"task_history_{uuid.uuid4().hex}"
        history_path = base / "history.jsonl"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = TaskHistoryStore(history_path)
            store.record(
                session_id="s1",
                turn_id="t1",
                task_type="email",
                status="completed",
                query="send an email to the team",
                tool_used="send_email_now",
                outcome="Sent email to team@example.com",
            )
            store.record(
                session_id="s2",
                turn_id="t2",
                task_type="web",
                status="completed",
                query="search latest ai news",
                tool_used="search_duckduckgo",
                outcome="Returned search results",
            )

            results = store.search("what email task did you complete", max_results=5)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["task_type"], "email")

            formatted = store.format_search_results("email task")
            self.assertIn("Task history matches:", formatted)
            self.assertIn("send_email_now", formatted)
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
