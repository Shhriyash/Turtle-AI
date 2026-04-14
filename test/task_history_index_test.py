import json
import shutil
import unittest
import uuid
from pathlib import Path

from core.task_history import TaskHistoryStore
from core.task_history_index import TaskHistoryIndex


class TaskHistoryIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"task_history_index_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.history_path = self.base / "history.jsonl"

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def _seed_jsonl(self, records: list[dict]) -> None:
        with self.history_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_rebuild_from_jsonl_when_sqlite_missing(self) -> None:
        self._seed_jsonl(
            [
                {
                    "session_id": "s1",
                    "turn_id": "t1",
                    "task_type": "email",
                    "status": "completed",
                    "timestamp": "2026-04-10T10:00:00Z",
                    "query": "send quarterly report",
                    "tool_used": "send_email_now",
                    "outcome": "sent to team",
                },
                {
                    "session_id": "s2",
                    "turn_id": "t2",
                    "task_type": "web",
                    "status": "completed",
                    "timestamp": "2026-04-11T10:00:00Z",
                    "query": "latest python release",
                    "tool_used": "search_duckduckgo",
                    "outcome": "ok",
                },
            ]
        )
        store = TaskHistoryStore(self.history_path)
        sqlite_path = self.history_path.with_suffix(".sqlite")
        self.assertTrue(sqlite_path.exists())
        self.assertEqual(len(store.search("quarterly report")), 1)
        self.assertEqual(store.search("quarterly report")[0]["task_type"], "email")

    def test_search_ranking_prefers_stronger_match(self) -> None:
        store = TaskHistoryStore(self.history_path)
        store.record(
            session_id="s1",
            turn_id="t1",
            task_type="email",
            status="completed",
            query="draft email to finance team about invoice",
            tool_used="send_email_now",
            outcome="sent invoice email",
        )
        store.record(
            session_id="s2",
            turn_id="t2",
            task_type="email",
            status="completed",
            query="birthday note",
            tool_used="send_email_now",
            outcome="sent",
        )
        results = store.search("invoice email", max_results=5)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["turn_id"], "t1")

    def test_list_by_session_returns_only_session_records(self) -> None:
        store = TaskHistoryStore(self.history_path)
        store.record(
            session_id="alpha",
            turn_id="t1",
            task_type="email",
            status="completed",
            query="q1",
            timestamp="2026-04-10T10:00:00Z",
        )
        store.record(
            session_id="beta",
            turn_id="t2",
            task_type="web",
            status="completed",
            query="q2",
            timestamp="2026-04-10T10:01:00Z",
        )
        store.record(
            session_id="alpha",
            turn_id="t3",
            task_type="email",
            status="completed",
            query="q3",
            timestamp="2026-04-10T10:02:00Z",
        )
        alpha = store.list_by_session("alpha")
        self.assertEqual([r["turn_id"] for r in alpha], ["t1", "t3"])

    def test_rebuild_is_idempotent(self) -> None:
        store = TaskHistoryStore(self.history_path)
        store.record(
            session_id="s1",
            turn_id="t1",
            task_type="email",
            status="completed",
            query="quarterly revenue summary",
        )
        store.rebuild_index()
        store.rebuild_index()
        self.assertEqual(len(store.search("quarterly revenue")), 1)

    def test_sqlite_stays_in_sync_with_jsonl_on_init(self) -> None:
        store = TaskHistoryStore(self.history_path)
        store.record(
            session_id="s1",
            turn_id="t1",
            task_type="email",
            status="completed",
            query="first task",
        )
        with self.history_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {
                        "session_id": "s2",
                        "turn_id": "t2",
                        "task_type": "web",
                        "status": "completed",
                        "timestamp": "2026-04-10T12:00:00Z",
                        "query": "out-of-band second task",
                        "tool_used": "",
                        "outcome": "",
                    }
                )
                + "\n"
            )

        reopened = TaskHistoryStore(self.history_path)
        second_hits = reopened.search("out-of-band")
        self.assertEqual(len(second_hits), 1)
        self.assertEqual(second_hits[0]["turn_id"], "t2")

    def test_payload_survives_index_roundtrip(self) -> None:
        store = TaskHistoryStore(self.history_path)
        store.record(
            session_id="s1",
            turn_id="t1",
            task_type="email",
            status="completed",
            query="send",
            payload={"recipients": ["a@example.com"], "retries": 0},
        )
        results = store.search("send")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["payload"]["recipients"], ["a@example.com"])
        self.assertEqual(results[0]["payload"]["retries"], 0)

    def test_index_direct_search_matches_store(self) -> None:
        store = TaskHistoryStore(self.history_path)
        store.record(
            session_id="s1",
            turn_id="t1",
            task_type="email",
            status="completed",
            query="send meeting recap",
            tool_used="send_email_now",
        )
        index = TaskHistoryIndex(self.history_path.with_suffix(".sqlite"))
        self.assertEqual(len(index.search("meeting recap")), 1)
        index.close()


if __name__ == "__main__":
    unittest.main()
