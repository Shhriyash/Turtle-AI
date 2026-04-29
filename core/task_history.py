from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.task_history_index import TaskHistoryIndex


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class TaskHistoryRecord:
    session_id: str
    turn_id: str
    task_type: str
    status: str
    timestamp: str
    query: str = ""
    tool_used: str = ""
    outcome: str = ""
    payload: dict[str, Any] | None = None


class TaskHistoryStore:
    """Append-only operational task history, separate from personalization memory.

    JSONL on disk is the source of truth. A co-located SQLite FTS5 index
    (``history.sqlite``) is a rebuildable cache used for search and
    per-session lookups. If the index is missing, empty, or falls behind
    the JSONL row count, it is rebuilt from the JSONL on init.
    """

    def __init__(self, history_path: Path) -> None:
        self.history_path = history_path
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_path.exists():
            self.history_path.touch()

        index_path = self.history_path.with_suffix(".sqlite")
        self._index = TaskHistoryIndex(index_path)
        self._sync_index_if_stale()

    def record(
        self,
        *,
        session_id: str,
        turn_id: str,
        task_type: str,
        status: str,
        query: str = "",
        tool_used: str = "",
        outcome: str = "",
        payload: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> TaskHistoryRecord:
        record = TaskHistoryRecord(
            session_id=session_id,
            turn_id=turn_id,
            task_type=task_type,
            status=status,
            timestamp=timestamp or _utc_now(),
            query=query,
            tool_used=tool_used,
            outcome=outcome,
            payload=dict(payload) if payload else None,
        )
        payload_record = _record_to_payload(record)
        with self.history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload_record, ensure_ascii=False) + "\n")
        self._index.insert_record(payload_record)
        return record

    def load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not self.history_path.exists():
            return records
        with self.history_path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    def list_by_session(self, session_id: str) -> list[dict[str, Any]]:
        target = str(session_id).strip()
        return self._index.list_by_session(target)

    def search(self, query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
        return self._index.search(query, max_results=max_results)

    def format_search_results(self, query: str, *, max_results: int = 5) -> str:
        results = self.search(query, max_results=max_results)
        if not results:
            return ""

        lines = ["Task history matches:"]
        for record in results:
            timestamp = str(record.get("timestamp", "")).strip()
            status = str(record.get("status", "")).strip() or "unknown"
            task_type = str(record.get("task_type", "")).strip() or "task"
            tool_used = str(record.get("tool_used", "")).strip()
            outcome = str(record.get("outcome", "")).strip()
            query_text = str(record.get("query", "")).strip()

            summary = f"- [{timestamp}] {task_type} ({status})"
            if tool_used:
                summary += f" via {tool_used}"
            if outcome:
                summary += f": {outcome}"
            elif query_text:
                summary += f": {query_text}"
            lines.append(summary)

        return "\n".join(lines)

    def rebuild_index(self) -> int:
        return self._index.rebuild(self.load_records())

    def _sync_index_if_stale(self) -> None:
        jsonl_count = self._count_jsonl_records()
        index_count = self._index.row_count()
        if index_count != jsonl_count:
            self.rebuild_index()

    def _count_jsonl_records(self) -> int:
        if not self.history_path.exists():
            return 0
        count = 0
        with self.history_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    count += 1
        return count


def _record_to_payload(record: TaskHistoryRecord) -> dict[str, Any]:
    payload = {
        "session_id": record.session_id,
        "turn_id": record.turn_id,
        "task_type": record.task_type,
        "status": record.status,
        "timestamp": record.timestamp,
        "query": record.query,
        "tool_used": record.tool_used,
        "outcome": record.outcome,
    }
    if record.payload:
        payload["payload"] = record.payload
    return payload
