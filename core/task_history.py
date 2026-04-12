from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
    """Append-only operational task history, separate from personalization memory."""

    def __init__(self, history_path: Path) -> None:
        self.history_path = history_path
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_path.exists():
            self.history_path.touch()

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
        with self.history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_record_to_payload(record), ensure_ascii=False) + "\n")
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
        return [record for record in self.load_records() if str(record.get("session_id", "")).strip() == target]

    def search(self, query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
        terms = _tokenize(query)
        if not terms:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for record in self.load_records():
            haystack = " ".join(
                [
                    str(record.get("task_type", "")),
                    str(record.get("status", "")),
                    str(record.get("query", "")),
                    str(record.get("tool_used", "")),
                    str(record.get("outcome", "")),
                    json.dumps(record.get("payload", {}), ensure_ascii=False) if record.get("payload") else "",
                ]
            ).lower()
            haystack_terms = set(_tokenize(haystack))
            score = sum(1 for term in terms if term in haystack_terms)
            if score <= 0:
                continue
            scored.append((score, record))

        scored.sort(
            key=lambda item: (
                item[0],
                str(item[1].get("timestamp", "")),
            ),
            reverse=True,
        )
        return [record for _, record in scored[:max_results]]

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


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9@._-]{3,}", str(text).lower())]
