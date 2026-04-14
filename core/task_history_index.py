from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable


_FTS_TABLE = "task_history_fts"
_META_TABLE = "task_history_meta"


class TaskHistoryIndex:
    """SQLite FTS5 index over task history records.

    This is a rebuildable cache. The JSONL on disk is the source of truth.
    If the sqlite file is deleted or falls behind, call rebuild().
    """

    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.sqlite_path))
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        try:
            self._connection.close()
        except Exception:
            pass

    def _ensure_schema(self) -> None:
        self._connection.executescript(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE}
            USING fts5(
                session_id UNINDEXED,
                turn_id UNINDEXED,
                task_type,
                status UNINDEXED,
                timestamp UNINDEXED,
                query,
                tool_used,
                outcome,
                payload_json,
                tokenize = 'unicode61 remove_diacritics 2'
            );

            CREATE TABLE IF NOT EXISTS {_META_TABLE} (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self._connection.commit()

    def row_count(self) -> int:
        cursor = self._connection.execute(f"SELECT COUNT(*) AS n FROM {_FTS_TABLE}")
        row = cursor.fetchone()
        return int(row["n"]) if row else 0

    def insert_record(self, record: dict[str, Any]) -> None:
        self._connection.execute(
            f"""
            INSERT INTO {_FTS_TABLE}
            (session_id, turn_id, task_type, status, timestamp, query, tool_used, outcome, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _record_columns(record),
        )
        self._connection.commit()

    def rebuild(self, records: Iterable[dict[str, Any]]) -> int:
        self._connection.execute(f"DELETE FROM {_FTS_TABLE}")
        rows = [_record_columns(record) for record in records]
        if rows:
            self._connection.executemany(
                f"""
                INSERT INTO {_FTS_TABLE}
                (session_id, turn_id, task_type, status, timestamp, query, tool_used, outcome, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        self._connection.commit()
        return len(rows)

    def search(self, query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
        fts_query = _compile_fts_query(query)
        if not fts_query:
            return []

        try:
            cursor = self._connection.execute(
                f"""
                SELECT session_id, turn_id, task_type, status, timestamp,
                       query, tool_used, outcome, payload_json,
                       bm25({_FTS_TABLE}) AS score
                FROM {_FTS_TABLE}
                WHERE {_FTS_TABLE} MATCH ?
                ORDER BY score ASC, timestamp DESC
                LIMIT ?
                """,
                (fts_query, max_results),
            )
        except sqlite3.OperationalError:
            return []

        return [_row_to_record(row) for row in cursor.fetchall()]

    def list_by_session(self, session_id: str) -> list[dict[str, Any]]:
        cursor = self._connection.execute(
            f"""
            SELECT session_id, turn_id, task_type, status, timestamp,
                   query, tool_used, outcome, payload_json
            FROM {_FTS_TABLE}
            WHERE session_id = ?
            ORDER BY timestamp ASC
            """,
            (str(session_id).strip(),),
        )
        return [_row_to_record(row) for row in cursor.fetchall()]


def _record_columns(record: dict[str, Any]) -> tuple[str, ...]:
    payload = record.get("payload")
    if payload:
        try:
            payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except Exception:
            payload_json = ""
    else:
        payload_json = ""

    return (
        str(record.get("session_id", "")),
        str(record.get("turn_id", "")),
        str(record.get("task_type", "")),
        str(record.get("status", "")),
        str(record.get("timestamp", "")),
        str(record.get("query", "")),
        str(record.get("tool_used", "")),
        str(record.get("outcome", "")),
        payload_json,
    )


def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    record: dict[str, Any] = {
        "session_id": row["session_id"],
        "turn_id": row["turn_id"],
        "task_type": row["task_type"],
        "status": row["status"],
        "timestamp": row["timestamp"],
        "query": row["query"],
        "tool_used": row["tool_used"],
        "outcome": row["outcome"],
    }
    payload_json = row["payload_json"]
    if payload_json:
        try:
            record["payload"] = json.loads(payload_json)
        except Exception:
            pass
    return record


_TERM_RE = re.compile(r"[A-Za-z0-9@._-]{2,}")
_FTS_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "was", "were", "to", "of", "and", "or", "for",
        "on", "in", "at", "by", "with", "from", "this", "that", "it", "its",
        "did", "do", "does", "you", "your", "we", "our", "i", "me", "my",
        "what", "which", "when", "who", "whom", "how", "why", "be", "been",
    }
)


def _compile_fts_query(query: str) -> str:
    tokens = [
        token.lower() for token in _TERM_RE.findall(query or "")
        if token.lower() not in _FTS_STOPWORDS
    ]
    unique_tokens: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        unique_tokens.append(token)
    if not unique_tokens:
        return ""
    quoted = [f'"{token}"' for token in unique_tokens]
    return " OR ".join(quoted)
