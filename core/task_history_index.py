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
        # MULTI-TENANCY: user_id scopes every row. Without it, search() was an
        # unfiltered FTS MATCH over one global history.sqlite shared by every
        # user, so tier 4 of the retrieval broker could splice one user's task
        # text into another user's memory context. It is UNINDEXED (we filter on
        # it, never full-text search it).
        self._connection.executescript(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE}
            USING fts5(
                user_id UNINDEXED,
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
        self._migrate_add_user_id()

    def _migrate_add_user_id(self) -> None:
        """Rebuild a pre-tenancy index that has no user_id column.

        FTS5 cannot ALTER TABLE ADD COLUMN, and this index is a rebuildable
        cache (the JSONL is the source of truth), so the safe migration is to
        drop it and let the store's staleness check refill it. Rows written
        before tenancy existed carry no owner and MUST NOT be served to anyone.
        """
        try:
            cols = {
                row[1]
                for row in self._connection.execute(f"PRAGMA table_info({_FTS_TABLE})")
            }
        except sqlite3.OperationalError:
            return
        if not cols or "user_id" in cols:
            return
        print("LOG: task history index missing user_id — rebuilding for tenancy")
        self._connection.executescript(f"DROP TABLE IF EXISTS {_FTS_TABLE};")
        self._connection.commit()
        self._ensure_schema()

    def row_count(self) -> int:
        cursor = self._connection.execute(f"SELECT COUNT(*) AS n FROM {_FTS_TABLE}")
        row = cursor.fetchone()
        return int(row["n"]) if row else 0

    def insert_record(self, record: dict[str, Any]) -> None:
        self._connection.execute(
            f"""
            INSERT INTO {_FTS_TABLE}
            (user_id, session_id, turn_id, task_type, status, timestamp, query, tool_used, outcome, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                (user_id, session_id, turn_id, task_type, status, timestamp, query, tool_used, outcome, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        self._connection.commit()
        return len(rows)

    def search(
        self, query: str, *, max_results: int = 5, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Full-text search, ALWAYS scoped to one tenant.

        user_id is required in practice: this index is one global file shared by
        every user, so an unscoped MATCH leaks one user's task text into
        another's retrieval context. Passing user_id=None returns nothing rather
        than silently searching everyone (fail closed, not open).
        """
        owner = str(user_id or "").strip()
        if not owner:
            return []
        fts_query = _compile_fts_query(query)
        if not fts_query:
            return []

        try:
            cursor = self._connection.execute(
                f"""
                SELECT user_id, session_id, turn_id, task_type, status, timestamp,
                       query, tool_used, outcome, payload_json,
                       bm25({_FTS_TABLE}) AS score
                FROM {_FTS_TABLE}
                WHERE {_FTS_TABLE} MATCH ? AND user_id = ?
                ORDER BY score ASC, timestamp DESC
                LIMIT ?
                """,
                (fts_query, owner, max_results),
            )
        except sqlite3.OperationalError:
            return []

        return [_row_to_record(row) for row in cursor.fetchall()]

    def list_by_session(
        self, session_id: str, *, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        owner = str(user_id or "").strip()
        sql = f"""
            SELECT user_id, session_id, turn_id, task_type, status, timestamp,
                   query, tool_used, outcome, payload_json
            FROM {_FTS_TABLE}
            WHERE session_id = ?
        """
        params: list[Any] = [str(session_id).strip()]
        if owner:
            sql += " AND user_id = ?"
            params.append(owner)
        sql += " ORDER BY timestamp ASC"
        cursor = self._connection.execute(sql, tuple(params))
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
        str(record.get("user_id", "")),
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
        "user_id": row["user_id"] if "user_id" in row.keys() else "",
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
