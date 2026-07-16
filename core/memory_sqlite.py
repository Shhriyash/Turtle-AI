"""SQLite FTS5 lexical index over the personal-memory journal.

A second derived projection of the append-only journal (alongside the markdown
topic files). The journal stays the source of truth; this index is rebuildable
at any time via ``backfill_from_journal``.

One SQLite file per user at ``data/memory/personal/{user_id}/memory.sqlite``.
Sync ``sqlite3`` (FTS5 inserts are microsecond-scale and the journal append
path is already synchronous). WAL mode + ``synchronous=NORMAL``.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.paths import personal_memory_dir

if TYPE_CHECKING:
    from core.memory_journal import JournalStore, MemoryEvent


# Bounds for value/evidence flattening so FTS storage stays reasonable.
_FLATTEN_MAX_DEPTH = 4
_FLATTEN_MAX_CHARS = 2048


class MemoryEventRow:
    """A row returned by ``MemorySQLiteIndex.search``.

    Carries the typed event fields plus the BM25 ``rank`` (more-negative is a
    better match, per SQLite convention).
    """

    __slots__ = (
        "event_id",
        "topic",
        "key",
        "value_text",
        "value",
        "evidence_text",
        "confidence",
        "source",
        "observed_at",
        "applied",
        "rank",
    )

    def __init__(
        self,
        *,
        event_id: str,
        topic: str,
        key: str,
        value_text: str,
        value: dict[str, Any],
        evidence_text: str,
        confidence: float,
        source: str,
        observed_at: str,
        applied: bool,
        rank: float,
    ) -> None:
        self.event_id = event_id
        self.topic = topic
        self.key = key
        self.value_text = value_text
        self.value = value
        self.evidence_text = evidence_text
        self.confidence = confidence
        self.source = source
        self.observed_at = observed_at
        self.applied = applied
        self.rank = rank


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    turn_id       TEXT NOT NULL,
    observed_at   TEXT NOT NULL,
    kind          TEXT NOT NULL,
    topic         TEXT NOT NULL,
    key           TEXT NOT NULL,
    value_json    TEXT NOT NULL,
    value_text    TEXT NOT NULL,
    confidence    REAL NOT NULL,
    source        TEXT NOT NULL,
    extractor     TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    evidence_text TEXT NOT NULL DEFAULT '',
    supersedes    TEXT,
    applied       INTEGER NOT NULL,
    rejected      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_topic_applied ON events(topic, applied);
CREATE INDEX IF NOT EXISTS idx_events_observed_at ON events(observed_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    topic, key, value_text, evidence_text,
    content='events', content_rowid='rowid',
    tokenize = "porter unicode61 remove_diacritics 2"
);

CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
    INSERT INTO events_fts(rowid, topic, key, value_text, evidence_text)
    VALUES (new.rowid, new.topic, new.key, new.value_text, new.evidence_text);
END;
"""


def _flatten_value(value: Any) -> str:
    """Recursively collect leaf strings/numbers from a value/evidence dict.

    Bounded in depth and total length so the FTS index stays small.
    """
    parts: list[str] = []

    def _walk(node: Any, depth: int) -> None:
        if depth > _FLATTEN_MAX_DEPTH:
            return
        if isinstance(node, dict):
            for val in node.values():
                _walk(val, depth + 1)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item, depth + 1)
        elif isinstance(node, bool):
            return
        elif isinstance(node, (str, int, float)):
            text = str(node).strip()
            if text:
                parts.append(text)

    _walk(value, 0)
    return " ".join(parts)[:_FLATTEN_MAX_CHARS]


def _escape_fts_query(query: str) -> str:
    """Turn a free-text query into a safe FTS5 MATCH expression.

    Each whitespace-separated word becomes a double-quoted token (so reserved
    FTS5 syntax in user text can't break the parse), joined with implicit AND
    via OR-less juxtaposition. We use OR so partial overlap still returns rows;
    BM25 ranks the better matches first.
    """
    words = [w for w in "".join(c if c.isalnum() else " " for c in query).split() if w]
    if not words:
        return ""
    return " OR ".join(f'"{w}"' for w in words)


class MemorySQLiteIndex:
    def __init__(self, user_id: str = "default", *, db_path: Path | None = None) -> None:
        self.user_id = user_id
        if db_path is None:
            db_path = personal_memory_dir(user_id) / "memory.sqlite"
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA synchronous = NORMAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def index_event(self, event: MemoryEvent) -> None:
        """Write-through, idempotent on event_id. Never raises on duplicate."""
        value_text = _flatten_value(event.value)
        evidence_text = _flatten_value(event.evidence)
        self._conn.execute(
            """
            INSERT OR IGNORE INTO events (
                event_id, session_id, turn_id, observed_at, kind, topic, key,
                value_json, value_text, confidence, source, extractor,
                evidence_json, evidence_text, supersedes, applied, rejected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.session_id,
                event.turn_id,
                event.observed_at,
                event.kind,
                event.topic,
                event.key,
                json.dumps(event.value, ensure_ascii=False, sort_keys=True),
                value_text,
                float(event.confidence),
                event.source,
                event.extractor,
                json.dumps(event.evidence, ensure_ascii=False, sort_keys=True),
                evidence_text,
                event.supersedes,
                1 if event.applied else 0,
                1 if event.rejected else 0,
            ),
        )
        self._conn.commit()

    def search(
        self,
        query: str,
        *,
        topic: str | None = None,
        applied_only: bool = True,
        limit: int = 10,
    ) -> list[MemoryEventRow]:
        """FTS5 MATCH with optional topic filter, BM25-ranked (best first)."""
        match_expr = _escape_fts_query(query)
        if not match_expr:
            return []

        sql = [
            "SELECT e.event_id, e.topic, e.key, e.value_text, e.value_json,",
            "       e.evidence_text, e.confidence, e.source, e.observed_at,",
            "       e.applied, bm25(events_fts) AS rank",
            "FROM events_fts",
            "JOIN events e ON e.rowid = events_fts.rowid",
            "WHERE events_fts MATCH ?",
        ]
        params: list[Any] = [match_expr]
        if applied_only:
            sql.append("AND e.applied = 1")
        if topic:
            sql.append("AND e.topic = ?")
            params.append(topic)
        sql.append("AND e.rejected = 0")
        sql.append("ORDER BY rank ASC")
        sql.append("LIMIT ?")
        params.append(int(limit))

        rows = self._conn.execute("\n".join(sql), params).fetchall()
        results: list[MemoryEventRow] = []
        for row in rows:
            try:
                value = json.loads(row["value_json"])
            except Exception:
                value = {}
            results.append(
                MemoryEventRow(
                    event_id=row["event_id"],
                    topic=row["topic"],
                    key=row["key"],
                    value_text=row["value_text"],
                    value=value if isinstance(value, dict) else {},
                    evidence_text=row["evidence_text"],
                    confidence=float(row["confidence"]),
                    source=row["source"],
                    observed_at=row["observed_at"],
                    applied=bool(row["applied"]),
                    rank=float(row["rank"]),
                )
            )
        return results

    def backfill_from_journal(self, journal_store: "JournalStore") -> int:
        """One-shot, idempotent. Returns the number of new rows inserted."""
        existing = self._conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
        for event in journal_store.iter_events():
            self.index_event(event)
        # ``index_event`` uses INSERT OR IGNORE, so re-runs against an already
        # populated DB simply skip duplicates. Report rows now present minus the
        # count we started with to give an accurate "newly inserted" number.
        total = self._conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
        return max(0, total - existing)

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"])

    def checkpoint(self) -> None:
        """Fold the WAL into the main database file (TRUNCATE mode).

        Called from session-finalization and shutdown paths so a copy of
        ``memory.sqlite`` alone is complete. Failures (e.g. cross-thread close
        during interpreter shutdown) are non-fatal by design.
        """
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            self._conn.commit()
        except Exception:
            pass

    def close(self) -> None:
        self.checkpoint()
        try:
            self._conn.close()
        except Exception:
            pass
