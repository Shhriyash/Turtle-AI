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
    """A row returned by ``MemorySQLiteIndex`` (search and the read-model queries).

    Carries the typed event fields plus the BM25 ``rank`` (more-negative is a
    better match, per SQLite convention). The read-model queries added in
    Phase 2 W3 (``get_event``/``latest_for_key``/``events_for_key``) also
    populate the identity fields (``kind``/``session_id``/``turn_id``/
    ``extractor``) and the derived read-model columns (``statement``/``status``/
    ``superseded_by``) so a caller can reconstruct the event without walking the
    journal. ``search`` leaves the identity fields at their defaults — it only
    needs the projected columns — but does populate ``statement``/``status``.
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
        "kind",
        "session_id",
        "turn_id",
        "extractor",
        "statement",
        "status",
        "superseded_by",
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
        kind: str = "",
        session_id: str = "",
        turn_id: str = "",
        extractor: str = "",
        statement: str = "",
        status: str = "",
        superseded_by: str = "",
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
        self.kind = kind
        self.session_id = session_id
        self.turn_id = turn_id
        self.extractor = extractor
        self.statement = statement
        self.status = status
        self.superseded_by = superseded_by


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
    rejected      INTEGER NOT NULL,
    statement     TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT '',
    superseded_by TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_events_topic_applied ON events(topic, applied);
CREATE INDEX IF NOT EXISTS idx_events_observed_at ON events(observed_at DESC);
-- Read-model lookup: latest_for_key filters on (topic, key) among applied,
-- non-rejected, non-superseded rows; the hot path is an existence/latest probe.
CREATE INDEX IF NOT EXISTS idx_events_topic_key ON events(topic, key);

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


_FTS_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "do", "does", "did", "i", "im", "my", "me", "mine",
        "you", "your", "yours", "we", "our", "us",
        "what", "whats", "who", "whos", "where", "when", "which", "how", "why",
        "to", "of", "in", "on", "at", "for", "and", "or",
        "it", "its", "this", "that", "these", "those",
        "have", "has", "had", "can", "could", "should", "would", "will",
        "please", "tell", "about",
    }
)


def _escape_fts_query(query: str, *, operator: str = "AND") -> str:
    """Turn a free-text query into a safe FTS5 MATCH expression.

    Each whitespace-separated word becomes a double-quoted token (so reserved
    FTS5 syntax in user text can't break the parse). Stopwords are removed for
    precision, with an all-stopword fallback so short queries still work. The
    search path uses AND first for precision; callers may retry with OR for
    recall when AND finds nothing.
    """
    words = [w for w in "".join(c if c.isalnum() else " " for c in query).split() if w]
    if not words:
        return ""
    filtered = [w for w in words if w.lower() not in _FTS_STOPWORDS]
    terms = filtered or words
    joiner = f" {operator.upper()} "
    return joiner.join(f'"{w}"' for w in terms)


def _status_for(event: "MemoryEvent") -> str:
    """Derive the read-model ``status`` for an event.

    A rejected event outranks everything (it was tombstoned); otherwise an
    applied event is "applied" and an unconfirmed candidate is "pending".
    """
    if event.rejected:
        return "rejected"
    return "applied" if event.applied else "pending"


# Column list shared by the read-model queries so they hydrate identical rows.
_READ_MODEL_COLUMNS = (
    "event_id, session_id, turn_id, observed_at, kind, topic, key,"
    " value_json, value_text, evidence_text, confidence, source, extractor,"
    " applied, statement, status, superseded_by"
)


def _build_event_row(row: sqlite3.Row, *, rank: float = 0.0) -> MemoryEventRow:
    """Hydrate a MemoryEventRow from a sqlite Row selecting _READ_MODEL_COLUMNS."""
    try:
        value = json.loads(row["value_json"])
    except Exception:
        value = {}
    return MemoryEventRow(
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
        rank=rank,
        kind=row["kind"],
        session_id=row["session_id"],
        turn_id=row["turn_id"],
        extractor=row["extractor"],
        statement=row["statement"],
        status=row["status"],
        superseded_by=row["superseded_by"],
    )


class MemorySQLiteIndex:
    def __init__(self, user_id: str = "default", *, db_path: Path | None = None) -> None:
        self.user_id = user_id
        if db_path is None:
            db_path = personal_memory_dir(user_id) / "memory.sqlite"
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        # Set when a write-through (index_event / mark_rejected) fails after the
        # journal write succeeded — from that point the read model may be missing
        # events, and consumers that need journal-faithful answers (the
        # confirmation gate) must fall back to journal scans until a successful
        # full backfill clears the flag. The journal is the source of truth;
        # this flag is what keeps a silently-degraded index from changing gate
        # decisions (Codex review B#3).
        self._stale = False
        self._init_schema()

    @property
    def is_stale(self) -> bool:
        return self._stale

    def _init_schema(self) -> None:
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA synchronous = NORMAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # Additive, idempotent migration for the read-model columns introduced in
        # Phase 2 W3 (statement/status/superseded_by). Fresh DBs already carry them
        # from the CREATE above, so this is a no-op there; a pre-existing production
        # DB gains them via ALTER TABLE. Runs on every open so the projection can be
        # evolved without a bespoke migration step.
        self._migrate_columns()

    _READ_MODEL_MIGRATION_COLUMNS = ("statement", "status", "superseded_by")

    def _migrate_columns(self) -> None:
        """Add the Phase 2 read-model columns if an older DB predates them.

        SQLite has no ``ADD COLUMN IF NOT EXISTS``, so probe ``PRAGMA table_info``
        and only ALTER for columns that are actually missing. Only the
        duplicate-column race (concurrent open) is swallowed; any other failure
        (locked DB, corrupt file) must surface here, at init, instead of as a
        confusing ``no such column`` from a later query. A final probe verifies
        every required column actually exists before the index is used.
        ``DEFAULT ''`` backfills existing rows.
        """
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(events)").fetchall()
        }
        for column in self._READ_MODEL_MIGRATION_COLUMNS:
            if column in existing:
                continue
            try:
                self._conn.execute(
                    f"ALTER TABLE events ADD COLUMN {column} TEXT DEFAULT ''"
                )
            except sqlite3.OperationalError as exc:
                # Column already present (concurrent open) — migration is additive.
                if "duplicate column" not in str(exc).lower():
                    raise
        self._conn.commit()
        present = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(events)").fetchall()
        }
        missing = [c for c in self._READ_MODEL_MIGRATION_COLUMNS if c not in present]
        if missing:
            raise sqlite3.OperationalError(
                f"memory.sqlite read-model migration incomplete; missing columns: {missing}"
            )

    def index_event(self, event: MemoryEvent) -> None:
        """Write-through, idempotent on event_id. Never raises on duplicate.

        Any operational failure marks the index stale before propagating: the
        caller (the journal's on_append hook) swallows the exception, so the
        flag is the only record that the read model just missed an event.
        """
        try:
            self._index_event_inner(event)
        except Exception:
            self._stale = True
            raise

    def _index_event_inner(self, event: MemoryEvent) -> None:
        value_text = _flatten_value(event.value)
        evidence_text = _flatten_value(event.evidence)
        # Read-model projection snapshotted at write time. ``superseded_by`` is
        # left at its default here — it is a cross-event relationship the journal
        # backfill resolves in one pass (see backfill_from_journal); the
        # write-through path has no cheap way to know its successor yet.
        status = _status_for(event)
        self._conn.execute(
            """
            INSERT OR IGNORE INTO events (
                event_id, session_id, turn_id, observed_at, kind, topic, key,
                value_json, value_text, confidence, source, extractor,
                evidence_json, evidence_text, supersedes, applied, rejected,
                statement, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                event.statement or "",
                status,
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
        requested_limit = int(limit)
        if requested_limit <= 0:
            return []

        def _run(match_expr: str) -> list[Any]:
            sql = [
                "SELECT e.event_id, e.topic, e.key, e.value_text, e.value_json,",
                "       e.evidence_text, e.confidence, e.source, e.observed_at,",
                "       e.applied, e.statement, e.status, bm25(events_fts) AS rank",
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
            # Over-fetch so the latest-per-key collapse below has material.
            params.append(max(requested_limit * 4, requested_limit))

            return self._conn.execute("\n".join(sql), params).fetchall()

        # AND-first for precision (stopwords stripped); OR fallback for recall.
        match_expr = _escape_fts_query(query, operator="AND")
        if not match_expr:
            return []

        rows = _run(match_expr)
        if not rows:
            or_expr = _escape_fts_query(query, operator="OR")
            if or_expr and or_expr != match_expr:
                rows = _run(or_expr)

        # A superseded event must never outrank (or accompany) its correction.
        superseded = {
            row["supersedes"]
            for row in self._conn.execute(
                "SELECT supersedes FROM events WHERE supersedes IS NOT NULL"
            ).fetchall()
        }
        rows = [row for row in rows if row["event_id"] not in superseded]

        # Collapse to the latest event per (topic, key) — retrieval must serve
        # the current value of a fact, not its whole history.
        latest_by_key: dict[tuple[str, str], Any] = {}
        for row in rows:
            key = (row["topic"], row["key"])
            current = latest_by_key.get(key)
            if current is None:
                latest_by_key[key] = row
                continue
            if row["observed_at"] > current["observed_at"]:
                latest_by_key[key] = row
            elif row["observed_at"] == current["observed_at"] and row["rank"] < current["rank"]:
                latest_by_key[key] = row

        survivors = set(id(row) for row in latest_by_key.values())
        rows = [row for row in rows if id(row) in survivors][:requested_limit]
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
                    statement=row["statement"],
                    status=row["status"],
                )
            )
        return results

    def backfill_from_journal(self, journal_store: "JournalStore") -> int:
        """One-shot, idempotent. Returns the number of new rows inserted."""
        existing = self._conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
        for event in journal_store.iter_events():
            self.index_event(event)
        # Resolve the two cross-event relationships the write-through path can't
        # see, so the read model can answer "is this rejected / what replaced it"
        # without walking the journal:
        #   1. Rejection tombstones — mark the target rejected and record the
        #      tombstone as its successor. Without this a rebuilt index would
        #      resurrect rejected facts.
        #   2. Correction chains — any event that ``supersedes`` another stamps
        #      its own event_id onto the target's ``superseded_by``.
        for event in journal_store.iter_events():
            rejected_event_id = event.value.get("rejected_event_id")
            if rejected_event_id:
                self._conn.execute(
                    "UPDATE events SET rejected = 1, status = 'rejected',"
                    " superseded_by = ? WHERE event_id = ?",
                    (event.event_id, rejected_event_id),
                )
            if event.supersedes:
                self._conn.execute(
                    "UPDATE events SET superseded_by = ? WHERE event_id = ?",
                    (event.event_id, event.supersedes),
                )
        self._conn.commit()
        # A completed full replay makes the read model journal-faithful again,
        # so a staleness flag set by an earlier write-through failure clears.
        self._stale = False
        # ``index_event`` uses INSERT OR IGNORE, so re-runs against an already
        # populated DB simply skip duplicates. Report rows now present minus the
        # count we started with to give an accurate "newly inserted" number.
        total = self._conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
        return max(0, total - existing)

    def mark_rejected(self, event_id: str) -> None:
        """Flag one event rejected in the index (pair with a journal tombstone
        via JournalStore.append_rejection so rebuilds stay consistent)."""
        try:
            self._conn.execute(
                "UPDATE events SET rejected = 1 WHERE event_id = ?",
                (event_id,),
            )
            self._conn.commit()
        except Exception:
            self._stale = True
            raise

    # ------------------------------------------------------------------
    # Read-model queries. These serve the hot-path existence / silence /
    # latest-by-key checks that used to O(n) scan the journal every turn
    # (autopsy DELTA-07). Indexed lookups over the derived projection.
    # ------------------------------------------------------------------

    def event_exists(self, event_id: str) -> bool:
        """True if an event with this id has been indexed (primary-key lookup)."""
        row = self._conn.execute(
            "SELECT 1 FROM events WHERE event_id = ? LIMIT 1", (event_id,)
        ).fetchone()
        return row is not None

    def get_event(self, event_id: str) -> MemoryEventRow | None:
        """Fetch a single event by id, or None if it isn't indexed."""
        row = self._conn.execute(
            f"SELECT {_READ_MODEL_COLUMNS} FROM events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        return _build_event_row(row)

    def latest_for_key(self, topic: str, key: str) -> MemoryEventRow | None:
        """Current served value for a (topic, key): the newest applied event
        that is neither rejected nor superseded. None when nothing qualifies."""
        # event_id tie-break matches journal replay, which picks the max
        # (observed_at, event_id) tuple when timestamps collide.
        row = self._conn.execute(
            f"SELECT {_READ_MODEL_COLUMNS} FROM events"
            " WHERE topic = ? AND key = ? AND applied = 1 AND rejected = 0"
            " AND superseded_by = ''"
            " ORDER BY observed_at DESC, event_id DESC LIMIT 1",
            (topic, key),
        ).fetchone()
        if row is None:
            return None
        return _build_event_row(row)

    def events_for_key(
        self, topic: str, key: str, *, limit: int | None = None
    ) -> list[MemoryEventRow]:
        """All indexed events for a (topic, key), oldest-first.

        Unfiltered by status on purpose: the confirmation gate's silence check
        scans these for contradiction tombstones, so candidates, corrections and
        contradictions must all be visible. ``limit`` caps the scan for callers
        that only need the most recent handful (oldest are dropped first).
        """
        sql = f"SELECT {_READ_MODEL_COLUMNS} FROM events WHERE topic = ? AND key = ?"
        params: list[Any] = [topic, key]
        if limit is not None and limit > 0:
            # Keep the most-recent ``limit`` rows but still hand them back
            # oldest-first, matching the journal iteration order the gate expects.
            sql += (
                " AND rowid IN (SELECT rowid FROM events WHERE topic = ? AND key = ?"
                " ORDER BY observed_at DESC, event_id DESC LIMIT ?)"
            )
            params.extend([topic, key, int(limit)])
        # Deterministic same-timestamp ordering (matches replay's tuple order);
        # bare observed_at would hand back whatever SQLite happened to scan first.
        sql += " ORDER BY observed_at ASC, event_id ASC"
        return [
            _build_event_row(row)
            for row in self._conn.execute(sql, params).fetchall()
        ]

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
