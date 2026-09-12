"""
core/storage/cloud/journal_store.py
--------------------------------------
Cloud (TURTLE_DEPLOY=cloud) storage backend for core/memory_journal.py's
JournalStore. The local backend shards an append-only JSONL file per
(user, year-month) under personal_journal_dir(user_id) — durable on a real
disk, but gone on the next serverless cold start. This is a straight
Postgres port of the same append-only, idempotent-by-event_id semantics,
one row per event.

SYNCHRONOUS on purpose: JournalStore's public methods (append/iter_events/
load_all/flush) are themselves plain sync methods today, called directly
(unawaited) from ~30 call sites across the codebase (routine scheduler,
account linking, every channel adapter, admin routes, ...). Keeping this
backend sync means JournalStore's public signatures need not change at all —
zero call-site edits anywhere, matching the local backend's own blocking-I/O
posture (fsync'd disk writes) rather than introducing a new async contract
this migration would then have to thread through every caller.
"""
from __future__ import annotations

import json
from typing import Any, Iterator, TYPE_CHECKING

from core.storage.cloud import get_pg_sync_pool

if TYPE_CHECKING:
    from core.memory_journal import MemoryEvent

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS journal_events (
    user_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (user_id, event_id)
)
"""
_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_journal_events_user_observed "
    "ON journal_events(user_id, observed_at)"
)


class PostgresJournalBackend:
    """Drop-in for core.memory_journal.JournalStore's internal storage: one
    row per event, keyed by (user_id, event_id) exactly like the local
    backend is keyed by event_id within a user's own journal directory.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id or "default"
        self._initialized = False

    def _ensure_init(self) -> Any:
        pool = get_pg_sync_pool()
        if not self._initialized:
            with pool.connection() as conn:
                conn.execute(_CREATE_TABLE_SQL)
                conn.execute(_CREATE_INDEX_SQL)
            self._initialized = True
        return pool

    def event_exists(self, event_id: str) -> bool:
        pool = self._ensure_init()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM journal_events WHERE user_id = %s AND event_id = %s",
                (self.user_id, event_id),
            ).fetchone()
        return row is not None

    def append_line(self, event: "MemoryEvent") -> None:
        """Idempotent insert — ON CONFLICT DO NOTHING mirrors the local
        backend's _event_exists-then-skip check, but atomically (no
        check-then-write race across concurrent serverless invocations)."""
        pool = self._ensure_init()
        payload = event.to_payload()
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO journal_events (user_id, event_id, observed_at, payload) "
                "VALUES (%s, %s, %s, %s::jsonb) "
                "ON CONFLICT (user_id, event_id) DO NOTHING",
                (self.user_id, event.event_id, event.observed_at, json.dumps(payload)),
            )

    def iter_events(self) -> Iterator["MemoryEvent"]:
        from core.memory_journal import MemoryEvent

        pool = self._ensure_init()
        with pool.connection() as conn:
            cur = conn.execute(
                "SELECT payload FROM journal_events WHERE user_id = %s "
                "ORDER BY observed_at, event_id",
                (self.user_id,),
            )
            rows = cur.fetchall()
        for (payload,) in rows:
            data = payload if isinstance(payload, dict) else json.loads(payload)
            try:
                yield MemoryEvent.from_payload(data)
            except Exception:
                continue

    def created_at_timestamp(self) -> float | None:
        """Epoch seconds of this user's earliest journal event — the cloud
        equivalent of the local backend's journal-directory ctime (there is
        no filesystem here to have one). A user's first-ever journal write
        happens moments after their journal directory would have been
        created locally, so this is a faithful substitute, not just a
        fallback, for confirmation_gate.py's first-session/account-age check.
        """
        pool = self._ensure_init()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT MIN(observed_at) FROM journal_events WHERE user_id = %s",
                (self.user_id,),
            ).fetchone()
        if not row or row[0] is None:
            return None
        return row[0].timestamp()

    def total_bytes(self) -> int:
        """Approximate on-disk size of this user's journal, for the storage
        cap check (mirrors the local backend's directory byte count)."""
        pool = self._ensure_init()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(pg_column_size(payload)), 0) "
                "FROM journal_events WHERE user_id = %s",
                (self.user_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def flush(self) -> None:
        pass  # Postgres commits are already durable; nothing to flush.
