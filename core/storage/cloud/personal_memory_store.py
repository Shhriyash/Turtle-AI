"""
core/storage/cloud/personal_memory_store.py
-----------------------------------------------
Cloud (TURTLE_DEPLOY=cloud) backend for core.personal_memory_store.PersonalMemoryStore.

THE critical gap this closes: the topic markdown files (identity.md,
preferences.md, ...) and the MEMORY.md index are what
core/personal_memory_prompt.py::PersonalMemoryPromptBuilder actually reads to
build the memory-context block injected into every chat turn's prompt. The
journal (source of truth for the underlying facts) has been Postgres-backed
since Phase 1, but nothing re-derived this rendered projection from it —
local disk was the ONLY place it lived, so a serverless cold start meant
Turtle's memory of a user (name, preferences, everything) silently vanished
until the next fact-storing turn happened to rewrite it. This was found
during a post-migration audit, not in the original plan's Phase 1 table.

SYNCHRONOUS (psycopg), matching PersonalMemoryStore's own plain-sync,
call-anywhere methods (write_topic/load_topic/save_index/append_daily_log
are called directly, unawaited, from ~5 modules including inside
core/memory_replayer.py's replay(), itself called synchronously from
core/confirmation_gate.py) — the established pattern throughout this
migration of matching the driver to the call site.
"""
from __future__ import annotations

from typing import Any, Optional

from core.storage.cloud import get_pg_sync_pool

_CREATE_TABLES_SQL = (
    """
    CREATE TABLE IF NOT EXISTS personal_memory_topics (
        user_id TEXT NOT NULL,
        topic_name TEXT NOT NULL,
        content TEXT NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (user_id, topic_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS personal_memory_daily_logs (
        user_id TEXT NOT NULL,
        log_date TEXT NOT NULL,
        content TEXT NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (user_id, log_date)
    )
    """,
)

# Sentinel topic_name for the MEMORY.md index — stored in the same table as
# real topics rather than a separate one, matching the local backend's own
# "index_path is just another file under base_dir" treatment.
_INDEX_TOPIC_NAME = "__index__"

_initialized = False


def _ensure_init() -> Any:
    global _initialized
    pool = get_pg_sync_pool()
    if not _initialized:
        with pool.connection() as conn:
            for statement in _CREATE_TABLES_SQL:
                conn.execute(statement)
        _initialized = True
    return pool


class PostgresPersonalMemoryBackend:
    """Drop-in for core.personal_memory_store._LocalPersonalMemoryBackend:
    same method surface, Postgres-backed. One row per (user, topic); the
    MEMORY.md index and each topic file share the same table, keyed apart by
    topic_name (see _INDEX_TOPIC_NAME); daily logs get their own table since
    they're keyed by date, not topic.
    """

    def __init__(self, user_id: str) -> None:
        if not user_id:
            raise ValueError("PostgresPersonalMemoryBackend requires a user_id")
        self.user_id = user_id

    def read_topic(self, topic_name: str) -> Optional[str]:
        pool = _ensure_init()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT content FROM personal_memory_topics WHERE user_id = %s AND topic_name = %s",
                (self.user_id, topic_name),
            ).fetchone()
        return row[0] if row else None

    def write_topic(self, topic_name: str, content: str) -> None:
        pool = _ensure_init()
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO personal_memory_topics (user_id, topic_name, content, updated_at) "
                "VALUES (%s, %s, %s, now()) "
                "ON CONFLICT (user_id, topic_name) DO UPDATE SET content = EXCLUDED.content, "
                "updated_at = now()",
                (self.user_id, topic_name, content),
            )

    def delete_topic(self, topic_name: str) -> bool:
        """Returns True if a row actually existed and was deleted."""
        pool = _ensure_init()
        with pool.connection() as conn:
            cur = conn.execute(
                "DELETE FROM personal_memory_topics WHERE user_id = %s AND topic_name = %s",
                (self.user_id, topic_name),
            )
            return cur.rowcount > 0

    def read_index(self) -> Optional[str]:
        return self.read_topic(_INDEX_TOPIC_NAME)

    def write_index(self, content: str) -> None:
        self.write_topic(_INDEX_TOPIC_NAME, content)

    def read_daily_log(self, log_date: str) -> Optional[str]:
        pool = _ensure_init()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT content FROM personal_memory_daily_logs WHERE user_id = %s AND log_date = %s",
                (self.user_id, log_date),
            ).fetchone()
        return row[0] if row else None

    def write_daily_log(self, log_date: str, content: str) -> None:
        pool = _ensure_init()
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO personal_memory_daily_logs (user_id, log_date, content, updated_at) "
                "VALUES (%s, %s, %s, now()) "
                "ON CONFLICT (user_id, log_date) DO UPDATE SET content = EXCLUDED.content, "
                "updated_at = now()",
                (self.user_id, log_date, content),
            )

    def total_bytes(self) -> int:
        """Approximate storage used by this user, for the storage-cap check
        (mirrors the local backend's directory byte count)."""
        pool = _ensure_init()
        with pool.connection() as conn:
            topics_row = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM personal_memory_topics WHERE user_id = %s",
                (self.user_id,),
            ).fetchone()
            logs_row = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM personal_memory_daily_logs WHERE user_id = %s",
                (self.user_id,),
            ).fetchone()
        return int(topics_row[0] or 0) + int(logs_row[0] or 0)
