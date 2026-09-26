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

import contextlib
from contextvars import ContextVar
from typing import Any, Iterator, Optional

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


# ---------------------------------------------------------------------------
# WP3.D (ledger 3.8, transaction half): ambient per-replay transaction.
#
# Every method below used to open its OWN `pool.connection()` and let that
# context manager's built-in commit-on-success/rollback-on-exception behavior
# apply per call (see psycopg_pool.ConnectionPool.connection's docstring: "the
# normal connection context behaviour (commit/rollback the transaction in
# case of success/error)"). core/memory_replayer.py::replay() calls
# write_topic/delete_topic once per topic (~11 topics) with nothing tying
# those calls together, so a crash between topic 3 and topic 4 left topics
# 1-3 committed and 4-11 stale, with nothing recording that it happened.
#
# `transaction()` below checks out ONE connection from the pool and stashes
# it in a contextvar for the duration of the `with` block. Every read/write
# method first asks `_connection_scope()` for a connection: if the contextvar
# is set (we are inside a `transaction()` block), it reuses that SAME
# connection instead of asking the pool for another one — critical on this
# pool (`max_size=5`, core/storage/cloud/__init__.py::get_pg_sync_pool): a
# naive per-call `pool.connection()` while a transaction is already holding
# one open would not deadlock outright (the pool has spare slots), but it
# WOULD silently run that write on a second, separate connection outside the
# open transaction — an unnoticed no-op fix that still lets a mid-replay
# crash strand a half-written projection, just with an extra commit before
# the crash instead of after. Reusing the ambient connection is therefore not
# an optimization, it is the fix.
#
# Contextvars (not a plain module global) because this is synchronous code
# invoked directly inside async request handlers on the shared event-loop
# thread (see this module's own docstring); asyncio gives each Task its own
# copy of the context, so two concurrent requests replaying for two different
# users never see each other's ambient connection.
# ---------------------------------------------------------------------------

_active_conn: "ContextVar[Optional[Any]]" = ContextVar("_pmst_active_conn", default=None)


@contextlib.contextmanager
def _connection_scope() -> Iterator[Any]:
    """Yield the ambient transaction connection if `transaction()` is open;
    otherwise check one out of the pool for just this call (old behavior,
    unchanged for every call site outside of replay())."""
    conn = _active_conn.get()
    if conn is not None:
        yield conn
        return
    pool = _ensure_init()
    with pool.connection() as fresh_conn:
        yield fresh_conn


@contextlib.contextmanager
def transaction() -> Iterator[None]:
    """All personal-memory writes made by ANY PostgresPersonalMemoryBackend
    instance while this context is open share one Postgres connection/
    transaction: they all commit together on clean exit, or all roll back
    together if the block raises. Intended caller: core/memory_replayer.py's
    replay(), wrapping its whole per-topic write loop.

    Re-entrant: a nested `transaction()` call (defensive — no known caller
    does this today) reuses the already-open connection rather than trying to
    check out a second one from the same small pool.
    """
    if _active_conn.get() is not None:
        yield
        return
    pool = _ensure_init()
    with pool.connection() as conn:
        token = _active_conn.set(conn)
        try:
            yield
        finally:
            _active_conn.reset(token)


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

    def transaction(self) -> "contextlib.AbstractContextManager[None]":
        """See module-level `transaction()`: exposed as an instance method so
        callers holding a PostgresPersonalMemoryBackend (e.g. replay(), via
        PersonalMemoryStore's cloud backend) don't need their own import of
        this module's free function. The pool/contextvar backing it is
        process-wide, not per-instance — every instance shares the same
        ambient transaction slot, which is correct: two PostgresPersonal-
        MemoryBackend objects opened for two different users must never
        accidentally share one transaction anyway (replay() only ever has one
        store/user in scope at a time)."""
        return transaction()

    def read_topic(self, topic_name: str) -> Optional[str]:
        with _connection_scope() as conn:
            row = conn.execute(
                "SELECT content FROM personal_memory_topics WHERE user_id = %s AND topic_name = %s",
                (self.user_id, topic_name),
            ).fetchone()
        return row[0] if row else None

    def write_topic(self, topic_name: str, content: str) -> None:
        with _connection_scope() as conn:
            conn.execute(
                "INSERT INTO personal_memory_topics (user_id, topic_name, content, updated_at) "
                "VALUES (%s, %s, %s, now()) "
                "ON CONFLICT (user_id, topic_name) DO UPDATE SET content = EXCLUDED.content, "
                "updated_at = now()",
                (self.user_id, topic_name, content),
            )

    def delete_topic(self, topic_name: str) -> bool:
        """Returns True if a row actually existed and was deleted."""
        with _connection_scope() as conn:
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
        with _connection_scope() as conn:
            row = conn.execute(
                "SELECT content FROM personal_memory_daily_logs WHERE user_id = %s AND log_date = %s",
                (self.user_id, log_date),
            ).fetchone()
        return row[0] if row else None

    def write_daily_log(self, log_date: str, content: str) -> None:
        with _connection_scope() as conn:
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
        with _connection_scope() as conn:
            topics_row = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM personal_memory_topics WHERE user_id = %s",
                (self.user_id,),
            ).fetchone()
            logs_row = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM personal_memory_daily_logs WHERE user_id = %s",
                (self.user_id,),
            ).fetchone()
        return int(topics_row[0] or 0) + int(logs_row[0] or 0)
