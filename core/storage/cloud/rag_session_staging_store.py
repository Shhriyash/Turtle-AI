"""
core/storage/cloud/rag_session_staging_store.py
----------------------------------------------------
Cloud (TURTLE_DEPLOY=cloud) backend for rag/system/complete_rag.py's
TurtleRAGSystem staging buffer.

Found in a post-migration audit: RAG_DATA_DIR/{user_id}/current_session.json
was assumed to be same-request scratch space, but start_session() actually
reads it back to recover conversations accumulated by a PRIOR
TurtleRAGSystem instantiation ("leftover staging from a crashed/previous
run"), and add_conversation() rewrites it every turn until end_session()
finally indexes the whole thing into the vector store. On serverless, if a
later turn of the same session lands on a different/cold instance, this
buffer was silently lost — end_session()'s episodic summary would be built
from an incomplete or empty conversation set.

SYNCHRONOUS (psycopg): add_conversation() is itself a plain sync method
called directly, unawaited, from the request path
(apps/turtle_server.py:4443/4668) — already a blocking call today (its local
file I/O), so keeping this backend sync there is parity, not a regression.
start_session()/end_session() are async methods; TurtleRAGSystem wraps their
calls into this backend in asyncio.to_thread so a sync Postgres round trip
never blocks the event loop from those two (mirrors
apps/calendar_oauth_routes.py's own to_thread-around-sync-psycopg pattern).
"""
from __future__ import annotations

import json
from typing import Any, Optional

from core.storage.cloud import get_pg_sync_pool

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rag_session_staging (
    user_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    creation_time TEXT NOT NULL,
    conversations JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

_initialized = False


def _ensure_init() -> Any:
    global _initialized
    pool = get_pg_sync_pool()
    if not _initialized:
        with pool.connection() as conn:
            conn.execute(_CREATE_TABLE_SQL)
        _initialized = True
    return pool


class PostgresRagSessionStaging:
    """One row per user (a user has at most one in-flight staging session at
    a time, matching the local file's own "one temp_session_file" model)."""

    def __init__(self, user_id: str) -> None:
        if not user_id:
            raise ValueError("PostgresRagSessionStaging requires a user_id")
        self.user_id = user_id

    def read(self) -> Optional[dict[str, Any]]:
        pool = _ensure_init()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT session_id, creation_time, conversations FROM rag_session_staging "
                "WHERE user_id = %s",
                (self.user_id,),
            ).fetchone()
        if row is None:
            return None
        session_id, creation_time, conversations = row
        conversations = conversations if isinstance(conversations, list) else json.loads(conversations or "[]")
        return {"session_id": session_id, "creation_time": creation_time, "conversations": conversations}

    def write(self, session_data: dict[str, Any]) -> None:
        pool = _ensure_init()
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO rag_session_staging (user_id, session_id, creation_time, conversations, updated_at) "
                "VALUES (%s, %s, %s, %s::jsonb, now()) "
                "ON CONFLICT (user_id) DO UPDATE SET session_id = EXCLUDED.session_id, "
                "creation_time = EXCLUDED.creation_time, conversations = EXCLUDED.conversations, "
                "updated_at = now()",
                (
                    self.user_id,
                    session_data.get("session_id", ""),
                    session_data.get("creation_time", ""),
                    json.dumps(session_data.get("conversations", [])),
                ),
            )

    def clear(self) -> None:
        pool = _ensure_init()
        with pool.connection() as conn:
            conn.execute("DELETE FROM rag_session_staging WHERE user_id = %s", (self.user_id,))
