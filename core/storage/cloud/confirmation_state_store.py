"""
core/storage/cloud/confirmation_state_store.py
--------------------------------------------------
Cloud (TURTLE_DEPLOY=cloud) backend for core.confirmation_gate.ConfirmationGate's
pending-candidate queue. Implements the ConfirmationStateBackend Protocol
(load/save a ``{"pending": [event_id, ...]}`` dict) so ConfirmationGate itself
stays free of any storage-backend-specific code.

This is the direct fix for the documented /api/memory/confirm 404: the local
JSON file lives under personal_memory_dir(user_id)/confirmation_state.json,
readable only by whichever process/instance wrote it. On a real multi-worker
deploy (the Dockerfile's own -w 1 comment says this already happens today at
-w 2) or any serverless topology, a POST /api/memory/confirm landing on a
DIFFERENT instance than the one that queued the prompt would find an empty
state file and 404. Postgres is the same store from every instance, so the
bug class is structurally impossible here — apps/turtle_server.py's
/api/memory/pending and /api/memory/confirm additionally construct a fresh
ConfirmationGate per request (no process-cached SharedState lookup) so this
fix applies regardless of deploy mode, not just in cloud.

SYNCHRONOUS (psycopg): ConfirmationGate's load/save calls happen inline
inside otherwise-synchronous methods (queue_candidate, record_response, ...)
that are themselves called from async request handlers without their own
await — matching this migration's established per-call-site driver choice.
"""
from __future__ import annotations

import json
from typing import Any

from core.storage.cloud import get_pg_sync_pool

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS confirmation_state (
    user_id TEXT PRIMARY KEY,
    pending JSONB NOT NULL DEFAULT '[]'::jsonb,
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


class PostgresConfirmationState:
    """One row per user; implements ConfirmationStateBackend (load/save)."""

    def __init__(self, user_id: str) -> None:
        if not user_id:
            raise ValueError("PostgresConfirmationState requires a user_id")
        self.user_id = user_id

    def load(self) -> dict[str, Any]:
        pool = _ensure_init()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT pending FROM confirmation_state WHERE user_id = %s", (self.user_id,)
            ).fetchone()
        if row is None:
            return {"pending": []}
        pending = row[0]
        pending = pending if isinstance(pending, list) else json.loads(pending or "[]")
        return {"pending": [str(item) for item in pending if item]}

    def save(self, state: dict[str, Any]) -> None:
        pool = _ensure_init()
        pending = list(state.get("pending", []))
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO confirmation_state (user_id, pending, updated_at) "
                "VALUES (%s, %s::jsonb, now()) "
                "ON CONFLICT (user_id) DO UPDATE SET pending = EXCLUDED.pending, "
                "updated_at = now()",
                (self.user_id, json.dumps(pending)),
            )
