"""
core/storage/cloud/calendar_token_store.py
---------------------------------------------
Cloud (TURTLE_DEPLOY=cloud) replacement for the per-user Google Calendar
OAuth token file at personal_memory_dir(user_id)/google_calendar_token.json
(apps/calendar_oauth_routes.py, tools/calendar_tool.py). That file does not
survive a serverless cold start — a user who connects their calendar would
find it silently disconnected on the very next invocation, since the token
lives only on the ephemeral local disk of whichever instance handled the
OAuth callback.

The credentials refresh-token flow never rewrites this value after the
initial OAuth connect (tools/calendar_tool.py deliberately omits the
short-lived access_token and always refreshes via refresh_token — see its
_load_credentials docstring), so this is a simple read-mostly key/value store,
not something needing the richer session/journal semantics elsewhere in this
migration.

SYNCHRONOUS on purpose, matching PgChunkVectorStore/the Redis backends: the
read call site (tools/calendar_tool.py::_load_token_json) already runs inside
asyncio.to_thread (every calendar tool wraps its Google API call that way),
and the write/delete/exists call sites in apps/calendar_oauth_routes.py are
wrapped in asyncio.to_thread at their call sites for the same reason — a
consistent posture across both the async-route and to-thread-worker contexts
without a second (async) driver need.

Encryption at rest (core/calendar_token_crypto.py): this module is
deliberately encryption-agnostic — token_json is an opaque string as far as
this table is concerned. It holds either a bare plaintext token JSON blob
(pre-encryption tokens, and any local-mode token written with no
CALENDAR_TOKEN_KEY configured) or a `{"key_version": N, "blob": "..."}`
envelope. There is no separate `key_version` SQL column on purpose: the
value already has to round-trip through a single TEXT column on the local
disk (a JSON file, no schema at all) via the exact same envelope shape, so
key_version is carried inside the stored value itself instead of splitting
it across a Postgres-only column that the local backend could not mirror.
apps/calendar_oauth_routes.py's _read_token/_write_token do the
encrypt/decrypt/migrate-on-next-write around calls to this module.
"""
from __future__ import annotations

from typing import Any, Optional

from core.storage.cloud import get_pg_sync_pool

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS calendar_tokens (
    user_id TEXT PRIMARY KEY,
    token_json TEXT NOT NULL,
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


def get_token_json(user_id: str) -> Optional[str]:
    """Drop-in for reading token_path_for_user(user_id).read_text()."""
    if not user_id:
        return None
    pool = _ensure_init()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT token_json FROM calendar_tokens WHERE user_id = %s", (user_id,)
        ).fetchone()
    return row[0] if row else None


def put_token_json(user_id: str, token_json: str) -> None:
    """Drop-in for token_path_for_user(user_id).write_text(token_json)."""
    pool = _ensure_init()
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO calendar_tokens (user_id, token_json, updated_at) "
            "VALUES (%s, %s, now()) "
            "ON CONFLICT (user_id) DO UPDATE SET token_json = EXCLUDED.token_json, "
            "updated_at = now()",
            (user_id, token_json),
        )


def delete_token_json(user_id: str) -> None:
    """Drop-in for token_path_for_user(user_id).unlink()."""
    pool = _ensure_init()
    with pool.connection() as conn:
        conn.execute("DELETE FROM calendar_tokens WHERE user_id = %s", (user_id,))


def token_exists(user_id: str) -> bool:
    """Drop-in for token_path_for_user(user_id).exists()."""
    return get_token_json(user_id) is not None
