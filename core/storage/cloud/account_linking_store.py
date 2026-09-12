"""
core/storage/cloud/account_linking_store.py
------------------------------------------------
Cloud (TURTLE_DEPLOY=cloud) backend for core.account_linking.LinkCodeStore.

The local class stores claim codes in a SQLite table alongside
users.sqlite, at identity_manager.db_path — an attribute that only exists on
the LOCAL core.identity.IdentityManager. In cloud mode, identity_manager is a
PostgresIdentityManager (core/storage/cloud/identity_store.py), which has no
db_path at all: apps/turtle_server.py's two LinkCodeStore(identity_manager.db_path)
call sites would raise AttributeError outright — a hard crash on the
account-linking code path, not just silent data loss, and worse than every
other gap this migration found. core/storage/factory.get_link_code_store()
is the fix: it selects this class in cloud mode without either call site
needing to know which store type it's holding (both expose the identical
issue/peek/reserve/release_reservation/consume/purge_expired surface).

SYNCHRONOUS (psycopg): the local class is itself built on plain sqlite3 (not
aiosqlite), and its own module-level wrapper functions
(peek/reserve/release_reservation/mark_consumed) exist specifically so the
redemption route can call them via asyncio.to_thread — i.e. every caller
already treats this as blocking work to be offloaded, so a sync driver here
requires zero call-site changes.
"""
from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from core.storage.cloud import get_pg_sync_pool

# Mirror core.account_linking's constants exactly so codes minted/redeemed
# through either backend behave identically.
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_CODE_LENGTH = 8
LINK_CODE_TTL_MINUTES = 15
RESERVATION_TTL_SECONDS = 60

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS link_codes (
    code TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    channel_user_id TEXT NOT NULL,
    source_user_id TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ,
    reserved_for TEXT,
    reserved_at TIMESTAMPTZ
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


def _normalize_code(code: str) -> str:
    return (code or "").strip().upper().replace(" ", "").replace("-", "")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(dt) -> str:
    return dt.isoformat(timespec="seconds") if dt else ""


class PostgresLinkCodeStore:
    """Drop-in for core.account_linking.LinkCodeStore — see module docstring.
    Returns core.account_linking.LinkCode instances so callers (including
    the module-level peek/reserve/release_reservation/mark_consumed thin
    wrappers) need no changes at all.
    """

    def __init__(self) -> None:
        pass  # No per-instance state — every method opens its own connection,
        # matching the local class's own "open/close per operation" posture.

    def issue(self, *, channel: str, channel_user_id: str, source_user_id: str):
        from core.account_linking import LinkCode

        pool = _ensure_init()
        code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))
        expires = _utc_now() + timedelta(minutes=LINK_CODE_TTL_MINUTES)
        with pool.connection() as conn:
            conn.execute(
                "DELETE FROM link_codes WHERE channel = %s AND channel_user_id = %s "
                "AND consumed_at IS NULL",
                (channel, channel_user_id),
            )
            conn.execute(
                "INSERT INTO link_codes (code, channel, channel_user_id, source_user_id, expires_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (code, channel, channel_user_id, source_user_id, expires),
            )
        return LinkCode(code, channel, channel_user_id, source_user_id, _iso(expires))

    def peek(self, code: str):
        from core.account_linking import LinkCode

        normalized = _normalize_code(code)
        if not normalized:
            return None
        pool = _ensure_init()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT channel, channel_user_id, source_user_id, expires_at, consumed_at "
                "FROM link_codes WHERE code = %s",
                (normalized,),
            ).fetchone()
        if row is None or row[4] is not None:
            return None
        channel, channel_user_id, source_user_id, expires_at, _consumed = row
        if expires_at <= _utc_now():
            return None
        return LinkCode(normalized, channel, channel_user_id, source_user_id, _iso(expires_at))

    def reserve(self, code: str, target_user_id: str) -> tuple[str, Optional[Any]]:
        from core.account_linking import LinkCode

        normalized = _normalize_code(code)
        if not normalized or not target_user_id:
            return ("invalid", None)
        pool = _ensure_init()
        now = _utc_now()
        cutoff = now - timedelta(seconds=RESERVATION_TTL_SECONDS)
        with pool.connection() as conn:
            # Same one-statement conditional reservation as the local class:
            # reserve iff unconsumed AND not-expired AND (unreserved OR
            # expired reservation OR same target).
            cur = conn.execute(
                """
                UPDATE link_codes
                   SET reserved_for = %s, reserved_at = %s
                 WHERE code = %s
                   AND consumed_at IS NULL
                   AND expires_at > %s
                   AND (reserved_for IS NULL
                        OR reserved_for = %s
                        OR reserved_at IS NULL
                        OR reserved_at < %s)
                """,
                (target_user_id, now, normalized, now, target_user_id, cutoff),
            )
            row = conn.execute(
                "SELECT channel, channel_user_id, source_user_id, expires_at, consumed_at "
                "FROM link_codes WHERE code = %s",
                (normalized,),
            ).fetchone()
        if row is None or row[4] is not None:
            return ("invalid", None)
        channel, channel_user_id, source_user_id, expires_at, _consumed = row
        if expires_at <= _utc_now():
            return ("invalid", None)
        if cur.rowcount == 0:
            # An active reservation for a different target is holding the
            # code — don't reveal who to the loser, same as the local class.
            return ("locked", None)
        claim = LinkCode(normalized, channel, channel_user_id, source_user_id, _iso(expires_at))
        return ("ok", claim)

    def release_reservation(self, code: str, target_user_id: str) -> None:
        normalized = _normalize_code(code)
        if not normalized or not target_user_id:
            return
        pool = _ensure_init()
        with pool.connection() as conn:
            conn.execute(
                "UPDATE link_codes SET reserved_for = NULL, reserved_at = NULL "
                "WHERE code = %s AND reserved_for = %s AND consumed_at IS NULL",
                (normalized, target_user_id),
            )

    def consume(self, code: str):
        from core.account_linking import LinkCode

        normalized = _normalize_code(code)
        if not normalized:
            return None
        pool = _ensure_init()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT channel, channel_user_id, source_user_id, expires_at, consumed_at "
                "FROM link_codes WHERE code = %s",
                (normalized,),
            ).fetchone()
            if row is None or row[4] is not None:
                return None
            channel, channel_user_id, source_user_id, expires_at, _consumed = row
            if expires_at <= _utc_now():
                return None
            cur = conn.execute(
                "UPDATE link_codes SET consumed_at = %s WHERE code = %s AND consumed_at IS NULL",
                (_utc_now(), normalized),
            )
            if cur.rowcount != 1:
                return None  # lost the race
        return LinkCode(normalized, channel, channel_user_id, source_user_id, _iso(expires_at))

    def purge_expired(self) -> int:
        pool = _ensure_init()
        with pool.connection() as conn:
            cur = conn.execute("DELETE FROM link_codes WHERE expires_at <= %s", (_utc_now(),))
            return cur.rowcount or 0
