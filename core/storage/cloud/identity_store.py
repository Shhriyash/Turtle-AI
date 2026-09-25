"""
core/storage/cloud/identity_store.py
--------------------------------------
Cloud (TURTLE_DEPLOY=cloud) counterpart to core/identity.py: IdentityManager
(users.sqlite, aiosqlite) and the account.json marker files under
personal_memory_dir(user_id) both live on local disk, which does not survive
a serverless cold start — every user would be minted a fresh user_id (and
orphan their entire memory tree) on the very next invocation without this.

Two pieces, matching the two different call-site conventions in core/identity.py:

- PostgresIdentityManager: async (asyncpg), matching IdentityManager's own
  fully-async surface (every caller already does `await identity_manager
  .resolve_user(...)` etc.) — a direct drop-in swap, zero call-site changes.
- account marker functions (write_account_marker/read/rebind lookup): SYNC
  (psycopg), because write_account_marker is a plain sync function called
  from many places (2 production call sites in apps/onboarding_routes.py
  plus several sync unit tests) with no `await` — matching this migration's
  established pattern of picking the driver that matches the call site
  rather than forcing every cloud backend onto one driver.

The marker's role differs slightly from its local-disk counterpart: a local
account.json's PARENT DIRECTORY NAME is the trusted identity (its payload's
embedded user_id is cross-checked against it — a tamper vector specific to
"a marker written inside one user's dir could claim a different user_id").
In Postgres, account_markers is keyed by user_id itself (the primary key IS
the identity, not a filesystem location), so that specific check has no
Postgres equivalent to port — there is no separate "location" to disagree
with the payload.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from core.storage.cloud import get_pg_pool, get_pg_sync_pool

_CREATE_TABLES_SQL = (
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        primary_email TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS channel_mappings (
        channel TEXT NOT NULL,
        channel_user_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        PRIMARY KEY (channel, channel_user_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS claimed_tokens (
        jti TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        claimed_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS account_markers (
        user_id TEXT PRIMARY KEY,
        email TEXT NOT NULL,
        email_verified BOOLEAN NOT NULL DEFAULT false,
        created_at TIMESTAMPTZ NOT NULL,
        channel TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_account_markers_email ON account_markers(email)",
)

WEB_EMAIL_CHANNEL = "web_email"


class PostgresIdentityManager:
    """Drop-in for core.identity.IdentityManager, same async method surface,
    Postgres-backed. See module docstring for why this mirrors IdentityManager
    method-for-method instead of sharing a base class: the two operate on
    entirely different drivers (aiosqlite vs. asyncpg) and there is exactly
    one instance of either alive in a process (the module-level singleton),
    so there is nothing a shared abstraction would buy here.
    """

    def __init__(self) -> None:
        self._initialized = False

    async def init_db(self) -> None:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            for statement in _CREATE_TABLES_SQL:
                await conn.execute(statement)
        self._initialized = True

    async def _ensure_init(self) -> Any:
        pool = await get_pg_pool()
        if not self._initialized:
            await self.init_db()
        return pool

    async def mark_token_claimed(self, jti: str, user_id: str) -> bool:
        pool = await self._ensure_init()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    "INSERT INTO claimed_tokens (jti, user_id) VALUES ($1, $2)", jti, user_id
                )
                return True
            except Exception as exc:
                # asyncpg raises UniqueViolationError on a PK conflict;
                # imported lazily to avoid a hard asyncpg dependency at module
                # load for callers that only ever run in local mode.
                import asyncpg

                if isinstance(exc, asyncpg.UniqueViolationError):
                    return False
                raise

    async def link_channel(
        self, *, user_id: str, channel: str, channel_user_id: str
    ) -> Optional[str]:
        target = (channel_user_id or "").strip()
        if not user_id or not channel or not target:
            return None
        pool = await self._ensure_init()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT user_id FROM channel_mappings WHERE channel = $1 AND channel_user_id = $2",
                channel, target,
            )
            previous = row["user_id"] if row else None
            if previous == user_id:
                return previous
            await conn.execute(
                "INSERT INTO users (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING", user_id
            )
            await conn.execute(
                "INSERT INTO channel_mappings (channel, channel_user_id, user_id) "
                "VALUES ($1, $2, $3) "
                "ON CONFLICT (channel, channel_user_id) DO UPDATE SET user_id = EXCLUDED.user_id",
                channel, target, user_id,
            )
        print(
            f"LOG: linked {channel}/{target} -> {user_id}"
            + (f" (was {previous})" if previous else "")
        )
        return previous

    async def lookup_user(self, channel: str, channel_user_id: str) -> Optional[str]:
        """Non-minting counterpart to resolve_user: returns the existing
        user_id for (channel, channel_user_id) or None on a miss. Never
        mints, never rebinds from account markers. See
        core.identity.IdentityManager.lookup_user for the shared rationale —
        resolve_user's own existing-mapping check delegates here.
        """
        from core.identity import normalize_email  # avoid a circular import at module load

        is_email = channel == WEB_EMAIL_CHANNEL
        lookup_id = normalize_email(channel_user_id) if is_email else channel_user_id
        pool = await self._ensure_init()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT user_id FROM channel_mappings WHERE channel = $1 AND channel_user_id = $2",
                channel, lookup_id,
            )
            return row["user_id"] if row else None

    async def resolve_user(self, channel: str, channel_user_id: str) -> str:
        from core.identity import normalize_email  # avoid a circular import at module load

        is_email = channel == WEB_EMAIL_CHANNEL
        lookup_id = normalize_email(channel_user_id) if is_email else channel_user_id

        pool = await self._ensure_init()
        existing = await self.lookup_user(channel, channel_user_id)
        if existing is not None:
            if is_email:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE users SET primary_email = $1 "
                        "WHERE user_id = $2 AND primary_email IS NULL",
                        lookup_id, existing,
                    )
            return existing

        if is_email:
            rebound = await self._rebind_from_markers(channel, lookup_id)
            if rebound is not None:
                return rebound

        async with pool.acquire() as conn:
            new_user_id = f"usr_{uuid.uuid4().hex[:12]}"
            async with conn.transaction():
                await conn.execute("INSERT INTO users (user_id) VALUES ($1)", new_user_id)
                if is_email:
                    await conn.execute(
                        "UPDATE users SET primary_email = $1 WHERE user_id = $2",
                        lookup_id, new_user_id,
                    )
                try:
                    await conn.execute(
                        "INSERT INTO channel_mappings (channel, channel_user_id, user_id) "
                        "VALUES ($1, $2, $3)",
                        channel, lookup_id, new_user_id,
                    )
                except Exception as exc:
                    import asyncpg

                    if not isinstance(exc, asyncpg.UniqueViolationError):
                        raise
                    # Two concurrent first-logins raced to mint; yield to the
                    # winner instead of surfacing a 500 (mirrors IdentityManager).
                    winner = await conn.fetchrow(
                        "SELECT user_id FROM channel_mappings WHERE channel = $1 "
                        "AND channel_user_id = $2",
                        channel, lookup_id,
                    )
                    if winner:
                        print(
                            f"LOG: identity mint race for {channel}:{lookup_id} — "
                            f"yielding to existing {winner['user_id']}"
                        )
                        return winner["user_id"]
                    raise
            return new_user_id

    async def _rebind_from_markers(self, channel: str, email: str) -> Optional[str]:
        pool = await self._ensure_init()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT user_id, email_verified FROM account_markers "
                "WHERE email = $1 ORDER BY created_at ASC LIMIT 1",
                email,
            )
            if row is None:
                return None
            # Cloud mode always requires verified proof — an unverified
            # marker (dev fast-path residue) must not silently rebind a
            # stranger, matching IdentityManager's own require_verified rule.
            if not row["email_verified"]:
                print(
                    f"LOG: identity rebind skipped (unverified marker in cloud) email={email}"
                )
                return None
            user_id = row["user_id"]
            await conn.execute(
                "INSERT INTO users (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING", user_id
            )
            await conn.execute(
                "UPDATE users SET primary_email = $1 WHERE user_id = $2", email, user_id
            )
            await conn.execute(
                "INSERT INTO channel_mappings (channel, channel_user_id, user_id) "
                "VALUES ($1, $2, $3) "
                "ON CONFLICT (channel, channel_user_id) DO UPDATE SET user_id = EXCLUDED.user_id",
                channel, email, user_id,
            )
        print(f"LOG: identity rebound from marker email={email} user_id={user_id} verified=True")
        return user_id


# --- Account markers (sync/psycopg — see module docstring) -----------------

_markers_initialized = False


def _ensure_markers_init() -> Any:
    global _markers_initialized
    pool = get_pg_sync_pool()
    if not _markers_initialized:
        with pool.connection() as conn:
            for statement in _CREATE_TABLES_SQL:
                conn.execute(statement)
        _markers_initialized = True
    return pool


def write_account_marker_pg(user_id: str, email: str, verified: bool, *, channel: str) -> None:
    """Cloud-mode body of core.identity.write_account_marker. Preserves the
    original created_at across re-writes (dev /start writes unverified, then
    /claim rewrites it verified — the account wasn't "created" twice), same
    as the local file version.
    """
    pool = _ensure_markers_init()
    with pool.connection() as conn:
        existing = conn.execute(
            "SELECT created_at FROM account_markers WHERE user_id = %s", (user_id,)
        ).fetchone()
        created_at = existing[0] if existing else datetime.now(timezone.utc)
        conn.execute(
            "INSERT INTO account_markers (user_id, email, email_verified, created_at, channel) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (user_id) DO UPDATE SET email = EXCLUDED.email, "
            "email_verified = EXCLUDED.email_verified, channel = EXCLUDED.channel",
            (user_id, email, verified, created_at, channel),
        )
