"""
core/tenant_purge.py
---------------------
Ledger item 2.3 (S-4.2): ONE purge_user(user_id) that both apps/admin_routes.py
routes (/forget-me/confirm, and any future admin-triggered purge) call
without branching on settings.is_cloud themselves.

Why this exists: apps/admin_routes.py's old ``_purge_user`` reached straight
for ``aiosqlite.connect(identity_manager.db_path)``. In cloud mode,
identity_manager is a PostgresIdentityManager, which has no db_path
attribute at all — every erasure request raised AttributeError in
production. Not partial deletion: no deletion, and a 500.

Local mode: unchanged rmtree-based behaviour (personal_memory_dir + RAG dir
+ the 2 users.sqlite rows), preserved here verbatim.

Cloud mode, in this order:
  1. Revoke the user's Google Calendar token FIRST (Phase 1's
     apps/calendar_oauth_routes.py::_revoke_at_google, reused via
     ``_read_token`` — imported, never edited, per WP 2.A's scope rules). A
     revoke failure (Google unreachable, non-200/400 status) is reported but
     never aborts the purge — mirrors /disconnect's own posture: the
     user-controllable, local half of an erasure request must not be held
     hostage by an upstream call.
  2. Delete every row keyed to this user across every purgeable cloud table,
     in ONE asyncpg transaction — hard delete is the only shape that can
     actually promise erasure. See _TABLE_USER_COLUMNS below for the
     enumeration (its count drifts as tables are added; don't hard-code it
     in prose) and test/tenant_purge_enumeration_test.py's DDL cross-check,
     which fails the next time a table is added anywhere in
     core/storage/cloud/ without a matching _TABLE_USER_COLUMNS entry (or a
     documented exclusion, for the rare table that must never be purged).
  3. Delete the user's Redis keys (spend counter, places-cap, ws-rate,
     live-delivery channel, channel-gate buffer, idempotency keys) — see
     _redis_patterns_for_user for the exact patterns and why 3 other key
     families are deliberately EXCLUDED.
  4. Write one content-free core/storage/cloud/purge_log_store.py row
     proving the purge happened, without keeping who it was (ledger 2.4).
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from typing import Any, Optional

from core.config import settings
from core.paths import PERSONAL_MEMORY_DIR, RAG_DATA_DIR

# ---------------------------------------------------------------------------
# Cloud table enumeration (ledger 2.3's "the enumeration IS the deliverable").
#
# One entry per CREATE TABLE IF NOT EXISTS found under core/storage/cloud/
# that holds DURABLE, PER-USER data an erasure request must cover.
# test/tenant_purge_enumeration_test.py::test_table_enumeration_matches_ddl
# greps every core/storage/cloud/*.py module for its own
# "CREATE TABLE IF NOT EXISTS <name>" statements and asserts this dict's key
# set is IDENTICAL (minus the deliberate exclusions documented at that
# test's _INTENTIONALLY_UNPURGED_TABLES) — a purge that silently omits a
# newly-added table is worse than one that fails loudly, so the next table
# added anywhere in core/storage/cloud/ without EITHER a matching entry here
# OR a documented exclusion at that test fails it instead of shipping a
# quiet gap. That test's own failure message points here for the fix.
#
# link_codes is the one two-column case: a code either ORIGINATES from this
# user (source_user_id) or is RESERVED for this user by a redemption in
# flight (reserved_for) — see core/storage/cloud/account_linking_store.py.
# Both must be cleared or a purge leaves the other side's row behind.
#
# telemetry_once (core/storage/cloud/telemetry_claim_store.py, WP2.C/ledger
# 2.6) records that ONE SPECIFIC USER reached ONE SPECIFIC funnel event —
# that is user data (who did what), not audit data about the purge itself
# (contrast with purge_log below, which is deliberately EXCLUDED because it
# IS the erasure's own proof and deleting it on every erasure would defeat
# it). A consequence worth flagging to whoever next reads the telemetry
# funnel: purging these rows means that if the SAME user_id ever recurred
# (e.g. a rebind from a surviving account.json marker, or — in principle —
# an id reused after a purge), first-run telemetry events would fire again
# for it. That is correct, not a bug: a purged identity must not carry
# residue that makes a fresh one look "already seen" — but it does mean a
# purge can be visible downstream as a dip-then-repeat in first-run funnel
# counts, which is worth knowing before chasing it as a tracking bug.
# ---------------------------------------------------------------------------
_TABLE_USER_COLUMNS: dict[str, tuple[str, ...]] = {
    "users": ("user_id",),
    "channel_mappings": ("user_id",),
    "claimed_tokens": ("user_id",),
    "account_markers": ("user_id",),
    "link_codes": ("source_user_id", "reserved_for"),
    "calendar_tokens": ("user_id",),
    "sessions": ("user_id",),
    "personal_memory_topics": ("user_id",),
    "personal_memory_daily_logs": ("user_id",),
    "routine_outbox": ("user_id",),
    "rag_session_staging": ("user_id",),
    "vector_docs": ("user_id",),
    "vector_chunks": ("user_id",),
    "routine_last_fired": ("user_id",),
    "journal_events": ("user_id",),
    "telemetry_once": ("user_id",),
    "confirmation_state": ("user_id",),
}


def table_enumeration() -> dict[str, tuple[str, ...]]:
    """Read-only accessor for tests — the enumeration above IS the contract."""
    return dict(_TABLE_USER_COLUMNS)


# ---------------------------------------------------------------------------
# Redis key patterns.
#
# Every family below is genuinely per-user (the key embeds user_id) and is
# durable/long-TTL state we own. Wildcards span exactly the nested segments
# each key shape actually has — a bare "turtle:*:{uid}*" (the ledger's
# original wording) matches the first 4 but MISSES turtle:gate:{uid}:{channel}
# and turtle:idem:{uid}:cal:{sha1} because the user_id isn't the LAST segment
# there; those two need their own "...{uid}:*" wildcard.
# ---------------------------------------------------------------------------
def _redis_patterns_for_user(user_id: str) -> list[str]:
    return [
        f"turtle:spend:{user_id}:*",        # apps/turtle_server.py per-day LLM spend counter
        f"turtle:places_cap:v1:{user_id}",  # tools/places_guardrails.py per-user Places/Routes daily cap
        f"turtle:ws_rate:{user_id}",        # core/storage/cloud/redis_backends.py inbound-message rate limiter
        f"turtle:live:{user_id}",           # core/storage/cloud/live_delivery.py per-user pub/sub channel name
        f"turtle:gate:{user_id}:*",         # core/storage/cloud/redis_backends.py channel-gate buffer, nested by channel
        f"turtle:idem:{user_id}:*",         # tools/idempotency.py dedup keys, nested by tool (":cal:", email hash, ...)
    ]


# Deliberately NEVER purged per-user — each reason is load-bearing, do not
# "fix" this by adding them to _redis_patterns_for_user:
#
#   turtle:places_cache:v1:*     — tools/places_guardrails.py's own docstring:
#                                   this cache is GLOBAL by design, no tenant
#                                   identifier is ever part of the key (two
#                                   users asking about the same place share a
#                                   cache hit). A per-user purge is meaningless
#                                   (nothing there is scoped to this user), and
#                                   deleting the whole family would evict every
#                                   OTHER tenant's cache too — collateral
#                                   damage this purge must never cause.
#   turtle:nonce:*                — core/internal_auth.py self-call replay
#                                   guard, keyed by a random nonce (not a user
#                                   id), 300s TTL. Expires on its own; there is
#                                   no per-user key to find here.
#   turtle:job:*                  — core/internal_auth.py job-result payload,
#                                   keyed by job id (not user id), 900s TTL.
#                                   Same reasoning as nonce above.
#   turtle:discord-interaction:*  — apps/channels/discord.py's interaction
#                                   dedup guard, keyed by Discord's own
#                                   interaction_id (not user id), 900s TTL.
#                                   Same reasoning.


async def purge_user(user_id: str) -> dict[str, Any]:
    """Hard-delete every artifact tied to ``user_id``. Local and cloud share
    this one entry point so callers (apps/admin_routes.py) never branch on
    settings.is_cloud themselves.
    """
    if settings.is_cloud:
        return await _purge_user_cloud(user_id)
    return await _purge_user_local(user_id)


# ---------------------------------------------------------------------------
# Local mode — unchanged rmtree + users.sqlite behaviour (moved verbatim out
# of apps/admin_routes.py::_purge_user).
# ---------------------------------------------------------------------------
async def _purge_user_local(user_id: str) -> dict[str, Any]:
    import aiosqlite

    from core.identity import identity_manager

    removed: dict[str, Any] = {"memory": False, "rag": False, "rows": 0}

    memory_dir = PERSONAL_MEMORY_DIR / user_id
    if memory_dir.exists():
        # This rmtree also removes account.json (it lives inside memory_dir).
        # That is REQUIRED, not incidental: the marker is the durable email->id
        # binding resolve_user() rebinds from, so it MUST die with the dir or a
        # purged user could be silently resurrected on the next onboarding.
        # Nothing else to delete for the marker — it has no separate location.
        shutil.rmtree(memory_dir, ignore_errors=True)
        removed["memory"] = not memory_dir.exists()

    rag_dir = RAG_DATA_DIR / user_id
    if rag_dir.exists():
        shutil.rmtree(rag_dir, ignore_errors=True)
        removed["rag"] = not rag_dir.exists()

    async with aiosqlite.connect(identity_manager.db_path) as db:
        cursor = await db.execute(
            "DELETE FROM channel_mappings WHERE user_id = ?", (user_id,)
        )
        removed["rows"] += cursor.rowcount or 0
        cursor = await db.execute(
            "DELETE FROM users WHERE user_id = ?", (user_id,)
        )
        removed["rows"] += cursor.rowcount or 0
        await db.commit()

    return removed


# ---------------------------------------------------------------------------
# Cloud mode.
# ---------------------------------------------------------------------------
async def _purge_user_cloud(user_id: str) -> dict[str, Any]:
    requested_at = datetime.now(timezone.utc)

    # 1. Revoke first — a user asking for erasure must not be blocked because
    #    Google is unreachable, so failure here is reported, never raised.
    revoked = await _revoke_calendar_token(user_id)

    # 2. Hard-delete every table row in one transaction.
    tables = await _purge_tables_cloud(user_id)

    # 3. Redis keys.
    redis_keys = await _purge_redis_keys(user_id)

    result: dict[str, Any] = {
        "revoked": revoked,
        "tables": tables,
        "redis_keys": redis_keys,
    }

    # 4. Content-free audit row (ledger 2.4) — never the plaintext user_id.
    from core.storage.cloud.purge_log_store import write_purge_log

    await write_purge_log(user_id, requested_at=requested_at, counts=result)

    return result


async def _revoke_calendar_token(user_id: str) -> Optional[bool]:
    """Best-effort Google Calendar token revoke. Returns True/False when a
    token existed and a revoke was attempted, None when there was no token
    to revoke. Never raises — a revoke failure must not abort the purge.

    Imports apps.calendar_oauth_routes's helpers rather than duplicating
    them (WP 2.A is not allowed to edit that file): _read_token already
    branches on settings.is_cloud internally and decrypts transparently, and
    _revoke_at_google already treats Google's 400 ("already invalid") as
    success — see that module's own docstrings.
    """
    from apps.calendar_oauth_routes import _read_token, _revoke_at_google

    token_json = await _read_token(user_id)
    if not token_json:
        return None
    revoked = await _revoke_at_google(token_json)
    if not revoked:
        print(
            f"LOG: tenant_purge could not confirm Google Calendar revoke for "
            f"user_id={user_id} — continuing the purge anyway; the grant may "
            f"still be live at Google."
        )
    return revoked


def _parse_delete_count(result: str) -> int:
    """asyncpg Connection.execute() returns a command-status string like
    "DELETE 3" — pull the row count back out of it."""
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


async def _purge_tables_cloud(user_id: str) -> dict[str, int]:
    from core.storage.cloud import get_pg_pool

    pool = await get_pg_pool()
    counts: dict[str, int] = {}
    async with pool.acquire() as conn:
        async with conn.transaction():
            for table, columns in _TABLE_USER_COLUMNS.items():
                where = " OR ".join(f"{column} = $1" for column in columns)
                result = await conn.execute(
                    f"DELETE FROM {table} WHERE {where}", user_id  # noqa: S608 — table/column names are our own fixed enum, never user input
                )
                counts[table] = _parse_delete_count(result)
    return counts


async def _purge_redis_keys(user_id: str) -> int:
    from core.storage.cloud import get_redis_client

    client = await get_redis_client()
    deleted = 0
    for pattern in _redis_patterns_for_user(user_id):
        matched = [key async for key in client.scan_iter(match=pattern, count=500)]
        if matched:
            deleted += await client.delete(*matched)
    return deleted
