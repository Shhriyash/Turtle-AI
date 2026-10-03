"""
core/storage/cloud/routine_last_fired_store.py
--------------------------------------------------
Exactly-once dedup for the cron-tick endpoint (apps/cron_tick_routes.py):
claims a (user_id, routine_key, fire_bucket) tuple before firing so a routine
fires exactly once per SCHEDULED occurrence even if two ticks both land
inside the same due window (misfire/retry), or a GitHub Actions run overlaps
the previous one. fire_bucket comes from core.routine_cron_tick
(Occurrence.fire_bucket) and identifies the scheduled time, not the tick that
observed it.

At-least-once (ledger 5.9). A claim row carries a `status`:
  claimed -> the occurrence was claimed but the fire has not been confirmed
             (the process may have died between claim and fire)
  fired   -> journal write + delivery both succeeded (mark_fired)
  missed  -> the occurrence was older than the late-fire limit; recorded, not
             fired
  failed  -> a stuck claim aged out of the retry window without ever firing
Each tick re-fires claims stuck in `claimed` (list_stuck_claims); the fire
event id is deterministic so a re-fire is a journal no-op.

SYNCHRONOUS (psycopg), called via asyncio.to_thread from the async cron-tick
route — matching this migration's established per-call-site driver choice.
"""
from __future__ import annotations

from typing import Any

from core.storage.cloud import get_pg_sync_pool

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS routine_last_fired (
    user_id TEXT NOT NULL,
    routine_key TEXT NOT NULL,
    fire_bucket TEXT NOT NULL,
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL DEFAULT 'fired',
    PRIMARY KEY (user_id, routine_key, fire_bucket)
)
"""
# CREATE TABLE IF NOT EXISTS never alters a table that already exists in
# production, so the status column is also added in place, idempotently (same
# pattern as account_linking_store._ADD_EXPECTED_EMAIL_SQL). The DEFAULT
# 'fired' is deliberate: every row that predates this column was written by
# the claim-then-fire code and must NOT be treated as stuck and re-fired on
# the first deploy. New claims insert status='claimed' explicitly.
_ADD_STATUS_SQL = (
    "ALTER TABLE routine_last_fired "
    "ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'fired'"
)
# Old buckets accumulate forever otherwise (one row per routine per scheduled
# fire) — bounded by an index on claimed_at so a periodic prune stays cheap.
_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_routine_last_fired_claimed_at "
    "ON routine_last_fired(claimed_at)"
)

_initialized = False


def _ensure_init() -> Any:
    global _initialized
    pool = get_pg_sync_pool()
    if not _initialized:
        with pool.connection() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_ADD_STATUS_SQL)
            conn.execute(_CREATE_INDEX_SQL)
        _initialized = True
    return pool


def try_claim_fire(
    user_id: str, routine_key: str, fire_bucket: str, status: str = "claimed"
) -> bool:
    """Atomically claim this (user, routine, scheduled-occurrence) tuple.

    Returns True the FIRST time it's called for a given tuple (proceed to
    fire) and False every subsequent time (already fired — skip). The
    ON CONFLICT DO NOTHING + rowcount check is the atomic part: two
    concurrent cron-tick invocations racing on the same tuple can only ever
    have one winner, unlike a SELECT-then-INSERT check.

    The default status 'claimed' means "fire pending, confirm with
    mark_fired". Pass status='missed' to record an occurrence that is being
    skipped as too late; the same exactly-once guarantee then makes the
    "I missed your routine" notice exactly-once too.
    """
    pool = _ensure_init()
    with pool.connection() as conn:
        cur = conn.execute(
            "INSERT INTO routine_last_fired (user_id, routine_key, fire_bucket, status) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (user_id, routine_key, fire_bucket, status),
        )
        return cur.rowcount == 1


def mark_fired(user_id: str, routine_key: str, fire_bucket: str) -> bool:
    """Flip a claim claimed -> fired once the fire is confirmed. Returns True
    when a row was flipped. The WHERE status='claimed' makes it a no-op for
    rows already fired/missed/failed."""
    pool = _ensure_init()
    with pool.connection() as conn:
        cur = conn.execute(
            "UPDATE routine_last_fired SET status = 'fired' "
            "WHERE user_id = %s AND routine_key = %s AND fire_bucket = %s "
            "AND status = 'claimed'",
            (user_id, routine_key, fire_bucket),
        )
        return cur.rowcount == 1


def mark_failed(user_id: str, routine_key: str, fire_bucket: str) -> bool:
    """Give up on a stuck claim (claimed -> failed). Returns True when flipped."""
    pool = _ensure_init()
    with pool.connection() as conn:
        cur = conn.execute(
            "UPDATE routine_last_fired SET status = 'failed' "
            "WHERE user_id = %s AND routine_key = %s AND fire_bucket = %s "
            "AND status = 'claimed'",
            (user_id, routine_key, fire_bucket),
        )
        return cur.rowcount == 1


def list_stuck_claims(
    min_age_s: int, max_age_s: int
) -> list[tuple[str, str, str]]:
    """Claims still in status 'claimed' whose claimed_at is between max_age_s
    and min_age_s old: old enough that the original attempt is surely dead
    (min_age_s), young enough to still be worth firing (max_age_s). Returns
    [(user_id, routine_key, fire_bucket), ...], oldest first. Ages are
    measured by the database clock."""
    pool = _ensure_init()
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT user_id, routine_key, fire_bucket FROM routine_last_fired "
            "WHERE status = 'claimed' "
            "AND claimed_at <= now() - make_interval(secs => %s) "
            "AND claimed_at > now() - make_interval(secs => %s) "
            "ORDER BY claimed_at",
            (float(min_age_s), float(max_age_s)),
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def fail_stale_claims(max_age_s: int) -> int:
    """Flip claims stuck in 'claimed' for longer than max_age_s to 'failed' so
    they stop being retried and stay visible as failures. Returns the count."""
    pool = _ensure_init()
    with pool.connection() as conn:
        cur = conn.execute(
            "UPDATE routine_last_fired SET status = 'failed' "
            "WHERE status = 'claimed' "
            "AND claimed_at <= now() - make_interval(secs => %s)",
            (float(max_age_s),),
        )
        return cur.rowcount


def prune_older_than(cutoff_iso: str) -> int:
    """Delete claim rows older than cutoff_iso (an ISO-8601 timestamp).
    Called by every cron tick (apps/cron_tick_routes.py) with a 7-day cutoff;
    also safe to invoke from an admin/maintenance task. Returns the number of
    rows deleted."""
    pool = _ensure_init()
    with pool.connection() as conn:
        cur = conn.execute(
            "DELETE FROM routine_last_fired WHERE claimed_at < %s", (cutoff_iso,)
        )
        return cur.rowcount
