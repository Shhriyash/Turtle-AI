"""
DDL-and-roundtrip test for core/storage/cloud/routine_last_fired_store.py
(table: routine_last_fired) against a real Postgres.
"""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.cloud


def test_try_claim_fire_is_exactly_once() -> None:
    from core.storage.cloud.routine_last_fired_store import (
        prune_older_than,
        try_claim_fire,
    )
    from core.storage.cloud import get_pg_sync_pool

    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    routine_key = "morning_briefing"
    fire_bucket = "2026-01-01T08:00"

    first = try_claim_fire(user_id, routine_key, fire_bucket)
    assert first is True

    second = try_claim_fire(user_id, routine_key, fire_bucket)
    assert second is False

    # A different bucket for the same routine is a separate claim.
    other_bucket = try_claim_fire(user_id, routine_key, "2026-01-02T08:00")
    assert other_bucket is True

    # prune_older_than's own DELETE, read back via a direct row count.
    pool = get_pg_sync_pool()
    with pool.connection() as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM routine_last_fired WHERE user_id = %s", (user_id,)
        ).fetchone()[0]
    assert before == 2

    deleted = prune_older_than("2099-01-01T00:00:00")  # everything is "older"
    assert deleted >= 2

    with pool.connection() as conn:
        after = conn.execute(
            "SELECT COUNT(*) FROM routine_last_fired WHERE user_id = %s", (user_id,)
        ).fetchone()[0]
    assert after == 0


# --- WP5-A1 (ledger 5.9 / 5.10): status column, stuck claims, pruning --------


def _uid() -> str:
    return f"usr_{uuid.uuid4().hex[:12]}"


def _status_of(user_id: str, routine_key: str, bucket: str) -> str:
    from core.storage.cloud import get_pg_sync_pool

    with get_pg_sync_pool().connection() as conn:
        row = conn.execute(
            "SELECT status FROM routine_last_fired "
            "WHERE user_id = %s AND routine_key = %s AND fire_bucket = %s",
            (user_id, routine_key, bucket),
        ).fetchone()
    return row[0] if row else None


def _set_claim_age(user_id: str, routine_key: str, bucket: str, age_s: int) -> None:
    from core.storage.cloud import get_pg_sync_pool

    with get_pg_sync_pool().connection() as conn:
        conn.execute(
            "UPDATE routine_last_fired SET claimed_at = now() - make_interval(secs => %s) "
            "WHERE user_id = %s AND routine_key = %s AND fire_bucket = %s",
            (float(age_s), user_id, routine_key, bucket),
        )


def test_status_column_is_added_in_place_to_a_pre_existing_table() -> None:
    """The production table predates the status column. CREATE TABLE IF NOT
    EXISTS cannot add it, so the store's ALTER must. Rows that already exist
    must come out as 'fired' (not 'claimed') or the first deploy would re-fire
    every historical claim. A TEMP table shadows the real one for this
    connection only, so nothing shared is touched."""
    from core.storage.cloud import get_pg_sync_pool
    from core.storage.cloud import routine_last_fired_store as store

    pool = get_pg_sync_pool()
    with pool.connection() as conn:
        conn.execute(
            "CREATE TEMP TABLE routine_last_fired ("
            " user_id TEXT NOT NULL, routine_key TEXT NOT NULL, fire_bucket TEXT NOT NULL,"
            " claimed_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            " PRIMARY KEY (user_id, routine_key, fire_bucket)) ON COMMIT DROP"
        )
        conn.execute(
            "INSERT INTO routine_last_fired (user_id, routine_key, fire_bucket) "
            "VALUES ('legacy', 'workflow.old', '2026-01-01T08:00')"
        )
        conn.execute(store._ADD_STATUS_SQL)
        conn.execute(store._ADD_STATUS_SQL)  # idempotent: second run must not error
        legacy_status = conn.execute(
            "SELECT status FROM routine_last_fired WHERE user_id = 'legacy'"
        ).fetchone()[0]
    assert legacy_status == "fired"


def test_ensure_init_twice_is_idempotent_on_the_real_table() -> None:
    from core.storage.cloud import get_pg_sync_pool
    from core.storage.cloud import routine_last_fired_store as store

    store._ensure_init()
    store._initialized = False  # force the DDL (incl. ALTER) to run again
    store._ensure_init()
    with get_pg_sync_pool().connection() as conn:
        cols = [
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'routine_last_fired' AND table_schema = current_schema()"
            ).fetchall()
        ]
    assert cols.count("status") == 1


def test_claim_lifecycle_claimed_fired_missed() -> None:
    from core.storage.cloud.routine_last_fired_store import mark_failed, mark_fired, try_claim_fire

    user = _uid()
    assert try_claim_fire(user, "r", "2026-01-01T08:00") is True
    assert _status_of(user, "r", "2026-01-01T08:00") == "claimed"
    assert mark_fired(user, "r", "2026-01-01T08:00") is True
    assert _status_of(user, "r", "2026-01-01T08:00") == "fired"
    assert mark_fired(user, "r", "2026-01-01T08:00") is False  # already fired
    assert mark_failed(user, "r", "2026-01-01T08:00") is False  # fired is final

    # A missed record keeps its status and still dedupes (PK unchanged).
    assert try_claim_fire(user, "r", "2026-01-02T08:00", status="missed") is True
    assert try_claim_fire(user, "r", "2026-01-02T08:00", status="missed") is False
    assert try_claim_fire(user, "r", "2026-01-02T08:00") is False
    assert mark_fired(user, "r", "2026-01-02T08:00") is False
    assert _status_of(user, "r", "2026-01-02T08:00") == "missed"


def test_list_stuck_claims_returns_only_claimed_rows_inside_the_age_window() -> None:
    from core.storage.cloud.routine_last_fired_store import (
        fail_stale_claims,
        list_stuck_claims,
        mark_fired,
        try_claim_fire,
    )

    user = _uid()
    for bucket in ("fresh", "stuck", "ancient", "done"):
        assert try_claim_fire(user, "r", bucket) is True
    _set_claim_age(user, "r", "fresh", 60)
    _set_claim_age(user, "r", "stuck", 15 * 60)
    _set_claim_age(user, "r", "ancient", 5 * 3600)
    _set_claim_age(user, "r", "done", 15 * 60)
    assert mark_fired(user, "r", "done") is True

    mine = [c for c in list_stuck_claims(600, 3600) if c[0] == user]
    assert mine == [(user, "r", "stuck")]

    assert fail_stale_claims(3600) >= 1
    assert _status_of(user, "r", "ancient") == "failed"
    assert _status_of(user, "r", "stuck") == "claimed"  # still retryable
    assert _status_of(user, "r", "done") == "fired"
    assert [c for c in list_stuck_claims(600, 3600) if c[0] == user] == [(user, "r", "stuck")]


def test_prune_with_a_seven_day_iso_cutoff_keeps_recent_rows() -> None:
    """Exactly what the tick passes: (now - 7d).isoformat() with a UTC offset."""
    from datetime import UTC, datetime, timedelta

    from core.storage.cloud.routine_last_fired_store import prune_older_than, try_claim_fire

    user = _uid()
    assert try_claim_fire(user, "r", "old") is True
    assert try_claim_fire(user, "r", "recent") is True
    _set_claim_age(user, "r", "old", 8 * 86400)
    _set_claim_age(user, "r", "recent", 6 * 86400)

    deleted = prune_older_than((datetime.now(UTC) - timedelta(days=7)).isoformat())
    assert deleted >= 1
    assert _status_of(user, "r", "old") is None
    assert _status_of(user, "r", "recent") == "claimed"


# --- WP5-A1 (ledger 5.7): cron_state ------------------------------------------


def test_cron_state_lock_commit_rollback_and_overlap() -> None:
    import threading
    from datetime import UTC, datetime, timedelta

    from core.storage.cloud.cron_state_store import get_last_tick_at, locked_tick_state

    original = get_last_tick_at()  # the row is global: restore it at the end
    t1 = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)
    t2 = t1 + timedelta(minutes=5)
    try:
        # Clean exit commits the new value.
        with locked_tick_state() as state:
            assert state is not None
            state.advance(t1)
        assert get_last_tick_at() == t1

        # An exception rolls the advance back and releases the lock.
        try:
            with locked_tick_state() as state:
                assert state.last_tick_at == t1
                state.advance(t2)
                raise RuntimeError("tick crashed")
        except RuntimeError:
            pass
        assert get_last_tick_at() == t1

        # While one tick holds the lock, an overlapping tick gets None
        # (FOR UPDATE SKIP LOCKED) instead of enumerating the same window.
        seen = {}
        with locked_tick_state() as outer:
            assert outer is not None

            def overlapping() -> None:
                with locked_tick_state() as inner:
                    seen["inner"] = inner

            worker = threading.Thread(target=overlapping)
            worker.start()
            worker.join(timeout=30)
            assert not worker.is_alive()
        assert seen["inner"] is None

        # Lock released: the next tick gets it again.
        with locked_tick_state() as again:
            assert again is not None
            assert again.last_tick_at == t1
    finally:
        from core.storage.cloud import get_pg_sync_pool

        with get_pg_sync_pool().connection() as conn:
            conn.execute("UPDATE cron_state SET last_tick_at = %s WHERE id = 1", (original,))


def test_cron_state_first_tick_reads_none_when_never_set() -> None:
    from core.storage.cloud import get_pg_sync_pool
    from core.storage.cloud.cron_state_store import get_last_tick_at, locked_tick_state

    original = get_last_tick_at()
    try:
        with get_pg_sync_pool().connection() as conn:
            conn.execute("UPDATE cron_state SET last_tick_at = NULL WHERE id = 1")
        with locked_tick_state() as state:
            assert state is not None
            assert state.last_tick_at is None
    finally:
        with get_pg_sync_pool().connection() as conn:
            conn.execute("UPDATE cron_state SET last_tick_at = %s WHERE id = 1", (original,))


def test_cron_state_table_holds_exactly_one_row() -> None:
    from core.storage.cloud import get_pg_sync_pool
    from core.storage.cloud.cron_state_store import _ensure_init

    _ensure_init()
    with get_pg_sync_pool().connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM cron_state").fetchone()[0]
        try:
            conn.execute("INSERT INTO cron_state (id, last_tick_at) VALUES (2, NULL)")
            second_row_allowed = True
        except Exception:
            second_row_allowed = False
    assert count == 1
    assert second_row_allowed is False
