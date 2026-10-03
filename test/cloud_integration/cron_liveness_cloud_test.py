"""
Real-Postgres test for the cron-tick liveness read (ledger 5.8):
core/storage/cloud/cron_state_store.get_last_tick_at (table: cron_state) and
apps/admin_routes.cron_liveness built on it.

NOTE: cron_state_store's own cloud tests have never run against a real
database before CI; this is also the first real execution of the
`SELECT last_tick_at FROM cron_state WHERE id = 1` read.

The test restores the singleton row's original last_tick_at when it finishes.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from test.cloud_integration.conftest import run_async

pytestmark = pytest.mark.cloud


def test_get_last_tick_at_roundtrip_and_liveness() -> None:
    from apps import admin_routes
    from core.storage.cloud import cron_state_store, get_pg_sync_pool

    cron_state_store.get_last_tick_at()  # forces table + seed row creation
    pool = get_pg_sync_pool()
    with pool.connection() as conn:
        original = conn.execute(
            "SELECT last_tick_at FROM cron_state WHERE id = 1"
        ).fetchone()[0]
    try:
        # Never ticked: NULL must come back as None and be reported "never".
        with pool.connection() as conn:
            conn.execute("UPDATE cron_state SET last_tick_at = NULL WHERE id = 1")
        assert cron_state_store.get_last_tick_at() is None
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(admin_routes.settings, "deploy_mode", "cloud")
            never = run_async(admin_routes.cron_liveness())
        assert never["status"] == "never"
        assert never["last_tick_age_s"] is None

        # Stale: 2h old.
        old = datetime.now(UTC) - timedelta(hours=2)
        with pool.connection() as conn:
            conn.execute("UPDATE cron_state SET last_tick_at = %s WHERE id = 1", (old,))
        got = cron_state_store.get_last_tick_at()
        assert got is not None and got.tzinfo is not None  # TIMESTAMPTZ
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(admin_routes.settings, "deploy_mode", "cloud")
            stale = run_async(admin_routes.cron_liveness())
        assert stale["status"] == "stale"
        assert 7100 < stale["last_tick_age_s"] < 7400

        # Fresh.
        with pool.connection() as conn:
            conn.execute(
                "UPDATE cron_state SET last_tick_at = %s WHERE id = 1",
                (datetime.now(UTC),),
            )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(admin_routes.settings, "deploy_mode", "cloud")
            fresh = run_async(admin_routes.cron_liveness())
        assert fresh["status"] == "ok"
    finally:
        with pool.connection() as conn:
            conn.execute(
                "UPDATE cron_state SET last_tick_at = %s WHERE id = 1", (original,)
            )
