"""
test/telemetry_test.py
-----------------------
WP2.C (ledger 2.6): core/telemetry.py's emit_once() dedup mechanism.

Local mode keeps the sentinel-file design but the exists()-then-write_text()
check-then-act is replaced with an atomic exclusive file create ("x" mode).
Cloud mode is backed by core/storage/cloud/telemetry_claim_store.py's
try_claim_once -- an INSERT ... ON CONFLICT DO NOTHING claim against
Postgres, mirroring core/storage/cloud/routine_last_fired_store.py's
try_claim_fire exactly (same _FakePool test-double pattern as
test/routine_cron_tick_test.py's TryClaimFireTest).

Decision pinned here: when the cloud claim store is unavailable (exception
from try_claim_once, e.g. DATABASE_URL unset or a connection failure),
emit_once falls back to emitting anyway rather than silently dropping the
funnel event -- the same posture the function already had for a failed local
sentinel write. A rare duplicate emission is cheaper than a rare gap for
analytics-grade data.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from core import telemetry


# ---------------------------------------------------------------------------
# Local mode: sentinel-file dedup
# ---------------------------------------------------------------------------

class EmitOnceLocalModeTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.object(telemetry, "emit", lambda *a, **k: None)
        self.addCleanup(patcher.stop)
        patcher.start()
        # is_cloud is a computed property off deploy_mode; force local.
        p_mode = patch("core.config.settings.deploy_mode", "local")
        self.addCleanup(p_mode.stop)
        p_mode.start()

    def test_second_call_same_user_event_is_a_no_op(self) -> None:
        user_id = "usr_telemetry_local_1"
        first = telemetry.emit_once(user_id, "onboarding_start")
        second = telemetry.emit_once(user_id, "onboarding_start")
        self.assertTrue(first)
        self.assertFalse(second)

    def test_different_event_is_a_new_claim(self) -> None:
        user_id = "usr_telemetry_local_2"
        telemetry.emit_once(user_id, "onboarding_start")
        self.assertTrue(telemetry.emit_once(user_id, "onboarding_complete"))

    def test_concurrent_calls_same_user_event_emit_once(self) -> None:
        """The exclusive-create fix: two threads racing the same (user_id,
        event) pair must not both win. Before the fix, exists()-then-
        write_text() was a check-then-act race a thread switch between the
        two calls could lose."""
        user_id = "usr_telemetry_local_race"
        results: list[bool] = []
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        def _call() -> None:
            barrier.wait()
            r = telemetry.emit_once(user_id, "first_message_sent")
            with lock:
                results.append(r)

        threads = [threading.Thread(target=_call) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(sorted(results), [False, True])


# ---------------------------------------------------------------------------
# Cloud mode: Postgres claim store, faked -- same pattern as
# test/routine_cron_tick_test.py's TryClaimFireTest.
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeConn:
    """A single lock around execute() stands in for Postgres's own row-level
    locking on the PRIMARY KEY -- the real guarantee that makes ONE
    INSERT ... ON CONFLICT DO NOTHING statement atomic across concurrent
    connections, which is what try_claim_once actually relies on."""

    _lock = threading.Lock()

    def __init__(self, claimed: set) -> None:
        self._claimed = claimed

    def execute(self, sql: str, params=None):
        sql_norm = " ".join(sql.split())
        with self._lock:
            if sql_norm.startswith("INSERT INTO telemetry_once"):
                key = tuple(params)
                if key in self._claimed:
                    return _FakeCursor(0)
                self._claimed.add(key)
                return _FakeCursor(1)
            return _FakeCursor(0)  # CREATE TABLE


class _FakeConnCtx:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


class _FakePool:
    def __init__(self) -> None:
        self.claimed: set = set()

    def connection(self):
        return _FakeConnCtx(_FakeConn(self.claimed))


class EmitOnceCloudModeTest(unittest.TestCase):
    def setUp(self) -> None:
        import core.storage.cloud.telemetry_claim_store as claim_store

        self.pool = _FakePool()
        claim_store._initialized = False
        p_pool = patch(
            "core.storage.cloud.telemetry_claim_store.get_pg_sync_pool",
            return_value=self.pool,
        )
        self.addCleanup(p_pool.stop)
        p_pool.start()

        p_emit = patch.object(telemetry, "emit", lambda *a, **k: None)
        self.addCleanup(p_emit.stop)
        p_emit.start()

        p_mode = patch("core.config.settings.deploy_mode", "cloud")
        self.addCleanup(p_mode.stop)
        p_mode.start()

    def test_try_claim_once_twice_claims_once(self) -> None:
        """Against the real claim mechanism (try_claim_once), not a mock of
        emit_once itself."""
        from core.storage.cloud.telemetry_claim_store import try_claim_once

        self.assertTrue(try_claim_once("usr_cloud_1", "onboarding_start"))
        self.assertFalse(try_claim_once("usr_cloud_1", "onboarding_start"))

    def test_emit_once_second_call_same_user_event_is_a_no_op(self) -> None:
        user_id = "usr_telemetry_cloud_1"
        first = telemetry.emit_once(user_id, "onboarding_start")
        second = telemetry.emit_once(user_id, "onboarding_start")
        self.assertTrue(first)
        self.assertFalse(second)

    def test_concurrent_emit_once_same_user_event_emit_once(self) -> None:
        """This phase has shipped four check-then-act races already -- this
        proves the fifth candidate (telemetry dedup) does not join them, by
        actually racing two threads against the atomic claim mechanism."""
        user_id = "usr_telemetry_cloud_race"
        results: list[bool] = []
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        def _call() -> None:
            barrier.wait()
            r = telemetry.emit_once(user_id, "memory_first_confirmed")
            with lock:
                results.append(r)

        threads = [threading.Thread(target=_call) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(sorted(results), [False, True])

    def test_claim_store_unavailable_falls_back_to_emit(self) -> None:
        """Pinned posture: a claim-store exception (DATABASE_URL unset,
        connection failure, ...) falls back to emitting anyway, mirroring
        the local sentinel-write fallback -- a rare duplicate is cheaper
        than a rare silent gap in analytics-grade funnel data."""
        with patch(
            "core.storage.cloud.telemetry_claim_store.try_claim_once",
            side_effect=RuntimeError("DATABASE_URL not set"),
        ):
            result = telemetry.emit_once("usr_telemetry_cloud_down", "onboarding_start")
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
