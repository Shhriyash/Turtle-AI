"""
test/storage_cloud_init_test.py
--------------------------------
Unit coverage for core/storage/cloud/__init__.py's shared pool/client
helpers (ledger item 3.5, "pools and timeouts").

Two things this file exists to prove:

1. get_pg_sync_pool() has a REAL cross-thread race. Eleven store modules
   call it, several reached via asyncio.to_thread from independent routes
   (apps/cron_tick_routes.py, apps/calendar_oauth_routes.py,
   apps/turtle_server.py, rag/system/complete_rag.py) — asyncio.to_thread
   dispatches to the default ThreadPoolExecutor, which runs real concurrent
   OS threads, not cooperative coroutines. Two such routes firing on a cold
   instance can both observe `_pg_sync_pool is None` and both construct a
   pool. test_concurrent_threads_construct_pool_exactly_once uses REAL
   threading.Thread workers (not asyncio) to reproduce this, and is written
   to FAIL against the unlocked code (see the module docstring's "run
   against unfixed code first" instruction in the work package brief).

2. The pool constructor kwargs the ledger asks for are actually passed
   through, via mocked constructors (never asserted on source text).
"""
from __future__ import annotations

import threading
import time
import unittest
from typing import Any, List
from unittest.mock import MagicMock, patch

import core.storage.cloud as cloud


class _FakeSecretStr:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


class _FakeSettings:
    database_url = _FakeSecretStr("postgresql://user:pass@fake-host/db")


class GetPgSyncPoolRaceTest(unittest.TestCase):
    """Reproduces the real cross-thread race in get_pg_sync_pool()."""

    def setUp(self) -> None:
        cloud._pg_sync_pool = None
        self.addCleanup(setattr, cloud, "_pg_sync_pool", None)
        # Reset the lock too, if the fix adds one, so each test starts clean.
        if hasattr(cloud, "_pg_sync_pool_lock"):
            cloud._pg_sync_pool_lock = threading.Lock()

    def test_concurrent_threads_construct_pool_exactly_once(self) -> None:
        construct_calls: List[Any] = []
        # All worker threads block here until every thread has been started,
        # so they all reach get_pg_sync_pool()'s cold-start check at roughly
        # the same moment — the real-world shape of the race (a calendar
        # OAuth callback and a cron tick both landing on a cold instance),
        # not something that merely happens to interleave by luck.
        n_threads = 8
        start_gate = threading.Barrier(n_threads)

        def fake_connection_pool(dsn, **kwargs):
            time.sleep(0.05)
            construct_calls.append((dsn, kwargs))
            return MagicMock()

        errors: List[BaseException] = []

        def worker() -> None:
            try:
                start_gate.wait(timeout=5)
                cloud.get_pg_sync_pool()
            except BaseException as exc:  # pragma: no cover - diagnostic
                errors.append(exc)

        with patch("core.storage.cloud.settings", _FakeSettings()), patch(
            "psycopg_pool.ConnectionPool", side_effect=fake_connection_pool
        ), patch("pgvector.psycopg.register_vector", return_value=None):
            threads = [threading.Thread(target=worker) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertEqual(
            len(construct_calls),
            1,
            f"expected exactly one ConnectionPool() construction under "
            f"{n_threads} concurrent threads, got {len(construct_calls)} — "
            f"the sync pool creation is not thread-safe",
        )


class GetPgSyncPoolKwargsTest(unittest.TestCase):
    def setUp(self) -> None:
        cloud._pg_sync_pool = None
        self.addCleanup(setattr, cloud, "_pg_sync_pool", None)

    def test_connection_pool_constructed_with_ledger_kwargs(self) -> None:
        captured = {}

        class _FakePool:
            def __init__(self, dsn, **kwargs):
                captured["dsn"] = dsn
                captured["kwargs"] = kwargs

            def open(self, *args, **kwargs):
                captured["opened"] = True

        with patch("core.storage.cloud.settings", _FakeSettings()), patch(
            "psycopg_pool.ConnectionPool", _FakePool
        ), patch("psycopg_pool.ConnectionPool.check_connection", "check-connection-sentinel", create=True), patch(
            "pgvector.psycopg.register_vector", return_value=None
        ):
            cloud.get_pg_sync_pool()

        kwargs = captured["kwargs"]
        self.assertEqual(kwargs.get("min_size"), 1)
        self.assertEqual(kwargs.get("max_size"), 5)
        self.assertEqual(kwargs.get("open"), False)
        # `timeout` on ConnectionPool bounds waiting for a free connection
        # FROM the pool; `connect_timeout` is a libpq per-connection param
        # and must travel inside `kwargs=`, not as a direct constructor
        # argument (ConnectionPool has no `connect_timeout` kwarg of its
        # own — see GetPgSyncPoolRealConstructionTest below, which caught
        # this exact shape being wrong against the real class).
        self.assertEqual(kwargs.get("timeout"), 10)
        self.assertEqual(kwargs.get("kwargs"), {"connect_timeout": 10})
        self.assertNotIn(
            "connect_timeout", kwargs,
            "connect_timeout must be nested inside kwargs=, not passed directly",
        )
        self.assertEqual(kwargs.get("check"), "check-connection-sentinel")
        self.assertTrue(captured.get("opened"), "pool.open() must be called explicitly after open=False")


class GetPgSyncPoolRealConstructionTest(unittest.TestCase):
    """A mocked constructor cannot validate a contract with a third-party
    library — it only re-asserts what the code already believes the
    signature is (this is exactly how a prior version of this WP shipped
    `connect_timeout` as a direct ConnectionPool kwarg, which is not a
    real parameter, and the mocked test above did not catch it). This test
    builds a REAL psycopg_pool.ConnectionPool with open=False and an
    unroutable DSN, so nothing actually connects, but construction alone
    proves every kwarg name/shape here is accepted by the installed
    psycopg_pool version."""

    def setUp(self) -> None:
        cloud._pg_sync_pool = None
        self.addCleanup(setattr, cloud, "_pg_sync_pool", None)

    def test_real_connection_pool_accepts_our_kwargs_unopened(self) -> None:
        # TEST-NET-1 (RFC 5737): guaranteed non-routable, so even if `open`
        # were mishandled this cannot reach a real network connection.
        settings_stub = type(
            "S", (), {"database_url": _FakeSecretStr("postgresql://u:p@192.0.2.1:5432/db")}
        )()
        with patch("core.storage.cloud.settings", settings_stub):
            pool = cloud.get_pg_sync_pool()
        try:
            self.assertEqual(type(pool).__module__.split(".")[0], "psycopg_pool")
        finally:
            pool.close()


class GetPgPoolKwargsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        cloud._pg_pool = None
        self.addCleanup(setattr, cloud, "_pg_pool", None)

    async def test_create_pool_called_with_ledger_kwargs(self) -> None:
        captured = {}

        async def fake_create_pool(dsn, **kwargs):
            captured["dsn"] = dsn
            captured["kwargs"] = kwargs
            return MagicMock()

        with patch("core.storage.cloud.settings", _FakeSettings()), patch(
            "asyncpg.create_pool", fake_create_pool
        ), patch("pgvector.asyncpg.register_vector", return_value=None):
            await cloud.get_pg_pool()

        kwargs = captured["kwargs"]
        self.assertEqual(kwargs.get("min_size"), 1)
        self.assertEqual(kwargs.get("max_size"), 5)
        self.assertEqual(kwargs.get("command_timeout"), 30)
        self.assertEqual(kwargs.get("statement_cache_size"), 0)
        self.assertEqual(kwargs.get("timeout"), 10)
        self.assertEqual(kwargs.get("max_inactive_connection_lifetime"), 60)
        self.assertEqual(kwargs.get("application_name"), "turtle")


class RedisTimeoutsUnchangedTest(unittest.IsolatedAsyncioTestCase):
    """WP3.B ruling: the ledger's 2/5 numbers are NOT adopted here — Phase 1's
    1.0/1.0 bound is deliberately kept (see core/storage/cloud/__init__.py's
    get_redis_client / get_redis_sync_client docstrings for the fail-fast
    rationale this preserves)."""

    async def asyncSetUp(self) -> None:
        cloud._redis_client = None
        cloud._redis_sync_client = None
        self.addCleanup(setattr, cloud, "_redis_client", None)
        self.addCleanup(setattr, cloud, "_redis_sync_client", None)

    async def test_async_redis_client_keeps_1s_timeouts(self) -> None:
        captured = {}

        def fake_from_url(url, **kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("core.storage.cloud.settings", _FakeSettings()):
            with patch("core.storage.cloud.settings.redis_url", "redis://fake", create=True):
                import redis.asyncio as redis_asyncio

                with patch.object(redis_asyncio, "from_url", fake_from_url):
                    await cloud.get_redis_client()

        self.assertEqual(captured.get("socket_connect_timeout"), 1.0)
        self.assertEqual(captured.get("socket_timeout"), 1.0)

    def test_sync_redis_client_keeps_1s_timeouts(self) -> None:
        captured = {}

        def fake_from_url(url, **kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("core.storage.cloud.settings", _FakeSettings()):
            with patch("core.storage.cloud.settings.redis_url", "redis://fake", create=True):
                import redis as redis_sync

                with patch.object(redis_sync, "from_url", fake_from_url):
                    cloud.get_redis_sync_client()

        self.assertEqual(captured.get("socket_connect_timeout"), 1.0)
        self.assertEqual(captured.get("socket_timeout"), 1.0)


class PgTransactionHelperTest(unittest.IsolatedAsyncioTestCase):
    """core.storage.cloud.pg_transaction() — the shared helper this WP
    defines (but does not adopt at any call site; see the module docstring
    and this WP's report for the sequencing note)."""

    async def test_sets_local_statement_timeout_then_yields_conn(self) -> None:
        executed = []

        class _FakeTxnCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        class _FakeConn:
            def transaction(self):
                return _FakeTxnCtx()

            async def execute(self, sql, *args):
                executed.append(sql)

        conn = _FakeConn()
        async with cloud.pg_transaction(conn) as txn_conn:
            self.assertIs(txn_conn, conn)

        self.assertTrue(
            any("SET LOCAL statement_timeout" in sql and "30s" in sql for sql in executed),
            f"expected a SET LOCAL statement_timeout call, got: {executed}",
        )


if __name__ == "__main__":
    unittest.main()
