"""
test/cloud_integration/conftest.py
-----------------------------------
WP0.C (ledger 0.4 / S-9.5): every test under this package executes the real
SQL/Redis commands issued by core/storage/cloud/*.py against a REAL Postgres
(pgvector/pgvector:pg16) and a REAL Redis (redis:7) — not the SQL-prefix fakes
in test/cloud_storage_test.py etc. That is the whole point: a wrong
placeholder or column name only turns CI red if something actually executes
the statement.

Self-skipping: this suite must not run (or import a cloud driver) during a
default local-mode run. DATABASE_URL/REDIS_URL unset means "no live cloud
services available" — every test here is skipped cleanly rather than failing
or erroring. The `cloud-tests` CI job (.github/workflows/tests.yml) is the
only place these two env vars are set together with TURTLE_DEPLOY=cloud.
"""
from __future__ import annotations

import asyncio
import os

import pytest

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
REDIS_URL = os.environ.get("REDIS_URL", "").strip()

_shared_loop: "asyncio.AbstractEventLoop | None" = None


def run_async(coro):
    """Run an async cloud-store coroutine on ONE persistent event loop shared
    across the whole test session/process.

    asyncpg's pool binds internal state (its DNS-resolution executor) to the
    loop it was created on. get_pg_pool()/get_redis_client() are process-wide
    singletons (core/storage/cloud/__init__.py), so if test A creates the pool
    inside its own asyncio.run() (which creates AND CLOSES a fresh loop), test
    B's own asyncio.run() call gets a second, different loop — and the
    already-built pool, still pointing at test A's now-closed loop, blows up
    with "RuntimeError: Event loop is closed" the moment it tries to open a
    new connection. Every async cloud-store test in this package must call
    this helper instead of asyncio.run() for that reason.
    """
    global _shared_loop
    if _shared_loop is None or _shared_loop.is_closed():
        _shared_loop = asyncio.new_event_loop()
    return _shared_loop.run_until_complete(coro)


@pytest.fixture(scope="session", autouse=True)
def _require_live_cloud_backends() -> None:
    """Session-wide gate: skip every test in this package when the live
    services this suite needs aren't configured, instead of importing
    asyncpg/psycopg/redis and failing to connect.

    Also bootstraps the pgvector extension via the ASYNC pool's own
    CREATE EXTENSION IF NOT EXISTS vector (core/storage/cloud/__init__.py's
    get_pg_pool) before any test runs. In a real cloud-mode boot something
    always touches the async pool first (e.g. PostgresSessionStore.init_db()
    at app startup); this test package has no such fixed boot order — several
    of the sync-psycopg stores (get_pg_sync_pool) eagerly open their own
    connection pool at first use, and psycopg's register_vector() on a
    connection opened before the extension exists cannot register the
    `vector` type, so a store touched first alphabetically could otherwise
    poison the pool for every pgvector-column store after it.
    """
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set — cloud integration suite skipped")
    if not REDIS_URL:
        pytest.skip("REDIS_URL not set — cloud integration suite skipped")

    from core.storage.cloud import get_pg_pool

    run_async(get_pg_pool())


@pytest.fixture(scope="session", autouse=True)
def _close_shared_pools():
    """Close the process-wide pg/redis pools (core/storage/cloud/__init__.py
    singletons) at session end so the pytest process exits cleanly instead of
    leaving open sockets/connections behind."""
    yield
    if not DATABASE_URL or not REDIS_URL:
        return
    from core.storage.cloud import (
        close_pg_sync_pool,
        close_redis_sync_client,
    )

    close_pg_sync_pool()
    close_redis_sync_client()

    # Async pool/client: only close if something actually created them, and on
    # the SAME shared loop they were created on (see run_async's docstring).
    import core.storage.cloud as cloud_mod

    if cloud_mod._pg_pool is not None or cloud_mod._redis_client is not None:
        from core.storage.cloud import close_pg_pool, close_redis_client

        async def _close_async() -> None:
            await close_pg_pool()
            await close_redis_client()

        run_async(_close_async())

    global _shared_loop
    if _shared_loop is not None and not _shared_loop.is_closed():
        _shared_loop.close()
        _shared_loop = None
