"""
core/storage/cloud/__init__.py
-------------------------------
Shared connection helpers for the cloud (TURTLE_DEPLOY=cloud) storage backends.

Every cloud store below (PostgresSessionStore, PgVectorStore, etc.) shares ONE
asyncpg pool and ONE Redis client per process, obtained here. Imports of
asyncpg/redis are lazy (inside the functions, not at module load) so a local
dev box that never installs these optional cloud dependencies still boots and
runs the full local/SQLite test suite untouched.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from core.config import settings


class CloudBackendUnavailable(RuntimeError):
    """Raised when a cloud store is used without its connection string set.

    Distinct from a bare ImportError/connection error so callers (and tests)
    can tell "not configured" apart from "configured but unreachable".
    """


_pg_pool: Optional[Any] = None
_pg_pool_lock: Optional[asyncio.Lock] = None

_pg_sync_pool: Optional[Any] = None

_redis_client: Optional[Any] = None
_redis_client_lock: Optional[asyncio.Lock] = None


def _get_pg_lock() -> asyncio.Lock:
    # Lazily created: an asyncio.Lock binds to the running loop, and this
    # module is imported long before the app loop exists (also re-used across
    # test loops, mirroring core/worker.py's _get_job_semaphore pattern).
    global _pg_pool_lock
    if _pg_pool_lock is None:
        _pg_pool_lock = asyncio.Lock()
    return _pg_pool_lock


async def get_pg_pool() -> Any:
    """Process-wide asyncpg pool, created on first use.

    Registers the pgvector type codec on every new connection so callers can
    pass/receive python lists (or numpy arrays) for a `vector` column without
    manual encode/decode.
    """
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    async with _get_pg_lock():
        if _pg_pool is not None:  # re-check: lost the race while awaiting the lock
            return _pg_pool
        dsn = settings.database_url.get_secret_value() if settings.database_url else ""
        if not dsn:
            raise CloudBackendUnavailable(
                "DATABASE_URL is not set — cannot create the Postgres pool. "
                "Set it to the Neon pooled connection string."
            )
        import asyncpg  # local import: optional dep, only needed in cloud mode
        from pgvector.asyncpg import register_vector

        async def _init_connection(conn: "asyncpg.Connection") -> None:
            await register_vector(conn)

        _pg_pool = await asyncpg.create_pool(
            dsn,
            min_size=1,
            max_size=5,
            init=_init_connection,
            # Serverless invocations are short-lived; don't hold connections
            # open indefinitely waiting on a command that will never finish.
            command_timeout=30,
        )
        return _pg_pool


def get_pg_sync_pool() -> Any:
    """Process-wide SYNCHRONOUS psycopg connection pool.

    Only for callers that are themselves synchronous and run directly on the
    event loop thread (rag/system/complete_rag.py's TurtleRAGSystem, an
    existing blocking-call pattern this preserves — see requirements.txt's
    comment on why asyncpg doesn't fit there). Everything else in
    core/storage/cloud uses the async pool above.

    No asyncio.Lock guard needed: this runs on the single event-loop thread in
    the app process (matching its caller's own execution model), so there is
    no concurrent-thread race to guard against here the way get_pg_pool()
    guards its own creation.
    """
    global _pg_sync_pool
    if _pg_sync_pool is not None:
        return _pg_sync_pool
    dsn = settings.database_url.get_secret_value() if settings.database_url else ""
    if not dsn:
        raise CloudBackendUnavailable(
            "DATABASE_URL is not set — cannot create the sync Postgres pool."
        )
    from psycopg_pool import ConnectionPool
    from pgvector.psycopg import register_vector

    def _configure(conn: Any) -> None:
        register_vector(conn)

    _pg_sync_pool = ConnectionPool(dsn, min_size=1, max_size=5, configure=_configure, open=True)
    return _pg_sync_pool


def close_pg_sync_pool() -> None:
    """Close the shared sync pool (test teardown / graceful shutdown)."""
    global _pg_sync_pool
    if _pg_sync_pool is not None:
        _pg_sync_pool.close()
        _pg_sync_pool = None


async def close_pg_pool() -> None:
    """Close the shared pool (test teardown / graceful shutdown)."""
    global _pg_pool
    if _pg_pool is not None:
        await _pg_pool.close()
        _pg_pool = None


def _get_redis_lock() -> asyncio.Lock:
    global _redis_client_lock
    if _redis_client_lock is None:
        _redis_client_lock = asyncio.Lock()
    return _redis_client_lock


async def get_redis_client() -> Any:
    """Process-wide async Redis client, created on first use."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    async with _get_redis_lock():
        if _redis_client is not None:
            return _redis_client
        url = settings.redis_url
        if not url:
            raise CloudBackendUnavailable(
                "REDIS_URL / UPSTASH_REDIS_URL is not set — cannot create the "
                "Redis client."
            )
        import redis.asyncio as redis  # local import: optional dep

        _redis_client = redis.from_url(url, decode_responses=True)
        return _redis_client


async def close_redis_client() -> None:
    """Close the shared client (test teardown / graceful shutdown)."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
