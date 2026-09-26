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
import contextlib
import logging
import math
import os
import threading
from typing import Any, AsyncIterator, Optional

from core.config import settings

logger = logging.getLogger(__name__)


class CloudBackendUnavailable(RuntimeError):
    """Raised when a cloud store is used without its connection string set.

    Distinct from a bare ImportError/connection error so callers (and tests)
    can tell "not configured" apart from "configured but unreachable".
    """


_pg_pool: Optional[Any] = None
_pg_pool_lock: Optional[asyncio.Lock] = None

_pg_sync_pool: Optional[Any] = None
# threading.Lock (NOT asyncio.Lock): get_pg_sync_pool() is called from real
# concurrent OS threads via asyncio.to_thread (see its docstring below), and
# an asyncio.Lock only serializes coroutines on ONE event loop thread — it
# provides no mutual exclusion at all across separate OS threads. Module-level
# (not lazily created like _pg_pool_lock above) because threading.Lock, unlike
# asyncio.Lock, doesn't bind to a running event loop, so there's no
# import-time hazard in constructing it eagerly.
_pg_sync_pool_lock = threading.Lock()

_redis_client: Optional[Any] = None
_redis_client_lock: Optional[asyncio.Lock] = None

_redis_sync_client: Optional[Any] = None


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
            # Neon databases don't have the pgvector extension enabled by
            # default; register_vector() raises ("vector type not found") if
            # it's missing, which took down every Postgres-backed route
            # (identity resolution included, since they all share this one
            # pool) rather than just the RAG vector-store paths that actually
            # need it. Idempotent and cheap, so just always ensure it here.
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await register_vector(conn)

        # asyncpg.create_pool() passes unrecognized **connect_kwargs straight
        # through to asyncpg.connect() for every new physical connection,
        # UNVALIDATED at pool-construction time — the pool object is handed
        # back immediately either way. A typo'd/nonexistent kwarg here only
        # raises the first time the pool actually opens a connection
        # (asyncpg/pool.py's _get_new_connection -> connect()), which in this
        # codebase means the first real request against a real Postgres.
        # command_timeout, statement_cache_size, and timeout below are all
        # confirmed present on asyncpg.connect()'s real signature (checked
        # via inspect.signature(asyncpg.connect)); min_size/max_size/
        # max_inactive_connection_lifetime are create_pool()'s own
        # (non-passthrough) parameters. application_name is NOT one of
        # connect()'s parameters — see its own comment below for what that
        # one actually needs. No offline/mocked test can validate any of
        # this list against the real library; only a real Postgres
        # connection can (the cloud-tests CI job, exercising this pool on
        # every cloud-marked test, is what caught application_name).
        _pg_pool = await asyncpg.create_pool(
            dsn,
            min_size=1,
            max_size=5,
            init=_init_connection,
            # Serverless invocations are short-lived; don't hold connections
            # open indefinitely waiting on a command that will never finish.
            command_timeout=30,
            # database_url is documented (core/config.py) as Neon's POOLED
            # (PgBouncer, transaction-mode) endpoint. asyncpg's own FAQ says
            # transaction-mode PgBouncer does not preserve a session across
            # statements, so its prepared-statement cache can be handed a
            # statement name it never actually prepared on the physical
            # connection the next call lands on ("prepared statement ...
            # does not exist"). statement_cache_size=0 disables that cache,
            # which is the FAQ's prescribed fix for exactly this setup.
            statement_cache_size=0,
            # Bound the one-off connect handshake itself, not just query
            # execution (command_timeout above) — a stalled TCP/TLS/auth
            # step during pool creation would otherwise hang unbounded.
            timeout=10,
            # Serverless: don't let an idle connection sit open past the
            # invocation that used it, waiting on a request that will never
            # come.
            max_inactive_connection_lifetime=60,
            # `application_name` is a Postgres SERVER setting, not a
            # parameter of asyncpg.connect() itself (confirmed via
            # inspect.signature(asyncpg.connect) — it has no
            # application_name kwarg, only server_settings). Passing it as
            # a bare kwarg above raised "connect() got an unexpected
            # keyword argument 'application_name'" the first time the pool
            # opened a real connection — asyncpg.create_pool() does NOT
            # validate connect kwargs at construction time, only when
            # _get_new_connection() actually calls connect(), so this only
            # surfaces against a real Postgres (which is what caught it:
            # the cloud-tests CI job, not any offline/mocked test — no
            # offline test can validate this; see
            # GetPgPoolKwargsTest.test_create_pool_called_with_ledger_kwargs
            # in test/storage_cloud_init_test.py for why, and don't add one
            # that pretends to).
            server_settings={"application_name": "turtle"},
        )
        return _pg_pool


def get_pg_sync_pool() -> Any:
    """Process-wide SYNCHRONOUS psycopg connection pool.

    Used by callers that are themselves synchronous (rag/system/
    complete_rag.py's TurtleRAGSystem, an existing blocking-call pattern this
    preserves — see requirements.txt's comment on why asyncpg doesn't fit
    there — plus core/storage/cloud/journal_store.py, identity_store.py,
    calendar_token_store.py, account_linking_store.py,
    confirmation_state_store.py, personal_memory_store.py, pgvector_store.py,
    rag_session_staging_store.py, routine_outbox_store.py,
    telemetry_claim_store.py, and routine_last_fired_store.py).

    IMPORTANT: despite this function's own execution being synchronous, it is
    NOT single-threaded in practice. Several of the routes that reach these
    sync stores do so via asyncio.to_thread (apps/cron_tick_routes.py,
    apps/calendar_oauth_routes.py, apps/turtle_server.py,
    rag/system/complete_rag.py) — asyncio.to_thread dispatches to the
    default ThreadPoolExecutor, which runs REAL concurrent OS threads, not
    cooperative coroutines. Two such routes firing on a cold instance (e.g.
    a calendar OAuth callback and a cron tick, or two concurrent callbacks)
    can both observe `_pg_sync_pool is None` before either finishes
    constructing one, both build a ConnectionPool, and one overwrites the
    other's global reference while its connections leak unclosed. A prior
    version of this docstring claimed this only ran on one event-loop thread
    with one caller — that was stale and wrong; hence the threading.Lock
    below (an asyncio.Lock would NOT protect across OS threads the way it
    needs to here — see get_pg_pool()'s asyncio.Lock, which is fine because
    that one only ever races coroutines on a single loop thread).
    """
    global _pg_sync_pool
    if _pg_sync_pool is not None:
        return _pg_sync_pool
    with _pg_sync_pool_lock:
        if _pg_sync_pool is not None:  # re-check: lost the race while blocked on the lock
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

        pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=5,
            configure=_configure,
            # Constructed closed, then opened explicitly below: psycopg_pool
            # warns that open=True (the implicit-open-in-constructor form)
            # can race the pool's own background connection-opening thread
            # against code that starts using the pool immediately, and is
            # deprecated in favor of this explicit two-step form.
            open=False,
            # ConnectionPool's own `timeout` bounds how long a CALLER waits
            # to be handed a connection out of the pool (e.g. all 5 are
            # checked out and busy) — it has nothing to do with the network
            # connect handshake. Kept short for the same serverless reason
            # as get_pg_pool()'s asyncpg timeout=10 above: don't let a
            # caller block indefinitely on pool exhaustion.
            timeout=10,
            # The actual connect-handshake bound (TCP/TLS/auth) is a libpq
            # connection parameter, not a ConnectionPool.__init__ argument —
            # psycopg_pool has no `connect_timeout` kwarg of its own; it
            # passes `kwargs` through to every new connection it opens.
            kwargs={"connect_timeout": 10},
            # psycopg_pool's own recommended liveness check: reject/replace
            # a connection from the pool that's gone stale (e.g. Neon
            # suspended the compute between invocations) instead of handing
            # a caller a dead connection that fails on first use.
            check=ConnectionPool.check_connection,
        )
        pool.open()
        _pg_sync_pool = pool
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


@contextlib.asynccontextmanager
async def pg_transaction(conn: Any) -> AsyncIterator[Any]:
    """Shared asyncpg transaction helper: wraps ``conn.transaction()`` and
    additionally sets a per-transaction statement timeout via
    ``SET LOCAL statement_timeout``.

    ``SET LOCAL`` scopes the setting to the current transaction only (it
    resets at COMMIT/ROLLBACK), which matters because every connection here
    comes out of a shared pool (get_pg_pool()) and is reused by unrelated
    callers afterwards — a bare (non-LOCAL) SET would leak the timeout onto
    whichever caller acquires that connection next.

    NOT yet adopted at any call site (ledger 3.5 asks this WP to define it,
    not to migrate every store's ad hoc ``async with conn.transaction():``
    onto it — see this WP's report for the exact list of call sites that
    should adopt it as a follow-up package, so as to avoid touching files
    other work packages currently own).

    Usage::

        async with pg_transaction(conn) as txn_conn:
            await txn_conn.execute(...)
    """
    async with conn.transaction():
        await conn.execute("SET LOCAL statement_timeout = '30s'")
        yield conn


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

        # WP 1.B / S-7.3 follow-up: this client is now awaited INSIDE a
        # request handler (apps/channels/discord.py's Discord Interactions
        # endpoint, via core/internal_auth.py's job-store/nonce calls) that
        # must answer within Discord's 3-second ACK deadline. Unbounded
        # socket timeouts mean a Redis that STALLS (rather than refusing)
        # would hang that request past the deadline and never reach
        # internal_auth's fail-closed handling at all — a refused connection
        # raises promptly and IS caught, a hung one previously wasn't bounded
        # here to raise at all. 1.0s connect + 1.0s socket read/write is a
        # 2.0s worst case for one Redis round trip, leaving roughly a third
        # of the 3s budget for building the payload, signing it, and the
        # subsequent self-invoke — generous for a healthy Upstash connection
        # (typically tens of ms) while still well short of the deadline.
        _redis_client = redis.from_url(
            url, decode_responses=True, socket_connect_timeout=1.0, socket_timeout=1.0
        )
        return _redis_client


async def close_redis_client() -> None:
    """Close the shared client (test teardown / graceful shutdown)."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


def get_redis_sync_client() -> Any:
    """Process-wide SYNCHRONOUS Redis client.

    Only for callers that are themselves synchronous and run directly on the
    event loop thread (core/guardrails.py's WebSocketRateLimiter,
    core/channel_gate.py's ChannelGateBuffer, tools/idempotency.py — all
    called with no `await` at their existing call sites, an existing
    blocking-call pattern this preserves rather than changes, matching
    get_pg_sync_pool()'s rationale). Everything else in core/storage/cloud
    uses the async client above.
    """
    global _redis_sync_client
    if _redis_sync_client is not None:
        return _redis_sync_client
    url = settings.redis_url
    if not url:
        raise CloudBackendUnavailable(
            "REDIS_URL / UPSTASH_REDIS_URL is not set — cannot create the "
            "sync Redis client."
        )
    import redis  # local import: optional dep

    # WP 1.B / S-7.3 follow-up (coordinator-flagged): this client runs its
    # commands SYNCHRONOUSLY, directly on the event-loop thread (see the
    # docstring above) — a stalling Redis here doesn't just hang the one
    # caller, it freezes the WHOLE server, every connected websocket user,
    # for as long as the stall lasts. That's a STRICTLY WORSE blast radius
    # than the async client's (bounded to Discord's 3s ACK deadline above),
    # which is why this uses the SAME 1.0s connect / 1.0s socket bound
    # rather than a more generous one: none of this client's callers (the
    # WS rate limiter, the channel-gate buffer, tools/idempotency.py's
    # Redis-backed dedup) sits under a hard external deadline the way
    # Discord's self-invoke does, but a longer timeout here would extend an
    # outage's freeze to every concurrent user rather than just the caller
    # that hit it, which is a worse trade than a tighter bound risking an
    # occasional false-positive timeout against a healthy backend. If a
    # real deploy ever needs slack beyond 1.0s/1.0s for a genuinely slow
    # (not stalled) Redis, the right fix is moving these callers off the
    # loop thread via asyncio.to_thread (the pattern ledger item 1a.7 used
    # for SMTP), not loosening this bound.
    _redis_sync_client = redis.from_url(
        url, decode_responses=True, socket_connect_timeout=1.0, socket_timeout=1.0
    )
    return _redis_sync_client


def close_redis_sync_client() -> None:
    """Close the shared sync client (test teardown / graceful shutdown)."""
    global _redis_sync_client
    if _redis_sync_client is not None:
        _redis_sync_client.close()
        _redis_sync_client = None


# ---------------------------------------------------------------------------
# Readiness probes (WP0.A / S-5.8)
# ---------------------------------------------------------------------------
# A module-level default so tests can shrink the budget (e.g. to prove a
# hanging backend doesn't make /readyz hang) without actually waiting out a
# real timeout. The /readyz route never hardcodes this value itself.
#
# 8.0s (raised from an original 2.0s): on a brand-new deployment, /readyz is
# the FIRST database call — it must build the asyncpg pool (DNS + TLS +
# connect) and wake a suspended Neon compute, both inside this one budget.
# Redis (Upstash, HTTP-ish, no suspend) comfortably beat 2s; Postgres cold
# start did not, producing a false-negative 503 on an otherwise-healthy
# deploy. probe_postgres() and probe_redis() are awaited concurrently via
# asyncio.gather() in the /readyz handler, so raising this to 8.0 does NOT
# make the route's worst case 16s — both probes share the one wall-clock
# budget, run in parallel, and the route returns as soon as both resolve.
#
# Overridable via TURTLE_READYZ_TIMEOUT_S so the budget can be tuned without
# a deploy. Read here (not core/config.py, which this WP does not own) and
# guarded so a malformed value falls back to the default rather than raising
# at import time — an exception here would break every cloud boot.
_READYZ_TIMEOUT_DEFAULT = 8.0


def _read_readyz_timeout_s() -> float:
    raw = os.environ.get("TURTLE_READYZ_TIMEOUT_S")
    if raw is None or not raw.strip():
        # Unset/empty is normal, not an error — no warning.
        return _READYZ_TIMEOUT_DEFAULT
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "TURTLE_READYZ_TIMEOUT_S=%r is not a valid number; falling back to "
            "the %.1fs default.",
            raw,
            _READYZ_TIMEOUT_DEFAULT,
        )
        return _READYZ_TIMEOUT_DEFAULT
    # asyncio.wait_for(timeout=0 or negative) raises TimeoutError instantly,
    # which would make /readyz a permanent 503 in cloud mode; a non-finite
    # value (inf/nan) either never times out (nan — the exact opposite of
    # this module's "a hanging backend doesn't make /readyz hang" promise)
    # or is simply nonsensical (inf). math.isfinite(nan) is False, so this
    # one check catches both non-finite cases; `value > 0` catches zero and
    # negatives WITHOUT relying on `value <= 0`, since every comparison
    # against nan (including `nan <= 0`) is False and would silently let
    # nan slip through a naive check.
    if not math.isfinite(value) or not value > 0:
        logger.warning(
            "TURTLE_READYZ_TIMEOUT_S=%r must be a finite, strictly positive "
            "number; falling back to the %.1fs default.",
            raw,
            _READYZ_TIMEOUT_DEFAULT,
        )
        return _READYZ_TIMEOUT_DEFAULT
    return value


READYZ_TIMEOUT_S = _read_readyz_timeout_s()


async def probe_postgres(timeout: Optional[float] = None) -> bool:
    """Run ``SELECT 1`` on the shared asyncpg pool, bounded by *timeout*.

    Never raises: a missing DATABASE_URL (CloudBackendUnavailable), a real
    connection failure, and exceeding the timeout budget all resolve to
    False — /readyz only needs a boolean per backend.
    """
    budget = READYZ_TIMEOUT_S if timeout is None else timeout

    async def _check() -> None:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")

    try:
        await asyncio.wait_for(_check(), timeout=budget)
        return True
    except Exception:
        return False


async def probe_redis(timeout: Optional[float] = None) -> bool:
    """Run ``PING`` on the shared async Redis client, bounded by *timeout*.

    Same never-raises contract as probe_postgres() above.
    """
    budget = READYZ_TIMEOUT_S if timeout is None else timeout

    async def _check() -> None:
        client = await get_redis_client()
        await client.ping()

    try:
        await asyncio.wait_for(_check(), timeout=budget)
        return True
    except Exception:
        return False
