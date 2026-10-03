"""
test/wp3c_failure_posture_test.py
----------------------------------
WP3.C (ledger 3.6, 3.9): failure posture + the query-shape half of the
pending-finalization sweep.

3.6 governing principle (see the WP brief / this module's targets): fail
CLOSED when proceeding causes an irreversible side effect (idempotency —
untouched here), fail OPEN when proceeding merely leaves a bounded,
recoverable limit unenforced (the rate limiter, the channel gate) while
failing closed would deny service to everyone.

These tests drive the REAL apps/turtle_server.py::websocket_endpoint
coroutine end-to-end in cloud mode (same harness style as
test/websocket_cloud_construction_test.py), with only the network edges
faked, so "the connection survives a Redis outage" is proven at the
user-visible layer, not just inside the Redis backend module.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

import redis as redis_module

from core.config import settings


# ---------------------------------------------------------------------------
# Shared fakes (mirrors test/websocket_cloud_construction_test.py)
# ---------------------------------------------------------------------------

class _FakeSyncCursor:
    def fetchone(self):
        return None

    def fetchall(self):
        return []

    @property
    def rowcount(self):
        return 0


class _FakeSyncConn:
    def execute(self, sql: str, params: Any = None):
        return _FakeSyncCursor()


class _FakeSyncConnCtx:
    def __enter__(self):
        return _FakeSyncConn()

    def __exit__(self, *exc):
        return False


class _FakeSyncPool:
    def connection(self):
        return _FakeSyncConnCtx()


class _FakeAsyncConn:
    async def execute(self, sql: str, *params: Any):
        return "OK"

    async def fetchrow(self, sql: str, *params: Any):
        return None

    async def fetch(self, sql: str, *params: Any):
        return []


class _FakeAsyncConnCtx:
    async def __aenter__(self):
        return _FakeAsyncConn()

    async def __aexit__(self, *exc):
        return False


class _FakeAsyncPool:
    def acquire(self):
        return _FakeAsyncConnCtx()


class _FakeWebSocket:
    """Duck-typed stand-in for starlette.websockets.WebSocket.

    Plays back a scripted sequence of receive() frames (defaulting to an
    immediate disconnect), and records every JSON frame sent.
    """

    def __init__(self, frames: list[dict] | None = None) -> None:
        self.sent: list[dict] = []
        self._frames = list(frames or [])
        self._frames.append({"type": "websocket.disconnect"})
        self.closed_with: tuple[int, str | None] | None = None

    async def accept(self) -> None:
        return None

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def receive(self) -> dict:
        return self._frames.pop(0)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed_with = (code, reason)


class _CloudWsHarness(unittest.TestCase):
    """Common cloud-mode /ws setup, shared by both tests below."""

    def setUp(self) -> None:
        self._orig_deploy_mode = settings.deploy_mode
        settings.deploy_mode = "cloud"
        self.assertTrue(settings.is_cloud)

        sync_pool = _FakeSyncPool()
        async_pool = _FakeAsyncPool()

        sync_targets = [
            "core.storage.cloud.personal_memory_store.get_pg_sync_pool",
            "core.storage.cloud.journal_store.get_pg_sync_pool",
            "core.storage.cloud.confirmation_state_store.get_pg_sync_pool",
            "core.storage.cloud.rag_session_staging_store.get_pg_sync_pool",
            "core.storage.cloud.pgvector_store.get_pg_sync_pool",
        ]
        async_targets = [
            "core.storage.cloud.postgres_store.get_pg_pool",
            "core.storage.cloud.pgvector_store.get_pg_pool",
        ]
        self._patchers = []
        for target in sync_targets:
            p = patch(target, return_value=sync_pool)
            p.start()
            self._patchers.append(p)
        for target in async_targets:
            p = patch(target, new=AsyncMock(return_value=async_pool))
            p.start()
            self._patchers.append(p)

        import apps.turtle_server as turtle_server

        self._turtle_server = turtle_server
        self._auth_patcher = patch.object(
            turtle_server, "authenticate_websocket",
            new=AsyncMock(return_value="usr_wp3c_smoke_test"),
        )
        self._auth_patcher.start()
        self._patchers.append(self._auth_patcher)

        import core.storage.factory as factory
        factory._rate_limiter = None
        factory._channel_gate_buffer = None
        factory._CLOUD_VECTOR_STORE_SINGLETONS.clear()

    def tearDown(self) -> None:
        for p in self._patchers:
            p.stop()
        settings.deploy_mode = self._orig_deploy_mode


class RateLimiterOutageSurvivesConnectionTest(_CloudWsHarness):
    """Ledger 3.6(a): the actual user-visible claim -- a live Redis outage
    during the per-message rate check must not take the WebSocket down."""

    def test_ws_connection_survives_rate_limiter_redis_outage(self) -> None:
        text_frame = {"type": "websocket.receive", "text": json.dumps({
            "type": "text", "content": "hello turtle",
        })}
        ws = _FakeWebSocket(frames=[text_frame])

        # The driver's REAL error type, raised on every Redis call the
        # cloud rate limiter makes -- not a mocked CloudBackendUnavailable.
        outage_patcher = patch(
            "core.storage.cloud.redis_backends.get_redis_sync_client",
            side_effect=redis_module.exceptions.ConnectionError("connection refused"),
        )
        outage_patcher.start()
        self._patchers.append(outage_patcher)

        # Stub the turn pipeline itself out -- this test's claim is about
        # connection survival through the rate-limiter check, not the full
        # LLM turn.
        handle_text_patcher = patch.object(
            self._turtle_server, "_handle_text_message",
            new=AsyncMock(return_value=None),
        )
        handle_text_patcher.start()
        self._patchers.append(handle_text_patcher)

        asyncio.run(self._turtle_server.websocket_endpoint(ws))

        # The connection must have run to its natural disconnect, not been
        # torn down by ws.close(code=1008, reason="rate_limited") or by an
        # uncaught exception reaching the outer handler.
        self.assertIsNone(ws.closed_with)
        self._turtle_server._handle_text_message.assert_awaited_once()
        error_frames = [f for f in ws.sent if f.get("type") == "error"]
        self.assertEqual(error_frames, [])


class SessionStoreOutageDegradesGracefullyTest(_CloudWsHarness):
    """Ledger 3.6(d): start_or_restore() had no error handling and ran
    OUTSIDE the receive loop's try/except, so a DB error here used to
    propagate straight out of websocket_endpoint uncaught."""

    def test_session_store_failure_sends_degraded_frame_not_a_crash(self) -> None:
        ws = _FakeWebSocket()

        restore_patcher = patch(
            "core.session_store.SessionStore.start_or_restore",
            new=AsyncMock(side_effect=ConnectionError("db unreachable")),
        )
        restore_patcher.start()
        self._patchers.append(restore_patcher)

        # Must not raise out of websocket_endpoint.
        asyncio.run(self._turtle_server.websocket_endpoint(ws))

        statuses = [f.get("status") for f in ws.sent if f.get("type") == "status"]
        self.assertIn("degraded", statuses)
        degraded_frame = next(f for f in ws.sent if f.get("status") == "degraded")
        self.assertEqual(degraded_frame.get("reason"), "session_store_unavailable")
        # "ready" still follows -- the connection is fully usable afterward,
        # just without restored history.
        self.assertIn("ready", statuses)


class PendingFinalizationSweepIsolationTest(unittest.IsolatedAsyncioTestCase):
    """Ledger 3.9 (query-shape half), exercised through the real caller:
    SessionStore.list_pending_finalization_archives() against
    PostgresSessionStore -- the same pairing apps/turtle_server.py's /ws
    connect handler uses in cloud mode. Confirms a second user's pending
    session is never returned once user_id is threaded through, and that
    legacy unowned rows (pre-tenancy) stay swept, exactly matching the
    SQLite-backed equivalents in test/phase3_sessions_test.py."""

    async def asyncSetUp(self) -> None:
        from test.cloud_storage_test import _FakePgPool

        self.pool = _FakePgPool()
        patcher = patch(
            "core.storage.cloud.postgres_store.get_pg_pool",
            new_callable=AsyncMock,
            return_value=self.pool,
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    async def test_second_users_pending_session_is_not_returned(self) -> None:
        from core.session_store import SessionStore
        from core.storage import Session
        from core.storage.cloud.postgres_store import PostgresSessionStore

        backend = PostgresSessionStore()
        await backend.put(Session(
            session_id="mine",
            data={"user_id": "usr_a", "status": "pending_finalization", "messages": []},
        ))
        await backend.put(Session(
            session_id="theirs",
            data={"user_id": "usr_b", "status": "pending_finalization", "messages": []},
        ))
        await backend.put(Session(
            session_id="legacy",
            data={"status": "pending_finalization", "messages": []},  # no user_id
        ))

        store = SessionStore(backend, user_id="usr_a")
        pending_ids = {sid for sid, _ in await store.list_pending_finalization_archives()}

        self.assertEqual(pending_ids, {"mine", "legacy"})
        self.assertNotIn("theirs", pending_ids)


if __name__ == "__main__":
    unittest.main()
