"""
test/websocket_cloud_construction_test.py
------------------------------------------
Regression test for the class of bug that broke every /ws connection on the
first real Vercel deploy: a class picks the correct Postgres/no-op backend in
cloud mode (settings.is_cloud), but something upstream of that branch still
touches the local filesystem unconditionally (an eager `path.mkdir()` while
computing a default, a helper that assumes local disk) and crashes with
"OSError: Read-only file system" before the branch is ever reached.

Found live, one at a time, across three separate deploys:
  1. core/personal_memory_store.py -> core/paths.py's personal_memory_dir()
  2. core/task_history.py's TaskHistoryStore (never had a cloud backend)
  3. rag/system/complete_rag.py's TurtleRAGSystem

This test drives the REAL apps/turtle_server.py::websocket_endpoint coroutine
end-to-end in cloud mode -- the same object-construction sequence that
crashed each time in production -- with only the network edges (Postgres,
auth) faked, so a fourth instance of this bug fails a local test run instead
of a live deploy.
"""
from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from core.config import settings


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

    Accepts the connection, records every JSON frame sent, then reports a
    disconnect on the first receive() so the handler runs its full setup and
    teardown without blocking on real client traffic.
    """

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._receive_called = False

    async def accept(self) -> None:
        return None

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def receive(self) -> dict:
        self._receive_called = True
        return {"type": "websocket.disconnect"}

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        return None


class WebSocketCloudConstructionTest(unittest.TestCase):
    """Exercises the full /ws setup+teardown sequence under settings.is_cloud."""

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
            new=AsyncMock(return_value="usr_cloud_smoke_test"),
        )
        self._auth_patcher.start()
        self._patchers.append(self._auth_patcher)

        # Process-wide singletons in core/storage/factory.py must not leak a
        # local-mode instance from an earlier (non-cloud) test in this run.
        import core.storage.factory as factory
        factory._rate_limiter = None
        factory._channel_gate_buffer = None
        factory._CLOUD_VECTOR_STORE_SINGLETONS.clear()

    def tearDown(self) -> None:
        for p in self._patchers:
            p.stop()
        settings.deploy_mode = self._orig_deploy_mode

    def test_full_connect_and_disconnect_does_not_crash(self) -> None:
        ws = _FakeWebSocket()
        asyncio.run(self._turtle_server.websocket_endpoint(ws))

        statuses = [f.get("status") for f in ws.sent if f.get("type") == "status"]
        self.assertIn("ready", statuses)


if __name__ == "__main__":
    unittest.main()
