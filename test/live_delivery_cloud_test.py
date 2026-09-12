"""
test/live_delivery_cloud_test.py
-----------------------------------
Unit coverage for core/storage/cloud/live_delivery.py (Vercel migration
Phase 3's cross-instance routine delivery) and
apps/turtle_server.deliver_routine_notice's cloud branch. Uses fakeredis for
publish_routine_frame_sync (real pub/sub semantics) and mocks for the async
subscription helper / the WS relay task (no live Redis in this environment).
"""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis

from core.storage.cloud.live_delivery import (
    _channel_name,
    open_user_subscription,
    publish_routine_frame_sync,
)


class PublishRoutineFrameSyncTest(unittest.TestCase):
    def test_publish_with_no_subscribers_returns_zero(self) -> None:
        fake = fakeredis.FakeStrictRedis(decode_responses=True)
        with patch(
            "core.storage.cloud.live_delivery.get_redis_sync_client", return_value=fake
        ):
            count = publish_routine_frame_sync("usr_a", {"type": "routine"})
        self.assertEqual(count, 0)

    def test_publish_reaches_a_subscriber(self) -> None:
        fake = fakeredis.FakeStrictRedis(decode_responses=True)
        pubsub = fake.pubsub()
        pubsub.subscribe(_channel_name("usr_a"))
        pubsub.get_message(timeout=1)  # consume the subscribe confirmation

        with patch(
            "core.storage.cloud.live_delivery.get_redis_sync_client", return_value=fake
        ):
            count = publish_routine_frame_sync("usr_a", {"type": "routine", "message": "hi"})
        self.assertEqual(count, 1)

        msg = pubsub.get_message(timeout=1)
        self.assertEqual(msg["type"], "message")
        self.assertEqual(json.loads(msg["data"]), {"type": "routine", "message": "hi"})

    def test_publish_never_raises_on_redis_error(self) -> None:
        with patch(
            "core.storage.cloud.live_delivery.get_redis_sync_client",
            side_effect=RuntimeError("connection refused"),
        ):
            count = publish_routine_frame_sync("usr_a", {"type": "routine"})
        self.assertEqual(count, 0)


class OpenUserSubscriptionTest(unittest.IsolatedAsyncioTestCase):
    async def test_subscribes_to_the_correct_channel(self) -> None:
        fake_pubsub = AsyncMock()
        fake_client = MagicMock()
        fake_client.pubsub.return_value = fake_pubsub

        with patch(
            "core.storage.cloud.live_delivery.get_redis_client",
            new_callable=AsyncMock,
            return_value=fake_client,
        ):
            result = await open_user_subscription("usr_a")

        fake_pubsub.subscribe.assert_awaited_once_with(_channel_name("usr_a"))
        self.assertIs(result, fake_pubsub)


class DeliverRoutineNoticeCloudBranchTest(unittest.TestCase):
    """apps.turtle_server.deliver_routine_notice must try a Redis publish
    before falling back to the outbox, but ONLY when no local socket exists
    (a local socket is delivered to directly, no cross-instance help needed)
    and ONLY in cloud mode."""

    def setUp(self) -> None:
        import apps.turtle_server as server

        self.server = server
        self._orig_live_sockets = dict(server._LIVE_SOCKETS)
        self._orig_app_loop = server._APP_LOOP
        server._LIVE_SOCKETS.clear()
        server._APP_LOOP = None

    def tearDown(self) -> None:
        self.server._LIVE_SOCKETS.clear()
        self.server._LIVE_SOCKETS.update(self._orig_live_sockets)
        self.server._APP_LOOP = self._orig_app_loop

    def test_cloud_mode_publishes_before_stashing(self) -> None:
        with patch.object(self.server, "settings") as fake_settings:
            fake_settings.is_cloud = True
            with patch(
                "core.storage.cloud.live_delivery.publish_routine_frame_sync", return_value=1
            ) as fake_publish, patch.object(
                self.server, "_stash_pending_routine_notice"
            ) as fake_stash:
                result = self.server.deliver_routine_notice("usr_a", {"type": "routine"})

        self.assertTrue(result)
        fake_publish.assert_called_once_with("usr_a", {"type": "routine"})
        fake_stash.assert_not_called()

    def test_cloud_mode_falls_back_to_outbox_when_no_subscribers(self) -> None:
        with patch.object(self.server, "settings") as fake_settings:
            fake_settings.is_cloud = True
            with patch(
                "core.storage.cloud.live_delivery.publish_routine_frame_sync", return_value=0
            ), patch.object(self.server, "_stash_pending_routine_notice") as fake_stash:
                result = self.server.deliver_routine_notice("usr_a", {"type": "routine"})

        self.assertFalse(result)
        fake_stash.assert_called_once_with("usr_a", {"type": "routine"})

    def test_local_socket_present_skips_redis_publish(self) -> None:
        # A local socket exists in _LIVE_SOCKETS -> the same-process bridge
        # handles it; no cross-instance publish is needed.
        self.server._LIVE_SOCKETS["usr_a"] = {object()}
        self.server._APP_LOOP = MagicMock()
        self.server._APP_LOOP.is_closed.return_value = False

        with patch.object(self.server, "settings") as fake_settings:
            fake_settings.is_cloud = True
            with patch(
                "core.storage.cloud.live_delivery.publish_routine_frame_sync"
            ) as fake_publish, patch("asyncio.run_coroutine_threadsafe"):
                self.server.deliver_routine_notice("usr_a", {"type": "routine"})

        fake_publish.assert_not_called()

    def test_local_mode_never_calls_redis(self) -> None:
        with patch.object(self.server, "settings") as fake_settings:
            fake_settings.is_cloud = False
            with patch(
                "core.storage.cloud.live_delivery.publish_routine_frame_sync"
            ) as fake_publish, patch.object(self.server, "_stash_pending_routine_notice"):
                self.server.deliver_routine_notice("usr_a", {"type": "routine"})

        fake_publish.assert_not_called()


class RelayRedisLiveFramesTest(unittest.IsolatedAsyncioTestCase):
    async def test_relay_forwards_messages_to_the_socket(self) -> None:
        import apps.turtle_server as server

        async def _messages():
            yield {"type": "subscribe", "data": 1}  # confirmation, must be skipped
            yield {"type": "message", "data": json.dumps({"type": "routine", "n": 1})}
            raise asyncio.CancelledError()

        fake_pubsub = MagicMock()
        fake_pubsub.listen = _messages
        fake_pubsub.unsubscribe = AsyncMock()
        fake_pubsub.aclose = AsyncMock()

        fake_ws = MagicMock()

        with patch(
            "core.storage.cloud.live_delivery.open_user_subscription",
            new_callable=AsyncMock,
            return_value=fake_pubsub,
        ), patch.object(server, "_ws_send_json", new_callable=AsyncMock) as fake_send:
            with self.assertRaises(asyncio.CancelledError):
                await server._relay_redis_live_frames("usr_a", fake_ws)

        fake_send.assert_awaited_once_with(fake_ws, {"type": "routine", "n": 1})
        fake_pubsub.unsubscribe.assert_awaited_once()
        fake_pubsub.aclose.assert_awaited_once()

    async def test_relay_exits_quietly_on_subscribe_failure(self) -> None:
        import apps.turtle_server as server

        with patch(
            "core.storage.cloud.live_delivery.open_user_subscription",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            await server._relay_redis_live_frames("usr_a", MagicMock())  # must not raise


if __name__ == "__main__":
    unittest.main()
