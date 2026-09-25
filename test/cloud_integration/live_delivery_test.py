"""
Real-Redis roundtrip test for core/storage/cloud/live_delivery.py: publish a
frame on a user's live channel and receive it via a real pub/sub
subscription (async client), proving both the sync publish path and the
async subscribe path against a real Redis.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from test.cloud_integration.conftest import run_async

pytestmark = pytest.mark.cloud


def test_publish_and_subscribe_roundtrip() -> None:
    from core.storage.cloud.live_delivery import (
        open_user_subscription,
        publish_routine_frame_sync,
    )

    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    frame = {"type": "routine_notice", "text": "time to stand up"}

    async def _run():
        pubsub = await open_user_subscription(user_id)
        try:
            # Drain the subscribe confirmation message.
            await pubsub.get_message(timeout=2)

            # publish_routine_frame_sync is sync — proceed via to_thread so
            # this stays a single event loop, matching how the app calls it.
            delivered = await asyncio.to_thread(publish_routine_frame_sync, user_id, frame)
            assert delivered == 1

            message = await pubsub.get_message(timeout=2)
            deadline_retries = 5
            while message is None and deadline_retries > 0:
                message = await pubsub.get_message(timeout=1)
                deadline_retries -= 1
            assert message is not None
            assert message["type"] == "message"

            import json

            assert json.loads(message["data"]) == frame
        finally:
            await pubsub.unsubscribe()
            await pubsub.aclose()

    run_async(_run())
