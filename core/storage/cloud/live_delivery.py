"""
core/storage/cloud/live_delivery.py
--------------------------------------
Cloud (TURTLE_DEPLOY=cloud) cross-instance live delivery for routine notices,
replacing apps/turtle_server.py's _LIVE_SOCKETS/_APP_LOOP for the case that
matters most in serverless: the WebSocket that should receive a live push
and the process that fires the routine (a cron-tick request) are almost
certainly DIFFERENT instances. _LIVE_SOCKETS is a process-local dict, so it
cannot answer "does ANY instance have this user's socket open?" — only "does
*this* instance." Redis pub/sub is the shared registry every instance can
publish to and every instance's open WS can subscribe to.

Two halves, matching the two different call-site conventions already
established across this migration:

- publish_routine_frame_sync: SYNCHRONOUS (sync Redis client), because it's
  called from apps.turtle_server.deliver_routine_notice, which itself runs
  synchronously on whatever thread fired the routine (a cron-tick request
  via asyncio.to_thread, or — in local/single-instance cloud testing — the
  same posture the local RoutineScheduler's worker thread already had).
- open_user_subscription: ASYNC (redis.asyncio pub/sub), because it backs a
  long-lived listen loop integrated into the WebSocket handler's own event
  loop for the life of that connection.

Design choice: PUBLISH, not a durable stream. A routine notice reaching zero
subscribers (no instance currently holds that user's socket open) is not lost
— apps.turtle_server.deliver_routine_notice still falls back to the Postgres
outbox (core/routine_outbox.py) exactly as it does today, so a publish
reaching nobody degrades to "delivered on next connect" rather than "lost".
"""
from __future__ import annotations

import json
from typing import Any

from core.storage.cloud import get_redis_client, get_redis_sync_client


def _channel_name(user_id: str) -> str:
    return f"turtle:live:{user_id}"


def publish_routine_frame_sync(user_id: str, frame: dict[str, Any]) -> int:
    """Publish a routine frame to user_id's live channel.

    Returns the number of subscribers Redis delivered it to (0 means no
    instance currently has this user's socket open — the caller should fall
    back to the durable outbox). Never raises: a Redis hiccup must degrade to
    the outbox fallback, not crash the routine fire.
    """
    try:
        client = get_redis_sync_client()
        return int(client.publish(_channel_name(user_id), json.dumps(frame)))
    except Exception as e:
        print(f"LOG: live_delivery publish failed user={user_id}: {e}")
        return 0


async def open_user_subscription(user_id: str) -> Any:
    """Open (and return) an async pubsub object subscribed to user_id's live
    channel. Caller is responsible for iterating pubsub.listen() and for
    calling await pubsub.unsubscribe() / aclose() on teardown.
    """
    client = await get_redis_client()
    pubsub = client.pubsub()
    await pubsub.subscribe(_channel_name(user_id))
    return pubsub
