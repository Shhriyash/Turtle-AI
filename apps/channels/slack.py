"""
apps/channels/slack.py
-----------------------
F3: Slack channel adapter — Events API webhook mode.

Endpoint: POST /channels/slack/events
  1. URL verification challenge (Slack sends {"type":"url_verification"}).
  2. HMAC-SHA256 request signature validated on every event.
  3. app_mention and direct message events dispatched to Turtle pipeline.
  4. Replies posted as threaded messages via chat.postMessage.

Slack app setup:
  - Create an app at api.slack.com/apps
  - Enable Event Subscriptions → set Request URL to https://<host>/channels/slack/events
  - Subscribe to bot events: app_mention, message.im
  - Install to workspace → copy Bot User OAuth Token

Required env vars:
  SLACK_BOT_TOKEN      xoxb-...
  SLACK_SIGNING_SECRET  (found in Basic Information → App Credentials)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

from apps.channels import TurtleEvent, TurtleResponse, dispatch_event
from core.config import settings
from core.identity import identity_manager

router = APIRouter(prefix="/channels/slack", tags=["slack"])

_SLACK_API_BASE = "https://slack.com/api"


def _signing_secret() -> str:
    return settings.slack_signing_secret.get_secret_value() if settings.slack_signing_secret else ""


def _bot_token() -> str:
    return settings.slack_bot_token.get_secret_value() if settings.slack_bot_token else ""


def _verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    """Validate X-Slack-Signature per Slack docs (v0 HMAC-SHA256)."""
    secret = _signing_secret()
    if not secret:
        return True  # dev mode

    # Reject replays older than 5 minutes
    try:
        if abs(time.time() - float(timestamp)) > 300:
            return False
    except ValueError:
        return False

    base = f"v0:{timestamp}:".encode() + body
    expected = "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


async def _post_slack_message(channel: str, text: str, thread_ts: str | None = None) -> None:
    """Send a message via Slack Web API."""
    token = _bot_token()
    if not token:
        print(f"[Slack] No bot token — skipping send. Message: {text!r}")
        return

    payload: dict = {"channel": channel, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_SLACK_API_BASE}/chat.postMessage",
            json=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=10.0,
        )
    data = resp.json()
    if not data.get("ok"):
        print(f"[Slack] chat.postMessage failed: {data.get('error')}")


@router.post("/events")
async def slack_events(request: Request):
    """Receive Slack Events API payloads."""
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if not _verify_slack_signature(body, timestamp, signature):
        raise HTTPException(status_code=403, detail="Invalid Slack signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event_type = payload.get("type")

    # 1. URL verification challenge
    if event_type == "url_verification":
        return {"challenge": payload.get("challenge")}

    # 2. Event callback
    if event_type != "event_callback":
        return Response(status_code=200)

    event = payload.get("event", {})
    sub_type = event.get("type", "")

    # Handle app_mention and direct messages (message.im)
    if sub_type not in ("app_mention", "message"):
        return Response(status_code=200)

    # Ignore bot messages to avoid loops
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return Response(status_code=200)

    slack_user_id: str = event.get("user", "")
    text: str = event.get("text", "").strip()
    channel_id: str = event.get("channel", "")
    thread_ts: str = event.get("thread_ts") or event.get("ts", "")
    event_ts: str = event.get("ts", "")

    # Strip bot mention from text (<@BOTID> prefix)
    import re
    text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()

    if not text:
        return Response(status_code=200)

    # Acknowledge immediately (Slack requires < 3 s)
    # Processing runs as a background task
    import asyncio

    async def _process():
        user_id = await identity_manager.resolve_user("slack", slack_user_id)
        turtle_event = TurtleEvent(
            user_id=user_id,
            channel="slack",
            modality="text",
            content=text,
            message_id=event_ts,
            thread_id=thread_ts,
        )
        response: TurtleResponse = await dispatch_event(turtle_event)
        await _post_slack_message(channel_id, response.content, thread_ts=thread_ts)

    asyncio.create_task(_process())
    return Response(status_code=200)
