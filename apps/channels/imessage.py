"""
apps/channels/imessage.py
--------------------------
F2: iMessage channel adapter via SendBlue cloud API.

Webhook flow:
  POST /channels/imessage
    ← SendBlue sends JSON payload with from_number, content, message_handle
    → 200 acknowledgement (reply sent async via SendBlue REST API)

Authentication:
  SendBlue signs each webhook with an HMAC-SHA256 signature in the
  X-SendBlue-Signature header.  Requests without a valid signature → 403.
  Signature check is skipped when SENDBLUE_API_KEY is not configured (dev).

Outbound reply:
  Turtle calls POST https://api.sendblue.co/api/send-message with the
  reply text to the originating number.

Required env vars:
  SENDBLUE_API_KEY
  SENDBLUE_API_SECRET
"""
from __future__ import annotations

import hashlib
import hmac
import json

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

from apps.channels import TurtleEvent, TurtleResponse, dispatch_event
from core.config import settings
from core.identity import identity_manager

router = APIRouter(prefix="/channels/imessage", tags=["imessage"])

_SENDBLUE_API_BASE = "https://api.sendblue.co/api"


def _get_api_creds() -> tuple[str, str]:
    key = settings.sendblue_api_key.get_secret_value() if settings.sendblue_api_key else ""
    secret = settings.sendblue_api_secret.get_secret_value() if settings.sendblue_api_secret else ""
    return key, secret


def _verify_sendblue_signature(body: bytes, signature: str) -> bool:
    """Validate X-SendBlue-Signature HMAC-SHA256."""
    _, secret = _get_api_creds()
    if not secret:
        # Dev no-op locally; FAIL CLOSED in cloud — see apps/channels/discord.py.
        # Accepting unsigned requests on a public webhook is unauthenticated
        # pipeline execution against an attacker-chosen tenant.
        return not settings.is_cloud

    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


async def _send_imessage_reply(to_number: str, text: str) -> None:
    """Send reply via SendBlue Messages API."""
    key, secret = _get_api_creds()
    if not (key and secret):
        print(f"[iMessage] Skipping send — SendBlue creds not configured. Reply: {text!r}")
        return

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_SENDBLUE_API_BASE}/send-message",
            json={"number": to_number, "content": text},
            headers={"sb-api-key-id": key, "sb-api-secret-key": secret},
            timeout=10.0,
        )
    if resp.status_code not in (200, 201):
        print(f"[iMessage] SendBlue send failed: {resp.status_code} {resp.text}")


@router.post("")
async def imessage_webhook(request: Request):
    """Receive an inbound iMessage from SendBlue."""
    body = await request.body()
    signature = request.headers.get("X-SendBlue-Signature", "")

    if not _verify_sendblue_signature(body, signature):
        raise HTTPException(status_code=403, detail="Invalid SendBlue signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    from_number: str = payload.get("from_number") or payload.get("number", "")
    content: str = payload.get("content", "").strip()
    message_handle: str = payload.get("message_handle", "")

    if not content:
        return Response(status_code=200)

    user_id = await identity_manager.resolve_user("imessage", from_number)

    event = TurtleEvent(
        user_id=user_id,
        channel="imessage",
        modality="text",
        content=content,
        message_id=message_handle,
    )
    response: TurtleResponse = await dispatch_event(event)

    await _send_imessage_reply(from_number, response.content)
    return Response(status_code=200)
