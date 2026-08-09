"""
apps/channels/whatsapp.py
--------------------------
F1: WhatsApp channel adapter via Twilio Cloud API.

Webhook flow:
  POST /channels/whatsapp
    ← Twilio sends form-encoded payload (From, Body, MessageSid, ...)
    → 200 empty TwiML (reply sent async via REST API)

Signature verification:
  Every request is validated with the Twilio request validator using
  TWILIO_AUTH_TOKEN.  Invalid signatures → 403.

Idempotency:
  MessageSid is used as the idempotency key.  Duplicate deliveries within
  60 s return the cached reply without re-running the pipeline.

Outbound reply:
  Turtle calls the Twilio Messages REST API to send a reply to the same
  From number via the configured TWILIO_WHATSAPP_NUMBER.

Required env vars (set in .env):
  TWILIO_ACCOUNT_SID
  TWILIO_AUTH_TOKEN
  TWILIO_WHATSAPP_NUMBER   e.g. whatsapp:+14155238886
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Annotated

import httpx
from fastapi import APIRouter, Form, Header, HTTPException, Request, Response

from apps.channels import TurtleEvent, dispatch_event, TurtleResponse
from core.config import settings
from core.identity import identity_manager

router = APIRouter(prefix="/channels/whatsapp", tags=["whatsapp"])

# In-process idempotency cache: {message_sid: (reply_text, timestamp)}
_IDEMPOTENCY_CACHE: dict[str, tuple[str, float]] = {}
_IDEMPOTENCY_TTL_S = 60.0


def _get_secret() -> str:
    if settings.twilio_auth_token:
        return settings.twilio_auth_token.get_secret_value()
    return ""


def _verify_twilio_signature(request_url: str, post_params: dict[str, str], signature: str) -> bool:
    """Validate X-Twilio-Signature per Twilio docs."""
    secret = _get_secret()
    if not secret:
        # Dev no-op locally; FAIL CLOSED in cloud. An unsigned-accepting public
        # webhook lets anyone drive the whole pipeline with a spoofed sender id
        # — unauthenticated LLM execution plus writes into an arbitrary tenant's
        # memory. The shipped container sets TURTLE_DEPLOY=cloud (Dockerfile)
        # and ships no channel secrets, so "no secret configured" is exactly the
        # production default. Mirrors apps/channels/discord.py.
        return not settings.is_cloud

    # Build validation string: URL + sorted POST params joined
    s = request_url
    for key in sorted(post_params.keys()):
        s += key + post_params[key]

    expected = hmac.new(secret.encode(), s.encode(), hashlib.sha1).digest()
    import base64
    expected_b64 = base64.b64encode(expected).decode()
    return hmac.compare_digest(expected_b64, signature)


def _check_idempotency(message_sid: str) -> str | None:
    """Return cached reply if this MessageSid was already processed."""
    now = time.monotonic()
    entry = _IDEMPOTENCY_CACHE.get(message_sid)
    if entry and (now - entry[1]) < _IDEMPOTENCY_TTL_S:
        return entry[0]
    return None


def _store_idempotency(message_sid: str, reply: str) -> None:
    _IDEMPOTENCY_CACHE[message_sid] = (reply, time.monotonic())
    # Evict old entries to bound memory
    now = time.monotonic()
    expired = [k for k, (_, ts) in _IDEMPOTENCY_CACHE.items() if now - ts > _IDEMPOTENCY_TTL_S]
    for k in expired:
        _IDEMPOTENCY_CACHE.pop(k, None)


async def _send_whatsapp_reply(to: str, body: str) -> None:
    """Send a reply via Twilio Messages REST API."""
    account_sid = settings.twilio_account_sid
    auth_token = settings.twilio_auth_token
    from_number = settings.twilio_whatsapp_number

    if not (account_sid and auth_token and from_number):
        print(f"[WhatsApp] Skipping send — Twilio creds not configured. Reply would be: {body!r}")
        return

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid.get_secret_value()}/Messages.json"
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            url,
            data={"From": f"whatsapp:{from_number}", "To": to, "Body": body},
            auth=(account_sid.get_secret_value(), auth_token.get_secret_value()),
            timeout=10.0,
        )
    if resp.status_code not in (200, 201):
        print(f"[WhatsApp] Twilio send failed: {resp.status_code} {resp.text}")


@router.post("")
async def whatsapp_webhook(
    request: Request,
    # Twilio form fields
    From: Annotated[str, Form()] = "",
    Body: Annotated[str, Form()] = "",
    MessageSid: Annotated[str, Form()] = "",
    x_twilio_signature: Annotated[str, Header(alias="X-Twilio-Signature")] = "",
):
    """Receive an inbound WhatsApp message from Twilio."""
    form = await request.form()
    post_params = {k: str(v) for k, v in form.items()}

    # Signature check
    url = str(request.url)
    if not _verify_twilio_signature(url, post_params, x_twilio_signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    if not Body.strip():
        return Response(content="<Response/>", media_type="application/xml")

    # Idempotency
    cached = _check_idempotency(MessageSid)
    if cached:
        await _send_whatsapp_reply(From, cached)
        return Response(content="<Response/>", media_type="application/xml")

    # Resolve user identity
    user_id = await identity_manager.resolve_user("whatsapp", From)

    # Dispatch
    event = TurtleEvent(
        user_id=user_id,
        channel="whatsapp",
        modality="text",
        content=Body.strip(),
        message_id=MessageSid,
    )
    response: TurtleResponse = await dispatch_event(event)
    reply_text = response.content

    # Cache + send
    _store_idempotency(MessageSid, reply_text)
    await _send_whatsapp_reply(From, reply_text)

    # Return empty TwiML — reply already sent via REST
    return Response(content="<Response/>", media_type="application/xml")
