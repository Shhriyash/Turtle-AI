"""
apps/channels/telegram_webhook.py
------------------------------------
F6: Telegram channel adapter — webhook mode (serverless-shaped).

apps/channels/telegram_gateway.py's long-polling Application.updater.start_polling()
holds a persistent HTTP long-poll connection open — that cannot survive a
serverless cold start (the connection dies with the invocation, and the next
invocation would have to re-poll from scratch, missing anything sent in
between). This module is the replacement: Telegram POSTs each Update to a
single HTTPS endpoint instead, the same shape as apps/channels/discord.py's
Interactions webhook and for the identical reason.

Zero extra dependency, matching discord.py's own posture: the raw Update JSON
is parsed by hand (no python-telegram-bot Update/Bot objects, which need
their own async init lifecycle this module has no use for) and replies are
sent via a plain httpx POST to the Bot API. The two pure string-formatting
helpers that don't touch any live connection — markdown_to_telegram_html and
_normalize_dashes — are imported from telegram_gateway.py rather than
duplicated, so the two paths render identically.

Endpoint: POST /channels/telegram/webhook

Setup (once, after deploying):
  1. Set TELEGRAM_WEBHOOK_SECRET to a random value (also used as the secret
     Telegram echoes back on every request — see _verify_secret_token).
  2. Call register_telegram_webhook() (or POST the Bot API's setWebhook
     yourself) pointing at https://<host>/channels/telegram/webhook with
     secret_token=<TELEGRAM_WEBHOOK_SECRET>.

Required env vars:
  TELEGRAM_BOT_TOKEN        Bot token from @BotFather (shared with the local
                             gateway — same bot, different transport).
  TELEGRAM_WEBHOOK_SECRET   Verified against Telegram's
                             X-Telegram-Bot-Api-Secret-Token header on every
                             request (Telegram docs: this header is set to
                             exactly the secret_token passed to setWebhook).
"""
from __future__ import annotations

import hmac

import httpx
from fastapi import APIRouter, HTTPException, Request

from apps.channels import TurtleEvent, TurtleResponse, dispatch_event
from apps.channels.telegram_gateway import _normalize_dashes, markdown_to_telegram_html
from core.config import settings
from core.identity import CHANNEL_INVITE_ONLY_MESSAGE, resolve_channel_user

router = APIRouter(prefix="/channels/telegram", tags=["telegram"])

_TELEGRAM_API_BASE = "https://api.telegram.org"
# Telegram's outbound message hard limit is 4096 chars (matches telegram_gateway.py).
_MAX_REPLY_CHARS = 4000

# Lazy, in-process cache of getMe()'s result (our own bot id/username) — used
# to detect an @mention of OUR bot in a group message. Not required for DMs.
# A cold-start-per-invocation cache is still worth it: cheap correctness win
# on any invocation that handles more than one group message.
_bot_info_cache: dict[str, str] | None = None


def _bot_token() -> str:
    return settings.telegram_bot_token.get_secret_value() if settings.telegram_bot_token else ""


def _webhook_secret() -> str:
    return (
        settings.telegram_webhook_secret.get_secret_value()
        if settings.telegram_webhook_secret
        else ""
    )


def _verify_secret_token(header_value: str) -> bool:
    """Validate Telegram's X-Telegram-Bot-Api-Secret-Token header.

    When no secret is configured we FAIL CLOSED unless TURTLE_DEV_ANON=1 AND
    not cloud — matching apps/channels/discord.py's own posture for the exact
    same reason: a public webhook accepting unauthenticated requests would
    let anyone drive the pipeline with a spoofed chat/user id.
    """
    expected = _webhook_secret()
    if not expected:
        return settings.dev_anon and not settings.is_cloud
    return hmac.compare_digest(header_value or "", expected)


async def _get_bot_username() -> str:
    global _bot_info_cache
    if _bot_info_cache is not None:
        return _bot_info_cache.get("username", "")
    token = _bot_token()
    if not token:
        return ""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{_TELEGRAM_API_BASE}/bot{token}/getMe", timeout=10.0)
        data = resp.json()
        if resp.status_code < 400 and data.get("ok"):
            username = str(data.get("result", {}).get("username", ""))
            _bot_info_cache = {"username": username}
            return username
    except Exception as e:
        print(f"LOG: telegram webhook getMe failed: {e}")
    return ""


def _extract_message_text(message: dict, bot_username: str) -> str:
    """Strip a leading @mention of our bot, mirroring
    telegram_gateway._extract_message_text but on a raw dict payload
    instead of a python-telegram-bot Message object."""
    text = message.get("text") or message.get("caption") or ""
    if not text:
        return ""
    if bot_username:
        needle = f"@{bot_username}"
        low = text.lower()
        low_needle = needle.lower()
        if low.startswith(low_needle):
            text = text[len(needle):]
        else:
            idx = low.find(low_needle)
            if idx != -1:
                text = text[:idx] + text[idx + len(needle):]
    return text.strip()


def _should_handle(chat_type: str, is_mention: bool, is_bot_author: bool) -> bool:
    """Mirrors telegram_gateway._should_handle exactly: not a bot author AND
    (a private DM OR the bot was @mentioned in a group)."""
    if is_bot_author:
        return False
    if chat_type == "private":
        return True
    return bool(is_mention)


async def _send_reply(chat_id: int | str, text: str) -> None:
    token = _bot_token()
    if not token:
        print(f"LOG: [Telegram webhook] No bot token — cannot deliver reply: {text!r}")
        return
    url = f"{_TELEGRAM_API_BASE}/bot{token}/sendMessage"
    html_reply = markdown_to_telegram_html(text)[:_MAX_REPLY_CHARS]
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url, json={"chat_id": chat_id, "text": html_reply, "parse_mode": "HTML"}, timeout=15.0
            )
        if resp.status_code >= 400:
            # HTML the converter produced was rejected (unbalanced entity,
            # same failure mode telegram_gateway.py guards against) — resend
            # as plain text so the user still gets the message.
            print(f"LOG: [Telegram webhook] HTML send rejected ({resp.status_code}); resending plain")
            plain = _normalize_dashes(text)[:_MAX_REPLY_CHARS]
            async with httpx.AsyncClient() as client:
                await client.post(url, json={"chat_id": chat_id, "text": plain}, timeout=15.0)
    except Exception as e:
        print(f"LOG: [Telegram webhook] send failed: {e}")


@router.post("/webhook")
async def telegram_webhook(request: Request):
    """Telegram Bot API webhook — the single URL registered via setWebhook."""
    secret_header = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not _verify_secret_token(secret_header):
        raise HTTPException(status_code=401, detail="Invalid secret token")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    message = payload.get("message") or payload.get("edited_message")
    if not message:
        # A non-message update (channel post, poll answer, etc.) — Telegram
        # only requires a 200 to stop retrying; nothing to act on.
        return {"ok": True}

    chat = message.get("chat") or {}
    user = message.get("from") or {}
    chat_type = str(chat.get("type", ""))
    is_bot_author = bool(user.get("is_bot", False))

    bot_username = await _get_bot_username()
    raw_text = message.get("text") or message.get("caption") or ""
    is_mention = bool(bot_username) and (f"@{bot_username}".lower() in raw_text.lower())

    print(
        f"LOG: telegram webhook message from {user.get('id')} chat_type={chat_type} "
        f"is_bot={is_bot_author} mention={is_mention} content_len={len(raw_text)}",
        flush=True,
    )

    if not _should_handle(chat_type, is_mention, is_bot_author):
        return {"ok": True}

    text = _extract_message_text(message, bot_username)
    if not text:
        return {"ok": True}

    try:
        tg_user_id = str(user.get("id", ""))
        user_id = await resolve_channel_user("telegram", tg_user_id)
        if user_id is None:
            # TURTLE_CHANNEL_SIGNUP=invite and this sender is unknown —
            # reply with the invite message and mint nothing.
            await _send_reply(chat.get("id"), CHANNEL_INVITE_ONLY_MESSAGE)
            return {"ok": True}
        sender_name = (
            str(user.get("first_name") or "").strip()
            or str(user.get("username") or "").strip()
        )
        turtle_event = TurtleEvent(
            user_id=user_id,
            channel="telegram",
            modality="text",
            content=text,
            message_id=str(message.get("message_id", "")),
            thread_id=str(chat.get("id", "")),
            sender_name=sender_name,
            channel_user_id=tg_user_id,
            is_private=(chat_type == "private"),
        )
        response: TurtleResponse = await dispatch_event(turtle_event)
        await _send_reply(chat.get("id"), response.content or "…")
        print(f"LOG: telegram webhook replied to {tg_user_id} ({len(response.content or '')} chars)")
    except Exception as e:
        print(f"LOG: telegram webhook processing failed: {e}")
        try:
            await _send_reply(chat.get("id"), "Sorry — something went wrong handling that.")
        except Exception:
            pass

    return {"ok": True}


async def register_telegram_webhook() -> None:
    """Register the webhook URL with Telegram (setWebhook). Best-effort and
    log-only; NOT called automatically on startup — invoke it from a script
    or a one-off admin action, same posture as
    apps/channels/discord.py::register_slash_commands.
    """
    token = _bot_token()
    secret = _webhook_secret()
    if not token or not secret:
        print("LOG: telegram register_telegram_webhook skipped (no bot token / no webhook secret)")
        return
    base = settings.public_base_url.rstrip("/")
    url = f"{base}/channels/telegram/webhook"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{_TELEGRAM_API_BASE}/bot{token}/setWebhook",
                json={
                    "url": url,
                    "secret_token": secret,
                    "allowed_updates": ["message", "edited_message"],
                    "drop_pending_updates": True,
                },
                timeout=15.0,
            )
        data = resp.json()
        if resp.status_code >= 400 or not data.get("ok"):
            print(f"LOG: telegram setWebhook failed: {resp.status_code} {data}")
        else:
            print(f"LOG: telegram webhook registered at {url}")
    except Exception as e:
        print(f"LOG: telegram setWebhook error: {e}")
