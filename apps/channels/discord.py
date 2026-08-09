"""
apps/channels/discord.py
------------------------
F6: Discord channel adapter — Interactions Endpoint (webhook) mode.

This is the ZERO-extra-dependency path. Discord posts every interaction (slash
command, message command, ping) to a single HTTPS endpoint; we verify the
Ed25519 request signature with `cryptography` (already a dependency — no
discord.py, no PyNaCl needed) and reply.

TURTLE is a registered BOT APPLICATION. This adapter NEVER acts as a user
account (a "self-bot"), which violates Discord's Terms of Service. Only the
bot path is implemented here.

Endpoint: POST /channels/discord  (this exact URL is the "Interactions
Endpoint URL" you paste into the Discord Developer Portal).

Flow:
  1. Discord signs every request with Ed25519. We verify
     X-Signature-Ed25519 over (X-Signature-Timestamp + raw body). Discord
     REQUIRES a 401 on a bad signature (it uses a deliberately-bad probe when
     you save the endpoint URL; anything but 401 fails validation).
  2. type==1 (PING) -> {"type": 1} (PONG) immediately.
  3. type==2 (APPLICATION_COMMAND): you have 3 seconds to ACK. We cannot run
     the Turtle pipeline inline, so we return {"type": 5} (DEFERRED —
     "Turtle is thinking…") right away and finish the work on a background
     task that PATCHes the original interaction response with the real reply.

Discord app setup (Developer Portal → https://discord.com/developers/applications):
  - General Information → "Public Key"        -> DISCORD_PUBLIC_KEY
  - Bot tab → "Reset Token" / copy Bot Token  -> DISCORD_BOT_TOKEN
  - General Information → "Application ID"     -> DISCORD_APPLICATION_ID
  - General Information → set "Interactions Endpoint URL" to
        https://<host>/channels/discord
    (Discord immediately probes it with a signed PING + a bad-signature probe;
    both the PONG and the 401 above are required for it to accept the URL.)
  - Register the /turtle slash command once via register_slash_commands().

Required env vars:
  DISCORD_PUBLIC_KEY       hex-encoded Ed25519 public key (signature verify)
  DISCORD_BOT_TOKEN        Bot token   (command registration only)
  DISCORD_APPLICATION_ID   Application (client) id
"""
from __future__ import annotations

import asyncio
import json

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi import APIRouter, HTTPException, Request

from apps.channels import TurtleEvent, TurtleResponse, dispatch_event
from core.config import settings
from core.identity import identity_manager

router = APIRouter(prefix="/channels/discord", tags=["discord"])

_DISCORD_API_BASE = "https://discord.com/api/v10"

# Discord's outbound message hard limit is 2000 chars; keep headroom.
_MAX_REPLY_CHARS = 1900

# Interaction type + response type constants (Discord Interactions API).
_INTERACTION_PING = 1
_INTERACTION_APPLICATION_COMMAND = 2
_RESPONSE_PONG = 1
_RESPONSE_CHANNEL_MESSAGE = 4
_RESPONSE_DEFERRED_CHANNEL_MESSAGE = 5

# Fallback strong-reference set for the deferred follow-up tasks, in case
# core.worker.track_task is unavailable for some reason (keeps parity with the
# event loop's weak task set so a follow-up is never GC'd mid-flight).
_PENDING_TASKS: set[asyncio.Task] = set()


def _public_key() -> str:
    return settings.discord_public_key.get_secret_value() if settings.discord_public_key else ""


def _bot_token() -> str:
    return settings.discord_bot_token.get_secret_value() if settings.discord_bot_token else ""


def _application_id() -> str:
    return settings.discord_application_id or ""


def _verify_discord_signature(body: bytes, signature_hex: str, timestamp: str) -> bool:
    """Validate the Ed25519 request signature Discord sends on every interaction.

    Signature is over (timestamp + raw body), verified against the app's
    hex-encoded public key. When no public key is configured we no-op like the
    other channel adapters — but ONLY in local/dev mode. In cloud we fail CLOSED
    (a public webhook that accepts unsigned requests would let anyone drive the
    pipeline with a spoofed user id), so a missing key rejects rather than admits.
    """
    pub = _public_key()
    if not pub:
        return not settings.is_cloud  # dev no-op locally; fail closed in cloud
    try:
        verify_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub))
        verify_key.verify(bytes.fromhex(signature_hex), timestamp.encode() + body)
        return True
    except (InvalidSignature, ValueError):
        return False


def _track(task_obj: asyncio.Task) -> None:
    """Retain a strong ref to a detached follow-up task so it isn't GC'd."""
    try:
        from core.worker import track_task
        track_task(task_obj)
    except Exception:
        _PENDING_TASKS.add(task_obj)
        task_obj.add_done_callback(_PENDING_TASKS.discard)


async def _send_followup(interaction_token: str, text: str) -> None:
    """Deliver the real reply by editing the deferred interaction response.

    The interaction token itself authorises this call — no Bot auth header is
    needed. Best-effort with a short timeout; swallow errors like slack's
    sender so a delivery hiccup never crashes the background task.
    """
    app_id = _application_id()
    if not app_id:
        print(f"[Discord] No application id — cannot deliver follow-up. Reply: {text!r}")
        return
    url = f"{_DISCORD_API_BASE}/webhooks/{app_id}/{interaction_token}/messages/@original"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                url,
                json={"content": text[:_MAX_REPLY_CHARS]},
                timeout=10.0,
            )
        if resp.status_code >= 400:
            print(f"[Discord] follow-up edit failed: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"[Discord] follow-up edit error: {e}")


def _extract_command_text(data: dict) -> str:
    """Pull the user's text out of an APPLICATION_COMMAND payload.

    Handles both a slash command with a "message" string option and a
    message-context command (type 3) that targets an existing message.
    """
    # Slash command: options -> [{name: "message", value: "..."}]
    for opt in data.get("options", []) or []:
        if opt.get("name") == "message" and isinstance(opt.get("value"), str):
            return opt["value"].strip()
    # Message-context command: resolved.messages -> {id: {content: "..."}}
    resolved = data.get("resolved", {}) or {}
    messages = resolved.get("messages", {}) or {}
    for msg in messages.values():
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


@router.post("")
async def discord_interactions(request: Request):
    """Discord Interactions Endpoint — the single URL for all interactions."""
    body = await request.body()
    signature = request.headers.get("X-Signature-Ed25519", "")
    timestamp = request.headers.get("X-Signature-Timestamp", "")

    # Discord REQUIRES 401 (not 403) on a bad signature — it probes the endpoint
    # with a deliberately-invalid signature when you save the URL.
    if not _verify_discord_signature(body, signature, timestamp):
        raise HTTPException(status_code=401, detail="Invalid request signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    interaction_type = payload.get("type")

    # 1. PING -> PONG (must be first, before any other handling).
    if interaction_type == _INTERACTION_PING:
        return {"type": _RESPONSE_PONG}

    # 2. APPLICATION_COMMAND (slash or message command).
    if interaction_type == _INTERACTION_APPLICATION_COMMAND:
        data = payload.get("data", {}) or {}
        interaction_id = str(payload.get("id", ""))
        interaction_token = payload.get("token", "")
        channel_id = str(payload.get("channel_id", ""))

        # Author id + bot flag live under member.user (guild) or user (DM).
        member = payload.get("member") or {}
        user_obj = member.get("user") or payload.get("user") or {}
        discord_user_id = str(user_obj.get("id", ""))
        author_is_bot = bool(user_obj.get("bot", False))

        # Bot-loop suppression: ignore bot authors and our own application.
        # An APPLICATION_COMMAND must be answered with a message-shaped response
        # (a PONG here would show "interaction failed"), so send a silent
        # ephemeral ack rather than acting on it.
        if author_is_bot or (discord_user_id and discord_user_id == _application_id()):
            return {
                "type": _RESPONSE_CHANNEL_MESSAGE,
                "data": {"content": "​", "flags": 64},  # zero-width, ephemeral
            }

        text = _extract_command_text(data)
        if not text:
            # Nothing to act on — a minimal ephemeral hint (flags=64 = ephemeral).
            return {
                "type": _RESPONSE_CHANNEL_MESSAGE,
                "data": {"content": "Send me a message with the command.", "flags": 64},
            }

        # 3-second ACK: cannot run the pipeline inline. Defer, then finish on a
        # background task that edits the original response with the real reply.
        async def _process() -> None:
            # Guard the whole body: if resolve/dispatch raises, the deferred
            # "Turtle is thinking…" placeholder would otherwise hang until
            # Discord's timeout. Deliver a graceful message instead.
            try:
                user_id = await identity_manager.resolve_user("discord", discord_user_id)
                sender_name = str(
                    user_obj.get("global_name")
                    or user_obj.get("username")
                    or ""
                )
                turtle_event = TurtleEvent(
                    user_id=user_id,
                    channel="discord",
                    modality="text",
                    content=text,
                    message_id=interaction_id,
                    thread_id=channel_id,
                    sender_name=sender_name,
                    channel_user_id=discord_user_id,
                    # A guild interaction carries "member"; a DM carries only
                    # "user". The deferred follow-up here is NOT ephemeral, so
                    # a guild reply is readable by everyone in the channel —
                    # treat it as public and let secret-bearing tools refuse.
                    is_private=not bool(payload.get("member")),
                )
                response: TurtleResponse = await dispatch_event(turtle_event)
                await _send_followup(interaction_token, response.content or "…")
            except Exception as e:
                print(f"[Discord] interaction processing failed: {e}")
                await _send_followup(interaction_token, "Sorry — something went wrong handling that.")

        _track(asyncio.create_task(_process()))
        return {"type": _RESPONSE_DEFERRED_CHANNEL_MESSAGE}

    # Anything else — acknowledge with a harmless PONG-shaped 200.
    return {"type": _RESPONSE_PONG}


async def register_slash_commands() -> None:
    """Register the global /turtle command (and an "Ask Turtle" message command).

    PUTs the command set to Discord with Bot auth. Idempotent on Discord's side.
    Best-effort and log-only; gated on bot token + application id being present.
    NOT called automatically on startup — invoke it from a script or a one-off
    admin action so you don't hammer Discord on every boot.
    """
    token = _bot_token()
    app_id = _application_id()
    if not token or not app_id:
        print("LOG: Discord register_slash_commands skipped (no bot token / application id)")
        return

    commands = [
        {
            "name": "turtle",
            "type": 1,  # CHAT_INPUT (slash command)
            "description": "Ask Turtle anything",
            "options": [
                {
                    "name": "message",
                    "description": "What do you want to ask Turtle?",
                    "type": 3,  # STRING
                    "required": True,
                }
            ],
        },
        {
            "name": "Ask Turtle",
            "type": 3,  # MESSAGE (context-menu command on a message)
        },
    ]

    url = f"{_DISCORD_API_BASE}/applications/{app_id}/commands"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.put(
                url,
                json=commands,
                headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
                timeout=15.0,
            )
        if resp.status_code >= 400:
            print(f"LOG: Discord command registration failed: {resp.status_code} {resp.text[:200]}")
        else:
            print("LOG: Discord slash commands registered")
    except Exception as e:
        print(f"LOG: Discord command registration error: {e}")
