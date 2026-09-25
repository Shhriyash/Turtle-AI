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
import time

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi import APIRouter, HTTPException, Request

from apps.channels import TurtleEvent, TurtleResponse, dispatch_event
from core.config import settings
from core.identity import CHANNEL_INVITE_ONLY_MESSAGE, resolve_channel_user
from core.internal_auth import (
    SignatureError,
    check_bearer,
    claim_once,
    sign_request,
    store_job_payload,
    take_job_payload,
    verify_request,
)
from core.storage.cloud import CloudBackendUnavailable

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

# WP 1.F (ledger 1b.4, S-7.8): a validly-signed interaction is proof Discord
# sent it AT SOME POINT, not that it's recent or hasn't already been acted
# on — a captured request replays forever against signature verification
# alone. Two independent defenses, both below:
#
#   1. Timestamp freshness — reject X-Signature-Timestamp older than 300s.
#      Symmetric (also rejects a timestamp implausibly far in the FUTURE),
#      matching core.internal_auth.verify_request's CLOCK_SKEW_S window.
#      Unlike that self-call envelope (where an attacker who leaks the
#      shared secret can mint a signature with ANY timestamp they choose,
#      making an unbounded-future allowance a real hole), Discord itself
#      signs this timestamp — an outside caller can't forge one without
#      Discord's private key, so the future side isn't closing a forgery
#      hole here. It's kept symmetric anyway as cheap, free defense-in-depth
#      against clock corruption producing a huge/garbage-but-numeric value,
#      and for consistency with the one other timestamp-window check in this
#      codebase. A real Discord request is always ~now on either endpoint,
#      so this never rejects legitimate traffic.
#   2. interaction_id dedup — SET NX EX claim, own prefix/TTL (see
#      _INTERACTION_CLAIM_PREFIX below), so even a replay INSIDE the 300s
#      freshness window is rejected the second time it's seen.
_TIMESTAMP_WINDOW_S = 300

# Own prefix + TTL, deliberately distinct from core.internal_auth's
# `turtle:nonce:` (300s) / `turtle:job:` (900s): a Discord interaction_id is
# a different security domain than Turtle's own self-call nonce (external
# party, no shared secret involved in minting it), and the two need to be
# rotatable/reasoned-about independently even though claim_once() is shared
# plumbing. 900s (matches the ledger's chosen TTL for this WP) is longer
# than the 300s freshness window on purpose: it isn't sized to the freshness
# check (an interaction already fails freshness well before its claim would
# expire) — it's sized to survive Discord's own retry/backoff behavior on a
# slow endpoint, which can resend the SAME interaction_id a little after the
# first attempt.
_INTERACTION_CLAIM_PREFIX = "turtle:discord-interaction:"
_INTERACTION_CLAIM_TTL_S = 900

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
        # FAIL CLOSED unless TURTLE_DEV_ANON=1 AND not cloud. A tunneled local
        # deploy (ngrok, cloudflared) is network-accessible and must not accept
        # unsigned interactions just because is_cloud is False. Aligned with
        # whatsapp / imessage / slack.
        return settings.dev_anon and not settings.is_cloud
    try:
        verify_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub))
        verify_key.verify(bytes.fromhex(signature_hex), timestamp.encode() + body)
        return True
    except (InvalidSignature, ValueError):
        return False


def _timestamp_is_fresh(timestamp: str) -> bool:
    """True iff ``timestamp`` (Discord's X-Signature-Timestamp: Unix seconds,
    as a string) is within _TIMESTAMP_WINDOW_S of now, in either direction.

    ``timestamp`` is external input from an unauthenticated-until-this-point
    caller (this check runs on the raw header, same as the signature check
    it accompanies) — missing, empty, non-numeric, or absurdly large/negative
    values must all resolve to "not fresh" rather than raise into the route
    handler. int() on an oversized-but-numeric string is fine (Python ints
    are unbounded); a garbage value just fails the window comparison.
    """
    if not timestamp:
        return False
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    now = int(time.time())
    return abs(now - ts) <= _TIMESTAMP_WINDOW_S


async def _claim_interaction_id(interaction_id: str) -> bool:
    """Dedup an APPLICATION_COMMAND's interaction_id via claim_once() (SET NX
    EX) under this module's own prefix/TTL (see the constants above).
    Returns True for a fresh (never-seen) id — proceed; False for a replay
    — reject.

    FAILS OPEN when the claim store can't be reached: an unset REDIS_URL
    (local mode always hits this — it has no Redis at all) raises
    CloudBackendUnavailable from get_redis_client(), and a live Redis outage
    in cloud raises the redis-py driver's own exception (ConnectionError /
    TimeoutError / ...) from the SET itself — claim_once() lets both through
    unchanged (see its docstring), and both are caught here, together,
    deliberately.

    This is the OPPOSITE posture from core.internal_auth.verify_request's
    nonce claim, which fails closed — and that's a deliberate choice, not a
    copy-paste of a different WP's trade-off:
      - verify_request's nonce claim is that self-call envelope's ONLY
        replay defense; losing it on a Redis outage would let a captured
        internal request replay freely, and that endpoint has an in-process
        fallback path on the CALLING side already, so failing closed there
        costs nothing but robustness on an internal path nobody outside
        Turtle can reach.
      - Here, the signature check is what actually authenticates the
        caller as Discord; the interaction_id dedup is a SECOND, narrower
        layer on top of it (closing the "same valid request replayed inside
        the freshness window" gap _timestamp_is_fresh doesn't cover). Failing
        closed would mean a Redis blip makes the PUBLIC, customer-facing
        Discord webhook reject every single interaction — not just replays —
        for as long as the outage lasts. Failing open instead only widens
        the replay window back to what _timestamp_is_fresh still bounds
        (<=300s); it never admits an unsigned or forged request. That is a
        strictly better trade for a public availability-sensitive endpoint
        than trading total channel downtime for closing an already-bounded
        window.
    """
    try:
        return await claim_once(_INTERACTION_CLAIM_PREFIX, interaction_id, _INTERACTION_CLAIM_TTL_S)
    except Exception as exc:
        print(f"[Discord] interaction dedup store unavailable, failing OPEN: {exc}")
        return True


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


async def _process_deferred_interaction(payload: dict) -> None:
    """The actual interaction processing: resolve identity, dispatch through
    the pipeline, PATCH the deferred response with the real reply.

    Guards the whole body: if resolve/dispatch raises, the deferred "Turtle
    is thinking…" placeholder would otherwise hang until Discord's timeout.
    Deliver a graceful message instead. Shared by both execution paths (see
    _kick_off_deferred_processing) — the payload dict is exactly what
    discord_interactions built, so this function is identical either way.
    """
    interaction_token = payload["interaction_token"]
    try:
        discord_user_id = payload["discord_user_id"]
        user_id = await resolve_channel_user("discord", discord_user_id)
        if user_id is None:
            # TURTLE_CHANNEL_SIGNUP=invite and this sender is unknown — reply
            # with the invite message and mint nothing (see core/identity.py).
            await _send_followup(interaction_token, CHANNEL_INVITE_ONLY_MESSAGE)
            return
        turtle_event = TurtleEvent(
            user_id=user_id,
            channel="discord",
            modality="text",
            content=payload["text"],
            message_id=payload["interaction_id"],
            thread_id=payload["channel_id"],
            sender_name=payload.get("sender_name", ""),
            channel_user_id=discord_user_id,
            is_private=bool(payload.get("is_private", False)),
        )
        response: TurtleResponse = await dispatch_event(turtle_event)
        await _send_followup(interaction_token, response.content or "…")
    except Exception as e:
        print(f"[Discord] interaction processing failed: {e}")
        await _send_followup(interaction_token, "Sorry — something went wrong handling that.")


async def _kick_off_deferred_processing(payload: dict) -> None:
    """Start the real (slower-than-3s) processing for a deferred interaction.

    Local mode: a detached asyncio task in this same process, tracked so it
    can't be GC'd mid-flight — proven correct there (a long-lived server
    process). Cloud mode: Vercel's docs describe post-response background
    work surviving ONLY when explicitly scheduled via its JS waitUntil()/
    after() API (@vercel/functions) — there is no documented Python
    equivalent, so relying on a bare detached asyncio.create_task here would
    be betting on unconfirmed platform behavior for exactly the feature this
    migration is meant to make reliable. Instead, cloud mode SELF-INVOKES a
    second, independent HTTP request to POST /channels/discord/process
    carrying this payload — Vercel runs that as a completely normal request
    with its own full timeout budget, so there is no ambiguity about it
    completing. We only need to know the request was SENT before this
    invocation's own response goes out, not that it finished — the short
    read timeout below (paired with a generous connect/write timeout) is
    exactly that: send fully, then stop waiting for a reply we don't need.
    """
    if not settings.is_cloud:
        _track(asyncio.create_task(_process_deferred_interaction(payload)))
        return

    secret = settings.internal_job_secret.get_secret_value() if settings.internal_job_secret else ""
    if not secret:
        # No internal-automation secret configured — degrade to the
        # in-process task rather than silently dropping the interaction.
        # Less robust on serverless, but strictly no worse than before this
        # fix existed, and it's a one-line env var away from the safe path.
        print("[Discord] INTERNAL_JOB_SECRET unset — falling back to in-process deferred task")
        _track(asyncio.create_task(_process_deferred_interaction(payload)))
        return

    # WP 1.B / S-7.3: payload-by-reference — stash the real payload (which
    # carries discord_user_id) in Redis and send only its id. If Redis isn't
    # reachable, that's the same "can't complete the internal-automation
    # round trip" case as no secret being set — degrade the same way.
    try:
        job_id = await store_job_payload(payload)
        body = json.dumps({"job_id": job_id}).encode("utf-8")
        envelope = sign_request(secret, body)
    except CloudBackendUnavailable as e:
        print(f"[Discord] job store unavailable, falling back to in-process: {e}")
        _track(asyncio.create_task(_process_deferred_interaction(payload)))
        return

    url = f"{settings.public_base_url.rstrip('/')}/channels/discord/process"
    headers = {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
        **envelope.headers(),
    }
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                url,
                content=body,
                headers=headers,
                timeout=httpx.Timeout(connect=5.0, read=0.1, write=5.0, pool=5.0),
            )
    except httpx.ReadTimeout:
        pass  # Expected: the request was sent; we deliberately don't await its reply.
    except Exception as e:
        print(f"[Discord] self-invoke for deferred processing failed: {e}")
        _track(asyncio.create_task(_process_deferred_interaction(payload)))


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

    # WP 1.F: freshness check runs SECOND, right after signature
    # verification and before anything else (including the PING
    # short-circuit) — a replay check must never run against an
    # unauthenticated request (that's its own DoS: burning claims for a
    # caller who hasn't proven they're Discord), so it can only go after the
    # signature check succeeds. It applies uniformly to every interaction
    # type, PING included: a real endpoint-validation PING is always signed
    # with a ~now timestamp, so this never breaks Discord's own probe / the
    # "save the Interactions Endpoint URL" flow. Reject with the exact same
    # 401 shape as a bad signature (per the WP: keep the handler's contract
    # with Discord — which only distinguishes "not a valid signature" from
    # everything else — consistent).
    if not _timestamp_is_fresh(timestamp):
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

        # WP 1.F: dedup claim runs right after we have interaction_id (still
        # before the bot-loop check / text extraction below — no point doing
        # more work for a request we're about to reject), and only for
        # APPLICATION_COMMAND — a PING carries no interaction id worth
        # claiming and must stay a fast, unconditional PONG for Discord's
        # endpoint-registration probe. A repeat of the same interaction_id
        # (a captured-and-replayed request, or Discord's own retry landing
        # after we already accepted the first delivery) is rejected with the
        # same 401 shape the signature/freshness checks use.
        if interaction_id and not await _claim_interaction_id(interaction_id):
            raise HTTPException(status_code=401, detail="Invalid request signature")

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

        # 3-second ACK: cannot run the pipeline inline. Defer, then finish the
        # real work out-of-band — see _kick_off_deferred_processing for how
        # that differs between local and cloud mode.
        deferred_payload = {
            "interaction_token": interaction_token,
            "interaction_id": interaction_id,
            "channel_id": channel_id,
            "discord_user_id": discord_user_id,
            "text": text,
            "sender_name": str(user_obj.get("global_name") or user_obj.get("username") or ""),
            # A guild interaction carries "member"; a DM carries only "user".
            # The deferred follow-up here is NOT ephemeral, so a guild reply
            # is readable by everyone in the channel — treat it as public and
            # let secret-bearing tools refuse.
            "is_private": not bool(payload.get("member")),
        }
        await _kick_off_deferred_processing(deferred_payload)
        return {"type": _RESPONSE_DEFERRED_CHANNEL_MESSAGE}

    # Anything else — acknowledge with a harmless PONG-shaped 200.
    return {"type": _RESPONSE_PONG}


@router.post("/process")
async def discord_process_deferred(request: Request):
    """Internal-only: runs the deferred interaction processing as its OWN
    independent request — see _kick_off_deferred_processing's docstring for
    why cloud mode self-invokes this instead of a detached background task.

    Never called by Discord itself (it has no idea this route exists) —
    protected by INTERNAL_JOB_SECRET (WP 1.B / S-7.3: no longer the same
    secret apps/cron_tick_routes.py's /internal/cron-tick uses), checked as
    both a bearer token AND the key for a signed-request envelope (timestamp
    + nonce + raw body, core.internal_auth.verify_request) — not a Discord
    signature. The body itself carries only an opaque job id; the real
    payload (including discord_user_id) is read-and-deleted from Redis
    (core.internal_auth.take_job_payload), never trusted from the wire.
    """
    secret_value = settings.internal_job_secret.get_secret_value() if settings.internal_job_secret else ""
    authorization = request.headers.get("Authorization", "")
    if not check_bearer(secret_value, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.body()
    try:
        await verify_request(
            secret_value,
            request.headers.get("X-Turtle-Timestamp"),
            request.headers.get("X-Turtle-Nonce"),
            request.headers.get("X-Turtle-Signature"),
            body,
        )
    except SignatureError as exc:
        raise HTTPException(status_code=401, detail=f"Unauthorized: {exc}") from exc

    try:
        envelope = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    job_id = envelope.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id is required")

    try:
        payload = await take_job_payload(job_id)
    except CloudBackendUnavailable as exc:
        # Redis went from reachable (the signature check above needs it too)
        # to unreachable between there and here — surface as a clean 503,
        # never an unhandled 500 from a raw driver exception.
        raise HTTPException(status_code=503, detail=f"Job store unavailable: {exc}") from exc
    if payload is None:
        raise HTTPException(status_code=401, detail="Unknown or expired job id")

    required = {"interaction_token", "discord_user_id", "text", "interaction_id", "channel_id"}
    if not required.issubset(payload):
        raise HTTPException(status_code=400, detail=f"Missing required field(s): {required - set(payload)}")

    await _process_deferred_interaction(payload)
    return {"ok": True}


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
