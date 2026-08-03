"""
apps/channels/discord_gateway.py
--------------------------------
F6: Discord channel adapter — Gateway (WebSocket bot) mode.

This is the "natural conversation" path: TURTLE joins the Discord Gateway as a
real bot and replies to direct messages and @mentions without a slash command.
It requires discord.py, which is an OPTIONAL dependency — the import is guarded
so this module (and the whole app / test suite) loads cleanly when discord.py
is NOT installed. In that case the gateway simply no-ops.

TURTLE runs here strictly as a registered BOT client (discord.Client with a Bot
token). It is NEVER a user account. Driving a real user account over the
Gateway is a "self-bot" and violates Discord's Terms of Service; that path is
deliberately not implemented.

For the zero-dependency alternative (slash commands over an HTTPS webhook), see
apps/channels/discord.py.

Required env var:
  DISCORD_BOT_TOKEN   Bot token (Developer Portal → Bot → Reset/Copy Token).

Gateway intents: message_content (privileged — enable it under Bot → Privileged
Gateway Intents in the Developer Portal) and dm_messages.
"""
from __future__ import annotations

from core.config import settings

# Guarded import: discord.py is optional. Absence must be a clean no-op, so the
# app boots and the CI suite passes without it installed.
try:
    import discord  # type: ignore
    _DISCORD_IMPORT_OK = True
except Exception:  # pragma: no cover - exercised only when discord.py absent
    discord = None  # type: ignore
    _DISCORD_IMPORT_OK = False

# Module-level handles so shutdown can reach the running client + its task.
_client = None  # type: ignore[var-annotated]
_client_task = None  # type: ignore[var-annotated]

_MAX_REPLY_CHARS = 1900


def gateway_available() -> bool:
    """True when the gateway CAN run: discord.py importable AND a bot token set."""
    return _DISCORD_IMPORT_OK and bool(_bot_token())


def _bot_token() -> str:
    return settings.discord_bot_token.get_secret_value() if settings.discord_bot_token else ""


async def start_discord_gateway() -> None:
    """Start the Discord Gateway bot as a background task (best-effort).

    Graceful no-op when discord.py is not installed or no bot token is set.
    """
    global _client, _client_task
    if not _DISCORD_IMPORT_OK or not _bot_token():
        print("LOG: discord gateway disabled (no discord.py / no token)")
        return
    if _client is not None:
        print("LOG: discord gateway already running")
        return

    import asyncio

    # Import locally too so type-checkers/readers see the guarded module.
    from apps.channels import TurtleEvent, TurtleResponse, dispatch_event
    from core.identity import identity_manager

    intents = discord.Intents.default()
    intents.message_content = True  # privileged — enable in the Developer Portal
    intents.dm_messages = True

    # BOT client only. Never construct this with a user token (self-bot),
    # which violates Discord's ToS.
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready() -> None:  # pragma: no cover - needs a live gateway
        print(f"LOG: discord gateway connected as {client.user}")

    @client.event
    async def on_message(message) -> None:  # pragma: no cover - needs a live gateway
        # Ignore bots and our own messages to prevent loops.
        if message.author.bot:
            return
        if client.user is not None and message.author.id == client.user.id:
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mention = client.user is not None and client.user in message.mentions
        if not (is_dm or is_mention):
            return

        # Strip the leading @mention so the pipeline sees clean text.
        text = message.content or ""
        if client.user is not None:
            text = text.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "")
        text = text.strip()
        if not text:
            return

        try:
            user_id = await identity_manager.resolve_user("discord", str(message.author.id))
            turtle_event = TurtleEvent(
                user_id=user_id,
                channel="discord",
                modality="text",
                content=text,
                message_id=str(message.id),
                thread_id=str(message.channel.id),
            )
            response: TurtleResponse = await dispatch_event(turtle_event)
            await message.channel.send(response.content[:_MAX_REPLY_CHARS])
        except Exception as e:
            print(f"LOG: discord gateway on_message error: {e}")

    _client = client
    # client.start() blocks until disconnect — run it as a background task so
    # startup returns immediately.
    _client_task = asyncio.create_task(client.start(_bot_token()))

    def _log_gateway_exit(task: "asyncio.Task") -> None:
        # Surface a silent connection failure instead of letting the exception
        # die inside the detached task. The common causes are an invalid bot
        # token (LoginFailure) or the privileged message_content intent not
        # being enabled under Bot → Privileged Gateway Intents in the Developer
        # Portal (PrivilegedIntentsRequired) — both otherwise vanish here.
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            print(f"LOG: discord gateway stopped: {exc.__class__.__name__}: {exc}", flush=True)

    _client_task.add_done_callback(_log_gateway_exit)

    try:
        from core.worker import track_task
        track_task(_client_task)
    except Exception:
        pass

    print("LOG: discord gateway starting", flush=True)


async def stop_discord_gateway() -> None:
    """Close the gateway client and cancel its task (graceful no-op if not up)."""
    global _client, _client_task
    if _client is not None:
        try:
            await _client.close()
        except Exception as e:
            print(f"LOG: discord gateway close error: {e}")
    if _client_task is not None:
        try:
            _client_task.cancel()
        except Exception:
            pass
    _client = None
    _client_task = None
