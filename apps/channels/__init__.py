"""
apps/channels/__init__.py
-------------------------
Tier 3 — Channel adapter shared types and dispatch wiring.

Every channel adapter (WhatsApp, iMessage, Slack, Twilio Voice) normalizes
its inbound payload to TurtleEvent and calls dispatch_text().  The main
server wires the real handler at startup via set_channel_dispatch().

TurtleEvent mirrors the north-star architecture envelope:
    { user_id, channel, modality, content, message_id, thread_id, attachments }
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, Optional

# ---------------------------------------------------------------------------
# Canonical types
# ---------------------------------------------------------------------------

Modality = Literal["text", "voice"]
Channel = Literal["web", "whatsapp", "imessage", "slack", "twilio_voice", "discord"]


@dataclass
class TurtleEvent:
    """Normalised inbound event — channel-agnostic."""
    user_id: str
    channel: Channel
    modality: Modality
    content: str
    message_id: str = ""
    thread_id: str = ""
    attachments: list[dict] = field(default_factory=list)


@dataclass
class TurtleResponse:
    """Normalised outbound response — channel-agnostic."""
    content: str
    channel: Channel
    user_id: str
    message_id: str = ""        # echo of TurtleEvent.message_id for idempotency
    thread_id: str = ""


# ---------------------------------------------------------------------------
# Dispatch wiring — inversion of control, no circular imports
# ---------------------------------------------------------------------------

# Type alias for the real handler set at startup
_DispatchFn = Callable[[TurtleEvent], Awaitable[TurtleResponse]]

_dispatch_fn: Optional[_DispatchFn] = None


def set_channel_dispatch(fn: _DispatchFn) -> None:
    """Called once at app startup to wire the real pipeline handler."""
    global _dispatch_fn
    _dispatch_fn = fn


async def dispatch_event(event: TurtleEvent) -> TurtleResponse:
    """
    Dispatch a TurtleEvent through the Turtle pipeline.

    Falls back to a stub response when the handler has not been wired yet
    (e.g. during tests or startup).
    """
    if _dispatch_fn is not None:
        return await _dispatch_fn(event)
    # Stub fallback — used in tests and during cold startup
    return TurtleResponse(
        content="[Turtle not ready — dispatch handler not wired]",
        channel=event.channel,
        user_id=event.user_id,
        message_id=event.message_id,
        thread_id=event.thread_id,
    )


async def dispatch_text(
    text: str,
    *,
    user_id: str,
    channel: Channel,
    message_id: str = "",
    thread_id: str = "",
) -> str:
    """Convenience wrapper — returns just the response text string."""
    event = TurtleEvent(
        user_id=user_id,
        channel=channel,
        modality="text",
        content=text,
        message_id=message_id,
        thread_id=thread_id,
    )
    resp = await dispatch_event(event)
    return resp.content
