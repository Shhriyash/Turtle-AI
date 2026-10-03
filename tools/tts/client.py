from __future__ import annotations

import asyncio
import os
import threading
import weakref

import httpx
from groq import AsyncGroq, Groq

# Explicit client-level timeouts (ledger 6.10). Nothing used to set these: the
# Groq SDK silently defaulted to 60 s and httpx.AsyncClient() to 5 s overall.
# ``connect`` stays short so a dead provider fails over quickly; ``read`` is the
# per-response ceiling (callers add their own wait_for budgets on top).
GROQ_ASYNC_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
DEEPGRAM_ASYNC_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY or GROQ_API_KEY2 for Groq TTS fallback.")
    return Groq(api_key=api_key)


def get_deepgram_client():
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPGRAM_API_KEY for Deepgram TTS.")

    try:
        from deepgram import DeepgramClient
    except Exception as exc:
        raise RuntimeError("deepgram-sdk is not available in this environment.") from exc

    return DeepgramClient(api_key=api_key)


# ---------------------------------------------------------------------------
# Async clients (cancellation reaches the HTTP request) -- built once per
# process, per event loop.
#
# httpx connection pools are bound to the loop they first ran on, so a client
# built under one loop must never be handed to another (it raises "Event loop
# is closed" on a pooled connection). The server runs a single loop, so in
# production this is one client per key for the life of the process; a new loop
# (tests, a restarted worker) transparently gets a fresh client.
#
# Deepgram note: ``AsyncDeepgramClient.__init__`` stamps one random
# ``x-deepgram-session-id`` on every request. Sharing the client therefore
# shares one id process-wide instead of one per call. That header is a
# provider-side correlation id (it carries no credentials and no user data), so
# we accept it. This was read from the SDK source, not observed on the wire.
# ---------------------------------------------------------------------------

_ASYNC_CLIENTS: dict[tuple[str, str], tuple["weakref.ref[asyncio.AbstractEventLoop]", object]] = {}
_ASYNC_CLIENTS_LOCK = threading.Lock()


def _cached_for_running_loop(kind: str, api_key: str, factory):
    loop = asyncio.get_running_loop()
    with _ASYNC_CLIENTS_LOCK:
        entry = _ASYNC_CLIENTS.get((kind, api_key))
        if entry is not None:
            cached_loop = entry[0]()
            if cached_loop is loop and not loop.is_closed():
                return entry[1]
        client = factory()
        _ASYNC_CLIENTS[(kind, api_key)] = (weakref.ref(loop), client)
        return client


def get_async_groq_client(api_key: str | None = None) -> AsyncGroq:
    """Process-wide (per-loop) ``AsyncGroq`` with explicit timeouts."""
    key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY or GROQ_API_KEY2 for Groq.")
    return _cached_for_running_loop(
        "groq", key, lambda: AsyncGroq(api_key=key, timeout=GROQ_ASYNC_TIMEOUT)
    )


def get_async_deepgram_client():
    """Process-wide (per-loop) ``AsyncDeepgramClient`` with explicit timeouts."""
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        raise RuntimeError("Missing DEEPGRAM_API_KEY for Deepgram TTS.")

    try:
        from deepgram import AsyncDeepgramClient
    except Exception as exc:
        raise RuntimeError("deepgram-sdk is not available in this environment.") from exc

    def _build():
        return AsyncDeepgramClient(
            api_key=key,
            httpx_client=httpx.AsyncClient(
                timeout=DEEPGRAM_ASYNC_TIMEOUT, follow_redirects=True
            ),
        )

    return _cached_for_running_loop("deepgram", key, _build)
