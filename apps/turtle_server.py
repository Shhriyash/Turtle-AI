"""
Turtle AI — Web Chat Server

FastAPI + WebSocket backend that bridges the browser UI to the existing
Turtle agent pipeline.  Reuses all core modules (LLM client, RAG, memory,
session store, tools) but replaces CLI I/O with a WebSocket interface.

Start with:
    python apps/turtle_server.py
    # or: uvicorn apps.turtle_server:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import atexit
import asyncio
import functools
import base64
import hashlib
import hmac
import json
import os
import signal
import sys
import threading
import weakref
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Optional

# ---------------------------------------------------------------------------
# Path bootstrap (same as turtle_voice.py)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import settings
import core.background_tasks  # Register background tasks

import httpx
import numpy as np
from fastapi import FastAPI, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from apps.auth import authenticate_websocket
from fastapi.staticfiles import StaticFiles

from core.env import load_env

# Populate os.environ from .env so os.getenv() sees the same values pydantic-
# settings already reads directly. Without this, only fields on `settings` are
# populated — anything read via os.getenv (OPEN_ROUTER_API_KEY_*, MAIN_AGENT_MODEL
# env override, etc.) silently sees None on a freshly-launched uvicorn process.
load_env()

load_env(override=True)

# Core imports — identical to turtle_voice.py
from groq import Groq
from pydantic_ai import Agent, RunContext, ModelMessagesTypeAdapter
from dataclasses import replace as _dc_replace
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import UsageLimits, RunUsage

from core.llm_client import (
    CascadeStats,
    get_groq_model,
    get_groq_models,
    get_google_models,
    get_openrouter_models,
    get_groq_fallback_model,
    run_agent_with_fallbacks,
    stream_agent_text_with_fallbacks,
    StreamCollector,
)
from core.email_flow import (
    build_compose_email_prompt,
    combine_extracted_email_details,
    derive_fallback_subject,
    extract_deterministic_email_details,
    format_email_draft,
    format_missing_email_prompt,
    merge_email_details,
    missing_email_fields,
    parse_email_extraction_response,
    resolve_suggested_recipient,
    send_email_now,
    suggest_recipient_completion,
    validate_recipients,
    validate_send_email_args,
)
from core.output_clean import (
    clean_text_for_model,
    clean_text_for_tts,
    clean_text_for_display,
    sanitize_for_envelope,
    wrap_untrusted,
    extract_tool_result_urls,
)
from core.confirmation_gate import ConfirmationGate
from core.guardrails import StorageCapExceededError, WebSocketRateLimitExceeded
from core.storage.factory import get_channel_gate_buffer, get_ws_rate_limiter
from core.telemetry import emit as emit_event, emit_once as emit_event_once
from core.memory_journal import JournalStore, make_event
from core.memory_schema import decide_write_policy, statement_for
from core.memory_extractor import extract_memory_event_specs
from core.memory_replayer import replay
from core.observability import ATTR_TOKENS_IN, ATTR_TOKENS_OUT, trace_sink
from core.personal_memory_extract import (
    PersonalMemoryCandidate,
    extract_memory_candidates_from_messages,
    extract_memory_candidates_from_messages_async,
    run_stage_b_session_extractor,
)
from core.periodic_reflector import PeriodicReflector
from core.personal_memory_prompt import PersonalMemoryPromptBuilder, PersonalMemoryPromptConfig
from core.personal_memory_store import PersonalMemoryStore
from core.task_history import TaskHistoryStore
from core.paths import (
    TASK_HISTORY_FILE,
    TEMP_AUDIO_DIR,
    ensure_dirs,
    personal_memory_dir,
)
from core.session_store import SessionStore
from core.system_prompts import load_prompt
from core.openrouter_tts import synthesize_speech
from core.stt_fastrtc import FastRTCSTT
from core.web_search import format_search_results, search_duckduckgo
from rag.system.complete_rag import TurtleRAGSystem
from tools.url_tools import fetch_url_content_async
from tools.contracts import (
    ToolResult,
    WebSearchArgs,
    UrlFetchArgs,
    EmailArgs,
    RecallArgs,
    CalendarCreateArgs,
    CalendarListArgs,
    FindPlaceArgs,
    PlaceDetailsArgs,
    GetDirectionsArgs,
)

try:
    import logfire
    logfire.configure(send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)
    _logfire_loaded = True
except Exception:
    _logfire_loaded = False

ensure_dirs()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = ROOT_DIR / "config" / "turtle_config.json"
STATIC_DIR = ROOT_DIR / "web"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765


def _load_config() -> dict[str, Any]:
    """Load turtle_config.json with defaults."""
    defaults = {
        "OPEN_ROUTER_MODEL": "nvidia/llama-3.1-nemotron-70b-instruct:free",
        "GROQ_PRIMARY_MODEL": "llama-3.3-70b-versatile",
        "GROQ_FALLBACK_MODEL": "llama-3.1-8b-instant",
        "DEEPGRAM_TTS_MODEL": "aura-2-orion-en",
        "DEEPGRAM_TTS_ENCODING": "linear16",
        "DEEPGRAM_TTS_CONTAINER": "wav",
        "DEEPGRAM_TTS_SAMPLE_RATE": 24000,
        "TURTLE_TTS_SPEED": 1.2,
        "GROQ_TTS_MODEL": "canopylabs/orpheus-v1-english",
        "GROQ_TTS_VOICE": "orion",
        "GROQ_TTS_FORMAT": "wav",
        "temperature": 0.2,
        "max_tokens": 1024,
        "TURTLE_HISTORY_MAX_TURNS": 12,
        "ACTIVE_HISTORY_MAX_MESSAGES": 40,
        "TURTLE_HISTORY_MAX_TOKENS": 4000,
        "TURTLE_MEMORY_FLUSH_TURNS": 8,
        "TURTLE_MEMORY_FLUSH_TOKENS": 6000,
        "TURTLE_MEMORY_PROFILE_MAX_LINES": 6,
        "TTS_DEBUG": False,
        "STT_MODEL": "whisper-large-v3-turbo",
        "MAIN_AGENT_MODEL": "groq:openai/gpt-oss-120b",
        "EMAIL_AGENT_MODEL": "groq:llama-3.3-70b-versatile",
        "SERVER_HOST": SERVER_HOST,
        "SERVER_PORT": SERVER_PORT,
    }
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            defaults.update(saved)
    except Exception:
        pass
    return defaults


def _save_config(cfg: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


config = _load_config()

# ---------------------------------------------------------------------------
# Shared constants (mirrored from turtle_voice.py)
# ---------------------------------------------------------------------------
EMAIL_PROMPT = load_prompt("email_agent").replace("{bot_email}", settings.bot_email)
_MAIN_ASSISTANT_PROMPT_TEMPLATE = load_prompt("main_assistant")

# Identity handed to the email agent when it composes a body on the user's
# behalf. Lets "tell about yourself" produce a self-description, and other
# requests be written as the user's assistant.
EMAIL_SENDER_IDENTITY = (
    "You are Turtle, the user's personal AI assistant. Emails you send come "
    f"from {settings.bot_email}. If the user asks you to write about yourself, "
    "describe Turtle; otherwise write the message on the user's behalf."
)


def _build_main_assistant_prompt(
    *,
    timezone: str = "UTC",
    channel: str = "web",
    user_greeting_block: str = "",
) -> str:
    """Inject runtime context into the main assistant system prompt (C2).

    The {runtime_context} and {user_greeting_block} placeholders in
    main_assistant.txt are retained for compatibility. Per-turn runtime values
    now come from _build_turn_instructions so the static prompt never bakes a
    frozen clock or timezone.
    """
    runtime_context = "Runtime context (current date/time, timezone, user memory) is provided in per-turn instructions."
    return (
        _MAIN_ASSISTANT_PROMPT_TEMPLATE
        .replace("{runtime_context}", runtime_context)
        .replace("{user_greeting_block}", user_greeting_block)
    )


def _extract_user_name(identity_doc) -> str | None:
    """Pull the user's first name from an identity.md MarkdownMemoryDocument."""
    try:
        for raw in identity_doc.lines:
            line = str(raw).strip().lstrip("-").strip()
            if line.lower().startswith("name:"):
                return line.split(":", 1)[1].strip() or None
    except Exception:
        return None
    return None


def _recent_conversation_context(state: "SharedState", *, max_chars: int = 4000, max_messages: int = 10) -> str:
    """Recent conversation text + this session's fetched content, for the email composer.

    The email sub-agent runs without the main conversation history, so a delegated
    email that references earlier content ("email me the fetched news", "send that
    summary") would be authored blind. Gather the tail of the conversation (user
    turns, assistant replies, tool results) plus cached search/URL results so the
    composer writes from the real content instead of an empty placeholder.
    """
    chunks: list[str] = []
    seen: set[str] = set()

    def _add(role: str, text: Any) -> None:
        s = str(text).strip() if text else ""
        if not s or s in seen:
            return
        seen.add(s)
        chunks.append(f"[{role}] {s}")

    try:
        history = list(getattr(state.session_store, "message_history", None) or [])
    except Exception:
        history = []
    for msg in history[-max_messages:]:
        for part in getattr(msg, "parts", None) or []:
            if isinstance(part, TextPart):
                _add("assistant", part.content)
            elif isinstance(part, ToolReturnPart):
                _add("fetched", part.content if isinstance(part.content, str) else part.content)
            elif isinstance(part, UserPromptPart):
                if isinstance(part.content, str):
                    _add("user", part.content)
    # Same-turn fetched content may not be persisted into history yet.
    try:
        for val in list((state.search_cache or {}).values())[-3:]:
            _add("fetched", val)
    except Exception:
        pass

    if not chunks:
        return ""
    context = "\n\n".join(chunks)
    if len(context) > max_chars:
        context = context[-max_chars:]  # keep the most recent
    return context


def _build_user_greeting_block(user_id: str) -> str:
    """Render the per-user greeting block injected into <runtime_context>."""
    if not user_id:
        return (
            "You don't yet know this user's name. If they haven't introduced "
            "themselves and the moment is right, ask once — naturally."
        )
    try:
        store = PersonalMemoryStore(user_id=user_id)
        identity = store.load_topic("identity")
        name = _extract_user_name(identity)
    except Exception:
        name = None
    if name:
        return f"You are speaking with {name}."
    return (
        "You don't yet know this user's name. If they haven't introduced "
        "themselves and the moment is right, ask once — naturally."
    )


def _build_turn_instructions(state: "SharedState") -> str:
    """Per-turn dynamic instructions: greeting, live clock/timezone, and the
    memory block. Runs on every model call (including fallback rungs), so the
    model always sees the CURRENT memory snapshot exactly once, never baked
    into persisted user turns where stale copies accumulate and contradict
    corrections."""
    import datetime as _dt
    uid = state.user_id if state is not None else ""
    parts: list[str] = [_build_user_greeting_block(uid)]
    tz_name = "UTC"
    try:
        identity = PersonalMemoryStore(user_id=uid).load_topic("identity") if uid else None
        if identity is not None:
            for raw in identity.lines:
                line = str(raw).strip().lstrip("-").strip()
                if line.lower().startswith("timezone:"):
                    tz_name = line.split(":", 1)[1].strip() or "UTC"
                    break
    except Exception:
        pass
    now_utc = _dt.datetime.now(_dt.UTC).strftime("%A, %d %B %Y, %H:%M UTC")
    parts.append(f"Current date and time: {now_utc}")
    parts.append(f"User timezone: {tz_name}")

    # Channel-aware formatting hint: the static system prompt describes both
    # the voice rules and the text markdown rules; this line tells the model
    # which one is active RIGHT NOW so it never fires the wrong ruleset. Voice
    # channels get plain speakable text; every text/chat channel gets markdown.
    channel = str(getattr(state, "channel", "") or "").strip() if state is not None else ""
    if channel in _VOICE_CHANNELS:
        parts.append(
            "Output channel: voice. Follow <voice_first_output_rules>: plain "
            "spoken English, no markdown, no bullet characters, no em/en dashes."
        )
    elif channel:
        parts.append(
            f"Output channel: {channel} (text). Follow <text_formatting_rules>: "
            "markdown IS rendered here, use **bold** labels, `- ` bullets and blank "
            "lines to make multi-part answers scannable. Never use em (—) or en (–) "
            "dashes anywhere in the reply."
        )

    memory_block = (state.memory_context or "").strip() if state is not None else ""
    if memory_block:
        parts.append(
            "Current user memory (authoritative; this is the only current copy, "
            "any memory blocks inside older conversation turns are stale snapshots):\n"
            + memory_block
        )
    return "\n".join(p for p in parts if p)


# Static fallback used by AgentManager.rebuild (agents are built at startup);
# per-request prompts are built via _build_main_assistant_prompt() per-turn.
MAIN_ASSISTANT_PROMPT = _build_main_assistant_prompt()

# Retries: 1 for plain-string output (chitchat — no validator, extra retries are
# a pure latency tax).  Kept at 2 for typed outputs (router/extractor/email).
# See H3 in arch_improve.md.
OUTPUT_RETRIES = 1
# Env var wins (deployment override); otherwise honour turtle_config.json so the
# committed default actually takes effect. "resume_if_active" lets a dropped or
# refreshed WebSocket rejoin the live session instead of starting empty.
SESSION_RESTORE_MODE = os.getenv("SESSION_RESTORE_MODE") or str(
    config.get("SESSION_RESTORE_MODE", "resume_if_active")
)
ACTIVE_HISTORY_MAX_TURNS = int(config.get("TURTLE_HISTORY_MAX_TURNS", 12))
ACTIVE_HISTORY_MAX_MESSAGES = int(config.get("ACTIVE_HISTORY_MAX_MESSAGES", 40))
ACTIVE_HISTORY_MAX_TOKENS = int(config.get("TURTLE_HISTORY_MAX_TOKENS", 4000))
MEMORY_FLUSH_TURNS = int(config.get("TURTLE_MEMORY_FLUSH_TURNS", 8))
MEMORY_FLUSH_TOKENS = int(config.get("TURTLE_MEMORY_FLUSH_TOKENS", 6000))
MEMORY_PROFILE_MAX_LINES = int(config.get("TURTLE_MEMORY_PROFILE_MAX_LINES", 6))
PERSONAL_MEMORY_ENABLED = settings.personal_memory_enabled
PERSONAL_MEMORY_MAX_BYTES = settings.personal_memory_max_bytes
PERSONAL_MEMORY_MAX_TOPIC_FILES = settings.personal_memory_max_topic_files
TOOL_OUTPUT_MAX_CHARS = settings.tool_output_max_chars

_groq_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else (settings.groq_api_key2.get_secret_value() if settings.groq_api_key2 else None)
groq_client = Groq(api_key=_groq_key)


# ---------------------------------------------------------------------------
# SharedState — same dataclass as turtle_voice.py
# ---------------------------------------------------------------------------
@dataclass
class SharedState:
    http_client: httpx.AsyncClient
    session_store: SessionStore
    personal_memory_store: PersonalMemoryStore
    personal_memory_prompt: PersonalMemoryPromptBuilder
    journal_store: JournalStore
    confirmation_gate: ConfirmationGate
    task_history_store: TaskHistoryStore
    rag_system: TurtleRAGSystem
    sqlite_index: Any | None = None   # MemorySQLiteIndex; closed on shutdown
    retrieval_broker: Any | None = None   # D4: wired in setup_shared_state
    reflector: PeriodicReflector | None = None
    search_cache: dict[str, str] = field(default_factory=dict)
    turn_counter: int = 0
    user_id: str = ""
    # Which surface this state is serving, and the platform-side id there.
    # Needed by account linking: a claim code is bound to the CHANNEL identity
    # (e.g. discord/759…), not to the Turtle user_id. Empty on the web path.
    channel: str = ""
    channel_user_id: str = ""
    # True only when this surface is readable solely by the sender (a DM).
    # Secret-bearing tools (link_account) refuse on anything else.
    channel_is_private: bool = False
    # Phase 1: the memory block for the current turn. Delivered to the model
    # via per-turn instructions (never inside the persisted user prompt).
    memory_context: str = ""
    # WP1.H: URLs a TOOL actually returned this turn (extracted from the
    # sanitised, pre-envelope text at the registration-loop wrapper). Reset
    # at the top of each turn; the "done" frame carries this list so the
    # client can allow-list anchors instead of trusting anything the model
    # wrote in prose.
    tool_sourced_urls: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Robust shutdown wiring (Phase 3)
# ---------------------------------------------------------------------------
_ACTIVE_STATES: dict[int, "SharedState"] = {}
_SHUTDOWN_LOCK = threading.Lock()
_SHUTDOWN_REQUESTED = False


def _register_shutdown_state(state: "SharedState") -> None:
    _ACTIVE_STATES[id(state)] = state


def _unregister_shutdown_state(state: "SharedState") -> None:
    _ACTIVE_STATES.pop(id(state), None)


# ---------------------------------------------------------------------------
# Phase 5 (W2): live-socket registry for routine delivery
# ---------------------------------------------------------------------------
# Maps user_id -> set of open WebSockets for that user. The routine scheduler
# fires on a ThreadPoolExecutor worker thread and reads this registry from there
# (deliver_routine_notice), so every access is guarded by a threading.Lock.
_LIVE_SOCKETS: dict[str, set[Any]] = {}
_LIVE_SOCKETS_LOCK = threading.Lock()


def _register_live_socket(user_id: str, ws: Any) -> None:
    """Record an open socket so a firing routine can find it (cross-thread)."""
    if not user_id:
        return
    with _LIVE_SOCKETS_LOCK:
        _LIVE_SOCKETS.setdefault(user_id, set()).add(ws)


def _discard_live_socket(user_id: str, ws: Any) -> None:
    """Drop a closed socket; remove the user key entirely when its last socket
    goes so the registry doesn't accumulate empty sets."""
    if not user_id:
        return
    with _LIVE_SOCKETS_LOCK:
        sockets = _LIVE_SOCKETS.get(user_id)
        if sockets is None:
            return
        sockets.discard(ws)
        if not sockets:
            _LIVE_SOCKETS.pop(user_id, None)


async def _shutdown_state(state: "SharedState") -> None:
    session_id = state.session_store.session_id
    if not session_id:
        return
    try:
        await state.session_store.archive_active(status="pending_finalization")
    except Exception as exc:
        print(f"LOG: Shutdown archive failed for {session_id}: {exc}")
    try:
        state.journal_store.flush()
    except Exception as exc:
        print(f"LOG: Shutdown journal flush failed for {session_id}: {exc}")
    if state.sqlite_index is not None:
        try:
            state.sqlite_index.close()
        except Exception as exc:
            print(f"LOG: Shutdown SQLite index close failed for {session_id}: {exc}")


async def _shutdown_all_states() -> None:
    for state in list(_ACTIVE_STATES.values()):
        await _shutdown_state(state)


def _run_shutdown_sync() -> None:
    global _SHUTDOWN_REQUESTED
    with _SHUTDOWN_LOCK:
        if _SHUTDOWN_REQUESTED:
            return
        _SHUTDOWN_REQUESTED = True

    def _runner() -> None:
        try:
            asyncio.run(_shutdown_all_states())
        except Exception as exc:
            print(f"LOG: Shutdown handler failed: {exc}")

    if not _ACTIVE_STATES:
        return
    thread = threading.Thread(target=_runner, name="turtle_shutdown")
    thread.start()
    thread.join()


def _call_prev_handler(prev: object, signum: int, frame: object | None) -> None:
    if callable(prev):
        try:
            prev(signum, frame)
        except Exception:
            pass


_PREV_SIGINT = signal.getsignal(signal.SIGINT)
_PREV_SIGTERM = signal.getsignal(signal.SIGTERM)


def _on_shutdown(signum, frame) -> None:
    _run_shutdown_sync()
    if signum == signal.SIGINT:
        _call_prev_handler(_PREV_SIGINT, signum, frame)
    elif signum == signal.SIGTERM:
        _call_prev_handler(_PREV_SIGTERM, signum, frame)


signal.signal(signal.SIGINT, _on_shutdown)
signal.signal(signal.SIGTERM, _on_shutdown)
atexit.register(_run_shutdown_sync)


# ---------------------------------------------------------------------------
# Helper functions (copied from turtle_voice.py to keep server standalone)
# ---------------------------------------------------------------------------

def _is_user_turn_request(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) for part in message.parts
    )


def _estimate_message_tokens(message: ModelMessage) -> int:
    """Approximate a message's token cost from its CONTENT, not its repr.

    The old estimator was ``len(str(m)) // 4``. ``str()`` on a pydantic-ai
    message renders the whole dataclass repr — every message dragged in
    ~130-155 chars of ``datetime.datetime(...)`` / ``RequestUsage()`` /
    part-class scaffolding, i.e. ~35 PHANTOM tokens each. Across the 40-message
    window that is ~1,450 tokens — 36% of ACTIVE_HISTORY_MAX_TOKENS spent on
    timestamps the model never sees, and for short conversational turns the
    over-count is ~10x. The trim therefore evicted real conversation long before
    the model was anywhere near its context limit: the mechanism behind
    "Turtle forgot what I said a few messages ago".

    Count the actual payload instead — text/content, tool args, tool names —
    plus a small per-part allowance for the role framing a provider adds.
    """
    parts = getattr(message, "parts", None) or []
    chars = 0
    for part in parts:
        content = getattr(part, "content", None)
        if content is not None:
            chars += len(content) if isinstance(content, str) else len(str(content))
        args = getattr(part, "args", None)
        if args is not None:
            chars += len(args) if isinstance(args, str) else len(str(args))
        tool_name = getattr(part, "tool_name", None)
        if tool_name:
            chars += len(str(tool_name))
    return (chars // 4) + (2 * len(parts))


def _trim_history_for_context(history: list[ModelMessage]) -> list[ModelMessage]:
    # Cost each message ONCE: the old code re-ran the estimator over the whole
    # window on every iteration of the shrink loop (O(n^2) re-serialization on
    # the critical path of every turn).
    costs: dict[int, int] = {id(m): _estimate_message_tokens(m) for m in history}

    def _window_tokens(msgs: list[ModelMessage]) -> int:
        return sum(costs.get(id(m)) or _estimate_message_tokens(m) for m in msgs)

    if len(history) <= ACTIVE_HISTORY_MAX_MESSAGES:
        if _window_tokens(history) <= ACTIVE_HISTORY_MAX_TOKENS:
            return history

    user_turns_seen = 0
    start_index = 0
    for index in range(len(history) - 1, -1, -1):
        if _is_user_turn_request(history[index]):
            user_turns_seen += 1
            if user_turns_seen >= ACTIVE_HISTORY_MAX_TURNS:
                start_index = index
                break

    trimmed = history[start_index:]
    if len(trimmed) > ACTIVE_HISTORY_MAX_MESSAGES:
        trimmed = trimmed[-ACTIVE_HISTORY_MAX_MESSAGES:]

    running = _window_tokens(trimmed)
    while trimmed and running > ACTIVE_HISTORY_MAX_TOKENS:
        running -= costs.get(id(trimmed[0])) or _estimate_message_tokens(trimmed[0])
        trimmed = trimmed[1:]

    # ── Front normalization (pair-aware) ──────────────────────────────────
    # Gemini rejects a window that starts with an ORPHAN function_response
    # ("Please ensure that function response turn comes immediately after a
    # function call turn"), and pydantic-ai's empty-user-turn prepend repairs a
    # leading function_CALL but NOT a leading function_RESPONSE. Two rules:
    #   1. Drop a leading dangling assistant turn (ModelResponse) — UNLESS it
    #      carries a ToolCallPart whose ToolReturnPart still survives, because
    #      dropping it would orphan that return (the exact bug that made the
    #      token-trim above turn [user, call, return] into a lone [return] and
    #      400 Gemini on every memory-context-bloated tool turn).
    #   2. Drop leading orphan tool-return turns whose call was trimmed away.
    def _surviving_return_ids(msgs: list[ModelMessage]) -> set[str]:
        return {
            p.tool_call_id
            for m in msgs if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart) and p.tool_call_id
        }

    def _is_leading_orphan(msg: ModelMessage) -> bool:
        if not isinstance(msg, ModelRequest):
            return False
        if not msg.parts:
            return True
        return all(
            isinstance(p, ToolReturnPart)
            or (isinstance(p, RetryPromptPart) and getattr(p, "tool_call_id", None))
            for p in msg.parts
        )

    def _front_normalize(msgs: list[ModelMessage]) -> list[ModelMessage]:
        out = list(msgs)
        while out:
            head = out[0]
            if isinstance(head, ModelResponse):
                rids = _surviving_return_ids(out[1:])
                if any(
                    isinstance(p, ToolCallPart) and p.tool_call_id in rids
                    for p in head.parts
                ):
                    break  # leading model(tool_call) paired with a surviving return — keep the pair
                out = out[1:]
            elif _is_leading_orphan(head):
                out = out[1:]  # orphan return (call already gone) — drop even at length 1
            else:
                break
        return out

    trimmed = _front_normalize(trimmed)

    # Validity guardrail. The window must be non-empty, contain a real user
    # prompt, and NOT start with an orphan tool-return. If trimming left it
    # empty/invalid, fall back to the tail of the ORIGINAL history — which keeps
    # the active turn's user→call→return intact. A slightly-over-budget but VALID
    # window always beats a malformed one that 400s the model.
    def _valid_window(msgs: list[ModelMessage]) -> bool:
        return (
            bool(msgs)
            and not _is_leading_orphan(msgs[0])
            and any(_is_user_turn_request(m) for m in msgs)
        )

    if not _valid_window(trimmed):
        tail = _front_normalize(history[-ACTIVE_HISTORY_MAX_MESSAGES:])
        return tail or history[-ACTIVE_HISTORY_MAX_MESSAGES:]

    return trimmed


def _persist_history(prior: list[ModelMessage] | None, response: Any) -> list[ModelMessage]:
    """Persistence must never shrink the conversation of record.

    history_processors trim the per-call view, and pydantic_ai writes the
    processed list back into run state, so ``response.all_messages()`` returns
    the TRIMMED history. Persisting that erases old turns before the
    reflector / session-end extraction ever read them. Append only this run's
    new messages to the untouched prior history instead.
    """
    prior_list = list(prior or [])
    try:
        new_msgs = list(response.new_messages())
    except Exception:
        return list(response.all_messages())
    return prior_list + new_msgs


def _sanitize_tool_pairs(history: list[ModelMessage]) -> list[ModelMessage]:
    """Drop orphan tool-call / tool-return parts so Gemini accepts the history.

    Gemini direct enforces that every function-call turn is followed by a
    matching function-response turn (and rejects with HTTP 400 INVALID_ARGUMENT
    otherwise). Groq / OpenRouter tolerate gaps. Trimming the context window
    can leave orphans at either boundary — e.g. the front-trim lands on a
    ModelRequest containing a ToolReturnPart whose originating ToolCallPart
    was dropped, or keeps a ModelResponse containing a ToolCallPart whose
    return has been discarded.

    Strategy: compute the set of tool_call_ids that appear as BOTH a call and
    a return anywhere in the surviving history. Keep only those; drop the
    rest. Untagged parts (UserPromptPart, TextPart, SystemPromptPart, plain
    RetryPromptPart without a tool_call_id) pass through untouched. Messages
    left with zero parts are dropped entirely.

    This is provider-agnostic: Groq/OpenRouter are unaffected (they already
    accepted the pairs); Gemini stops 400ing.
    """
    call_ids: set[str] = set()
    return_ids: set[str] = set()
    for msg in history:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_call_id:
                    call_ids.add(part.tool_call_id)
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                tcid = getattr(part, "tool_call_id", None)
                if tcid and isinstance(part, (ToolReturnPart, RetryPromptPart)):
                    return_ids.add(tcid)
    paired = call_ids & return_ids

    def _keep_part(part: Any) -> bool:
        if isinstance(part, ToolCallPart):
            return bool(part.tool_call_id) and part.tool_call_id in paired
        if isinstance(part, ToolReturnPart):
            return bool(part.tool_call_id) and part.tool_call_id in paired
        if isinstance(part, RetryPromptPart):
            tcid = getattr(part, "tool_call_id", None)
            # Plain retries (no tool_call_id) are model-level retry signals,
            # not tied to a specific call — keep them.
            return tcid is None or tcid in paired
        return True

    cleaned: list[ModelMessage] = []
    for msg in history:
        kept_parts = [p for p in msg.parts if _keep_part(p)]
        if not kept_parts:
            continue
        if len(kept_parts) == len(msg.parts):
            cleaned.append(msg)
        else:
            cleaned.append(_dc_replace(msg, parts=kept_parts))

    # Fail-open: pydantic-ai raises UserError("Processed history cannot be
    # empty") if we hand it []. That can happen when trim lands deep in a
    # tool chain whose surviving window is exclusively orphan call/return
    # turns. Returning the input unchanged lets the model attempt the request
    # — Gemini may 400 on adjacency, but `is_key_failure_error` now treats
    # that as fallback-eligible, so the cascade recovers instead of crashing.
    if not cleaned:
        return history
    return cleaned


# Modality-aware turn deadlines (ISSUE-022).
#
# The old code used a flat 60 s for every surface — a value inherited from the
# deleted graph layer. That is roughly FIFTY TIMES the voice-to-voice target: a
# user speaking to Turtle could sit in silence for a minute. Meanwhile a text
# turn that spends 40 s working through a cascade and returns a good answer is a
# perfectly good outcome. One number cannot serve both.
#
# Measured before this change (data/traces/traces.jsonl, 80 turns):
# median 5.7 s, mean 10.5 s, max 43.6 s, with 26/80 turns over 10 s.
_TURN_DEADLINE_VOICE_S = float(os.getenv("TURTLE_TURN_DEADLINE_VOICE_S", "15"))
_TURN_DEADLINE_TEXT_S = float(os.getenv("TURTLE_TURN_DEADLINE_TEXT_S", "60"))
_VOICE_CHANNELS = frozenset({"web_voice", "twilio_voice"})


def _turn_deadline_for(channel: str) -> float:
    """Total cascade budget for a turn on *channel*.

    Voice gets a tight budget because the user is listening to silence; text and
    chat channels keep the historical ceiling because a slow-but-correct answer
    still lands well there.
    """
    return _TURN_DEADLINE_VOICE_S if channel in _VOICE_CHANNELS else _TURN_DEADLINE_TEXT_S


def _detect_task_type(user_text: str) -> str:
    lowered = user_text.lower()
    if "email" in lowered or "mail" in lowered:
        return "email"
    if "http://" in lowered or "https://" in lowered:
        return "url"
    if any(token in lowered for token in ["search", "latest", "news", "top ", "price"]):
        return "web"
    return "general"


def _truncate_tool_output(text: str, *, label: str) -> str:
    if len(text) <= TOOL_OUTPUT_MAX_CHARS:
        return text
    return (
        f"{text[:TOOL_OUTPUT_MAX_CHARS]}\n\n"
        f"[Output truncated: {label} was too long. Ask follow-up questions for specific details.]"
    )


def _wrap_tool_with_envelope(name: str, fn):
    """Wrap a registered tool's coroutine so its return is always the
    sanitised, enveloped ``<untrusted source="name">...</untrusted>`` text —
    see the WP1.H comment at the call site (``_register_tools``) for why this
    is the one place that can cover all twelve tools regardless of what each
    closure returns internally.

    ``functools.wraps`` matters beyond cosmetics here: pydantic-ai builds each
    tool's arguments schema from ``inspect.signature``/``get_type_hints`` on
    the function object passed to ``Agent.tool()``. Both follow the
    ``__wrapped__`` pointer ``functools.wraps`` sets, so the wrapper — despite
    taking ``*args, **kwargs`` — is introspected as if it were the original
    ``(ctx, args)`` closure, and pydantic-ai builds the correct schema.
    """

    @functools.wraps(fn)
    async def _wrapped(*args, **kwargs):
        raw = await fn(*args, **kwargs)
        raw = raw if isinstance(raw, str) else str(raw)
        sanitized = sanitize_for_envelope(raw, max_chars=TOOL_OUTPUT_MAX_CHARS)

        # Collect this turn's tool-sourced URLs onto SharedState (first
        # positional arg is always `ctx: RunContext[SharedState]` per the
        # tool signatures above) so the "done" frame can allow-list them.
        ctx = args[0] if args else kwargs.get("ctx")
        deps = getattr(ctx, "deps", None)
        url_bucket = getattr(deps, "tool_sourced_urls", None)
        if url_bucket is not None:
            try:
                for _url in extract_tool_result_urls(sanitized):
                    if _url not in url_bucket:
                        url_bucket.append(_url)
            except Exception:
                pass

        return wrap_untrusted(name, sanitized)

    return _wrapped


def _compose_prompt_with_memory(user_text: str, memory_context: str | list[str]) -> str:
    if isinstance(memory_context, list):
        context = "\n".join(memory_context).strip()
    else:
        context = str(memory_context).strip()
    if not context:
        return user_text
    return (
        "Relevant user memory:\n"
        f"{context}\n\n"
        "User request:\n"
        f"{user_text}"
    )


async def _resolve_memory_context(state: SharedState, *, task_type: str, user_text: str) -> str:
    """D4: Use RetrievalBroker (4-tier, 400-token budget) as the primary memory source.

    Falls back to PersonalMemoryPromptBuilder. The bypass path
    (_compose_prompt_with_memory calling build_memory_block directly) is
    replaced by this function.
    """
    # Tier 1: RetrievalBroker (4-tier budget-aware retrieval)
    if PERSONAL_MEMORY_ENABLED and state.retrieval_broker is not None:
        try:
            block = await state.retrieval_broker.build_context(
                task_type=task_type,
                query=user_text,
            )
            if block:
                return block
        except Exception as exc:
            print(f"LOG: RetrievalBroker failed ({exc}), falling back to prompt builder")

    # Tier 2: PersonalMemoryPromptBuilder (legacy fallback)
    if PERSONAL_MEMORY_ENABLED:
        try:
            personal_block = state.personal_memory_prompt.build_memory_block(
                task_type=task_type,
                query=user_text,
            )
            if personal_block:
                return personal_block
        except Exception:
            pass

    return ""


def _new_turn_id(state: SharedState) -> str:
    state.turn_counter += 1
    return f"{state.session_store.session_id or 'session'}_turn_{state.turn_counter}"


def _normalize_url_for_cache(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit
    raw = " ".join(url.split())
    try:
        parsed = urlsplit(raw)
        normalized_path = parsed.path or "/"
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), normalized_path, parsed.query, ""))
    except Exception:
        return raw


# NOTE: chat-text confirmation parsing is gone entirely (Phase 4). A "yes" in
# chat is just a word the model answers; memory confirmations happen ONLY in
# the web UI panel via /api/memory/confirm. The old text-parsing path could
# silently promote a stale pending candidate when the user said "yes" to a
# completely unrelated question (Codex P4 review A#2/B#2).


def _queue_confirmation_candidates_from_turn(
    state: SharedState,
    *,
    session_id: str,
    user_text: str,
) -> int:
    """Phase 2 / B1+B2: schedule async multi-turn extraction in the background.

    Returns immediately. The actual extraction + journaling + queueing runs
    as an asyncio task so the user-facing turn doesn't block on the (often
    cheap regex but sometimes LLM) extraction path. Multi-turn flows like
    "save as routine" -> "every day" -> "8 am" need a window of recent
    turns, not the single current utterance.
    """
    if not PERSONAL_MEMORY_ENABLED:
        return 0

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Not in async context (test/voice path) — fall back to sync single-turn.
        return _queue_confirmation_candidates_sync(
            state, session_id=session_id, user_text=user_text
        )

    # RETAINED: asyncio holds only a weak ref to a bare create_task, so this —
    # the per-turn memory candidate extraction — could be GC'd mid-flight and
    # silently drop everything the user just disclosed. track_task keeps a
    # strong ref and logs failures instead of swallowing them.
    task = loop.create_task(
        _queue_confirmation_candidates_async(
            state, session_id=session_id, user_text=user_text
        )
    )
    try:
        from core.worker import track_task
        # Tag by user_id so account-link merge can drain THIS user's in-flight
        # extraction before snapshotting the source journal (Codex flagged that
        # detached extraction outlives the source lock and can append into a
        # now-unreachable journal after the mapping is re-pointed).
        track_task(task, user_id=getattr(state, "user_id", None) or None)
    except Exception:
        pass
    return 0


def _queue_confirmation_candidates_sync(
    state: SharedState,
    *,
    session_id: str,
    user_text: str,
) -> int:
    """Sync single-utterance fallback for environments without a running loop."""
    try:
        profile = state.personal_memory_store.load_profile_snapshot()
        fake_msg = ModelRequest(parts=[UserPromptPart(content=user_text)])
        candidates = extract_memory_candidates_from_messages(
            message_history=[fake_msg],
            session_id=session_id,
            profile=profile,
        )
        return _journal_and_queue_candidates(state, candidates, session_id=session_id)
    except Exception as e:
        print(f"LOG: Confirmation candidate queue (sync) failed for {session_id}: {e}")
        return 0


async def _queue_confirmation_candidates_async(
    state: SharedState,
    *,
    session_id: str,
    user_text: str,
) -> int:
    """B1+B2: windowed multi-turn extraction with LLM fallback. Background task."""
    try:
        profile = state.personal_memory_store.load_profile_snapshot()
        window = max(1, int(settings.memory_extract_window_turns))
        history_tail = list(state.session_store.message_history or [])[-window:]

        # Append the just-received user_text so it's always included even if
        # the session_store hasn't been flushed for this turn yet.
        history_with_current = history_tail + [
            ModelRequest(parts=[UserPromptPart(content=user_text)])
        ]

        candidates = await extract_memory_candidates_from_messages_async(
            message_history=history_with_current,
            session_id=session_id,
            profile=profile,
        )
        return _journal_and_queue_candidates(state, candidates, session_id=session_id)
    except Exception as e:
        print(f"LOG: Confirmation candidate queue (async) failed for {session_id}: {e}")
        return 0


def _journal_and_queue_candidates(
    state: SharedState,
    candidates: list[PersonalMemoryCandidate],
    *,
    session_id: str,
) -> int:
    if not candidates:
        return 0

    applied_events = []
    pending_events = []
    for idx, candidate in enumerate(candidates):
        try:
            event = _candidate_to_journal_event(
                candidate=candidate,
                session_id=session_id,
                ordinal=idx,
            )
        except Exception as exc:
            print(f"LOG: candidate->event conversion failed ({exc.__class__.__name__}: {exc}) for topic={candidate.topic!r} key={candidate.key!r}")
            continue

        if event is None:
            continue

        # llm_turn values are not guaranteed to be substrings of the user text
        # the way regex values are, so recompute evidence support and let the
        # single write-policy make the applied call (an explicit fact whose value
        # is absent from its evidence downgrades to pending).
        if candidate.extraction_source == "llm_turn" and event.applied:
            value_text = str(candidate.value).strip().lower()
            evidence_text = str(candidate.evidence or "").lower()
            evidence_supported = bool(value_text) and value_text in evidence_text
            policy = decide_write_policy(
                source=event.source,
                topic=event.topic,
                confidence=event.confidence,
                evidence_supported=evidence_supported,
            )
            if policy != "applied":
                event = _dc_replace(event, applied=False)

        if event.applied:
            applied_events.append(event)
        else:
            pending_events.append(event)

    events = applied_events + pending_events
    if not events:
        return 0

    try:
        state.journal_store.append_many(events)
    except StorageCapExceededError:
        # Cap hit on the journal append: nothing persisted. Notify the user and
        # bail — queuing pending candidates behind a gate we couldn't journal
        # would be dishonest bookkeeping.
        _notify_storage_cap(state)
        return 0
    try:
        if applied_events:
            result = replay(state.journal_store.load_all(), store=state.personal_memory_store)
            if result.written_topics:
                print(f"LOG: Per-turn memory applied for {session_id}: {result.written_topics}")
                if "workflow" in result.written_topics:
                    _register_user_routines_safe(state.user_id)
    except StorageCapExceededError:
        # Distinct boundary from the append (Codex R1#2): the journal HAS the
        # events; only the rendered projection hit the cap. Notify but keep
        # going — the pending candidates below were journaled and belong in the
        # gate, and the projection regenerates on the next successful replay.
        _notify_storage_cap(state)

    queued = 0
    for event in pending_events:
        if state.confirmation_gate.queue_candidate(event):
            queued += 1

    if queued:
        print(f"LOG: Queued {queued} confirmation candidate(s) for {session_id}")
    return queued


# ---------------------------------------------------------------------------
# Personal memory helpers (mirrors turtle_voice.py — kept standalone)
# ---------------------------------------------------------------------------

def _kind_for_candidate(candidate: PersonalMemoryCandidate) -> str:
    if candidate.topic in {"identity", "contacts", "projects"}:
        return "fact"
    if candidate.topic == "corrections":
        return "correction"
    return "preference"


def _source_for_candidate(candidate: PersonalMemoryCandidate) -> str:
    source = str(candidate.source).strip().lower()
    if source in {"explicit", "inferred", "synthesized", "migration"}:
        return source
    return "inferred"


def _extractor_for_candidate(candidate: PersonalMemoryCandidate) -> str:
    extraction_source = str(candidate.extraction_source).strip().lower()
    if "dream" in extraction_source:
        return "dream_pass"
    if "llm" in extraction_source:
        return "llm_turn"
    return "deterministic"


def _confidence_for_candidate(candidate: PersonalMemoryCandidate) -> float:
    if candidate.confidence == "confirmed":
        return 1.0
    if candidate.confidence == "inferred":
        return 0.8
    return 0.5


def _deterministic_evidence_supported(candidate: PersonalMemoryCandidate) -> bool:
    """Evidence check for the deterministic sync path.

    Regex candidates are extracted from literal user text, but nothing at apply
    time enforced that (Codex review A#2). For the quote-shaped topics —
    identity and contacts, where a mis-parsed value auto-applied at high
    confidence does real damage — require the value to literally appear in the
    captured evidence. Derived values elsewhere (booleans, routine descriptors,
    counts) are not quotes of the user text; for those, non-empty evidence is
    the requirement, since the regex match window *is* the evidence.
    """
    evidence = str(candidate.evidence or "").strip().lower()
    if not evidence:
        return False
    if candidate.topic not in {"identity", "contacts"}:
        return True
    value = str(candidate.value or "").strip().lower()
    return bool(value) and value in evidence


def _should_auto_apply_event(
    *,
    kind: str,
    source: str,
    confidence: float,
    topic: str = "",
    evidence_supported: bool = True,
) -> bool:
    # Thin adapter over the single write-policy registry for the deterministic
    # sync path. Callers with a candidate in hand pass the computed
    # ``_deterministic_evidence_supported`` verdict; the default True is only
    # for legacy call shapes without one. (``kind`` is retained for signature
    # stability; the policy keys off source/topic/confidence/evidence.)
    return decide_write_policy(
        source=source,
        topic=topic,
        confidence=confidence,
        evidence_supported=evidence_supported,
    ) == "applied"


def _candidate_to_journal_event(
    *,
    candidate: PersonalMemoryCandidate,
    session_id: str,
    ordinal: int,
) -> Any | None:
    topic = candidate.topic
    key = candidate.key
    value_text = str(candidate.value).strip()
    value_lower = value_text.lower()
    if not value_text:
        return None

    event_key = ""
    event_value: dict[str, object] = {}

    if topic == "identity" and key == "name":
        event_key = "identity.name"
        event_value = {"name": value_text}
    elif topic == "identity" and key == "home_city":
        event_key = "identity.home_city"
        event_value = {"home_city": value_text}
    elif topic == "identity" and key == "current_city":
        event_key = "identity.current_city"
        event_value = {"current_city": value_text}
    elif topic == "identity" and key == "country":
        event_key = "identity.country"
        event_value = {"country": value_text}
    elif topic == "identity" and key == "timezone":
        event_key = "identity.timezone"
        event_value = {"timezone": value_text}
    elif topic == "identity" and key == "preferred_language":
        event_key = "identity.preferred_language"
        event_value = {"preferred_language": value_text}
    elif topic == "identity" and key == "occupation":
        event_key = "identity.occupation"
        event_value = {"occupation": value_text}
    elif topic == "identity" and key == "company":
        event_key = "identity.company"
        event_value = {"company": value_text}
    elif topic == "identity" and key == "primary_email":
        event_key = "identity.primary_email"
        event_value = {"primary_email": value_lower}
    elif topic == "identity" and key.startswith("known_email:"):
        email = key.split(":", 1)[1].strip().lower() or value_lower
        if not email:
            return None
        event_key = f"identity.known_email.{email}"
        event_value = {"email": email}
    elif topic == "preferences" and key == "response_style":
        event_key = "preferences.response_style"
        event_value = {"response_style": value_text}
    elif topic == "preferences" and key == "humor_level":
        event_key = "preferences.humor_level"
        event_value = {"humor_level": value_text}
    elif topic == "preferences" and key == "email_tone":
        event_key = "preferences.email_tone"
        event_value = {"email_tone": value_text}
    elif topic == "workflow" and key == "prefers_draft_before_send":
        event_key = "workflow.prefers_draft_before_send"
        event_value = {"prefers_draft_before_send": value_lower in {"true", "1", "yes", "y"}}
    elif topic == "workflow" and key == "primary_llm":
        event_key = "workflow.primary_llm"
        event_value = {"primary_llm": value_text}
    elif topic == "workflow" and key in {"morning_routine", "daily_briefing"} or (
        topic == "workflow" and key.startswith("recurring_request")
    ):
        # D1 fix: routine candidates carry a structured dict in value_struct.
        struct = getattr(candidate, "value_struct", None) or {}
        if not isinstance(struct, dict) or "cadence" not in struct:
            return None
        if key.startswith("recurring_request"):
            slug = key.split(".", 1)[1] if "." in key else "routine"
            event_key = f"workflow.recurring_request.{slug}"
        else:
            event_key = f"workflow.{key}"
        event_value = dict(struct)
    elif topic == "relations":
        slug = key.strip().replace(" ", "_") or "person"
        event_key = f"relations.{slug}"
        event_value = {"role": slug, "name": value_text}
    elif topic == "contacts" and key.startswith("frequent_recipient:"):
        email = key.split(":", 1)[1].strip().lower() or value_lower
        if not email:
            return None
        event_key = f"contacts.frequent_recipient.{email}"
        event_value = {"email": email}
    elif topic == "projects" and key.startswith("project:"):
        slug = key.split(":", 1)[1].strip().lower().replace(" ", "_")
        if not slug:
            return None
        event_key = f"projects.project.{slug}"
        event_value = {"name": value_text}
    elif topic == "corrections":
        slug = key.strip().replace(" ", "_") or "note"
        event_key = f"corrections.{slug}"
        event_value = {"summary": value_text}
    elif topic == "working_style":
        slug = key.strip().replace(" ", "_") or "note"
        event_key = f"working_style.{slug}"
        event_value = {"note": value_text}
    elif topic == "communication_style":
        slug = key.strip().replace(" ", "_") or "note"
        event_key = f"communication_style.{slug}"
        event_value = {"note": value_text}
    elif topic == "tool_preferences":
        slug = key.strip().replace(" ", "_") or "tool"
        event_key = f"tool_preferences.{slug}"
        event_value = {"tool": value_text}
    elif topic == "decision_style":
        slug = key.strip().replace(" ", "_") or "note"
        event_key = f"decision_style.{slug}"
        event_value = {"note": value_text}
    else:
        # Generic fallback: any candidate whose topic the journal accepts is
        # persistable — silently dropping unknown keys is how facts like
        # preferences.favourite_editor vanished. Unknown topics still bail.
        import re
        from core.memory_journal import ALLOWED_TOPICS
        if topic not in ALLOWED_TOPICS or not value_text:
            return None
        slug = re.sub(r"[^a-z0-9]+", "_", str(key or "note").strip().lower()).strip("_") or "note"
        event_key = f"{topic}.{slug}"
        event_value = {"value": value_text}

    stable_payload = {
        "session": session_id,
        "topic": topic,
        "key": event_key,
        "value": event_value,
        "evidence": candidate.evidence,
        "ord": ordinal,
    }
    digest = hashlib.sha1(
        json.dumps(stable_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    kind = _kind_for_candidate(candidate)
    source = _source_for_candidate(candidate)
    confidence = _confidence_for_candidate(candidate)

    return make_event(
        event_id=f"sync_{digest[:22]}",
        kind=kind,
        topic=topic,
        key=event_key,
        value=event_value,
        confidence=confidence,
        source=source,
        extractor=_extractor_for_candidate(candidate),
        session_id=session_id,
        turn_id=f"{session_id}_sync_{ordinal}",
        evidence={
            "user_text": candidate.evidence,
            "observation_count": 1,
        },
        applied=_should_auto_apply_event(
            kind=kind,
            source=source,
            confidence=confidence,
            topic=topic,
            evidence_supported=_deterministic_evidence_supported(candidate),
        ),
        # Snapshot the projection on the assembled event so the replayer renders
        # it verbatim (statement-based rendering).
        statement=statement_for(topic, event_key, event_value),
    )


def _sync_personal_memory_from_messages(
    state: "SharedState",
    *,
    session_id: str | None,
    message_history: list[ModelMessage],
) -> None:
    """Extract memory candidates from message history and write applied facts to the journal."""
    if not PERSONAL_MEMORY_ENABLED or not session_id or not message_history:
        return
    try:
        profile = state.personal_memory_store.load_profile_snapshot()
        candidates = extract_memory_candidates_from_messages(
            message_history=message_history,
            session_id=session_id,
            profile=profile,
        )
        if not candidates:
            return

        events = [
            event
            for index, candidate in enumerate(candidates)
            for event in [_candidate_to_journal_event(candidate=candidate, session_id=session_id, ordinal=index)]
            if event is not None
        ]
        if not events:
            return

        state.journal_store.append_many(events)

        queued_candidates = 0
        for event in events:
            if event.applied:
                continue
            if event.source == "explicit":
                continue
            if state.confirmation_gate.queue_candidate(event):
                queued_candidates += 1

        replay_result = replay(state.journal_store.load_all(), store=state.personal_memory_store)
        if replay_result.written_topics or replay_result.cleared_topics:
            topics = ", ".join(replay_result.written_topics) if replay_result.written_topics else "none"
            print(
                f"LOG: Personal memory updated for {session_id} "
                f"({len(events)} events -> {replay_result.resolved_event_count} resolved entries across {topics})"
            )
            if queued_candidates:
                print(f"LOG: Queued {queued_candidates} inferred memory candidate(s) for confirmation")
    except Exception as e:
        print(f"LOG: Personal memory sync failed for {session_id}: {e}")
        traceback.print_exc()


async def _sync_personal_memory_from_archive(
    state: "SharedState",
    *,
    session_id: str | None,
    archive_path: Path,
) -> None:
    """Read archived session messages and extract personal memory into the journal."""
    if not PERSONAL_MEMORY_ENABLED or not session_id:
        return
    messages_path = archive_path / "messages.json"
    if not messages_path.exists():
        print(f"LOG: No messages file for personal memory sync {session_id}")
        return
    try:
        message_history = ModelMessagesTypeAdapter.validate_json(messages_path.read_bytes())
    except Exception as e:
        print(f"LOG: Unable to read archived messages for personal memory sync {session_id}: {e}")
        return
    _sync_personal_memory_from_messages(state, session_id=session_id, message_history=message_history)
    try:
        await run_stage_b_session_extractor(
            state,
            session_id=session_id,
            message_history=message_history,
        )
    except Exception as e:
        print(f"LOG: Stage B session extractor failed for {session_id}: {e}")


def _register_user_routines_safe(user_id: str) -> None:
    """Phase 4 / E1: re-scan + register a user's routines after a write.

    Idempotent — APScheduler replaces existing job ids on re-registration.
    In cloud mode get_routine_scheduler() is always None (RoutineScheduler is
    gated off there — see the cloud branch in _start_routine_scheduler), so
    this is a correct no-op: the cron-tick endpoint discovers routines fresh
    on every tick (core.routine_scheduler.get_active_routines_for_user)
    without any registration step to re-run.
    """
    if not user_id:
        return
    try:
        sched = get_routine_scheduler()
        if sched is None:
            return
        n = sched.register_for_user(user_id)
        if n:
            print(f"LOG: Re-registered {n} routine(s) for {user_id}")
    except Exception as e:
        print(f"LOG: routine registration failed for {user_id}: {e}")


# ---------------------------------------------------------------------------
# Storage-cap user notification (W4 / Phase 3)
#
# core.guardrails.enforce_storage_cap raises StorageCapExceededError when a
# user's memory dir is at its cap (settings.user_storage_cap_mb). Before this,
# breaches vanished into broad `except Exception` logs and the user's memory
# writes failed silently. These helpers surface the breach:
#   * always a LOG line (server-side, works today);
#   * a WS "notice" frame when a websocket is reachable (see delivery note).
#
# Delivery mechanism (documented for the integrator finishing the concurrent
# handler refactor): the per-turn write funnels this fires from
# (_apply_explicit_facts_from_turn, _journal_and_queue_candidates) are sync,
# take no `ws`, and their call sites live in the handler region another agent
# owns — and SharedState can't be extended here. So today the frame is stashed
# in a per-session pending registry and always logged; the remember tool
# (which returns a string the model relays) delivers a user-visible message
# directly. When the refactor lands, the handler should drain the pending
# notice next to its other _ws_send_json calls, e.g.:
#     notice = pop_pending_storage_cap_notice(state.session_store.session_id or state.user_id)
#     if notice: await _ws_send_json(ws, notice)
# Passing a live `ws` into _notify_storage_cap also sends immediately (dormant
# today because no caller has one to pass).
_STORAGE_CAP_NOTICE_CODE = "storage_cap"
_STORAGE_CAP_NOTICE_MESSAGE = (
    "Memory storage is full — new facts can't be saved. "
    "Ask me to forget things, or contact the admin to raise the cap."
)
# Bounded to keep these from growing without limit on a long-lived process.
_STORAGE_CAP_REGISTRY_CAP = 512
# Once-per-session guard so a user isn't spammed every failing write this turn.
_STORAGE_CAP_NOTIFIED: dict[str, float] = {}
# Frames awaiting a websocket to carry them (drained by the handler; see note).
_PENDING_STORAGE_CAP_NOTICES: dict[str, dict[str, Any]] = {}


def build_storage_cap_notice() -> dict[str, Any]:
    """The WS frame the browser renders as a toast (see websocket.js 'notice')."""
    return {
        "type": "notice",
        "code": _STORAGE_CAP_NOTICE_CODE,
        "message": _STORAGE_CAP_NOTICE_MESSAGE,
    }


def _storage_cap_key(state: "SharedState") -> str:
    """Stable per-session identity, defensive about partial test doubles."""
    session_store = getattr(state, "session_store", None)
    session_id = getattr(session_store, "session_id", None) if session_store else None
    return str(session_id or getattr(state, "user_id", "") or f"state_{id(state)}")


def _bounded_put(registry: dict[str, Any], key: str, value: Any) -> None:
    """Insert into a dict with a hard size cap (clears wholesale when full)."""
    if key not in registry and len(registry) >= _STORAGE_CAP_REGISTRY_CAP:
        registry.clear()
    registry[key] = value


def pop_pending_storage_cap_notice(key: str) -> dict[str, Any] | None:
    """Handler-facing: fetch and clear a session's pending storage-cap notice."""
    return _PENDING_STORAGE_CAP_NOTICES.pop(key, None)


def _notify_storage_cap(state: "SharedState", ws: Any | None = None) -> bool:
    """Surface a storage-cap breach to the user, at most once per session.

    Always prints a LOG line. Stashes a WS notice frame for the handler to
    deliver, and—if a live ws is supplied—schedules an immediate send.
    Returns True on the first call for a session, False on subsequent ones.
    """
    key = _storage_cap_key(state)
    if key in _STORAGE_CAP_NOTIFIED:
        return False
    _bounded_put(_STORAGE_CAP_NOTIFIED, key, time.time())

    print(f"LOG: storage cap reached — memory write blocked for session {key}")
    frame = build_storage_cap_notice()

    if ws is not None:
        # Immediate delivery when a caller has a websocket in hand. Send OR
        # queue, never both — queueing too would double-toast the user when
        # the turn-end drain fires (Codex R1#6).
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(_ws_send_json(ws, frame))
            # Swallow send failures; an unobserved task exception would log noisily.
            task.add_done_callback(lambda t: t.exception())
            return True
        except RuntimeError:
            pass  # no running loop — fall through to the pending registry
    _bounded_put(_PENDING_STORAGE_CAP_NOTICES, key, frame)
    return True


def _store_remembered_fact(
    state: "SharedState",
    *,
    topic: str,
    key_slug: str,
    value_text: str,
    mode: str = "replace",
) -> str:
    """Persist an explicit user-stated fact; return the agent-facing result string.

    Extracted from the `remember` tool (a closure, so otherwise untestable) so
    the storage-cap failure path can be unit tested. On a cap breach the fact is
    NOT saved: the user is notified and an honest failure string is returned
    instead of a fabricated "Stored".

    mode="add" (model-chosen) stores an ADDITIONAL value for a multi-valued fact
    (another email, phone, project…): the value is slugged into the key so each
    distinct value gets its own (topic,key) slot and survives the latest-per-key
    projection collapse — instead of overwriting the previous value. mode="replace"
    (default) keeps the single canonical slot for single-valued facts/corrections.
    """
    from core.memory_journal import generate_event_id
    import re as _re

    additive = (mode or "replace").strip().lower() == "add"
    if additive:
        value_slug = _re.sub(r"[^a-z0-9]+", "_", value_text.lower()).strip("_")[:40] or "value"
        stored_key_slug = f"{key_slug}.{value_slug}"
    else:
        stored_key_slug = key_slug

    # Bug B backstop: the model sometimes calls `remember` twice for one fact
    # under two keys (e.g. projects.codename_atlas="I'm working on a project
    # codenamed Atlas." AND projects.project_codename="Atlas"). Collapse a
    # restated fact — same topic, same session, where one applied value contains
    # the other — into the entry already stored. Length-gated (>=4 chars) and
    # skip-not-supersede so short distinct facts ("Sam" vs "Sam Smith") and
    # already-applied data are never clobbered. SKIPPED for mode="add": additive
    # values are meant to accumulate, and one email being a substring of another
    # must never silently drop it.
    new_norm = (value_text or "").strip().lower()
    session_id = state.session_store.session_id or "unknown_session"
    if not additive and len(new_norm) >= 4:
        try:
            recent = state.journal_store.load_all()[-50:]
        except Exception:
            recent = []
        for ev in recent:
            if ev.topic != topic or not getattr(ev, "applied", False):
                continue
            if getattr(ev, "session_id", None) != session_id:
                continue
            existing = " ".join(str(v) for v in (ev.value or {}).values()).strip().lower()
            if len(existing) < 4:
                continue
            if existing == new_norm or existing in new_norm or new_norm in existing:
                return ToolResult.ok(
                    f"Already remembered ({ev.key}): {topic}.{key_slug} = {value_text}"
                ).to_agent_string()

    try:
        event = make_event(
            event_id=generate_event_id(),
            kind="fact",
            topic=topic,
            key=f"{topic}.{stored_key_slug}",
            value={"value": value_text},
            confidence=1.0,
            source="explicit",
            extractor="deterministic",
            session_id=state.session_store.session_id or "unknown_session",
            turn_id=f"remember_{generate_event_id()[:8]}",
            evidence={"note": "user asked Turtle to remember this"},
            applied=True,
        )
        state.journal_store.append_many([event])
    except StorageCapExceededError:
        # Cap hit on the journal append — the fact truly was NOT saved. Honest
        # failure the model relays to the user instead of a fake "Stored".
        _notify_storage_cap(state)
        return (
            "I couldn't save that — memory storage is at its cap. "
            "Ask me to forget something to free up space."
        )
    except Exception as e:
        return ToolResult.upstream_error(f"Could not store the memory: {e}").to_agent_string()

    try:
        replay(state.journal_store.load_all(), store=state.personal_memory_store)
    except StorageCapExceededError:
        # Distinct boundary from the append (Codex R1#2): the journal — the
        # source of truth — HAS the fact; only the rendered projection failed.
        # Claiming "couldn't save" here would be false, and the projection
        # regenerates on the next successful replay.
        _notify_storage_cap(state)
        return (
            f"Noted: {topic}.{stored_key_slug} = {value_text}. But memory storage is at "
            "its cap, so my memory files couldn't refresh — ask me to forget "
            "things to free up space."
        )
    except Exception as e:
        print(f"LOG: remember-tool replay failed after journal append: {e}")

    return ToolResult.ok(f"Stored: {topic}.{stored_key_slug} = {value_text}").to_agent_string()


def _apply_explicit_facts_from_turn(
    state: "SharedState",
    *,
    session_id: str,
    turn_id: str,
    user_text: str,
    task_type: str,
) -> None:
    """Immediately write high-confidence explicit facts (email, name) to the journal.

    Called per-turn so disclosures like 'my email is X' are reflected in the
    next turn's memory context without waiting for session-end replay.

    Phase 4: the old workflow auto-promote-on-"yes" step is gone — chat text
    never confirms pending memory anymore; /api/memory/confirm is the one
    confirmation surface.
    """
    if not PERSONAL_MEMORY_ENABLED:
        return
    try:
        profile = state.personal_memory_store.load_profile_snapshot()
        # Use a minimal single-message history to reuse candidate extraction + dedup
        fake_msg = ModelRequest(parts=[UserPromptPart(content=user_text)])
        candidates = extract_memory_candidates_from_messages(
            message_history=[fake_msg],
            session_id=session_id,
            profile=profile,
        )
        # Only auto-apply explicit high-confidence facts; behaviors stay in the gate
        explicit_candidates = [
            c for c in candidates
            if c.source == "explicit" and c.topic in {
                "identity",
                "preferences",
                "workflow",
                "contacts",
                "projects",
                "corrections",
            }
        ]
        if not explicit_candidates:
            return

        events = [
            event
            for idx, candidate in enumerate(explicit_candidates)
            for event in [_candidate_to_journal_event(candidate=candidate, session_id=session_id, ordinal=idx)]
            if event is not None and _should_auto_apply_event(
                kind=_kind_for_candidate(candidate),
                source=_source_for_candidate(candidate),
                confidence=_confidence_for_candidate(candidate),
                topic=candidate.topic,
                evidence_supported=_deterministic_evidence_supported(candidate),
            )
        ]
        if not events:
            return

        state.journal_store.append_many(events)
        result = replay(state.journal_store.load_all(), store=state.personal_memory_store)
        if result.written_topics:
            print(f"LOG: Per-turn memory applied for {session_id}: {result.written_topics}")
            if "workflow" in result.written_topics:
                _register_user_routines_safe(state.user_id)
    except StorageCapExceededError:
        # The write funnel hit the per-user storage cap. Tell the user their
        # memory is full instead of swallowing it as a generic failure below.
        _notify_storage_cap(state)
    except Exception as e:
        print(f"LOG: Per-turn fact extraction failed for {session_id}: {e}")


def _runtime_agent_registry() -> list[dict[str, Any]]:
    main_model = str(config.get("MAIN_AGENT_MODEL") or f"groq:{config.get('GROQ_PRIMARY_MODEL', 'llama-3.3-70b-versatile')}")
    email_model = str(config.get("EMAIL_AGENT_MODEL") or main_model)
    stage_b_model = f"groq:{settings.personal_memory_stage_b_model}"

    return [
        {
            "id": "main_assistant",
            "label": "Main Assistant",
            "model": main_model,
            "editable": True,
            "config_key": "MAIN_AGENT_MODEL",
            "status": "active",
        },
        {
            "id": "email_specialist",
            "label": "Email Specialist",
            "model": email_model,
            "editable": True,
            "config_key": "EMAIL_AGENT_MODEL",
            "status": "active",
        },
        {
            "id": "stage_b_extractor",
            "label": "Stage B Memory Extractor",
            "model": stage_b_model,
            "editable": False,
            "status": "active" if PERSONAL_MEMORY_ENABLED else "disabled",
        },
        {
            "id": "main_fallback_chain",
            "label": "Main Fallback Chain",
            "model": f"{len(agents_mgr.main_assistant_fallbacks)} model(s)",
            "editable": False,
            "status": "derived",
        },
        {
            "id": "email_fallback_chain",
            "label": "Email Fallback Chain",
            "model": f"{len(agents_mgr.email_agent_fallbacks)} model(s)",
            "editable": False,
            "status": "derived",
        },
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# remember tool args
# ---------------------------------------------------------------------------
from pydantic import BaseModel as _RememberBaseModel, Field as _RememberField


class RememberArgs(_RememberBaseModel):
    topic: str = _RememberField(
        ...,
        description=(
            "Memory topic: identity|preferences|workflow|contacts|relations|projects|"
            "corrections|working_style|communication_style|tool_preferences|decision_style"
        ),
    )
    key: str = _RememberField(
        ...,
        description="Short snake_case identifier, e.g. favourite_editor.",
    )
    value: str = _RememberField(
        ...,
        description="The fact as stated by the user.",
    )
    mode: str = _RememberField(
        "replace",
        description=(
            "'replace' (default) for a single-valued fact or a correction — the "
            "new value supersedes any previous value under this key. 'add' when "
            "the user is providing ANOTHER value for something they can have "
            "several of (e.g. an additional email, phone, address, or project) — "
            "it accumulates alongside the existing values instead of overwriting "
            "them. Use 'add' whenever the user says things like 'also', 'another', "
            "'add', or lists more than one."
        ),
    )


# ---------------------------------------------------------------------------
# link_account tool args (WP1.J — two-sided link-code binding, ledger 1b.6)
# ---------------------------------------------------------------------------
class LinkAccountArgs(_RememberBaseModel):
    expected_email: str = _RememberField(
        ...,
        description=(
            "The email address of the Turtle WEB account the user will sign in "
            "with to finish linking. Ask for it if they haven't given it. This is "
            "never treated as proof of anything by itself — the authenticated web "
            "session is still what proves ownership — it only lets the server "
            "refuse a code redeemed by the wrong account."
        ),
    )


def _build_model_from_str(model_str: str, settings: Any) -> Any | None:
    """Parse 'provider:model_name' and return a pydantic-ai model object."""
    if not model_str:
        return None
    if model_str.startswith("groq:"):
        return get_groq_model(model_name=model_str[5:], settings=settings)
    if model_str.startswith("openrouter:"):
        models = get_openrouter_models(model_name=model_str[11:], settings=settings)
        return models[0] if models else None
    if model_str.startswith("gemini:"):
        models = get_google_models(model_name=model_str[7:], settings=settings)
        return models[0] if models else None
    return None


# ---------------------------------------------------------------------------
# Agent builder — creates agent chain from current config, supports hot-reload
# ---------------------------------------------------------------------------

class AgentManager:
    """Builds and hot-reloads the Pydantic AI agent chain."""

    def __init__(self) -> None:
        self.model_settings: dict[str, Any] = {}
        self.main_assistant: Agent | None = None
        self.main_assistant_fallbacks: list[Agent] = []
        self.email_agent: Agent | None = None
        self.email_agent_fallbacks: list[Agent] = []
        # WP1.D2 (ledger 1a.4 part 4): total_tokens_limit bounds ONE turn's
        # worst case, independent of the daily budget check (which only sees
        # spend recorded by turns that already finished). 100_000 is ~10% of
        # the default daily budget (TURTLE_DAILY_TOKEN_BUDGET=1_000_000) — a
        # single pathological cascade (many fallback rungs, each burning a
        # fraction of the 30-request cap with its own prompt+context) can cap
        # out there without ever touching it in normal 1-3 rung turns, while
        # still keeping the worst-case daily-budget overshoot from one turn
        # to a tenth of a day's allowance rather than unbounded.
        self.usage_limits = UsageLimits(request_limit=30, total_tokens_limit=100_000)
        self.stt = FastRTCSTT(groq_client=groq_client)
        self.rebuild(config)  # stt rebuilt inside rebuild()

    def rebuild(self, cfg: dict[str, Any]) -> None:
        """Rebuild all agents from the given config dict."""
        self.model_settings = {
            "temperature": float(cfg.get("temperature", 0.2)),
            "max_tokens": int(cfg.get("max_tokens", 1024)),
        }
        settings = self.model_settings

        # Update STT model on every rebuild
        stt_model = cfg.get("STT_MODEL", "whisper-large-v3-turbo")
        self.stt = FastRTCSTT(groq_client=groq_client, model=stt_model)

        # Model pools — one model object per provider API key. Each pool is
        # ordered so the first element is the preferred entry-point, and a pool
        # with N keys lets the cascade retry the SAME model on the next key
        # before it drops to a different provider.
        #
        # Roster (2026-08-25 directive): ox-alpha discarded (it intermittently
        # returned empty output on multi-tool synthesis, timing out the turn).
        # Gemini 2.5 Flash on every Google key is the primary; gpt-oss-20b on
        # every Groq key is the second rung; Gemini via OpenRouter (every OR key)
        # is the last-resort safety net. GEMINI_MODEL / GROQ_PRIMARY_MODEL /
        # OPEN_ROUTER_MODEL in turtle_config.json name gemini-2.5-flash /
        # gpt-oss-20b / gemini-2.5-flash so these pools carry the whole roster.
        openrouter_models = get_openrouter_models(
            model_name=cfg.get("OPEN_ROUTER_MODEL"), settings=settings,
        )
        gemini_models = get_google_models(
            model_name=cfg.get("GEMINI_MODEL"), settings=settings,
        )
        # gpt-oss-20b across ALL Groq keys — replaces the decommissioned llama
        # slugs (Groq 404'd every llama-3.x call). GROQ_FALLBACK_MODEL now points
        # at the same slug, so both keys of gpt-oss-20b are covered by one pool.
        groq_gpt_oss = get_groq_models(
            model_name=cfg.get("GROQ_PRIMARY_MODEL"), settings=settings,
        )

        if not (openrouter_models or gemini_models or groq_gpt_oss):
            raise RuntimeError(
                "No model providers available. Set GEMINI_API_KEY, "
                "OPEN_ROUTER_API_KEY_*, or GROQ_API_KEY."
            )

        # Per-agent cascades. Each is a flat list [primary, *fallbacks] composed
        # in priority order; build_chain() drops Nones and de-dupes identity.
        def build_chain(*candidates: Any) -> list[Any]:
            seen: list[Any] = []
            for c in candidates:
                if c is None:
                    continue
                items = c if isinstance(c, list) else [c]
                for item in items:
                    if item is not None and id(item) not in {id(s) for s in seen}:
                        seen.append(item)
            return seen

        # If the per-agent override resolves to the same provider+model as the
        # head of a pool, skip the override (it'd create a redundant first-rung
        # retry on the same API key). Pool ordering already encodes the
        # desired primary anyway.
        def _override_redundant(override: Any, pool: list[Any]) -> bool:
            if override is None or not pool:
                return False
            head = pool[0]
            return (
                type(override) is type(head)
                and getattr(override, "model_name", None) == getattr(head, "model_name", None)
            )

        # main_assistant cascade (2026-08-25 roster, ox-alpha discarded):
        #   Gemini 2.5 Flash (all Google keys) → gpt-oss-20b (all Groq keys) →
        #   Gemini via OpenRouter (all OR keys, last-resort safety net).
        #   - Gemini direct leads: proven tool-caller with a large free tier. Its
        #     occasional function-call-adjacency 400 is a harmony-class 400 that
        #     run_agent_with_fallbacks + health_tracker treat as fallback-eligible
        #     and cool at family scope, so a rejected tool turn drops to gpt-oss.
        #   - gpt-oss-20b is the second rung across every Groq key.
        #   - Gemini via OpenRouter reuses the OpenRouter keys as a different route
        #     to the same capable model, so a total Google+Groq outage still
        #     answers. (OPEN_ROUTER_MODEL now names gemini-2.5-flash, not ox-alpha.)
        # An explicit MAIN_AGENT_MODEL override (env or config) still leads if set;
        # when it names the Gemini pool head it collapses onto the pool instead of
        # prepending a duplicate model on the same key.
        main_override = _build_model_from_str(
            os.getenv("MAIN_AGENT_MODEL") or cfg.get("MAIN_AGENT_MODEL", ""),
            settings,
        )
        if _override_redundant(main_override, gemini_models):
            main_override = None
        main_head: Any = main_override or (
            gemini_models[0] if gemini_models
            else (groq_gpt_oss[0] if groq_gpt_oss else (openrouter_models[0] if openrouter_models else None))
        )
        main_chain = build_chain(
            main_head, gemini_models, groq_gpt_oss, openrouter_models,
        )

        # email_agent: same roster as main — Gemini → gpt-oss-20b → Gemini(OR).
        # Email composition is structure-heavy; Gemini's instruction-following
        # handles it and the OpenRouter route is the final rung when Google + Groq
        # are both unavailable.
        email_override = _build_model_from_str(
            os.getenv("EMAIL_AGENT_MODEL") or cfg.get("EMAIL_AGENT_MODEL", ""),
            settings,
        )
        if _override_redundant(email_override, gemini_models):
            email_override = None
        email_head: Any = email_override or (
            gemini_models[0] if gemini_models
            else (groq_gpt_oss[0] if groq_gpt_oss else (openrouter_models[0] if openrouter_models else None))
        )
        email_chain = build_chain(
            email_head, gemini_models, groq_gpt_oss, openrouter_models,
        )

        if not main_chain or not email_chain:
            raise RuntimeError(
                "Cannot build agent chain — no usable model for main/email agent."
            )

        # Main assistant. Every tool is offered on every turn and every rung of
        # the cascade — no per-intent tool scoping. The _register_tools loop
        # (F5) registers the identical toolset on each fallback so a model swap
        # never silently loses a capability.
        self.main_assistant = Agent(
            main_chain[0],
            deps_type=SharedState,
            output_type=str,
            output_retries=OUTPUT_RETRIES,
            instructions=MAIN_ASSISTANT_PROMPT,
            history_processors=[_trim_history_for_context, _sanitize_tool_pairs],
        )
        self.main_assistant_fallbacks = [
            Agent(m, deps_type=SharedState, output_type=str,
                  output_retries=OUTPUT_RETRIES, instructions=MAIN_ASSISTANT_PROMPT,
                  history_processors=[_trim_history_for_context, _sanitize_tool_pairs])
            for m in main_chain[1:]
        ]

        # Email agent
        self.email_agent = Agent(
            email_chain[0],
            deps_type=SharedState,
            output_type=str,
            output_retries=OUTPUT_RETRIES,
            instructions=EMAIL_PROMPT,
            history_processors=[_trim_history_for_context, _sanitize_tool_pairs],
        )
        self.email_agent_fallbacks = [
            Agent(m, deps_type=SharedState, output_type=str,
                  output_retries=OUTPUT_RETRIES, instructions=EMAIL_PROMPT,
                  history_processors=[_trim_history_for_context, _sanitize_tool_pairs])
            for m in email_chain[1:]
        ]

        # Register tools on the main assistant
        self._register_tools()
        print(
            f"LOG: Agent chain rebuilt — "
            f"main={os.getenv('MAIN_AGENT_MODEL') or cfg.get('MAIN_AGENT_MODEL') or cfg.get('GROQ_PRIMARY_MODEL', 'default')}, "
            f"email={os.getenv('EMAIL_AGENT_MODEL') or cfg.get('EMAIL_AGENT_MODEL') or 'same'}, "
            f"stt={stt_model}, "
            f"temp={settings.get('temperature')}, max_tokens={settings.get('max_tokens')}"
        )

    def _register_tools(self) -> None:
        """Register all tools on the main assistant and every fallback rung."""
        from pathlib import Path as _Path

        def _load_tool_contract(name: str) -> str:
            """Load tool contract markdown as the tool description."""
            md_path = (
                _Path(__file__).resolve().parents[1]
                / "core" / "system_prompts" / "tools" / f"{name}.md"
            )
            try:
                return md_path.read_text(encoding="utf-8")
            except Exception:
                return f"Tool: {name}"  # graceful fallback

        agent = self.main_assistant

        # Per-turn dynamic instructions: inject the user-specific greeting
        # block so a freshly-onboarded user gets greeted by name and a stranger
        # gets a gentle "ask once" hint. Runs once per turn against the live
        # SharedState (which carries the resolved user_id).
        def _attach_user_greeting(target_agent: Agent) -> None:
            @target_agent.instructions
            async def _turn_instructions(ctx: RunContext[SharedState]) -> str:
                try:
                    return _build_turn_instructions(ctx.deps)
                except Exception:
                    return ""

        _attach_user_greeting(agent)
        for fb in self.main_assistant_fallbacks:
            _attach_user_greeting(fb)

        async def search_web(ctx: RunContext[SharedState], args: WebSearchArgs) -> str:
            """Search the web for real-time information. See tool contract for full spec."""
            query = args.query.strip()
            if not query:
                return ToolResult.invalid("query must not be empty", code="invalid_args").to_agent_string()
            print(f"\nSEARCHING: Web search for: {query!r}")
            normalized_query = " ".join(query.split())
            cache_key = f"web::{normalized_query}"
            cached = ctx.deps.search_cache.get(cache_key)
            if cached:
                return cached
            try:
                results = await search_duckduckgo(ctx.deps.http_client, normalized_query, max_results=10)
                formatted = format_search_results(normalized_query, results)
                if not results:
                    return ToolResult.empty("No search results found for this query.").to_agent_string()
            except Exception as e:
                return ToolResult.upstream_error(f"Web search failed: {e}").to_agent_string()
            cleaned = clean_text_for_model(formatted)
            trimmed = _truncate_tool_output(cleaned, label="web search results")
            ctx.deps.search_cache[cache_key] = trimmed
            return trimmed

        async def search_url(ctx: RunContext[SharedState], args: UrlFetchArgs) -> str:
            """Fetch and extract content from a specific URL. See tool contract for full spec."""
            url = args.url.strip()
            if not url:
                return ToolResult.invalid("url must not be empty").to_agent_string()
            print(f"\nANALYZING: URL content extraction from {url}")
            normalized_url = _normalize_url_for_cache(url)
            cache_key = f"url::{normalized_url}"
            cached = ctx.deps.search_cache.get(cache_key)
            if cached:
                return cached
            try:
                result = await fetch_url_content_async(ctx.deps.http_client, normalized_url)
                cleaned = clean_text_for_model(result.to_formatted_string())
                trimmed = _truncate_tool_output(cleaned, label="url analysis")
            except Exception as e:
                return ToolResult.upstream_error(f"URL fetch failed: {e}").to_agent_string()
            ctx.deps.search_cache[cache_key] = trimmed
            return trimmed

        async def send_email_assistant(ctx: RunContext[SharedState], args: EmailArgs) -> str:
            """Send emails on behalf of the user. See tool contract for full spec."""
            query = args.query.strip()
            if not query:
                return ToolResult.invalid("query describing email request must not be empty").to_agent_string()
            print(f"\nEMAIL: Delegating to email specialist")
            pending_email = ctx.deps.session_store.get_pending_email()
            deterministic = extract_deterministic_email_details(query)
            # "x@gmail" (no ".com") extracts zero recipients — the strict
            # regex correctly refuses to guess a TLD. If the user is simply
            # confirming a completion we proposed last turn ("yes"), use it
            # now instead of asking them to retype the address.
            if not deterministic["recipients"]:
                confirmed_recipient = resolve_suggested_recipient(query, pending_email)
                if confirmed_recipient:
                    deterministic["recipients"] = [confirmed_recipient]
            # The email sub-agent runs WITHOUT the main conversation, so a request
            # like "email me the fetched news" would otherwise lose the content it
            # names. Capture the recent conversation + fetched results to hand to
            # the composer (see _recent_conversation_context).
            conversation_context = _recent_conversation_context(ctx.deps)

            known_contacts: dict[str, Any] = {}
            try:
                _snapshot = ctx.deps.personal_memory_store.load_profile_snapshot()
                known_contacts = {
                    "contacts": _snapshot.get("contacts") or {},
                    "relations": _snapshot.get("relations") or {},
                }
            except Exception:
                known_contacts = {}

            context_section = (
                "Recent conversation and fetched content (for resolving references "
                "only). Use it to understand what the user points at ('the news', "
                "'that summary', 'it', 'send it') and to fill recipients/subject when "
                "they are clear from it. Do NOT dump this content into 'content': if "
                "the user DELEGATED the writing (e.g. 'email me the news', 'write it "
                "for me'), leave 'content' EMPTY so it can be authored properly; only "
                "fill 'content' with wording the user DICTATED verbatim.\n"
                "--- context ---\n"
                f"{conversation_context}\n"
                "--- end context ---\n\n"
                if conversation_context and conversation_context.strip()
                else ""
            )
            extraction_prompt = (
                "Extract only email send fields from the latest user request.\n"
                "Rules:\n"
                "- Do not invent values that are not present in latest message or clear context.\n"
                "- Return recipients as a list of email strings.\n"
                "- Return cc_recipients as a list of email strings when user specifies cc.\n"
                "- Return bcc_recipients as a list of email strings when user specifies bcc.\n"
                "- Return empty strings for missing subject/content.\n"
                "- send_intent should be true only when user asks to send now.\n"
                "- If the user names a person (e.g. 'my manager', 'Keshav') and Known contacts below contains a matching address, use it; never invent addresses.\n\n"
                f"{context_section}"
                f"Known contacts:\n{json.dumps(known_contacts, ensure_ascii=False)}\n\n"
                f"Current pending email state:\n{json.dumps(pending_email, ensure_ascii=False)}\n\n"
                f"Deterministic extraction hints:\n{json.dumps(deterministic, ensure_ascii=False)}\n\n"
                f"Latest user request:\n{query}"
            )

            extraction_result = await run_agent_with_fallbacks(
                agents_mgr.email_agent,
                agents_mgr.email_agent_fallbacks,
                extraction_prompt,
                deps=ctx.deps,
                usage=ctx.usage,
            )
            llm_extraction = parse_email_extraction_response(extraction_result.output).model_dump()
            latest_fields = combine_extracted_email_details(deterministic, llm_extraction)
            merged = merge_email_details(pending_email, latest_fields)

            valid_recipients, invalid_recipients = validate_recipients(merged["recipients"])
            valid_cc, invalid_cc = validate_recipients(merged.get("cc_recipients", []))
            valid_bcc, invalid_bcc = validate_recipients(merged.get("bcc_recipients", []))
            merged["recipients"] = valid_recipients
            merged["cc_recipients"] = valid_cc
            merged["bcc_recipients"] = valid_bcc

            if invalid_recipients or invalid_cc or invalid_bcc:
                await ctx.deps.session_store.set_pending_email(
                    recipients=valid_recipients, cc_recipients=valid_cc,
                    bcc_recipients=valid_bcc, subject=merged["subject"], content=merged["content"],
                    suggested_recipient="",
                )
                parts = []
                if invalid_recipients:
                    parts.append(f"to: {', '.join(invalid_recipients)}")
                if invalid_cc:
                    parts.append(f"cc: {', '.join(invalid_cc)}")
                if invalid_bcc:
                    parts.append(f"bcc: {', '.join(invalid_bcc)}")
                return clean_text_for_model(f"I found invalid email format: {'; '.join(parts)}. Please provide the address again.")

            # Composition pass: when the user delegated authoring ("tell about
            # yourself", "pick a subject"), the extractor leaves subject/content
            # empty — historically that looped back asking the user to type
            # them. Instead, let the agent AUTHOR the missing pieces. Subject is
            # auto-derived and never blocks; we only fall back to asking when
            # the request gave no basis to write a body.
            profile = ctx.deps.personal_memory_store.load_profile_snapshot()
            email_tone = (profile.get("preferences") or {}).get("email_tone") or ""
            content_before_compose = merged["content"]

            if missing_email_fields(merged):
                compose_prompt = build_compose_email_prompt(
                    user_request=query,
                    merged=merged,
                    email_tone=email_tone,
                    sender_identity=EMAIL_SENDER_IDENTITY,
                    conversation_context=conversation_context,
                )
                compose_result = await run_agent_with_fallbacks(
                    agents_mgr.email_agent,
                    agents_mgr.email_agent_fallbacks,
                    compose_prompt,
                    deps=ctx.deps,
                    usage=ctx.usage,
                )
                composed = parse_email_extraction_response(compose_result.output).model_dump()
                # Fill only the gaps — never overwrite anything the user dictated.
                if not merged["content"] and composed.get("content"):
                    merged["content"] = str(composed["content"]).strip()
                if not merged["subject"] and composed.get("subject"):
                    merged["subject"] = str(composed["subject"]).strip()
                # Subject must never block a send once we have a body.
                if not merged["subject"] and merged["content"]:
                    merged["subject"] = derive_fallback_subject(merged["content"])

            missing = missing_email_fields(merged)
            if missing:
                suggested_recipient = (
                    suggest_recipient_completion(query) if "recipients" in missing else None
                )
                await ctx.deps.session_store.set_pending_email(
                    recipients=merged["recipients"], cc_recipients=merged["cc_recipients"],
                    bcc_recipients=merged["bcc_recipients"], subject=merged["subject"], content=merged["content"],
                    suggested_recipient=suggested_recipient[1] if suggested_recipient else "",
                )
                return clean_text_for_model(format_missing_email_prompt(missing, merged, suggested_recipient))

            # Draft-before-send: when the body was authored by Turtle this turn
            # (not dictated by the user) and the user prefers drafts, show the
            # draft and hold for a 'send' confirmation instead of sending now.
            # On the follow-up turn the content is already pending, so this
            # branch is skipped and the send proceeds. (Finally activates the
            # previously-captured-but-unenforced prefers_draft_before_send.)
            authored_this_turn = not content_before_compose and bool(merged["content"])
            prefers_draft = bool((profile.get("workflow") or {}).get("prefers_draft_before_send"))
            # Hold for confirmation when Turtle authored the body this turn and
            # the user prefers drafts, OR when the user never actually said to
            # send (send_intent was extracted but previously ignored).
            if (authored_this_turn and prefers_draft) or not merged.get("send_intent"):
                await ctx.deps.session_store.set_pending_email(
                    recipients=merged["recipients"], cc_recipients=merged["cc_recipients"],
                    bcc_recipients=merged["bcc_recipients"], subject=merged["subject"], content=merged["content"],
                    suggested_recipient="",
                )
                return clean_text_for_model(
                    "Here's the draft:\n\n"
                    + format_email_draft(merged)
                    + "\n\nReply \"send\" to send it, or tell me what to change."
                )

            from pydantic_ai.exceptions import ModelRetry as _ModelRetry

            try:
                validate_send_email_args(
                    merged["recipients"], merged["subject"], merged["content"],
                    merged["cc_recipients"], merged["bcc_recipients"],
                )
                # B5: idempotency check — prevent double-sends within 60 s.
                # Reservation-based (see tools/idempotency.py): the key is
                # claimed BEFORE the send, keyed per-user so two tenants
                # sending an identical email never collide, and the actual
                # SMTP call is pushed off the event loop so one hung mail
                # server can't freeze every connected user.
                from tools.idempotency import (
                    IdempotencyReservationError,
                    build_email_idempotency_key,
                    is_duplicate_invocation,
                    record_invocation,
                    send_with_reservation,
                )
                idem_key = build_email_idempotency_key(
                    ctx.deps.user_id,
                    merged["recipients"],
                    merged["subject"],
                    merged["content"],
                    cc=merged["cc_recipients"],
                    bcc=merged["bcc_recipients"],
                )
                try:
                    cached_result = is_duplicate_invocation(idem_key)
                except IdempotencyReservationError:
                    print(f"LOG: Email idempotency store unavailable — refusing send ({idem_key[:12]}...)")
                    return clean_text_for_model(
                        "I could not verify this wasn't a duplicate send (the safety "
                        "check is temporarily unavailable), so I did NOT send this "
                        "email. Please try again in a moment."
                    )
                if cached_result is not None:
                    print(f"LOG: Email idempotency hit — skipping duplicate send ({idem_key[:12]}...)")
                    return clean_text_for_model(cached_result)

                # send_with_reservation guarantees the reservation is always
                # finalized or released, however the send exits (including
                # asyncio.CancelledError on a client disconnect mid-send) —
                # see its docstring in tools/idempotency.py.
                send_result = await send_with_reservation(
                    idem_key, lambda: asyncio.to_thread(send_email_now, merged)
                )
            except _ModelRetry:
                # pydantic_ai's retry protocol — swallowing it hands the model
                # a prose failure instead of a structured retry.
                raise
            except Exception as e:
                await ctx.deps.session_store.set_pending_email(
                    recipients=merged["recipients"], cc_recipients=merged["cc_recipients"],
                    bcc_recipients=merged["bcc_recipients"], subject=merged["subject"], content=merged["content"],
                    suggested_recipient="",
                )
                return clean_text_for_model(f"Failed to send email: {e}")

            if send_result.startswith("Email sent successfully!"):
                await ctx.deps.session_store.clear_pending_email()
                # When Turtle authored the body, show it so the user sees what
                # went out (send_email_now echoes only the headers).
                if authored_this_turn:
                    send_result = f"{send_result}\n\nBody:\n{merged['content']}"
            else:
                await ctx.deps.session_store.set_pending_email(
                    recipients=merged["recipients"], cc_recipients=merged["cc_recipients"],
                    bcc_recipients=merged["bcc_recipients"], subject=merged["subject"], content=merged["content"],
                    suggested_recipient="",
                )
            try:
                # recall(scope="tasks") finally has data: record the action.
                ctx.deps.task_history_store.record(
                    session_id=ctx.deps.session_store.session_id or "unknown_session",
                    turn_id=f"email_{int(time.time())}",
                    task_type="email",
                    status="completed" if send_result.startswith("Email sent successfully") else "failed",
                    query=query[:200],
                    tool_used="send_email_assistant",
                    outcome=send_result[:200],
                )
            except Exception as _e:
                print(f"LOG: task history record failed: {_e}")
            return clean_text_for_model(send_result)


        async def recall(ctx: RunContext[SharedState], args: RecallArgs) -> str:
            """Recall personal, episodic, task, or working context. See tool contract for full spec."""
            query = args.query.strip()
            scope = str(args.scope or "").strip().lower()
            if not query:
                return ToolResult.invalid("query must not be empty").to_agent_string()
            if scope not in {"personal", "episodic", "tasks", "working"}:
                return ToolResult.invalid("scope must be personal, episodic, tasks, or working").to_agent_string()
            broker = ctx.deps.retrieval_broker
            if broker is None:
                return ToolResult.empty("Recall is not available.").to_agent_string()
            try:
                recall_text = await broker.recall(
                    query=query,
                    scope=scope,
                    message_history=ctx.deps.session_store.message_history,
                    trim_fn=_trim_history_for_context,
                )
            except Exception as e:
                return ToolResult.upstream_error(f"Recall failed: {e}").to_agent_string()
            if not recall_text:
                return ToolResult.empty("No relevant information found.").to_agent_string()
            return ToolResult.ok(recall_text).to_agent_string()

        async def link_account(ctx: RunContext[SharedState], args: LinkAccountArgs) -> str:
            """Issue a claim code to link this channel identity to a web account."""
            deps = ctx.deps
            channel = str(getattr(deps, "channel", "") or "")
            channel_uid = str(getattr(deps, "channel_user_id", "") or "")
            if not channel or not channel_uid:
                return ToolResult.invalid(
                    "Account linking is only available from a channel like Discord. "
                    "On the web you are already signed in."
                ).to_agent_string()
            # NEVER emit a claim code into a shared channel. The code is a
            # BEARER credential: whoever redeems it first gets THIS sender's
            # channel identity re-pointed at THEIR account, and merge_memory
            # copies THIS sender's journal into it. So an observer in a public
            # channel who races the code steals the sender's memory — it is not
            # (as the original design note claimed) harmless to leak.
            if not getattr(deps, "channel_is_private", False):
                return ToolResult.invalid(
                    "I can't start account linking in a shared channel — the code "
                    "would be visible to everyone here, and anyone who used it "
                    "first would end up with your memory. Send me a direct "
                    "message and I'll set it up there."
                ).to_agent_string()
            # Two-sided binding (WP1.J, ledger 1b.6): the caller states which web
            # account they intend to redeem with. This is NOT trusted as proof of
            # anything — the authenticated web session at redemption is still the
            # only thing that proves account ownership — it only lets redemption
            # REFUSE a session that doesn't match. See core/account_linking.py's
            # module docstring for the full threat-model writeup.
            from core.identity import normalize_email

            expected_email = normalize_email(args.expected_email)
            if not expected_email or "@" not in expected_email:
                return ToolResult.invalid(
                    "I need the email address of the Turtle web account you'll "
                    "sign in with to finish linking — that's what lets me refuse "
                    "the code if anyone but you tries to redeem it."
                ).to_agent_string()
            try:
                from core.account_linking import LINK_CODE_TTL_MINUTES
                from core.storage.factory import get_link_code_store

                store = get_link_code_store()
                issued = await asyncio.to_thread(
                    store.issue,
                    channel=channel,
                    channel_user_id=channel_uid,
                    source_user_id=deps.user_id,
                    expected_email=expected_email,
                )
            except Exception as e:
                return ToolResult.upstream_error(
                    f"Could not create a link code: {e}"
                ).to_agent_string()
            return ToolResult.ok(
                f"Link code: {issued.code}\n"
                f"To finish linking, sign in to Turtle on the web as {expected_email} and "
                f"enter this code in Settings -> Link account. It expires in "
                f"{LINK_CODE_TTL_MINUTES} minutes, can only be used once, and only that "
                f"account can redeem it. Signing in is what proves the web account is "
                f"yours — I can't link on an email address alone."
            ).to_agent_string()

        async def calendar_create(ctx: RunContext[SharedState], args: CalendarCreateArgs) -> str:
            """Stage a Google Calendar event as a draft. See tool contract for full spec.

            Does NOT touch Google Calendar. Stores the proposed event as
            pending_calendar and asks the user to confirm; calendar_confirm
            is the tool that actually creates it (mirrors the email
            draft/send flow's pending_email)."""
            from tools.calendar_tool import CalendarCreateArgs as _CalendarCreateArgs
            from tools.calendar_tool import render_calendar_draft

            inner = _CalendarCreateArgs(
                title=args.title,
                start_iso=args.start_iso,
                end_iso=args.end_iso,
                attendee_emails=args.attendee_emails,
                description=args.description,
                add_google_meet=args.add_google_meet,
                notify_attendees=args.notify_attendees,
            )
            await ctx.deps.session_store.set_pending_calendar(
                title=inner.title,
                start_iso=inner.start_iso,
                end_iso=inner.end_iso,
                attendee_emails=inner.attendee_emails,
                description=inner.description,
                add_google_meet=inner.add_google_meet,
                notify_attendees=inner.notify_attendees,
            )
            return clean_text_for_model(render_calendar_draft(inner))

        async def calendar_confirm(ctx: RunContext[SharedState]) -> str:
            """Create the previously drafted calendar event. See tool contract for full spec."""
            from tools.calendar_tool import CalendarCreateArgs as _CalendarCreateArgs
            from tools.calendar_tool import create_calendar_event
            from tools.idempotency import (
                IdempotencyReservationError,
                build_calendar_idempotency_key,
                is_duplicate_invocation,
                send_with_reservation,
            )

            pending = ctx.deps.session_store.get_pending_calendar()
            if not pending.get("title") or not pending.get("start_iso") or not pending.get("end_iso"):
                return clean_text_for_model(
                    "There's no pending calendar event to confirm (it may have expired — "
                    "drafts last an hour). Ask me to create the event again first."
                )

            inner = _CalendarCreateArgs(
                title=pending.get("title", ""),
                start_iso=pending.get("start_iso", ""),
                end_iso=pending.get("end_iso", ""),
                attendee_emails=list(pending.get("attendee_emails") or []),
                description=pending.get("description", ""),
                add_google_meet=bool(pending.get("add_google_meet", True)),
                notify_attendees=bool(pending.get("notify_attendees", False)),
            )

            idem_key = build_calendar_idempotency_key(
                ctx.deps.user_id, inner.title, inner.start_iso, inner.end_iso, inner.attendee_emails,
            )
            try:
                cached_result = is_duplicate_invocation(idem_key)
            except IdempotencyReservationError:
                print(f"LOG: Calendar idempotency store unavailable — refusing create ({idem_key[:12]}...)")
                return clean_text_for_model(
                    "I could not verify this wasn't a duplicate create (the safety "
                    "check is temporarily unavailable), so I did NOT create this "
                    "event. Please try again in a moment."
                )
            if cached_result is not None:
                print(f"LOG: Calendar idempotency hit — skipping duplicate create ({idem_key[:12]}...)")
                return clean_text_for_model(cached_result)

            success_holder: dict[str, bool] = {"ok": False}

            async def _do_create() -> str:
                result = await create_calendar_event(inner, user_id=ctx.deps.user_id)
                success_holder["ok"] = result.status == "ok"
                if result.status == "invalid" and result.error_code == "credentials_missing":
                    return (
                        f"{result.to_agent_string()} Ask the user to connect their calendar at "
                        f"{settings.public_base_url.rstrip('/')}/integrations/google_calendar/connect"
                    )
                return result.to_agent_string()

            try:
                # send_with_reservation guarantees the reservation is always
                # finalized or released, however the create exits (including
                # asyncio.CancelledError on a client disconnect) — see its
                # docstring in tools/idempotency.py. On any exception the
                # draft must stay armed for retry (do not clear pending_calendar).
                create_result = await send_with_reservation(
                    idem_key, _do_create, is_success=lambda _r: success_holder["ok"],
                )
            except Exception as e:
                return clean_text_for_model(f"Failed to create calendar event: {e}")

            if success_holder["ok"]:
                await ctx.deps.session_store.clear_pending_calendar()
            return clean_text_for_model(create_result)

        async def calendar_list(ctx: RunContext[SharedState], args: CalendarListArgs) -> str:
            """List upcoming Google Calendar events. See tool contract for full spec."""
            from tools.calendar_tool import list_upcoming_events
            from tools.calendar_tool import CalendarListArgs as _CalendarListArgs
            inner = _CalendarListArgs(
                max_results=args.max_results,
                time_min_iso=args.time_min_iso or None,
            )
            result = await list_upcoming_events(inner, user_id=ctx.deps.user_id)
            if result.status == "invalid" and result.error_code == "credentials_missing":
                return (
                    f"{result.to_agent_string()} Ask the user to connect their calendar at "
                    f"{settings.public_base_url.rstrip('/')}/integrations/google_calendar/connect"
                )
            return result.to_agent_string()

        async def find_place(ctx: RunContext[SharedState], args: FindPlaceArgs) -> str:
            """Look up real-world places via Google Places. See tool contract for full spec."""
            from tools.places_tool import (
                find_place as _find_place,
                render_find_place,
                FindPlaceArgs as _FindPlaceArgs,
            )
            print(f"\nPLACES: find_place query={args.query!r}")
            inner = _FindPlaceArgs(
                query=args.query,
                max_results=args.max_results,
                location_bias=args.location_bias,
            )
            # Reuse the shared http_client so we inherit the app's pool and timeouts.
            result = await _find_place(inner, http_client=ctx.deps.http_client, user_id=ctx.deps.user_id)
            if result.status == "ok" and result.data is not None:
                rendered = render_find_place(result.data)
                return _truncate_tool_output(rendered, label="places search")
            return result.to_agent_string()

        async def place_details(ctx: RunContext[SharedState], args: PlaceDetailsArgs) -> str:
            """Fetch full details for a Google place_id. See tool contract for full spec."""
            from tools.places_tool import (
                place_details as _place_details,
                render_place_details,
                PlaceDetailsArgs as _PlaceDetailsArgs,
            )
            print(f"\nPLACES: place_details id={args.place_id!r}")
            inner = _PlaceDetailsArgs(place_id=args.place_id)
            result = await _place_details(inner, http_client=ctx.deps.http_client, user_id=ctx.deps.user_id)
            if result.status == "ok" and result.data is not None:
                rendered = render_place_details(result.data)
                return _truncate_tool_output(rendered, label="place details")
            return result.to_agent_string()

        async def get_directions(ctx: RunContext[SharedState], args: GetDirectionsArgs) -> str:
            """Compute a route between two locations via Google Routes. See tool contract."""
            from tools.places_tool import (
                get_directions as _get_directions,
                render_directions,
                GetDirectionsArgs as _GetDirectionsArgs,
            )
            print(
                f"\nPLACES: directions {args.origin!r} -> {args.destination!r} "
                f"mode={args.travel_mode}"
            )
            inner = _GetDirectionsArgs(
                origin=args.origin,
                destination=args.destination,
                travel_mode=args.travel_mode,
            )
            result = await _get_directions(inner, http_client=ctx.deps.http_client, user_id=ctx.deps.user_id)
            if result.status == "ok" and result.data is not None:
                rendered = render_directions(result.data)
                return _truncate_tool_output(rendered, label="directions")
            return result.to_agent_string()

        async def remember(ctx: RunContext[SharedState], args: RememberArgs) -> str:
            """Explicitly store a user-stated fact in personal memory. See tool contract."""
            from core.memory_journal import ALLOWED_TOPICS

            topic = args.topic.strip().lower()
            key_slug = args.key.strip().lower()
            value_text = args.value.strip()

            if topic not in ALLOWED_TOPICS:
                return ToolResult.invalid(
                    f"topic must be one of: {', '.join(sorted(ALLOWED_TOPICS))}"
                ).to_agent_string()
            if not key_slug or not value_text:
                return ToolResult.invalid("key and value must not be empty").to_agent_string()

            import re as _re
            key_slug = _re.sub(r"[^a-z0-9]+", "_", key_slug).strip("_") or "note"
            mode = (args.mode or "replace").strip().lower()
            if mode not in {"replace", "add"}:
                mode = "replace"

            # Store + storage-cap handling lives in a module-level helper so the
            # cap-failure path is unit testable (this tool is a closure).
            return _store_remembered_fact(
                ctx.deps, topic=topic, key_slug=key_slug, value_text=value_text, mode=mode
            )

        # Register the identical toolset on every rung of the cascade:
        # run_agent_with_fallbacks swaps Agent objects on failure, and a rung
        # without tools silently loses every capability while the shared
        # system prompt still commands tool use.
        _tool_registry = [
            ("search_web", search_web),
            ("search_url", search_url),
            ("send_email_assistant", send_email_assistant),
            ("recall", recall),
            ("calendar_create", calendar_create),
            ("calendar_confirm", calendar_confirm),
            ("calendar_list", calendar_list),
            ("find_place", find_place),
            ("place_details", place_details),
            ("get_directions", get_directions),
            ("remember", remember),
            ("link_account", link_account),
        ]
        # WP1.H (ledger 1b.2): every tool result is attacker-influenced text
        # (a fetched page, a search snippet, a place review) handed to the
        # model with nothing marking it as data. Of the twelve tools above,
        # five never call ToolResult.to_agent_string() on their happy path,
        # two never touch ToolResult at all, and one re-wraps a
        # to_agent_string() value through clean_text_for_model — four
        # distinct return shapes. An envelope placed inside any one of those
        # shapes would cover under half the surface while looking complete.
        # This registration loop is the one place every tool's return passes
        # through regardless of its internal shape, so the envelope is
        # applied here, once, to all twelve.
        _tool_registry = [
            (_name, _wrap_tool_with_envelope(_name, _fn))
            for _name, _fn in _tool_registry
        ]
        for _target_agent in [self.main_assistant, *self.main_assistant_fallbacks]:
            for _contract_name, _tool_fn in _tool_registry:
                _target_agent.tool(description=_load_tool_contract(_contract_name))(_tool_fn)


# ---------------------------------------------------------------------------
# Global agent manager
# ---------------------------------------------------------------------------
agents_mgr = AgentManager()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Turtle AI", docs_url=None, redoc_url=None)

# Phase 4 / E1: routine scheduler — singleton, started/stopped with the app.
_routine_scheduler = None  # type: ignore[var-annotated]

# Phase 5 (W2): the event loop the app runs on. The scheduler fires routines on
# a worker thread; that thread needs this loop to bridge sends back onto the app
# loop via asyncio.run_coroutine_threadsafe. None until startup captures it.
_APP_LOOP: "asyncio.AbstractEventLoop | None" = None


@app.on_event("startup")
async def _warn_on_vercel_deploy_mode_mismatch() -> None:
    """Loudly flag it when TURTLE_DEPLOY isn't "cloud" on a real Vercel deploy.

    Found 2026-09-13: TURTLE_DEPLOY was configured as an empty string in the
    Vercel production environment (not "cloud", not unset — an explicit blank
    value some earlier migration step left behind). settings.is_cloud gates
    the Postgres/Redis storage backends, core/paths.py's read-only-filesystem
    mkdir skip, and the Discord/embed self-invoke dispatch — with it False on
    Vercel, the app runs against local SQLite/FAISS paths on a filesystem
    that's read-only outside /tmp (core/paths.py:ensure_dirs already documents
    the resulting OSError crash) while the real Postgres/Redis sit unused.
    Warn (don't hard-fail) so a still-misconfigured deploy stays diagnosable
    in logs instead of silently doing the wrong thing AND instead of taking
    the whole app down if this check itself lands before the env var is fixed.
    """
    if os.environ.get("VERCEL") and not settings.is_cloud:
        print(
            f"LOG: MISCONFIGURED DEPLOY - running on Vercel (VERCEL env set) but "
            f"TURTLE_DEPLOY={settings.deploy_mode!r} (not 'cloud'). Storage will "
            f"use local SQLite/FAISS paths instead of the provisioned "
            f"Postgres/Redis, and filesystem writes outside /tmp WILL crash "
            f"with 'Read-only file system'. Set TURTLE_DEPLOY=cloud in the "
            f"Vercel project's environment variables.",
            flush=True,
        )
    if os.environ.get("VERCEL") and settings.public_base_url.rstrip("/") in (
        "http://127.0.0.1:8765",
        "http://localhost:8765",
    ):
        print(
            f"LOG: MISCONFIGURED DEPLOY - running on Vercel but "
            f"TURTLE_PUBLIC_BASE_URL is unset (defaulting to "
            f"{settings.public_base_url!r}). Magic-link/forget-me emails and "
            f"the Google Calendar OAuth redirect_uri will point at localhost. "
            f"Set TURTLE_PUBLIC_BASE_URL to the deployed domain.",
            flush=True,
        )


@app.on_event("startup")
async def _refuse_forgeable_binding() -> None:
    """Refuse to serve a network-reachable interface without a real AUTH_SECRET_KEY.

    Codex adversarial review found that missing-secret deployments accepted
    tokens signed with the repo-known dev fallback for any user_id. The fallback
    is now a per-process random secret (core/auth_secret.py) instead of a
    literal, which fixes forgery — but a non-loopback bind without a stable
    AUTH_SECRET_KEY still invalidates every cookie on every restart, which is
    worse UX than refusing to start. Cloud is already fail-fast in auth_secret().
    """
    import os
    from core.auth_secret import is_using_fallback_secret

    if not is_using_fallback_secret():
        return  # explicit secret set — safe to bind anywhere
    # ASGI adapters expose the bound host via env or arg. UVICORN_HOST / HOST
    # are the common ones; if neither is set, we can't prove it's non-loopback,
    # so err on the side of allowing (the local dev case).
    host = (os.environ.get("UVICORN_HOST") or os.environ.get("HOST") or "").strip()
    if host and host not in ("127.0.0.1", "::1", "localhost"):
        raise RuntimeError(
            f"Refusing to bind {host!r} without AUTH_SECRET_KEY set. Any leaked "
            "session cookie becomes forgeable after restart, and the fallback "
            "is a process-random dev secret. Set AUTH_SECRET_KEY, or bind "
            "127.0.0.1."
        )


@app.on_event("startup")
async def _require_cloud_backends_configured() -> None:
    """Cloud mode with a missing DATABASE_URL/REDIS_URL cannot serve a single
    request — every Postgres/Redis-backed store raises CloudBackendUnavailable
    lazily on first use (core/storage/cloud/__init__.py). Move that failure to
    boot: a deploy that cannot work should not serve.

    No-op unless settings.is_cloud is true, so this never fires in the test
    suite's default local configuration (no Postgres/Redis there).
    """
    if not settings.is_cloud:
        return
    # bool(SecretStr("")) is False, so a plain unset/blank DATABASE_URL is
    # already caught by `not settings.database_url` — the 71c3378 class of
    # bug (Vercel once held TURTLE_DEPLOY="" rather than unset). But
    # bool(SecretStr(" ")) is True, so a whitespace-only value would sail
    # past that check unless we strip first. settings.redis_url already
    # strips internally (core/config.py); mirror that here rather than add a
    # field there.
    database_url_value = (
        settings.database_url.get_secret_value().strip() if settings.database_url else ""
    )
    if not database_url_value:
        raise RuntimeError(
            "TURTLE_DEPLOY=cloud requires DATABASE_URL to be set (the Neon "
            "pooled connection string) — refusing to start without it."
        )
    if not settings.redis_url:
        raise RuntimeError(
            "TURTLE_DEPLOY=cloud requires REDIS_URL (or UPSTASH_REDIS_URL) to "
            "be set — refusing to start without it."
        )


@app.on_event("startup")
async def _validate_google_calendar_credentials() -> None:
    """Fail loudly (but not fatally) at boot if GOOGLE_CALENDAR_CREDENTIALS_JSON
    is malformed, instead of only surfacing it on the first calendar tool call
    or OAuth connect attempt.

    Concretely catches the mistake of pasting the bare client-secret string
    (e.g. "GOCSPX-...") instead of the full OAuth client JSON downloaded from
    Google Cloud Console — that used to fail silently until a user tried to
    connect their calendar.
    """
    raw = (settings.google_calendar_credentials_json or "").strip()
    if not raw:
        return  # Calendar integration is optional; unset is not an error.
    try:
        from apps.calendar_oauth_routes import validate_credentials_json
        ok, message, _config = validate_credentials_json(raw)
    except Exception as e:
        print(f"LOG: GOOGLE_CALENDAR_CREDENTIALS_JSON validation skipped: {e}")
        return
    if ok:
        print("LOG: GOOGLE_CALENDAR_CREDENTIALS_JSON is valid (client_id + client_secret present).")
    else:
        print(f"LOG: WARNING - GOOGLE_CALENDAR_CREDENTIALS_JSON is misconfigured: {message}")


@app.on_event("startup")
async def _warn_on_missing_calendar_token_key() -> None:
    """Loudly flag CALENDAR_TOKEN_KEY being unset in cloud mode.

    core/config.py's field_validator already catches a MALFORMED key at
    settings-construction time (before this even runs) — that one fails
    process boot outright, because a malformed key is unambiguously a typo
    to fix before anything else happens. An UNSET key in cloud is different:
    it is a valid, working configuration for a deploy that simply hasn't
    wired up Calendar OAuth token encryption yet, and calendar is one
    optional integration — the rest of the app (chat, memory, every other
    tool) works fine without it. So this warns instead of refusing to start,
    mirroring _warn_on_vercel_deploy_mode_mismatch above rather than
    _require_cloud_backends_configured's refuse-to-boot posture: without
    this warning, the only signal an operator gets is a user hitting a 503
    on /integrations/google_calendar/callback — possibly days after
    deploying, and reported as a complaint rather than caught in logs.

    No-op in local mode: CALENDAR_TOKEN_KEY there is optional by design
    (falls back to plaintext-on-disk, today's dev-box behaviour — see
    core/calendar_token_crypto.py), so an unset key locally is not
    noteworthy.
    """
    if not settings.is_cloud:
        return
    if settings.calendar_token_key is not None:
        return
    print(
        "LOG: WARNING - CALENDAR_TOKEN_KEY is unset in cloud mode. Google "
        "Calendar connections will be refused (503 on "
        "/integrations/google_calendar/callback) rather than silently "
        "storing tokens unencrypted in Postgres. Generate one with: "
        "python -c \"import secrets, base64; print(base64.urlsafe_b64encode"
        "(secrets.token_bytes(32)).decode())\" and set CALENDAR_TOKEN_KEY.",
        flush=True,
    )


@app.on_event("startup")
async def _start_routine_scheduler() -> None:
    global _routine_scheduler, _APP_LOOP
    # Capture the running app loop so the scheduler thread can bridge routine
    # notices back onto it (run_coroutine_threadsafe). Must happen on the loop.
    try:
        _APP_LOOP = asyncio.get_running_loop()
    except RuntimeError:
        _APP_LOOP = None
    if settings.is_cloud:
        # Cloud mode has no persistent process to hold a live scheduler in —
        # the periodic GitHub Actions cron-tick (apps/cron_tick_routes.py)
        # replaces it. Starting RoutineScheduler here too would double-fire
        # every routine (both paths would append the same scheduled_fire
        # event and push the same delivery).
        print("LOG: RoutineScheduler skipped in cloud mode — see apps/cron_tick_routes.py")
        return
    try:
        from core.routine_scheduler import RoutineScheduler
        _routine_scheduler = RoutineScheduler()
        _routine_scheduler.start()
    except Exception as e:
        print(f"LOG: RoutineScheduler failed to start: {e}")
        _routine_scheduler = None


@app.on_event("shutdown")
async def _stop_routine_scheduler() -> None:
    global _routine_scheduler
    if _routine_scheduler is not None:
        try:
            _routine_scheduler.shutdown()
        except Exception as e:
            print(f"LOG: RoutineScheduler shutdown error: {e}")
    _routine_scheduler = None


@app.on_event("startup")
async def _start_discord_gateway_hook() -> None:
    # Optional natural-DM/@mention bot. Guarded so absence of discord.py or a
    # bot token is a clean no-op (see apps/channels/discord_gateway.py). The
    # zero-dependency slash-command webhook (/channels/discord) works regardless.
    #
    # Never open a REAL gateway connection under pytest: many tests enter the
    # app lifespan via `with TestClient(app)`, and with a live token present in
    # the environment each would connect+disconnect a bot session. That churn
    # can exhaust Discord's session-start/IDENTIFY allowance and temporarily
    # lock the live bot out of the gateway. Tests that need the gateway patch it.
    import sys
    if "pytest" in sys.modules:
        return
    if settings.is_cloud:
        # A persistent Gateway WebSocket cannot survive a serverless cold
        # start — this connection would be torn down (and re-IDENTIFY'd,
        # burning Discord's rate-limited session-start allowance) on every
        # invocation. apps/channels/discord.py's Interactions webhook
        # (POST /channels/discord) is the serverless-shaped replacement;
        # register it as the app's Interactions Endpoint URL in the
        # Developer Portal instead of running this gateway.
        print("LOG: discord gateway skipped in cloud mode — use the /channels/discord webhook")
        return
    try:
        from apps.channels.discord_gateway import start_discord_gateway
        # start_discord_gateway spawns the gateway client as its own background
        # task and returns immediately, so awaiting it here is safe (no startup
        # block) and avoids leaving an untracked outer task.
        await start_discord_gateway()
    except Exception as e:
        print(f"LOG: discord gateway startup skipped: {e}")


@app.on_event("shutdown")
async def _stop_discord_gateway_hook() -> None:
    try:
        from apps.channels.discord_gateway import stop_discord_gateway
        await stop_discord_gateway()
    except Exception as e:
        print(f"LOG: discord gateway shutdown error: {e}")


@app.on_event("startup")
async def _start_telegram_gateway_hook() -> None:
    # Optional natural-DM/@mention Telegram bot. Guarded so absence of the
    # python-telegram-bot library or a bot token is a clean no-op (see
    # apps/channels/telegram_gateway.py).
    #
    # Same pytest guard as Discord: don't open a real long-poll connection
    # from inside a test's app lifespan — one bot session, many test entries.
    import sys
    if "pytest" in sys.modules:
        return
    if settings.is_cloud:
        # A long-poll loop needs a persistent connection, same reasoning as
        # Discord's gateway above. apps/channels/telegram_webhook.py's
        # webhook (POST /channels/telegram/webhook) is the serverless-shaped
        # replacement; register it via Telegram's setWebhook instead.
        print("LOG: telegram gateway skipped in cloud mode — use the /channels/telegram/webhook endpoint")
        return
    try:
        from apps.channels.telegram_gateway import start_telegram_gateway
        await start_telegram_gateway()
    except Exception as e:
        print(f"LOG: telegram gateway startup skipped: {e}")


@app.on_event("shutdown")
async def _stop_telegram_gateway_hook() -> None:
    try:
        from apps.channels.telegram_gateway import stop_telegram_gateway
        await stop_telegram_gateway()
    except Exception as e:
        print(f"LOG: telegram gateway shutdown error: {e}")


@app.on_event("shutdown")
async def _flush_trace_spans() -> None:
    # Drain the BatchSpanProcessor's in-memory buffer; without this, spans from
    # the final minutes before shutdown never reach data/traces/traces.jsonl.
    try:
        from core.observability import flush_traces
        flush_traces()
    except Exception as e:
        print(f"LOG: trace flush on shutdown failed: {e}")


def get_routine_scheduler():
    return _routine_scheduler

if _logfire_loaded:
    try:
        import logfire as _lf
        _lf.instrument_fastapi(app)
    except Exception as _lfe:
        print(f"LOG: logfire.instrument_fastapi skipped ({_lfe.__class__.__name__}: {_lfe})")


@app.middleware("http")
async def no_cache_js(request: Request, call_next):
    """Prevent browsers from caching JS/CSS so dev changes take effect immediately."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/js/") or path.startswith("/static/css/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


# Serve static files from web/ directory
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Channel adapters — Tier 3 (F1/F2/F3/E5)
# ---------------------------------------------------------------------------
from apps.channels import TurtleEvent, TurtleResponse, set_channel_dispatch
from apps.channels.whatsapp import router as _whatsapp_router
from apps.channels.imessage import router as _imessage_router
from apps.channels.slack import router as _slack_router
from apps.channels.twilio_voice import router as _twilio_voice_router
from apps.channels.discord import router as _discord_router

app.include_router(_whatsapp_router)
app.include_router(_imessage_router)
app.include_router(_slack_router)
# apps/channels/twilio_voice.py: NOT MOUNTED.
# The WS at /channels/twilio/voice/stream has no signature verification and
# takes the tenant straight from client-supplied `start.customParameters.from`,
# streaming replies back over the caller's own socket. That is an
# unauthenticated, bidirectional cross-tenant read channel — setting
# TWILIO_AUTH_TOKEN does not close it because the WS route never checks any
# signature. Flagged critical by both audits; remounting requires (a) validating
# the /incoming HTTP hop and (b) issuing a short-lived server-signed stream
# token that binds the socket to the validated call SID / caller. Until that
# lands, keep the router unmounted so no code path can reach it.
# app.include_router(_twilio_voice_router)  # DO NOT UNCOMMENT WITHOUT AUTH.
app.include_router(_discord_router)

from apps.onboarding_routes import router as _onboarding_router, verify_session_cookie
app.include_router(_onboarding_router)

from apps.admin_routes import router as _admin_router
app.include_router(_admin_router)

from apps.calendar_oauth_routes import router as _calendar_oauth_router
app.include_router(_calendar_oauth_router)

from apps.cron_tick_routes import router as _cron_tick_router
app.include_router(_cron_tick_router)

from apps.channels.telegram_webhook import router as _telegram_webhook_router
app.include_router(_telegram_webhook_router)


# Per-(user_id, channel) SharedState cache. Channels now run through the SAME
# turn pipeline as the WebSocket path, so their conversation of record lives in
# the session store (like web), not a bare capped list. The cache preserves
# session continuity across webhook turns; a per-key lock serialises turns for
# the same (user, channel) so the shared, lazily-assigned http_client and the
# session-store writes never race.
#
# Bounded + idle-TTL'd (Codex review R2#8/#9): a public webhook can mint
# unbounded unique sender ids, and a cached state pins its SessionStore's
# resumed session forever — evicting after idle both bounds memory and forces
# start_or_restore (with its session age cap) to re-run for returning users.
# Entries store (state, last_used_monotonic).
_CHANNEL_STATES: dict[tuple[str, str], tuple[SharedState, float]] = {}
_CHANNEL_STATE_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}
_CHANNEL_STATE_CAP = 64
# MUST stay comfortably BELOW SessionStore's resume_window_seconds (1800s).
# These were both 30 min, which made idle-eviction deterministically the one
# case that CANNOT resume: evict at 1800s -> next message calls start_or_restore
# -> the session's age is by construction just past the 1800s window -> no
# resume, brand-new empty session, conversation context gone. Evicting earlier
# means a returning channel user rebuilds state and STILL lands inside the
# resume window, so the session (and its history) comes back warm.
_CHANNEL_STATE_IDLE_TTL_S = 10 * 60

# Channel-native confirmation-gate answer buffer (ISSUE-011) — tracks the one
# outstanding memory-gate prompt per (user_id, channel) so a plain "yes"/"no"
# chat reply can answer it. See core/channel_gate.py for the narrow-match
# rules. Process-local singleton in local mode; Redis-backed (shared across
# invocations) in cloud mode — see core/storage/factory.get_channel_gate_buffer.
_CHANNEL_GATE_BUFFER = get_channel_gate_buffer()


def _channel_state_lock(key: tuple[str, str]) -> asyncio.Lock:
    lock = _CHANNEL_STATE_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _CHANNEL_STATE_LOCKS[key] = lock
    return lock


def _evict_stale_channel_states(now: float) -> None:
    """Drop idle/over-cap cached channel states.

    Never evicts an entry whose lock is currently held: popping a held lock
    would let the next same-key event mint a fresh lock and run concurrently
    with the holder. Durability does not depend on the cache — every turn
    journals and persists through its stores before returning, so an evicted
    state is simply rebuilt (with a fresh age-capped start_or_restore) on the
    user's next event."""
    def _evict(key: tuple[str, str]) -> None:
        lock = _CHANNEL_STATE_LOCKS.get(key)
        if lock is not None and lock.locked():
            return  # a turn is running for this key — skip this round
        _CHANNEL_STATES.pop(key, None)
        _CHANNEL_STATE_LOCKS.pop(key, None)

    for key in [
        key for key, (_, last_used) in _CHANNEL_STATES.items()
        if now - last_used > _CHANNEL_STATE_IDLE_TTL_S
    ]:
        _evict(key)
    if len(_CHANNEL_STATES) > _CHANNEL_STATE_CAP:
        # Still over cap after TTL: drop the least-recently-used entries.
        by_age = sorted(_CHANNEL_STATES.items(), key=lambda kv: kv[1][1])
        for key, _ in by_age[: len(_CHANNEL_STATES) - _CHANNEL_STATE_CAP]:
            _evict(key)


async def _build_channel_state(user_id: str, channel: str) -> SharedState:
    """Construct a full SharedState for a channel turn.

    This mirrors the WebSocket connection setup (SessionStore + PersonalMemory
    + sqlite index with None-degrade + JournalStore write-through +
    ConfirmationGate + RetrievalBroker + reflector) so channel turns get memory
    context, journaling, the gate, extraction, and session continuity — parity
    with web. It does NOT run the WS path's pending-finalization sweep; that is
    a connection-lifecycle concern and unnecessary for stateless webhooks.

    ``http_client`` is left ``None``; the dispatcher assigns a live client for
    the duration of each turn (search/url tools need one) and clears it after.
    """
    session_store = SessionStore(user_id=user_id)
    restore_result = await session_store.start_or_restore(mode=SESSION_RESTORE_MODE)
    # Personal memory lives under personal_memory_dir(user_id); there is no
    # single-tenant store to construct.
    personal_memory_store = PersonalMemoryStore(user_id=user_id)
    # The read-model index is a local SQLite file — pointless to build on
    # ephemeral serverless disk every cold start (it would just be rebuilt
    # from scratch next invocation), and every consumer already accepts
    # sqlite_index=None (falls back to a plain journal scan, which in cloud
    # mode already reads from Postgres — see core/memory_journal.py).
    sqlite_index = None
    if not settings.is_cloud:
        from core.memory_sqlite import MemorySQLiteIndex
        # Same derived-read-model None-degrade as the WS path.
        try:
            sqlite_index = MemorySQLiteIndex(user_id=user_id)
        except Exception as exc:
            print(f"LOG: SQLite memory index unavailable for {user_id}: {exc}; falling back to journal scans")
            sqlite_index = None
    journal_store = JournalStore(
        user_id=user_id,
        on_append=sqlite_index.index_event if sqlite_index is not None else None,
    )
    if sqlite_index is not None:
        try:
            sqlite_index.backfill_from_journal(journal_store)
        except Exception as exc:
            print(f"LOG: SQLite memory index backfill failed for {user_id}: {exc}")
    from core.storage.factory import get_confirmation_state_backend
    confirmation_gate = ConfirmationGate(
        journal=journal_store,
        store=personal_memory_store,
        state_path=personal_memory_dir(user_id) / "confirmation_state.json",
        state_backend=get_confirmation_state_backend(user_id),
        sqlite_index=sqlite_index,
    )
    personal_memory_prompt = PersonalMemoryPromptBuilder(
        personal_memory_store,
        config=PersonalMemoryPromptConfig(
            max_bytes=PERSONAL_MEMORY_MAX_BYTES,
            max_topic_files=PERSONAL_MEMORY_MAX_TOPIC_FILES,
        ),
    )
    task_history_store = TaskHistoryStore(TASK_HISTORY_FILE, user_id=user_id)
    rag_system = TurtleRAGSystem(user_id=user_id)

    from core.storage.factory import get_vector_store
    from core.retrieval_broker import RetrievalBroker
    # Process singleton, not per-connection: the store is already keyed by
    # user_id internally, so a fresh instance per socket duplicated every
    # tenant's index in RAM and split the per-tenant locks. See
    # core/storage/factory.get_vector_store (FAISS locally, pgvector in cloud).
    vector_store = get_vector_store()
    retrieval_broker = RetrievalBroker(
        store=personal_memory_store,
        task_store=task_history_store,
        journal_store=journal_store,
        sqlite_index=sqlite_index,
        session_store=session_store,
        rag_system=rag_system,
        vector_store=vector_store,
        user_id=user_id,
    )

    state = SharedState(
        http_client=None,
        session_store=session_store,
        personal_memory_store=personal_memory_store,
        personal_memory_prompt=personal_memory_prompt,
        journal_store=journal_store,
        confirmation_gate=confirmation_gate,
        task_history_store=task_history_store,
        rag_system=rag_system,
        sqlite_index=sqlite_index,
        retrieval_broker=retrieval_broker,
        reflector=PeriodicReflector(),
        user_id=user_id,
    )
    try:
        await rag_system.start_session(session_id=restore_result.session_id)
    except Exception as exc:
        print(f"LOG: Channel rag start_session failed for {user_id}: {exc}")
    # Register for graceful-shutdown journal flush / index checkpoint.
    _register_shutdown_state(state)
    return state


async def _channel_dispatch_handler(event: TurtleEvent) -> TurtleResponse:
    """Channel-agnostic dispatch — every adapter funnels through the ONE
    canonical turn pipeline (_execute_turn) with a full per-(user, channel)
    SharedState.

    Channels therefore get exactly what web gets: the single agent call with
    fallbacks, memory context, the trace span, journaling, the confirmation gate,
    explicit facts, silent candidate queuing, and cross-turn continuity via the
    session store. No websocket exists here, so ``_execute_turn`` runs with ``ws=None``
    and every frame no-ops; the terminal user-facing text comes back on
    ``reply_text``.

    Known limitation: sessions are tenant-scoped, not channel-scoped, so a user
    active on web and a channel at the same moment can resume the same session
    and interleave histories — identical to two simultaneous web tabs today.
    Channel-scoped session streams are future work, deliberately not bolted on
    here (a pseudo-tenant per channel would orphan its sessions from the
    web-connect finalization sweep).
    """
    now = time.monotonic()
    _evict_stale_channel_states(now)

    # Rate limit FIRST — before provisioning (disk I/O) and before the
    # per-(user, channel) lock, so a refused request costs neither. Keyed on
    # the raw CHANNEL identity ("<channel>:<channel_user_id>"), never on
    # event.user_id: (a) event.user_id can still be re-pointed by the
    # re-resolve below if an account-link redemption lands mid-flight, which
    # would let a link be used to dodge the limit; (b) once sign-up is
    # invite-only (TURTLE_CHANNEL_SIGNUP=invite) an uninvited caller has no
    # user_id at all — the channel identity is the only thing to key on, and
    # exactly the requests we most want to rate limit. Reuses the same
    # mode-aware limiter as the web WebSocket path (get_ws_rate_limiter());
    # see that path in this module for the sibling usage.
    channel_identity = f"{event.channel or ''}:{getattr(event, 'channel_user_id', '') or event.user_id}"
    try:
        get_ws_rate_limiter().check_and_record(channel_identity)
    except WebSocketRateLimitExceeded as exc:
        retry_text = (
            f"You're sending messages too quickly ({exc.limit}/{exc.window}). "
            "Please try again later."
        )
        return TurtleResponse(
            content=retry_text,
            channel=event.channel,
            user_id=event.user_id,
            message_id=event.message_id,
            thread_id=event.thread_id,
        )

    # First-contact provisioning. Web users are seeded at /onboarding/start;
    # channel users arrived as empty shells with no name and no identity.md,
    # so Turtle greeted a stranger every time and had nothing to personalise
    # with. Idempotent + non-destructive (never overrides a user-stated name),
    # and offloaded because it touches disk. Best-effort: never fail a turn.
    if getattr(event, "sender_name", ""):
        try:
            from core.user_provisioning import provision_channel_user

            await asyncio.to_thread(provision_channel_user, event)
        except Exception as exc:
            print(f"LOG: channel provisioning skipped: {exc}")

    # Codex verification: event.user_id was resolved by the adapter BEFORE
    # dispatch. If an account-link redemption re-points channel_mappings while
    # this event is in flight, we would run under the STALE source user_id and
    # our post-turn writers would append to the (now unreachable) source
    # journal. Lock on the CHANNEL identity, then re-resolve inside the lock so
    # a queued turn picks up the new mapping.
    lock_key = (str(event.channel or ""), str(getattr(event, "channel_user_id", "") or event.user_id))
    async with _channel_state_lock(lock_key):
        # Re-resolve inside the lock: if a link redemption committed while we
        # were queued, this returns the NEW target user_id.
        chan_uid = getattr(event, "channel_user_id", "") or ""
        if chan_uid:
            from core.identity import identity_manager as _idm

            resolved = await _idm.resolve_user(event.channel, chan_uid)
            if resolved != event.user_id:
                print(
                    f"LOG: turn re-resolved after link: {event.user_id} -> {resolved} "
                    f"({event.channel}/{chan_uid[:12]}***)"
                )
                event = TurtleEvent(
                    user_id=resolved,
                    channel=event.channel,
                    modality=event.modality,
                    content=event.content,
                    message_id=event.message_id,
                    thread_id=event.thread_id,
                    attachments=event.attachments,
                    sender_name=event.sender_name,
                    channel_user_id=event.channel_user_id,
                    is_private=event.is_private,
                )

        key = (event.user_id, event.channel)
        cached = _CHANNEL_STATES.get(key)
        state = cached[0] if cached is not None else None
        if state is None:
            state = await _build_channel_state(event.user_id, event.channel)
        _CHANNEL_STATES[key] = (state, now)

        # Bind the platform-side identity for this turn so the link_account tool
        # can issue a claim code for the right channel identity.
        state.channel = str(event.channel or "")
        state.channel_user_id = str(getattr(event, "channel_user_id", "") or "")
        state.channel_is_private = bool(getattr(event, "is_private", False))

        # Channel-native confirmation-gate answering (ISSUE-011). The web UI
        # answers a pending memory candidate through a dedicated REST call,
        # never by parsing chat text — a bare "yes" in ordinary conversation
        # could silently promote a stale candidate. Channels have no such
        # panel, so a candidate queued for a channel user was never asked
        # about at all. This reopens chat-text answering, but ONLY when a
        # prompt was actually surfaced to this (user, channel) moments ago
        # AND the reply parses as an unambiguous yes/no (see
        # core/channel_gate.py) — narrow enough to avoid the original hazard.
        #
        # Private-only: a group chat must never resolve a personal-fact
        # prompt, mirroring the is_private gate already used for account-link
        # claim codes (never let anyone but the sender answer, or see it).
        if state.channel_is_private:
            gate_key = (event.user_id, str(event.channel or ""))
            gate_answer = _CHANNEL_GATE_BUFFER.try_consume_answer(gate_key, event.content)
            if gate_answer is not None:
                accepted, answered_event_ids = gate_answer
                for eid in answered_event_ids:
                    state.confirmation_gate.record_response(eid, accepted=accepted)
                if accepted:
                    ack_text = (
                        "Got it — saved."
                        if len(answered_event_ids) == 1
                        else "Got it — saved all of those."
                    )
                else:
                    ack_text = (
                        "Okay, I won't save that."
                        if len(answered_event_ids) == 1
                        else "Okay, I won't save any of those."
                    )
                return TurtleResponse(
                    content=ack_text,
                    channel=event.channel,
                    user_id=event.user_id,
                    message_id=event.message_id,
                    thread_id=event.thread_id,
                )

        message_history = state.session_store.message_history or None

        # Tools need a live http client; lend the cached state one for this
        # turn only (async with so it's always closed).
        async with httpx.AsyncClient() as client:
            state.http_client = client
            try:
                outcome = await _execute_turn(
                    None,
                    state,
                    event.content,
                    message_history,
                    channel=event.channel,
                    send_status=False,
                )
            finally:
                state.http_client = None

    text = outcome.reply_text or outcome.output_text or ""

    # Surface the next pending memory-gate prompt as a trailing question, and
    # remember it so the user's next private reply can answer it in plain
    # chat (see the answer-consuming branch above + core/channel_gate.py).
    # Private-only for the same reason as that branch — never ask about (or
    # leak) a personal-fact candidate in a shared channel.
    if state.channel_is_private:
        pending_prompt = state.confirmation_gate.next_prompt()
        if pending_prompt is not None:
            gate_key = (event.user_id, str(event.channel or ""))
            _CHANNEL_GATE_BUFFER.note_prompt(gate_key, pending_prompt.all_event_ids)
            text = f"{text}\n\n📋 {pending_prompt.question}".strip()

    return TurtleResponse(
        content=text,
        channel=event.channel,
        user_id=event.user_id,
        message_id=event.message_id,
        thread_id=event.thread_id,
    )


# Wire the real handler — replaces the stub in apps/channels/__init__.py
set_channel_dispatch(_channel_dispatch_handler)


@app.get("/")
async def serve_index(request: Request):
    """Serve the chat UI when authenticated, otherwise serve onboarding.

    "Authenticated" = a valid turtle_uid cookie, OR the dev_anon escape hatch
    is on. Anything else lands on the onboarding form.
    """
    cookie_token = request.cookies.get("turtle_uid")
    authed = bool(cookie_token and verify_session_cookie(cookie_token))
    if not authed and not (settings.dev_anon and not settings.is_cloud):
        onboarding_path = STATIC_DIR / "onboarding.html"
        if onboarding_path.exists():
            return FileResponse(onboarding_path, media_type="text/html")
        return JSONResponse({"error": "Onboarding page missing"}, status_code=500)

    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"error": "Frontend not built yet"}, status_code=404)
    return FileResponse(index_path, media_type="text/html")


@app.get("/admin")
async def serve_admin():
    """Serve the operator admin dashboard.

    The PAGE itself is intentionally unauthenticated — it ships no secrets and
    no data. The gate is the API it calls: /admin/users requires a matching
    X-Admin-Token header (401/503 from apps.admin_routes._require_admin). The
    operator pastes the token into the page at runtime; it is never embedded
    here. Serving the static shell freely is harmless and keeps the login UX
    simple (the page renders, then asks for the token).
    """
    admin_path = STATIC_DIR / "admin.html"
    if not admin_path.exists():
        return JSONResponse({"error": "Admin page missing"}, status_code=404)
    return FileResponse(admin_path, media_type="text/html")


@app.get("/healthz")
async def healthz():
    """Liveness probe for container orchestration and CI boot smoke.

    Intentionally does no auth and no I/O — it only proves the ASGI app booted
    and is routing. The Dockerfile HEALTHCHECK and test/smoke_boot_test.py both
    hit this. `sha` is read directly from the environment (not core/config.py,
    a hotspot no WP owns this wave) so a deploy can prove which commit it is
    serving; present (as null) even when unset so callers can rely on the key.
    """
    return JSONResponse({"status": "ok", "sha": os.environ.get("TURTLE_BUILD_SHA")})


@app.get("/readyz")
async def readyz():
    """Readiness probe: proves the backends this deploy actually needs work.

    Local mode has no Postgres/Redis at all — report ready without any I/O
    (probing here would be pure overhead and would never apply to local's
    SQLite/JSONL/FAISS storage). Cloud mode runs both backend probes
    concurrently, each bounded by core.storage.cloud.READYZ_TIMEOUT_S, and
    returns 503 with a per-backend boolean if either fails — one dead backend
    cannot make this route hang, and a deploy that cannot work should not
    report itself ready.
    """
    if not settings.is_cloud:
        return JSONResponse({"status": "ok", "mode": "local"})

    from core.storage.cloud import probe_postgres, probe_redis

    postgres_ok, redis_ok = await asyncio.gather(probe_postgres(), probe_redis())
    ok = postgres_ok and redis_ok
    return JSONResponse(
        {"postgres": postgres_ok, "redis": redis_ok},
        status_code=200 if ok else 503,
    )


@app.get("/favicon.ico")
async def serve_favicon():
    favicon_svg = STATIC_DIR / "favicon.svg"
    if favicon_svg.exists():
        return FileResponse(favicon_svg, media_type="image/svg+xml")
    return RedirectResponse(url="/static/favicon.svg")


# ---------------------------------------------------------------------------
# REST: Dev-mode config endpoints
# ---------------------------------------------------------------------------
def _admin_token_matches(expected: str, provided: str | None) -> bool:
    """Timing-safe compare, guarding hmac.compare_digest's requirement that
    both operands be present and of the same type (it raises on None)."""
    if provided is None:
        return False
    return hmac.compare_digest(expected, provided)


@app.get("/api/config")
async def get_config(x_admin_token: str | None = Header(default=None)):
    """Return current config for the dev-mode panel.

    In cloud, this is gated behind the same X-Admin-Token as POST: local's
    dev panel is the only reader (web/js/devmode.js, which already prompts
    for the token on a 401), and D7 already refuses config edits in cloud —
    so an open GET in cloud serves no one but leaks config shape to anyone
    who hits the endpoint. Local stays open, unchanged from today.
    """
    if settings.is_cloud:
        expected = (
            settings.admin_token.get_secret_value()
            if settings.admin_token is not None
            else None
        )
        if not expected or not _admin_token_matches(expected, x_admin_token):
            return JSONResponse({"error": "Unauthorized."}, status_code=401)
    return JSONResponse(_load_config())


@app.post("/api/config")
async def update_config(
    body: dict[str, Any] | None = None,
    x_admin_token: str | None = Header(default=None),
):
    """Update config and hot-reload agent chain.

    Hot-swapping models is a privileged side effect reachable by any same-origin
    visitor, so gate it behind the admin token WHEN one is configured: with
    TURTLE_ADMIN_TOKEN set, every POST must carry a matching X-Admin-Token header
    (401 otherwise). When the token is unset (local dev) the endpoint stays open,
    preserving the current zero-config developer flow. GET /api/config now
    mirrors this gate in cloud (see get_config's docstring); local's GET stays
    open on purpose — the dev panel reads config to render, and it exposes no
    secrets.
    """
    global config
    expected = (
        settings.admin_token.get_secret_value()
        if settings.admin_token is not None
        else None
    )
    if not expected:
        # No admin token configured. Cloud has always failed closed here — but
        # a tunneled LOCAL deploy (ngrok, cloudflared) is equally reachable,
        # and hot-swapping the model / agent chain from an anon POST would let
        # an attacker downgrade every user to a chosen provider mid-conversation.
        # Require TURTLE_DEV_ANON=1 as the explicit "yes, this is unsafe" flag,
        # matching the auth path and the channel webhook verifiers.
        if not (settings.dev_anon and not settings.is_cloud):
            return JSONResponse(
                {"error": "Config updates are disabled (TURTLE_ADMIN_TOKEN not set)."},
                status_code=503,
            )
    if expected and not _admin_token_matches(expected, x_admin_token):
        return JSONResponse({"error": "Unauthorized."}, status_code=401)
    if not body:
        return JSONResponse({"error": "Empty body"}, status_code=400)

    current = _load_config()
    current.update(body)
    _save_config(current)
    config = current

    try:
        agents_mgr.rebuild(current)
    except Exception as e:
        return JSONResponse({"error": f"Agent rebuild failed: {e}"}, status_code=500)

    return JSONResponse({"status": "ok", "config": current})


@app.get("/api/models")
async def list_models():
    """List available model options for dev-mode dropdowns."""
    openrouter_models = [
        # OpenAI
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "openai/gpt-5",
        "openai/gpt-5-mini",
        "openai/gpt-4.1",
        "openai/gpt-4o-mini",
        # Anthropic Claude 4.x
        "anthropic/claude-opus-4.7",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-haiku-4.5",
        # Google Gemini
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "google/gemini-2.0-flash-001",
        # xAI / Mistral
        "x-ai/grok-4",
        "x-ai/grok-3-mini",
        "mistralai/mistral-large-2411",
        "mistralai/mistral-small-3.2-24b-instruct",
        # Llama 4 / 3.3
        "meta-llama/llama-4-scout",
        "meta-llama/llama-4-maverick",
        "meta-llama/llama-3.3-70b-instruct",
        # DeepSeek
        "deepseek/deepseek-r1",
        "deepseek/deepseek-chat-v3.1",
        # Qwen 3
        "qwen/qwen3-235b-a22b",
        "qwen/qwen3-30b-a3b",
        "qwen/qwen3-coder",
        # Moonshot
        "moonshotai/kimi-k2-0905",
        # Free tier picks
        "meta-llama/llama-4-scout:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-r1:free",
        "qwen/qwen3-30b-a3b:free",
        "google/gemma-3-27b-it:free",
        "nvidia/llama-3.1-nemotron-70b-instruct:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
    ]
    groq_models = [
        # GPT-OSS (OpenAI weights on Groq)
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        # Llama 4
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        # Llama 3.3 / 3.1
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        # Moonshot Kimi K2
        "moonshotai/kimi-k2-instruct-0905",
        # DeepSeek / Qwen
        "deepseek-r1-distill-llama-70b",
        "qwen/qwen3-32b",
        # Llama 3 legacy
        "llama3-70b-8192",
        "llama3-8b-8192",
        # Gemma
        "gemma2-9b-it",
    ]
    groq_stt_models = [
        "whisper-large-v3-turbo",
        "whisper-large-v3",
        "distil-whisper-large-v3-en",
    ]
    deepgram_tts_models = [
        "aura-asteria-en",
        "aura-luna-en",
        "aura-stella-en",
        "aura-athena-en",
        "aura-hera-en",
        "aura-orion-en",
        "aura-arcas-en",
        "aura-perseus-en",
        "aura-angus-en",
        "aura-orpheus-en",
        "aura-helios-en",
        "aura-zeus-en",
        # aura-2 series
        "aura-2-andromeda-en",
        "aura-2-arcas-en",
        "aura-2-asteria-en",
        "aura-2-luna-en",
        "aura-2-orion-en",
        "aura-2-zeus-en",
    ]
    groq_tts_voices = ["orion", "atlas", "vale", "celeste", "nova"]
    # Combined list for per-agent dropdowns (prefixed with provider)
    all_models = (
        [f"groq:{m}" for m in groq_models]
        + [f"openrouter:{m}" for m in openrouter_models]
    )
    return JSONResponse({
        "openrouter_models": openrouter_models,
        "groq_models": groq_models,
        "groq_stt_models": groq_stt_models,
        "deepgram_tts_models": deepgram_tts_models,
        "groq_tts_voices": groq_tts_voices,
        "all_models": all_models,
    })


@app.get("/api/agents")
async def list_agents():
    """List all runtime agents shown in the dev sidebar."""
    return JSONResponse({"agents": _runtime_agent_registry()})


# ---------------------------------------------------------------------------
# Memory confirmation endpoints
# ---------------------------------------------------------------------------

def _get_user_id_from_request(request: Request) -> str | None:
    """Resolve the HTTP caller's user_id.

    Mirrors ``authenticate_websocket`` (apps/auth.py:40) so cookies and the
    dev-anon escape hatch behave identically on WS and HTTP. Order:

      1. ``turtle_uid`` cookie minted by /onboarding/claim — this is what the
         browser actually sends (`credentials: 'same-origin'`). The previous
         implementation ignored it entirely, so every real logged-in user got
         401 from every memory endpoint in cloud, and off-cloud silently
         resolved to a shared literal ``local_dev_user`` regardless of who was
         logged in — which link redemption in particular must not do.
      2. ``Authorization: Bearer …`` header (legacy / API clients).
      3. Dev-only fallback to ``local_dev_user`` ONLY when
         ``TURTLE_DEV_ANON=1`` AND not cloud. Never the default off-cloud
         behaviour.
    """
    # 1. cookie
    cookie_token = request.cookies.get("turtle_uid")
    if cookie_token:
        try:
            from apps.onboarding_routes import verify_session_cookie
            user_id = verify_session_cookie(cookie_token)
            if user_id:
                return user_id
        except Exception:
            pass

    # 2. bearer
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
        try:
            from apps.auth import verify_token
            payload = verify_token(token)
            sub = payload.get("sub")
            if sub:
                return sub
        except Exception:
            pass

    # 3. explicit dev-anon opt-in only (mirrors WS auth)
    if settings.dev_anon and not settings.is_cloud:
        return "local_dev_user"

    return None


def _build_confirmation_gate_for_user(user_id: str) -> "ConfirmationGate":
    """Construct a ConfirmationGate fresh from durable storage (journal +
    confirmation state), with no dependency on a currently-live SharedState.

    This replaces looking the gate up in _ACTIVE_STATES_BY_USER (a
    process-local cache populated only while a WebSocket/channel turn for
    this user is live IN THIS PROCESS). That lookup is the documented root
    cause of a real bug: on a multi-worker deploy (the Dockerfile's own -w 1
    comment says this already happens at -w 2 today) a POST landing on a
    different worker than the one holding the user's SharedState found
    nothing and 404'd, even though the pending candidate genuinely existed in
    the journal. Building the gate straight from storage on every request
    makes that structurally impossible — the journal and confirmation state
    are the same store from every process/instance, cloud or not.

    Deliberately lightweight: unlike _build_channel_state, this does NOT
    construct a SessionStore (no start_or_restore side effect — a mere
    "check my pending confirmations" GET must never mutate session state),
    RAG system, or retrieval broker. It builds only what ConfirmationGate
    itself needs.
    """
    from core.storage.factory import get_confirmation_state_backend

    personal_memory_store = PersonalMemoryStore(user_id=user_id)
    journal_store = JournalStore(user_id=user_id)
    return ConfirmationGate(
        journal=journal_store,
        store=personal_memory_store,
        state_path=personal_memory_dir(user_id) / "confirmation_state.json",
        state_backend=get_confirmation_state_backend(user_id),
    )


@app.get("/api/memory/pending")
async def get_pending_memory(request: Request):
    """Return all queued memory candidates awaiting user confirmation."""
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    from core.confirmation_gate import _render_question  # noqa: PLC0415
    gate = _build_confirmation_gate_for_user(user_id)
    pending_ids = gate.get_pending_ids()
    items = []
    for event_id in pending_ids:
        event = gate._load_event(event_id)  # noqa: SLF001
        if event is None:
            continue
        items.append({
            "event_id": event.event_id,
            "question": _render_question(event),
            "topic": event.topic,
            "key": event.key,
        })
    return JSONResponse({"pending": items})


@app.post("/api/memory/confirm")
async def confirm_memory(request: Request):
    """Accept or reject a pending memory candidate by event_id."""
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    event_id = body.get("event_id")
    accepted = body.get("accepted")
    if not event_id or not isinstance(event_id, str):
        return JSONResponse({"error": "event_id required"}, status_code=400)
    if not isinstance(accepted, bool):
        return JSONResponse({"error": "accepted (bool) required"}, status_code=400)

    gate = _build_confirmation_gate_for_user(user_id)
    result = gate.record_response(event_id, accepted=accepted)
    if result is None:
        return JSONResponse({"error": "event_id not found in pending queue"}, status_code=404)
    # This panel is now the ONLY confirmation surface (chat-text confirmation
    # is gone entirely), so the side effects the old chat handler owned live
    # here: routine registration on a workflow accept, and the first-confirm
    # telemetry event.
    if accepted:
        emit_event_once(user_id, "memory_first_confirmed", topic=result.topic)
        if result.topic == "workflow":
            _register_user_routines_safe(user_id)
    return JSONResponse({"status": "ok", "applied": accepted})


async def _get_target_account_email(user_id: str) -> str | None:
    """The authoritative email on file for ``user_id`` (the ``users.primary_email``
    column) — used ONLY to check the two-sided link binding (WP1.J, ledger
    1b.6). Deliberately a raw query against the SAME `users` table
    core.identity.IdentityManager / core.storage.cloud.identity_store.
    PostgresIdentityManager already own, rather than a new method on either —
    both files belong to a parallel WP and are not touched here. This value
    is NOT something the redeemer supplies; it's whatever the web sign-in
    flow (magic-link claim / dev fast-path) already put in the users table,
    so there is nothing for a redeemer to spoof by typing a different email
    into this endpoint.
    """
    if not user_id:
        return None
    if settings.is_cloud:
        from core.storage.cloud import get_pg_pool

        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT primary_email FROM users WHERE user_id = $1", user_id
            )
            return row["primary_email"] if row else None

    import aiosqlite

    from core.identity import identity_manager

    async with aiosqlite.connect(identity_manager.db_path) as db:
        async with db.execute(
            "SELECT primary_email FROM users WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


@app.post("/api/account/link")
async def link_account_redeem(request: Request):
    """Redeem a channel claim code against the CALLER's authenticated account.

    This is the second, decisive half of account linking. The claim code proves
    the requester controls the channel identity (Discord etc.); this endpoint's
    authentication proves they own the target Turtle account. Neither alone is
    sufficient, which is why linking is never done from a self-claimed email —
    that would let anyone inherit another person's memory.

    WP1.J two-sided binding: if the code names an expected redeemer email
    (claim.expected_email), the authenticated caller's own account email must
    match it or redemption is refused — see the binding check below for the
    full reasoning.

    On success the channel mapping is re-pointed at the caller and the channel
    account's memory is folded into theirs.
    """
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "Sign in to link an account"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        body = {}
    code = str((body or {}).get("code", "")).strip()
    if not code:
        return JSONResponse({"error": "code is required"}, status_code=400)

    from core.account_linking import (
        mark_consumed, merge_memory, release_reservation, reserve,
    )
    from core.identity import identity_manager
    from core.storage.factory import get_link_code_store

    store = get_link_code_store()

    # ── ORDERING (post-Codex-verification-pass):
    #   1. RESERVE the code atomically for THIS target user_id. Same-target
    #      retries refresh the reservation; a DIFFERENT target within TTL is
    #      rejected. This closes the two-target race — two authenticated
    #      redeemers can no longer both merge and repoint.
    #   2. Take the source's channel lock so in-flight turns don't write into
    #      the snapshot we are about to copy.
    #   3. MERGE the source journal into the target.
    #   4. On merge failure: release the reservation (so THIS target can retry
    #      immediately) and return 503. Mapping and code are unchanged.
    #   5. On merge success: link_channel (idempotent) then mark_consumed
    #      (single-shot). A crash between them leaves the mapping correct AND
    #      the code burnable on retry — because link_channel is a no-op when
    #      already pointing at this user_id. ─────────────────────────────────
    status, claim = await asyncio.to_thread(reserve, store, code, user_id)
    if status == "invalid":
        return JSONResponse(
            {"error": "That code is invalid or has expired"}, status_code=400
        )
    if status == "locked":
        # Reserved for a different target — most likely an interception attempt.
        # Do not reveal that the code exists.
        return JSONResponse(
            {"error": "That code is invalid or has expired"}, status_code=400
        )
    assert claim is not None  # status=="ok" always yields a claim

    # ── TWO-SIDED BINDING (WP1.J, ledger 1b.6) ───────────────────────────────
    # claim.expected_email is what the channel-side issuer said their web
    # account would be — a self-claimed assertion that proves nothing by
    # itself, so it is never used to grant anything. It is only ever used to
    # REFUSE a redeemer whose actually-authenticated account doesn't match,
    # closing the gap where ANY authenticated session (not just the one the
    # channel user intended) could previously redeem a leaked/intercepted
    # code. A code minted before this shipped has expected_email=None and is
    # treated as unbound — see core/account_linking.py's module docstring for
    # why that's the deliberate, TTL-bounded choice, not an oversight.
    #
    # The mismatch response is byte-for-byte the SAME "invalid or expired"
    # 400 used above for an unknown/locked code (never a distinct message or
    # status): a prober holding a stolen code who tries authenticating as
    # different accounts must not be able to learn "wrong account" vs "code
    # doesn't exist" vs "someone else already claimed it" — that would turn
    # this endpoint into an oracle for which email a given code is bound to.
    if claim.expected_email:
        target_email = await _get_target_account_email(user_id)
        if not target_email or target_email.strip().lower() != claim.expected_email:
            # Release rather than let the wrong target squat on the
            # reservation for the full 60s TTL — the intended redeemer should
            # not have to wait out someone else's failed attempt.
            await asyncio.to_thread(release_reservation, store, code, user_id)
            return JSONResponse(
                {"error": "That code is invalid or has expired"}, status_code=400
            )

    if claim.source_user_id == user_id:
        # Nothing to merge; burn the reservation so it can't be replayed.
        await asyncio.to_thread(store.consume, code)
        return JSONResponse({"status": "ok", "already_linked": True, "merged": {}})

    # Lock on the CHANNEL IDENTITY (channel, channel_user_id), same key the
    # dispatch handler now uses. Any in-flight or queued turn for this Discord
    # user waits behind us — and on the other side of the lock, it re-resolves
    # user_id and picks up the NEW mapping. If we locked on the old user_id
    # instead, a queued turn resolving after our re-point could enter concurrently.
    source_lock = _channel_state_lock((claim.channel, claim.channel_user_id))
    async with source_lock:
        # Drain detached writers (per-turn extraction, reflector Stage-B +
        # rolling summary, embed jobs) that started under the source user_id.
        # They outlive _channel_dispatch_handler and would otherwise append
        # into a journal that is about to become unreachable, silently
        # stranding those writes. drain_user_tasks is bounded (5s) so a hung
        # task can't stall linking indefinitely. Evict the cached source
        # SharedState UP FRONT so anything that resolves after this sees the
        # new target on the next turn.
        from core.worker import drain_user_tasks

        _CHANNEL_STATES.pop((claim.source_user_id, claim.channel), None)
        drained = await drain_user_tasks(claim.source_user_id, timeout=5.0)
        if drained:
            print(f"LOG: link drained {drained} in-flight source task(s) for {claim.source_user_id}")

        merged = await asyncio.to_thread(merge_memory, claim.source_user_id, user_id)
        if not merged.get("ok", False):
            await asyncio.to_thread(release_reservation, store, code, user_id)
            print(
                f"LOG: link merge FAILED for {claim.source_user_id}->{user_id}: "
                f"{merged.get('error','?')} — mapping unchanged, reservation released"
            )
            return JSONResponse(
                {"error": "Could not import your channel memory — please try again in a minute"},
                status_code=503,
            )

        try:
            previous = await identity_manager.link_channel(
                user_id=user_id, channel=claim.channel, channel_user_id=claim.channel_user_id
            )
            consumed = await asyncio.to_thread(mark_consumed, store, code)
        except Exception:
            # link_channel or consume threw AFTER merge already succeeded. The
            # journal writes are idempotent by event_id and link_channel is a
            # no-op for a same-target rebind, so a retry from the SAME target
            # heals it — but only if their reservation stays alive. Release it
            # eagerly so the retry doesn't wait for the 60s TTL.
            await asyncio.to_thread(release_reservation, store, code, user_id)
            raise
        if not consumed:
            # Only reachable if someone raced the consume AFTER we reserved —
            # they'd have needed our target too (reservation blocks others), so
            # link_channel above already saw the same target and was a no-op.
            print(f"LOG: link code raced during consume: {code[:2]}***")

        # Cache was already evicted before the merge — nothing more to drop
        # here. We keep the LOCK in place so any queued turn for the source
        # awaits us and then rebuilds against the new mapping.

    print(
        f"LOG: account linked channel={claim.channel} external={claim.channel_user_id} "
        f"-> {user_id} (was {previous}) merged={merged}"
    )
    return JSONResponse(
        {
            "status": "ok",
            "channel": claim.channel,
            "linked_to": user_id,
            "previous_user_id": previous,
            "merged": merged,
        }
    )


@app.get("/api/memory/profile")
async def get_memory_profile(request: Request):
    """Return everything Turtle currently remembers ABOUT THE CALLER.

    Read straight from the rendered topic files on disk (the applied, confirmed
    projection), so it reflects the durable memory and works even without an
    active WebSocket session. This is the user-facing "what do you remember
    about me" view; /api/memory/pending is the separate confirm-these queue.
    """
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    from core.personal_memory_store import PersonalMemoryStore  # noqa: PLC0415
    from core.memory_schema import TOPICS  # noqa: PLC0415

    store = PersonalMemoryStore(user_id=user_id)
    topics: list[dict[str, Any]] = []
    for topic_key, spec in TOPICS.items():
        try:
            doc = store.load_topic(topic_key)
        except Exception:
            continue
        # Rendered lines look like "- Name: Maya Chen"; strip the bullet for a
        # clean display list, drop blanks.
        lines = [
            str(line).strip().lstrip("- ").strip()
            for line in (doc.lines or [])
            if str(line).strip()
        ]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue
        topics.append({
            "topic": topic_key,
            "title": spec.title,
            "summary": spec.summary,
            "lines": lines,
        })
    return JSONResponse({"topics": topics, "empty": len(topics) == 0})


# ---------------------------------------------------------------------------
# WebSocket: Main chat interface
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("LOG: WebSocket client connected")
    
    try:
        user_id = await authenticate_websocket(ws)
    except Exception:
        return

    # Build SharedState for this connection
    async with httpx.AsyncClient() as client:
        # Tenant-scoped: resume/sweep must never see another user's sessions.
        session_store = SessionStore(user_id=user_id)
        restore_result = await session_store.start_or_restore(mode=SESSION_RESTORE_MODE)
        # Personal memory lives under personal_memory_dir(user_id); there is no
        # single-tenant store to construct.
        personal_memory_store = PersonalMemoryStore(user_id=user_id)
        # The read-model index is a local SQLite file — pointless to build on
        # ephemeral serverless disk every cold start (rebuilt from scratch
        # next invocation anyway), and every consumer below already accepts
        # sqlite_index=None (falls back to a plain journal scan, which in
        # cloud mode already reads from Postgres — core/memory_journal.py).
        sqlite_index = None
        if not settings.is_cloud:
            from core.memory_sqlite import MemorySQLiteIndex
            # The index is a derived read model — if it can't open (locked file,
            # failed column migration), degrade to journal scans rather than kill
            # the session. Every consumer below accepts sqlite_index=None.
            try:
                sqlite_index = MemorySQLiteIndex(user_id=user_id)
            except Exception as exc:
                print(f"LOG: SQLite memory index unavailable for {user_id}: {exc}; falling back to journal scans")
                sqlite_index = None
        journal_store = JournalStore(
            user_id=user_id,
            on_append=sqlite_index.index_event if sqlite_index is not None else None,
        )
        # Backfill the FTS5 index from the journal (idempotent; no-op on restart).
        if sqlite_index is not None:
            try:
                sqlite_index.backfill_from_journal(journal_store)
            except Exception as exc:
                print(f"LOG: SQLite memory index backfill failed for {user_id}: {exc}")
        from core.storage.factory import get_confirmation_state_backend
        confirmation_gate = ConfirmationGate(
            journal=journal_store,
            store=personal_memory_store,
            state_path=personal_memory_dir(user_id) / "confirmation_state.json",
            state_backend=get_confirmation_state_backend(user_id),
            # Phase 2 W3: indexed hot-path lookups instead of O(n) journal scans.
            sqlite_index=sqlite_index,
        )
        personal_memory_prompt = PersonalMemoryPromptBuilder(
            personal_memory_store,
            config=PersonalMemoryPromptConfig(
                max_bytes=PERSONAL_MEMORY_MAX_BYTES,
                max_topic_files=PERSONAL_MEMORY_MAX_TOPIC_FILES,
            ),
        )
        task_history_store = TaskHistoryStore(TASK_HISTORY_FILE, user_id=user_id)
        rag_system = TurtleRAGSystem(user_id=user_id)

        # D4: construct RetrievalBroker for 4-tier memory context retrieval
        from core.storage.factory import get_vector_store
        from core.retrieval_broker import RetrievalBroker
        # Process singleton — see the channel-state twin above.
        vector_store = get_vector_store()
        retrieval_broker = RetrievalBroker(
            store=personal_memory_store,
            task_store=task_history_store,
            journal_store=journal_store,
            sqlite_index=sqlite_index,
            session_store=session_store,
            rag_system=rag_system,
            vector_store=vector_store,
            user_id=user_id,
        )

        state = SharedState(
            http_client=client,
            session_store=session_store,
            personal_memory_store=personal_memory_store,
            personal_memory_prompt=personal_memory_prompt,
            journal_store=journal_store,
            confirmation_gate=confirmation_gate,
            task_history_store=task_history_store,
            rag_system=rag_system,
            sqlite_index=sqlite_index,
            retrieval_broker=retrieval_broker,
            reflector=PeriodicReflector(),
            user_id=user_id,
        )
        _register_shutdown_state(state)
        # Phase 5 (W2): expose this socket to the routine scheduler so a fire can
        # reach the user live. Symmetrically discarded in the teardown finally.
        _register_live_socket(user_id, ws)

        # Cloud mode only: subscribe to this user's Redis live-delivery
        # channel so a routine fired by a DIFFERENT instance (almost always
        # true in serverless — see deliver_routine_notice's docstring) can
        # still reach this socket live instead of waiting for the outbox to
        # drain on next connect. Cancelled in the teardown finally below,
        # symmetrically with _register_live_socket/_discard_live_socket.
        redis_relay_task: "asyncio.Task | None" = None
        if settings.is_cloud:
            redis_relay_task = asyncio.create_task(
                _relay_redis_live_frames(user_id, ws), name=f"redis_relay_{user_id}"
            )

        # Process pending sessions from previous runs (personal memory finalization).
        # list_pending_finalization_archives now returns (session_id, message_history)
        # directly from SQLite — no file-based archive path needed.
        for pending_sid, pending_messages in await session_store.list_pending_finalization_archives():
            print(f"LOG: Finalizing pending session {pending_sid}")
            if pending_messages:
                _sync_personal_memory_from_messages(
                    state, session_id=pending_sid, message_history=pending_messages,
                )
                try:
                    await run_stage_b_session_extractor(
                        state, session_id=pending_sid, message_history=pending_messages,
                    )
                except Exception as _e:
                    print(f"LOG: Stage B error for pending session {pending_sid}: {_e}")
            try:
                await session_store.mark_finalized(pending_sid)
            except Exception as _e:
                print(f"LOG: mark_finalized failed for {pending_sid}: {_e}")
        if state.sqlite_index is not None:
            try:
                state.sqlite_index.checkpoint()
            except Exception:
                pass

        await rag_system.start_session(session_id=restore_result.session_id)
        message_history: list[ModelMessage] | None = session_store.message_history or None
        # Live Deepgram Flux streaming-STT session for this connection, when the
        # client opts in with a mic_open frame (only if the server enables it).
        mic_session: "_MicStreamSession | None" = None

        if restore_result.restored:
            await _ws_send_json(ws, {
                "type": "status",
                "status": "restored",
                "session_id": restore_result.session_id,
                "message_count": restore_result.message_count,
            })

        await _ws_send_json(ws, {
            "type": "status",
            "status": "ready",
            # Capability advertisement: the client only streams mic frames when the
            # server actually has streaming STT enabled.
            "stream_stt": _voice_stream_stt_enabled(),
        })

        # Phase 5 (W2): drain any routine fires that arrived while this user had
        # no live socket (queued by deliver_routine_notice). Verified sends: a
        # frame popped from the queue is only gone once a socket actually
        # accepted it — if this socket dies mid-drain, the remainder re-queues
        # for the next connect instead of vanishing (Codex P5 #2).
        _pending_routines = pop_pending_routine_notices(user_id)
        for _i, _pending_routine in enumerate(_pending_routines):
            try:
                async with _ws_send_lock(ws):
                    await ws.send_json(_pending_routine)
            except Exception:
                for _frame in _pending_routines[_i:]:
                    _stash_pending_routine_notice(user_id, _frame)
                break

        try:
            while True:
                raw = await ws.receive()

                # Starlette sends an explicit disconnect frame before closing.
                # Exit loop immediately to avoid a RuntimeError on next receive().
                if raw.get("type") == "websocket.disconnect":
                    break

                # Phase 6: per-user inbound rate limit. Helper bubbles
                # WebSocketRateLimitExceeded so we can close cleanly.
                async def _check_user_message_rate() -> bool:
                    try:
                        get_ws_rate_limiter().check_and_record(user_id)
                        return True
                    except WebSocketRateLimitExceeded as exc:
                        await _ws_send_json(ws, {
                            "type": "error",
                            "code": "rate_limited",
                            "window": exc.window,
                            "limit": exc.limit,
                            "message": (
                                f"Message rate limit reached "
                                f"({exc.limit}/{exc.window}). Try again later."
                            ),
                        })
                        await ws.close(code=1008, reason="rate_limited")
                        return False

                # Binary frame = audio data
                if "bytes" in raw and raw["bytes"]:
                    audio_bytes = raw["bytes"]
                    # Streaming mode: an open Flux session consumes incremental
                    # PCM frames. Turns are driven by Flux EndOfTurn events in the
                    # consumer, and rate-limited once at mic_open — not per frame.
                    if mic_session is not None:
                        await mic_session.stt.send_audio(audio_bytes)
                        continue
                    # Legacy one-shot mode: the whole utterance in a single frame.
                    if not await _check_user_message_rate():
                        break
                    message_history = await _handle_audio_message(
                        ws, state, audio_bytes, message_history
                    )
                    continue

                # Text frame = JSON message
                if "text" in raw and raw["text"]:
                    try:
                        msg = json.loads(raw["text"])
                    except json.JSONDecodeError:
                        await _ws_send_json(ws, {"type": "error", "message": "Invalid JSON"})
                        continue

                    msg_type = msg.get("type", "")

                    if msg_type == "text":
                        content = str(msg.get("content", "")).strip()
                        if content:
                            if not await _check_user_message_rate():
                                break
                            # Reclaim a finishing voice session so the text turn
                            # builds on the up-to-date conversation, not a stale copy.
                            if mic_session is not None:
                                message_history = await _reclaim_mic_session(mic_session)
                                mic_session = None
                            message_history = await _handle_text_message(
                                ws, state, content, message_history
                            )

                    elif msg_type == "audio":
                        # Base64-encoded audio fallback
                        audio_b64 = msg.get("data", "")
                        sample_rate = int(msg.get("sample_rate", 16000))
                        if audio_b64:
                            if not await _check_user_message_rate():
                                break
                            audio_bytes = base64.b64decode(audio_b64)
                            message_history = await _handle_audio_message(
                                ws, state, audio_bytes, message_history,
                                sample_rate=sample_rate,
                            )

                    elif msg_type == "mic_open":
                        # Begin streaming STT for this utterance/conversation.
                        if not _voice_stream_stt_enabled():
                            await _ws_send_json(ws, {
                                "type": "error",
                                "message": "Streaming STT is not enabled on the server.",
                            })
                        else:
                            # Reclaim any prior (still-finishing) session first so
                            # we carry its history into the new one.
                            if mic_session is not None:
                                message_history = await _reclaim_mic_session(mic_session)
                                mic_session = None
                            if not await _check_user_message_rate():
                                break
                            sample_rate = int(msg.get("sample_rate", 16000))
                            try:
                                mic_session = await _open_mic_stream(
                                    ws, state, message_history, sample_rate=sample_rate,
                                )
                            except Exception as mic_exc:
                                print(f"LOG: mic_open failed: {mic_exc}")
                                traceback.print_exc()
                                await _ws_send_json(ws, {
                                    "type": "error",
                                    "message": f"Could not start streaming STT: {mic_exc}",
                                })
                                mic_session = None

                    elif msg_type == "mic_close":
                        # Non-blocking: flush + finalise in the background so the
                        # receive loop stays free (an interrupt can still arrive).
                        # The session is reclaimed on the next mic_open / disconnect.
                        if mic_session is not None:
                            await _begin_mic_close(mic_session)

                    elif msg_type == "interrupt":
                        # Stop the agent mid-reply (barge-in / explicit stop).
                        if mic_session is not None and mic_session.interrupt():
                            print("LOG: interrupt — reply cancelled by user")

                    elif msg_type == "ping":
                        await _ws_send_json(ws, {"type": "pong"})

        except WebSocketDisconnect:
            print("LOG: WebSocket client disconnected")
        except RuntimeError as e:
            # Some disconnect paths surface as RuntimeError instead of WebSocketDisconnect.
            if "disconnect message has been received" in str(e):
                print("LOG: WebSocket client disconnected")
            else:
                print(f"LOG: WebSocket error: {e}")
                traceback.print_exc()
        except Exception as e:
            print(f"LOG: WebSocket error: {e}")
            traceback.print_exc()
        finally:
            # Symmetric with the redis_relay_task startup above.
            if redis_relay_task is not None:
                redis_relay_task.cancel()
                try:
                    await redis_relay_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Tear down any live/finishing streaming-STT session so its worker
            # threads and websocket don't leak when the client drops.
            if mic_session is not None:
                try:
                    mic_session.interrupt()
                    await mic_session.stt.aclose()
                    if mic_session.consumer_task is not None:
                        mic_session.consumer_task.cancel()
                except Exception:
                    pass
                mic_session = None

            # Session cleanup. The legacy single-tenant MemoryStore checkpoint
            # was dropped here — personal memory is journaled per-turn and needs
            # no session-end flush.
            session_id = state.session_store.session_id
            # Capture messages before archive_active() clears them.
            final_messages = list(state.session_store.message_history)
            await state.session_store.archive_active(status="pending_finalization")
            try:
                # Index this session's conversations into the per-user episodic
                # store NOW — end_session was previously only reachable from the
                # next start_session in the same process, so no web session was
                # ever indexed and cross-session recall returned nothing.
                await state.rag_system.end_session()
            except Exception as _e:
                print(f"LOG: episodic end_session failed for {session_id}: {_e}")
            if session_id and final_messages:
                _sync_personal_memory_from_messages(
                    state, session_id=session_id, message_history=final_messages,
                )
                try:
                    await run_stage_b_session_extractor(
                        state, session_id=session_id, message_history=final_messages,
                    )
                except Exception as _e:
                    print(f"LOG: Stage B error for session {session_id}: {_e}")
            if session_id:
                try:
                    # Extraction just ran on final_messages; without this flip the
                    # next connect re-extracts the same session.
                    await state.session_store.mark_finalized(session_id)
                except Exception as _e:
                    print(f"LOG: mark_finalized failed for {session_id}: {_e}")
            if state.sqlite_index is not None:
                try:
                    # CLOSE, not just checkpoint (close() checkpoints first).
                    # MemorySQLiteIndex opens a sqlite3 connection in __init__
                    # and one is built per WebSocket connect; this path used to
                    # call checkpoint() only, so the fd leaked on every
                    # connect/disconnect cycle and only _shutdown_state (process
                    # exit) ever reclaimed it. Page reloads are frequent, so it
                    # accumulated for the life of the process.
                    state.sqlite_index.close()
                except Exception:
                    pass
                # _shutdown_state guards on `is not None`, so clearing the ref
                # keeps the shutdown path from touching a closed connection.
                state.sqlite_index = None
            print("LOG: Session archived and cleaned up")
            _unregister_shutdown_state(state)
            # Phase 5 (W2): stop advertising this socket to the scheduler.
            _discard_live_socket(user_id, ws)


# ---------------------------------------------------------------------------
# Message handlers
# ---------------------------------------------------------------------------

async def _ws_send_json(ws: WebSocket, data: dict[str, Any]) -> None:
    """Send a JSON message to the WebSocket client.

    Serialized per socket: routine delivery runs as a separate loop task from
    the connection handler, and interleaved multi-writer sends on one Starlette
    websocket are not safe (Codex P5 #4).
    """
    try:
        async with _ws_send_lock(ws):
            await ws.send_json(data)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Phase 5 (W2): routine delivery
# ---------------------------------------------------------------------------
# When a routine fires but the user has no live socket (offline, or the app loop
# was never captured), its frame is stashed here and drained on the user's next
# connect. A small LIST per user so multiple fires queue instead of clobbering
# each other. Stashed from the scheduler worker thread, drained on the app loop,
# so access is lock-guarded.
_PENDING_ROUTINE_NOTICES: dict[str, list[dict[str, Any]]] = {}
_PENDING_ROUTINE_NOTICES_LOCK = threading.Lock()
# Cap on queued fires PER USER (most recent kept). The number of distinct users
# held is bounded by _STORAGE_CAP_REGISTRY_CAP, mirroring the storage-cap path.
_PENDING_ROUTINE_MAX_PER_USER = 5

# Durable write-through: the in-memory dict above is the hot cache; each user's
# queue is mirrored to <personal_memory_dir(user_id)>/routine_outbox.json
# (core.routine_outbox) so a process restart between a fire and its delivery no
# longer drops the notice. Best-effort — save/load only LOG on error.
from core.routine_outbox import load_outbox as _load_outbox, save_outbox as _save_outbox


def _stash_pending_routine_notice(user_id: str, frame: dict[str, Any]) -> None:
    """Queue a routine frame for delivery on the user's next connect (bounded).

    Durable write-through outbox under the user's memory dir; survives restarts;
    loaded lazily on next connect. The in-memory queue is the hot cache, but
    after mutating it we persist the user's (capped) queue to a small JSON outbox
    under their personal-memory dir (core.routine_outbox). A process restart
    between a fire and its delivery therefore no longer drops the notice — the
    frames survive on disk and are surfaced on the user's next connect (see
    pop_pending_routine_notices; no startup scan is needed). The persist happens
    under the same lock (the write is per-user and tiny — bounded work) and is
    best-effort: an I/O or storage-cap error is LOGged inside save_outbox and
    swallowed, leaving the in-memory queue intact (memory-only fallback).
    """
    with _PENDING_ROUTINE_NOTICES_LOCK:
        queue = _PENDING_ROUTINE_NOTICES.get(user_id)
        if queue is None:
            # Cap distinct users held in MEMORY. Evict the OLDEST user's queue
            # (dict preserves insertion order) rather than clearing wholesale — a
            # clear would drop every queued reminder for 500+ unrelated users
            # because one more showed up (Codex P5 review #4). The evicted user's
            # on-disk outbox is deliberately LEFT in place: memory eviction only
            # bounds RAM, and their frames should survive to their next connect
            # rather than be lost — memory eviction != data loss (Phase 6 W1).
            if len(_PENDING_ROUTINE_NOTICES) >= _STORAGE_CAP_REGISTRY_CAP:
                _PENDING_ROUTINE_NOTICES.pop(next(iter(_PENDING_ROUTINE_NOTICES)), None)
            # HYDRATE from disk before the first write-through for this user
            # (Codex P6 #2): after a restart or memory eviction the hot cache is
            # empty while the outbox file still holds frames — saving only the
            # new frame would clobber them. A load failure (None) hydrates
            # nothing; the subsequent save then overwrites, accepting the rare
            # loss over blocking the stash.
            disk_frames = _load_outbox(user_id)
            queue = list(disk_frames) if disk_frames else []
            _PENDING_ROUTINE_NOTICES[user_id] = queue
        queue.append(frame)
        # Keep only the most recent N fires for this user.
        if len(queue) > _PENDING_ROUTINE_MAX_PER_USER:
            del queue[:-_PENDING_ROUTINE_MAX_PER_USER]
        # Write-through the (capped) queue to the durable outbox.
        _save_outbox(user_id, queue)


def _routine_frame_identity(frame: dict[str, Any]) -> tuple:
    """Dedupe identity for a routine frame.

    (routine_key, fired_at) when either is present — real frames always carry
    both. Frames lacking both fall back to canonical content so a write-through
    mirror copy (same content, freshly deserialized → different object) still
    collapses against its in-memory twin.
    """
    rk = frame.get("routine_key")
    fa = frame.get("fired_at")
    if rk is not None or fa is not None:
        return ("k", rk, fa)
    try:
        return ("c", json.dumps(frame, sort_keys=True, ensure_ascii=False))
    except Exception:
        return ("i", id(frame))


def _merge_routine_frames(
    disk_frames: list[dict[str, Any]], mem_frames: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge disk-first + memory frames, deduping by frame identity.

    Normal write-through case: disk mirrors memory, so the dedupe collapses the
    mirror to one copy each (disk-order preserved). Restart case: memory is
    empty and every disk frame is surfaced.
    """
    merged: list[dict[str, Any]] = []
    seen: set = set()
    for frame in list(disk_frames) + list(mem_frames):
        ident = _routine_frame_identity(frame)
        if ident in seen:
            continue
        seen.add(ident)
        merged.append(frame)
    return merged


def pop_pending_routine_notices(user_id: str) -> list[dict[str, Any]]:
    """Handler-facing: fetch and clear a user's queued routine frames (drain).

    Consults BOTH stores: the in-memory hot cache AND the durable on-disk outbox.
    After a process restart the memory dict is empty but the outbox file still
    holds the frames, so the merged result is disk-frames-first + memory-frames,
    deduped by (routine_key, fired_at) — frames lacking both are deduped by
    canonical content so the write-through mirror collapses to one copy. Both the
    memory queue and the disk file are then cleared. This is what makes
    durability lazy: no startup scan is needed — a user's outbox loads on their
    next connect-time drain (which calls this).
    """
    # Memory mutation under the lock; disk I/O OUTSIDE it (Codex P6 #3 — pop
    # runs on the app loop, and file I/O under the global lock would stall both
    # the loop and every scheduler-thread stash). A stash landing between the
    # two steps keeps its frame in the memory dict (picked up by the next pop);
    # its write-through may be cleared below, but the frame itself survives.
    with _PENDING_ROUTINE_NOTICES_LOCK:
        mem_frames = _PENDING_ROUTINE_NOTICES.pop(user_id, [])
    disk_frames = _load_outbox(user_id)
    if disk_frames is None:
        # Load failed — the file may still hold frames we never read. Do NOT
        # clear it (Codex P6 #4); deliver what memory had and leave disk for a
        # later, healthier pop.
        return mem_frames
    merged = _merge_routine_frames(disk_frames, mem_frames)
    _save_outbox(user_id, [])  # clear disk; both stores now drained
    # Cap coherence after the merge: disk(≤5) + memory(≤5) could return 10;
    # keep the most-recent N like every other bound on this queue (Codex P6 #7).
    return merged[-_PENDING_ROUTINE_MAX_PER_USER:]


# Per-socket send locks. Routine delivery runs as its own loop task, concurrent
# with the connection handler's frame sends — Starlette websockets are not a
# safe multi-writer queue, so every writer serializes per socket (Codex P5 #4).
# WeakKeyDictionary: a closed socket's lock dies with it.
_WS_SEND_LOCKS: "weakref.WeakKeyDictionary[Any, asyncio.Lock]" = weakref.WeakKeyDictionary()


def _ws_send_lock(ws: Any) -> asyncio.Lock:
    lock = _WS_SEND_LOCKS.get(ws)
    if lock is None:
        lock = asyncio.Lock()
        _WS_SEND_LOCKS[ws] = lock
    return lock


async def _deliver_routine_on_loop(user_id: str, frame: dict[str, Any]) -> None:
    """Runs ON the app loop: try each of the user's sockets, verify the send.

    A registered-but-dead socket (abnormal disconnect whose teardown hasn't run
    yet) must not count as delivered — _ws_send_json swallows errors, so we call
    ws.send_json directly here, discard sockets whose send raises, and stash the
    frame for next-connect delivery when NO socket actually accepted it
    (Codex P5 review #2: scheduled != delivered).
    """
    with _LIVE_SOCKETS_LOCK:
        sockets = list(_LIVE_SOCKETS.get(user_id, ()))
    delivered = False
    for ws in sockets:
        try:
            async with _ws_send_lock(ws):
                await ws.send_json(frame)
            delivered = True
        except Exception:
            # Dead socket: stop advertising it so future fires skip it.
            _discard_live_socket(user_id, ws)
    if not delivered:
        _stash_pending_routine_notice(user_id, frame)


async def _drain_pending_routines_on_loop(user_id: str) -> None:
    """Runs ON the app loop: deliver any queued frames to a now-live socket.

    Used to close the connect-vs-stash race — same verified-send semantics as
    the connect-time drain (undelivered frames re-queue).
    """
    with _LIVE_SOCKETS_LOCK:
        sockets = list(_LIVE_SOCKETS.get(user_id, ()))
    if not sockets:
        return
    pending = pop_pending_routine_notices(user_id)
    for i, frame in enumerate(pending):
        delivered = False
        for ws in sockets:
            try:
                async with _ws_send_lock(ws):
                    await ws.send_json(frame)
                delivered = True
                break
            except Exception:
                _discard_live_socket(user_id, ws)
        if not delivered:
            for remaining in pending[i:]:
                _stash_pending_routine_notice(user_id, remaining)
            return


async def _relay_redis_live_frames(user_id: str, ws: WebSocket) -> None:
    """Cloud-mode WS task: forward this user's Redis live-delivery channel
    straight to their socket for the life of the connection.

    Started once per WS connect (websocket_endpoint) and cancelled on
    disconnect. This is what makes deliver_routine_notice's cross-instance
    Redis publish actually reach a browser tab: the publish alone only tells
    Redis "someone wants this," an active subscriber is what turns that into
    a delivered frame. Runs until cancelled; any error (Redis hiccup,
    connection drop) ends the loop quietly — a lost live push still has the
    Postgres outbox as a correctness backstop, so this task failing is a UX
    degrade, never a data-loss bug.
    """
    try:
        from core.storage.cloud.live_delivery import open_user_subscription

        pubsub = await open_user_subscription(user_id)
    except Exception as e:
        print(f"LOG: redis live relay subscribe failed user={user_id}: {e}")
        return
    try:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue  # subscribe/unsubscribe confirmations, not a payload
            try:
                frame = json.loads(message["data"])
            except Exception:
                continue
            await _ws_send_json(ws, frame)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"LOG: redis live relay error user={user_id}: {e}")
    finally:
        try:
            await pubsub.unsubscribe()
            await pubsub.aclose()
        except Exception:
            pass


def deliver_routine_notice(user_id: str, frame: dict[str, Any]) -> bool:
    """Push a routine frame to the user's live socket(s), or queue it.

    Called from the routine scheduler's worker thread when a routine fires
    (local mode) or from apps/cron_tick_routes.py's cron-tick request thread
    (cloud mode — see core.routine_scheduler._fire_routine). The ONLY safe
    cross-thread bridge for THIS process's own sockets is
    asyncio.run_coroutine_threadsafe against the captured app loop
    (_APP_LOOP) — we never create_task/get_running_loop from this thread, and
    never block on the returned future's .result(). The bridged coroutine
    verifies each send and stashes the frame itself when no socket accepted
    it, so a stale registry entry cannot lose a fire.

    In cloud mode, the firing process and the process holding the user's live
    WebSocket are almost always DIFFERENT instances — _LIVE_SOCKETS is
    process-local and cannot see across that boundary. Before falling back to
    the durable outbox, cloud mode first PUBLISHES to the user's Redis
    channel (core.storage.cloud.live_delivery); any OTHER instance with that
    socket open is subscribed and relays it live (see the WebSocket handler's
    _redis_live_relay task). A publish reaching zero subscribers (nobody has
    the socket open anywhere) still falls through to the outbox exactly as
    before — never lost, just delivered on next connect instead of live.

    Returns True when the delivery attempt was scheduled onto the app loop OR
    published to at least one cross-instance subscriber; False when it was
    stashed directly. Never raises — a delivery failure must never affect the
    journal write that already happened upstream.
    """
    try:
        loop = _APP_LOOP
        with _LIVE_SOCKETS_LOCK:
            has_sockets = bool(_LIVE_SOCKETS.get(user_id))
        if not has_sockets and settings.is_cloud:
            from core.storage.cloud.live_delivery import publish_routine_frame_sync

            if publish_routine_frame_sync(user_id, frame) > 0:
                return True  # delivered live by a different instance
        if loop is not None and not loop.is_closed() and has_sockets:
            try:
                asyncio.run_coroutine_threadsafe(
                    _deliver_routine_on_loop(user_id, frame), loop
                )
                return True
            except Exception as e:
                print(f"LOG: routine notice bridge failed user={user_id}: {e}")
        # No socket / no usable loop / bridge raised → queue for next connect.
        _stash_pending_routine_notice(user_id, frame)
        # Close the connect-vs-stash race (Codex P5 #3): if the user connected
        # between our no-socket snapshot and the stash above, their connect-time
        # drain may have already run against an empty queue — re-check and, if a
        # socket is now live, bridge a delivery pass so the frame doesn't wait
        # for a future reconnect.
        if loop is not None and not loop.is_closed():
            with _LIVE_SOCKETS_LOCK:
                connected_now = bool(_LIVE_SOCKETS.get(user_id))
            if connected_now:
                try:
                    asyncio.run_coroutine_threadsafe(
                        _drain_pending_routines_on_loop(user_id), loop
                    )
                except Exception:
                    pass  # queue still holds the frame for next connect
        return False
    except Exception as e:  # defensive: this path must never raise
        print(f"LOG: deliver_routine_notice error user={user_id}: {e}")
        try:
            _stash_pending_routine_notice(user_id, frame)
        except Exception:
            pass
        return False


def _classify_handler_error(exc: Exception) -> tuple[str, str]:
    """Phase 1 / F1+F2+F3: map an exception to a (code, user-friendly message).

    The raw exception string (e.g. pydantic-ai ModelHTTPError stack fragments
    with `status_code: 402, model_name: ...`) must never reach the UI toast.
    """
    try:
        from pydantic_ai.exceptions import ModelHTTPError
    except Exception:
        ModelHTTPError = None  # type: ignore

    if ModelHTTPError is not None and isinstance(exc, ModelHTTPError):
        status = getattr(exc, "status_code", None)
        body = str(exc).lower()
        if status == 402:
            return "credit_exhausted", (
                "I'm out of credits on a backend right now. Please ping the operator."
            )
        if status in (429, 503):
            return "upstream_overload", (
                "Search is unavailable right now. Try again in a minute."
            )
        if status == 400 and ("harmony" in body or "render tokens" in body or "tools should have a name" in body):
            return "serialization_bug", (
                "I hit an internal serialization bug — it's been logged."
            )
        if isinstance(status, int) and status >= 500:
            return "upstream_overload", (
                "A backend is having trouble. Try again in a moment."
            )
        return "upstream_error", "Something went wrong upstream. The error has been logged."

    msg = str(exc).lower()
    if "timeout" in msg or isinstance(exc, asyncio.TimeoutError):
        return "timeout", "That took too long to come back. Try again."
    return "internal_error", "Something went wrong. The error has been logged."


# ---------------------------------------------------------------------------
# Daily token budget (WP1.D2 / ledger 1a.4 part 4)
# ---------------------------------------------------------------------------
#
# Cloud-mode only: local has no Redis to reserve spend against, so
# _reserve_daily_spend/_finalize_daily_spend are no-ops off-cloud — every
# local turn is unmetered. This mirrors how get_ws_rate_limiter()/
# get_channel_gate_buffer() branch (Redis in cloud, an in-process/no-op
# stand-in locally): there is no local equivalent of a cross-tenant spend cap
# because a local deployment IS single-tenant already.
#
# RESERVATION, NOT CHECK-THEN-ACT. An earlier version of this budget read the
# user's spend, compared it to the limit, let the turn run, then recorded the
# real cost afterwards. That has the exact shape wave 1's email idempotency
# bug had before its SET-NX fix: N concurrent turns for the same user (extra
# browser tabs, a scripted client) all read the SAME pre-turn spend, all pass
# the check before any of them records, and the user's real budget becomes
# N x total_tokens_limit with N unbounded — a comment claiming "bounded to
# one turn" would simply be false under concurrency. Verified live: 20
# simultaneous turns against one fake Redis, all 20 passed the pre-check.
#
# So the limit is now Redis's invariant, not a stale application-level read.
# Before a turn runs, _reserve_daily_spend atomically INCRBYs the spend key
# by the turn's WORST-CASE ceiling (agents_mgr.usage_limits.total_tokens_
# limit) and inspects the value the increment itself returned. INCRBY is a
# single atomic Redis command, so under N concurrent reservations each caller
# sees the cumulative total AFTER its own increment — the decision of
# "did I push this over the limit" is made against a value only one turn
# could have produced, not a value someone else might race past next. If the
# post-increment total is over the limit, the reservation is refunded
# immediately (DECRBY) and the turn is refused BEFORE any LLM call. If it's
# within the limit, the reservation stands for the duration of the turn — the
# ceiling is genuinely held, not just assumed. After the turn, _finalize_
# daily_spend adjusts the key by (actual_tokens - reserved_ceiling), which
# self-corrects the estimate down to the real cost (or up, if a single call
# somehow exceeded its own ceiling, though pydantic-ai's UsageLimits should
# prevent that). Concurrent reservations for the same user still serialize
# correctly under this scheme because every adjustment is itself an atomic
# INCRBY of a (possibly negative) delta — never a read-modify-write.
#
# REFUND MUST BE GUARANTEED ON EVERY EXIT PATH. A stranded reservation (taken
# but never finalized) costs the user real budget until the UTC day rolls —
# a worse, longer-lived, user-visible bug than the overshoot this whole
# mechanism exists to close. _finalize_daily_spend therefore runs from
# _execute_turn's outer `finally`, which fires on a normal return, on the
# `except Exception` branch, AND on a bare BaseException that is not an
# Exception at all — asyncio.CancelledError (raised on a client disconnect
# mid-turn) inherits from BaseException specifically so it is NOT caught by
# `except Exception`, but `finally` still runs.
#
# REDIS-UNAVAILABLE POSTURE: fail OPEN (allow the turn, reserve nothing, log
# it). Wave 1's email reservation fails CLOSED because a duplicate send is
# unrecoverable — the cost of a false negative there is a real external side
# effect. Here the cost of a false negative is "this one turn goes unmetered
# during a Redis blip" — recoverable, bounded to that turn, and self-healing
# the moment Redis comes back. Fail-closed here would instead turn a Redis
# blip into a full outage for every metered user simultaneously, which is a
# strictly worse failure mode than an uncommon, temporary loss of metering
# precision. Confirmed against a real unroutable socket (not just the
# CloudBackendUnavailable early-exit), so this also covers a Redis that is up
# but unreachable/stalled, not just a missing REDIS_URL.
#
# WASTED TOKENS COUNT. cascade_stats.total_input_tokens/total_output_tokens
# (what _finalize_daily_spend is given as "actual") already fold in
# wasted_input_tokens/wasted_output_tokens — the real, billed cost of rungs
# that were tried and failed before a later rung succeeded (or before the
# whole cascade gave up). Those tokens were genuinely spent; excluding them
# would undercount exactly the expensive-failure case most worth capturing.
#
# STREAMING GAP: _execute_turn_streaming (the voice/streaming path) performs
# NEITHER a reservation NOR a spend record — it has no CascadeStats to read
# a real cost from at all. This BYPASSES the daily budget ENTIRELY for voice
# turns, not merely "leaves them uncounted": a user can exhaust every bit of
# ledger 1a.4's intended spend ceiling by speaking instead of typing, with no
# refusal, ever, on this path. See the comment at its call to
# stream_agent_text_with_fallbacks. The ledger accepts this explicitly
# ("streaming joins in Phase 4") — it is not an oversight, but whoever picks
# up Phase 4 should understand the size of the gap, not just its existence.

def _utc_day_str() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y%m%d")


def _spend_key(user_id: str) -> str:
    return f"turtle:spend:{user_id}:{_utc_day_str()}"


def _next_utc_midnight_str() -> str:
    """Human-readable UTC reset time for the refusal message."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return tomorrow.strftime("%H:%M UTC on %Y-%m-%d")


def _is_unmetered(user_id: str) -> bool:
    from core.config import parse_unmetered_user_ids

    return bool(user_id) and user_id in parse_unmetered_user_ids(settings.unmetered_user_ids)


# Nice-to-have (coordinator feedback): TURTLE_DAILY_TOKEN_BUDGET<=0 silently
# unmeters EVERY cloud user — one bad env var away from production running
# with no spend ceiling and no operational signal. Logged once per process,
# the first time a reservation attempt actually observes it (not just at
# settings-construction time), since settings.daily_token_budget can be
# reassigned after startup (hot config reload, tests).
_BUDGET_DISABLED_WARNED = False


def _warn_budget_disabled_once() -> None:
    global _BUDGET_DISABLED_WARNED
    if _BUDGET_DISABLED_WARNED:
        return
    _BUDGET_DISABLED_WARNED = True
    print(
        "LOG: TURTLE_DAILY_TOKEN_BUDGET <= 0 — the daily token budget is "
        "DISABLED for every user on this deployment. If that's not "
        "intentional, set TURTLE_DAILY_TOKEN_BUDGET to a positive value."
    )


def _reserve_daily_spend(user_id: str) -> tuple[bool, str | None, int, str]:
    """Atomically reserve one turn's worst-case token ceiling against
    ``user_id``'s daily budget, BEFORE the turn runs. See the module comment
    above for why this replaced a check-then-act read.

    Returns ``(allowed, refusal_message_or_None, reserved_amount, spend_key)``.
    ``reserved_amount`` is 0 whenever nothing was actually reserved — local
    mode, an unmetered user, a disabled budget, a zero/unset per-turn
    ceiling, Redis being unavailable (fail open), or a refusal (the
    reservation taken to find that out is refunded before returning). The
    caller MUST pass ``(spend_key, reserved_amount)`` to
    ``_finalize_daily_spend`` exactly once, on every exit path — see that
    function's docstring.
    """
    if not settings.is_cloud or not user_id:
        return True, None, 0, ""
    limit = int(settings.daily_token_budget or 0)
    if limit <= 0:
        _warn_budget_disabled_once()
        return True, None, 0, ""
    if _is_unmetered(user_id):
        return True, None, 0, ""
    ceiling = int(getattr(agents_mgr.usage_limits, "total_tokens_limit", 0) or 0)
    if ceiling <= 0:
        return True, None, 0, ""

    key = _spend_key(user_id)
    try:
        from core.storage.cloud import get_redis_sync_client

        client = get_redis_sync_client()
        pipe = client.pipeline()  # transaction=True by default: INCRBY + EXPIRE land atomically together
        pipe.incrby(key, ceiling)
        pipe.expire(key, 172800)  # 2 days, per the ledger's chosen option — refreshed on every touch
        results = pipe.execute()
        new_total = int(results[0])
    except Exception as exc:
        print(f"LOG: daily budget reserve: Redis unavailable ({exc}) — failing open, turn allowed unmetered")
        return True, None, 0, ""

    if new_total > limit:
        # Over budget: refund the reservation we just took so a refused turn
        # doesn't itself eat a ceiling-sized chunk of tomorrow's — today's,
        # rather — budget. If the refund itself fails, the reservation is
        # stranded until the UTC day rolls; logged loudly because that's a
        # real user-visible cost (wrongly-refused turns), not just a
        # metering-precision blip.
        try:
            client.decrby(key, ceiling)
        except Exception as exc:
            print(
                f"LOG: daily budget refund-on-refusal FAILED for user (Redis error: {exc}) — "
                f"a {ceiling}-token reservation is stranded on {key} until the UTC day rolls"
            )
        reset_at = _next_utc_midnight_str()
        return (
            False,
            f"You've reached today's usage limit for now. It resets at {reset_at}.",
            0,
            key,
        )

    return True, None, ceiling, key


def _finalize_daily_spend(key: str, reserved: int, actual_tokens: int) -> None:
    """Reconcile a reservation taken by ``_reserve_daily_spend``: adjust the
    spend key by ``(actual_tokens - reserved)`` so it ends up holding the
    turn's REAL cost rather than its worst-case ceiling. A no-op when
    ``reserved`` is 0 (nothing was reserved for this turn) or the delta is 0.

    MUST be called from a ``finally``, not a plain post-return statement, so
    it runs on every exit path — including a raised exception or a cancelled
    turn (``asyncio.CancelledError`` inherits from ``BaseException``, not
    ``Exception``; a bare ``except Exception: ... finalize()`` would miss
    it, silently stranding the reservation). ``actual_tokens`` should be 0
    for a turn that never reached the LLM call, which correctly refunds the
    reservation in full.

    Best-effort: a Redis failure here is logged, never raised — the turn
    itself already completed (or errored) and has nothing further to give
    the caller; see the module comment's REDIS-UNAVAILABLE POSTURE.
    """
    if reserved <= 0 or not key:
        return
    diff = int(actual_tokens) - reserved
    if diff == 0:
        return
    try:
        from core.storage.cloud import get_redis_sync_client

        client = get_redis_sync_client()
        pipe = client.pipeline()
        pipe.incrby(key, diff)  # negative diff decrements — INCRBY accepts negative deltas
        pipe.expire(key, 172800)
        pipe.execute()
    except Exception as exc:
        print(
            f"LOG: daily budget finalize failed for key={key} diff={diff} ({exc}) — "
            f"this turn's spend is left at its reserved ceiling estimate rather than "
            f"its actual cost; self-corrects on the next successful finalize for this "
            f"key, or when the UTC day rolls"
        )


class TurnOutcome(NamedTuple):
    """Return value of the canonical turn pipeline.

    ``new_history`` is the conversation of record to carry into the next turn.

    ``output_text`` is the cleaned *model* reply — what the voice path speaks.
    It is ``None`` when the turn produced no model reply: a confirmation
    intercept (which already sent its own ``done`` frame) or a pipeline error
    (which already sent an ``error`` frame). Voice skips TTS when it is ``None``.

    ``reply_text`` is the terminal user-facing text for this turn regardless of
    how it terminated — the model reply, the confirmation acknowledgement, or the
    friendly error string. WebSocket callers already received it as a frame, but
    a channel caller (``ws=None``) relays this to the adapter, so it never gets
    a silent empty reply.
    """
    new_history: list[ModelMessage] | None
    output_text: str | None
    reply_text: str = ""


async def _emit(ws: WebSocket | None, data: dict[str, Any]) -> None:
    """Send a WS frame, no-op when there is no websocket (channel callers).

    Every user-facing frame in the pipeline goes through here so a channel
    adapter can drive the exact same turn with ``ws=None``.
    """
    if ws is None:
        return
    await _ws_send_json(ws, data)


async def _execute_turn(
    ws: WebSocket | None,
    state: SharedState,
    user_text: str,
    message_history: list[ModelMessage] | None,
    *,
    channel: str,
    send_status: bool = True,
) -> TurnOutcome:
    """The one canonical turn pipeline shared by every entrypoint.

    web (text), web_voice (audio), and every channel adapter (WhatsApp, iMessage,
    Slack, Twilio Voice) all funnel through here so a single code path owns
    analytics, the confirmation sidecar, the heuristic task-type label, memory
    context, the per-turn trace span, the single agent call + fallbacks, output
    cleaning, persistence, explicit-fact application, silent candidate queuing,
    the reflector, timing, and classified error handling.

    All websocket sends go through ``_emit`` so a channel caller may pass
    ``ws=None``; ``send_status`` suppresses the "thinking" status frame for
    callers that manage their own status lifecycle (or have no UI to update).
    """
    timings: dict[str, float] = {}
    overall_start = time.time()
    # WP1.H: reset this turn's tool-sourced URL bucket. Populated by the
    # envelope wrapper (_wrap_tool_with_envelope) as tools run below, read
    # when the "done" frame is sent.
    state.tool_sourced_urls = []
    # Hoisted to the top (rather than created just before the agent call, as
    # it used to be) so it exists on EVERY exit path — including one that
    # raises before ever reaching the agent call — for
    # _finalize_daily_spend's `finally` below to read. Starts at all-zero
    # tokens, which is exactly right for a turn that never reached the LLM.
    cascade_stats = CascadeStats()

    if state.user_id:
        emit_event_once(state.user_id, "first_message_sent", channel=channel)

    # WP1.D2 (ledger 1a.4 part 4): reserve BEFORE any LLM call, atomically in
    # Redis — see the module comment above for why this replaced a
    # check-then-act read (concurrent turns for the same user could all pass
    # a pre-check before any of them recorded, multiplying the budget by
    # however many ran at once). A refusal here costs nothing further (no
    # "thinking" frame, no memory-context resolution, no agent call); an
    # allowed reservation MUST be reconciled via _finalize_daily_spend in the
    # `finally` below on every exit path, which is why send_status/the try
    # block start only after this succeeds.
    _budget_allowed, _budget_refusal, _budget_reserved, _budget_key = _reserve_daily_spend(state.user_id)
    if not _budget_allowed:
        await _emit(ws, {"type": "done", "content": _budget_refusal})
        timings["total_ms"] = round((time.time() - overall_start) * 1000)
        await _emit(ws, {"type": "timing", **timings})
        return TurnOutcome(message_history, _budget_refusal, _budget_refusal)

    if send_status:
        await _emit(ws, {"type": "status", "status": "thinking"})

    # final_output tracks whether the model already answered; the except block
    # uses it to keep a computed answer alive when only post-processing failed.
    final_output: str | None = None
    try:
        # The confirmation sidecar is websocket-only: a channel caller (ws=None)
        # has no way to render the prompt. If a memory-confirmation prompt is
        # pending, surface it as a sidecar frame before the agent reply so the
        # user can answer it via the web UI's confirm panel (/api/memory/confirm
        # — the ONLY confirmation surface). Chat turns are never intercepted or
        # parsed for confirmation: a bare "yes" is just a word the model answers.
        if ws is not None:
            pending_prompt = state.confirmation_gate.next_prompt()
            if pending_prompt is not None:
                await _emit(ws, {
                    "type": "confirmation_prompt",
                    "event_ids": list(pending_prompt.all_event_ids),
                    "topic": pending_prompt.topic,
                    "key": pending_prompt.key,
                    "message": pending_prompt.question,
                })

        # Heuristic task type steers memory retrieval + trace labels only. With
        # per-intent tool scoping gone, every tool is offered on every turn.
        task_type = _detect_task_type(user_text)
        # Pending-email bypass: a half-finished draft means an AMBIGUOUS turn is
        # almost certainly continuing it, even when the words don't say "email"
        # (e.g. "the subject is lunch"). Only override the heuristic's "general"
        # verdict — a turn that clearly asks for something else ("search the
        # web for X") keeps its own label so a stale draft can't relabel
        # unrelated work (Codex P4 review A#4/B#6). Drafts also TTL out.
        if task_type == "general":
            _pending_email = state.session_store.get_pending_email() or {}
            if _pending_email.get("recipients") or _pending_email.get("subject") or _pending_email.get("content"):
                task_type = "email"

        state.memory_context = await _resolve_memory_context(state, task_type=task_type, user_text=user_text)
        # Memory travels via per-turn instructions (_build_turn_instructions);
        # the persisted user turn stays the user's bare words.
        prompt_input = user_text
        turn_id = _new_turn_id(state)

        llm_start = time.time()
        # Phase 1: one local span per turn — the record that makes "why did
        # Turtle answer X" answerable from disk (data/traces/traces.jsonl).
        # (cascade_stats itself is created at the top of _execute_turn now —
        # see the WP1.D2 comment there — so it's available to
        # _finalize_daily_spend on every exit path, not just this one.)
        turn_deadline_s = _turn_deadline_for(channel)
        with trace_sink.span(
            "turtle.turn",
            user_id=state.user_id,
            session_id=state.session_store.session_id or "",
            turn_id=turn_id,
            intent=task_type,
            memory_context_chars=len(state.memory_context or ""),
            channel=channel,
        ) as _turn_span:
            # Preserve the rich Logfire span the deleted graph layer used to
            # emit, so a turn's model spans still nest under one logical unit.
            if _logfire_loaded:
                import logfire as _lf_turn
                _turn_span_cm = _lf_turn.span("turtle.turn", intent=task_type, channel=channel)
            else:
                from contextlib import nullcontext
                _turn_span_cm = nullcontext()
            try:
                with _turn_span_cm:
                    # ONE agent call with fallbacks. The cascade now owns its own
                    # total + per-rung budget (see core/llm_client._Budget), so
                    # the deadline here is modality-aware rather than the flat 60s
                    # the deleted graph layer left behind. The outer wait_for is
                    # belt-and-braces for a hang in the cascade machinery itself,
                    # hence the small margin over the cascade's own budget.
                    response = await asyncio.wait_for(
                        run_agent_with_fallbacks(
                            agents_mgr.main_assistant,
                            agents_mgr.main_assistant_fallbacks,
                            prompt_input,
                            deps=state,
                            message_history=message_history,
                            usage=RunUsage(),
                            usage_limits=agents_mgr.usage_limits,
                            deadline_s=turn_deadline_s,
                            stats=cascade_stats,
                        ),
                        timeout=turn_deadline_s + 5.0,
                    )
            finally:
                # Record spend even when the turn FAILED — a turn that burned
                # eight rungs and returned nothing is the most expensive kind
                # there is, and it was previously invisible. Attribute-setting
                # must never mask the real exception, so it is best-effort.
                try:
                    for _k, _v in cascade_stats.as_span_attrs().items():
                        _turn_span.set_attribute(_k, _v)
                    _turn_span.set_attribute(ATTR_TOKENS_IN, cascade_stats.total_input_tokens)
                    _turn_span.set_attribute(ATTR_TOKENS_OUT, cascade_stats.total_output_tokens)
                except Exception:
                    pass
                # WP1.D2 (ledger 1a.4 part 4): budget reconciliation used to
                # happen right here, but that only runs when execution
                # reaches this inner `with` block at all — an exception
                # raised earlier (memory-context resolution, task-type
                # detection, etc.) would skip it and strand the reservation
                # taken above. It now happens exactly once, in the OUTER
                # `finally` at the bottom of this function, which runs on
                # every exit path including one that never gets here.
        timings["llm_ms"] = round((time.time() - llm_start) * 1000)

        final_output = clean_text_for_model(response.output)

        # Send complete response. The chat gets the display-cleaned text (markdown
        # links preserved → clickable), while final_output (links flattened) feeds
        # TTS/RAG.
        await _emit(ws, {
            "type": "done",
            "content": clean_text_for_display(response.output),
            "tool_urls": list(state.tool_sourced_urls),
        })

        # Update session
        message_history = _persist_history(message_history, response)
        await state.session_store.replace_messages(message_history)
        state.rag_system.add_conversation(user_text, final_output)
        # NOTE: the legacy single-tenant memory_store.record_turn block was
        # dropped here. Personal memory now lives under
        # personal_memory_dir(user_id) and is journaled per-turn via the
        # post-step extraction pipeline, so the old block was dead on all paths.
        # Immediately apply explicit facts (email, name) so next turn sees them
        _apply_explicit_facts_from_turn(
            state,
            session_id=state.session_store.session_id or "unknown_session",
            turn_id=turn_id,
            user_text=user_text,
            task_type=task_type,
        )
        # D3: candidates queued silently — no in-turn confirmation interrupt.
        # The web UI /api/memory/pending endpoint will expose them for batch review.
        _queue_confirmation_candidates_from_turn(
            state,
            session_id=state.session_store.session_id or "unknown_session",
            user_text=user_text,
        )
        # Storage-cap breaches inside the sync extraction funnels can't reach
        # the websocket themselves — they stash a pending notice frame. Deliver
        # it now so the user learns their memory writes are failing. Only pop
        # when a websocket exists to carry it: popping with ws=None would
        # discard the notice unseen (Codex review R2#7); a channel user's
        # pending notice survives until their next web session drains it.
        if ws is not None:
            # Same key derivation as _notify_storage_cap — a hand-rolled key
            # here could diverge and strand the pending frame (Codex R1#3).
            cap_notice = pop_pending_storage_cap_notice(_storage_cap_key(state))
            if cap_notice:
                await _emit(ws, cap_notice)
        if state.reflector is not None:
            await state.reflector.on_turn(
                state,
                session_id=state.session_store.session_id or "",
                message_history=message_history or [],
            )

        timings["total_ms"] = round((time.time() - overall_start) * 1000)
        await _emit(ws, {"type": "timing", **timings})

        return TurnOutcome(message_history, final_output, final_output)

    except Exception as e:
        print(f"LOG: Turn pipeline error ({channel}): {e}")
        traceback.print_exc()
        # The model already answered and the failure happened in post-turn
        # bookkeeping (persistence, RAG, extraction, reflector). The websocket
        # user already has their done frame; a channel caller only ever sees
        # reply_text — so return the real answer, not an error that would
        # replace it (Codex review R2#6). The failure itself is logged above.
        if final_output is not None:
            return TurnOutcome(message_history, final_output, final_output)
        code, friendly = _classify_handler_error(e)
        if _logfire_loaded:
            try:
                import logfire as _lf
                _lf.error(
                    "turtle.turn_failed",
                    error_class=e.__class__.__name__,
                    error_code=code,
                    error_message=str(e),
                    channel=channel,
                )
            except Exception:
                pass
        await _emit(ws, {"type": "error", "code": code, "message": friendly})
        # No model reply, but a channel caller still needs the friendly message
        # relayed rather than a silent empty string.
        return TurnOutcome(message_history, None, friendly)

    finally:
        # WP1.D2 (ledger 1a.4 part 4): reconcile the reservation taken above,
        # on EVERY exit path out of the try/except above — normal return,
        # the `except Exception` branch, AND a BaseException that isn't an
        # Exception at all (asyncio.CancelledError, raised when a client
        # disconnects mid-turn on Vercel/uvicorn, inherits from
        # BaseException — a bare `except Exception` would never see it, but
        # `finally` always runs). A stranded reservation costs the user
        # budget until the UTC day rolls, which is a worse user-visible bug
        # than the overshoot this reservation scheme exists to close, so
        # this must not be skippable. No-ops when nothing was reserved (see
        # _reserve_daily_spend's return contract).
        _finalize_daily_spend(
            _budget_key,
            _budget_reserved,
            cascade_stats.total_input_tokens + cascade_stats.total_output_tokens,
        )


async def _handle_text_message(
    ws: WebSocket,
    state: SharedState,
    user_text: str,
    message_history: list[ModelMessage] | None,
) -> list[ModelMessage] | None:
    """Web text entrypoint — a thin wrapper over the canonical turn pipeline."""
    outcome = await _execute_turn(ws, state, user_text, message_history, channel="web")
    return outcome.new_history


def _voice_stream_llm_enabled() -> bool:
    """Whether the voice turn streams LLM tokens into TTS (opt-in).

    Off by default: the streamed path overlaps generation with synthesis to start
    speaking sooner, but it needs live validation against the full tool/memory
    turn before it can be the default. Enable via VOICE_STREAM_LLM=1 (env) or the
    same key in turtle_config.json.
    """
    raw = os.getenv("VOICE_STREAM_LLM")
    if raw is None:
        raw = str(config.get("VOICE_STREAM_LLM", "0"))
    return raw.strip().lower() in ("1", "true", "yes", "on")


class _StreamPreAudioError(Exception):
    """Streaming failed before any audio was sent — safe to retry via batch path.

    Once a sentence has been spoken we can't un-say it, so only failures raised
    before the first audio frame carry this type; the caller then falls back to
    the canonical batch turn for this same turn.
    """


async def _execute_turn_streaming(
    ws: WebSocket,
    state: SharedState,
    user_text: str,
    message_history: list[ModelMessage] | None,
    *,
    channel: str,
    timings: dict[str, float],
    overall_start: float,
) -> TurnOutcome:
    """Voice turn that streams LLM tokens into sentence-chunked TTS.

    Mirrors the pre-run (memory context, confirmation sidecar, turn id) and
    post-run (persistence, explicit facts, candidate queuing, reflector) stages
    of ``_execute_turn`` — reusing the same helpers so behaviour can't silently
    diverge — but replaces the single blocking agent call + separate TTS with a
    streamed run whose text is synthesised and sent as each sentence completes.

    Raises ``_StreamPreAudioError`` if it fails before the first audio frame, so
    the caller can transparently fall back to the batch path for this turn.
    """
    # --- pre-run: identical inputs to the batch path -----------------------
    # WP1.H: reset this turn's tool-sourced URL bucket (mirrors _execute_turn).
    state.tool_sourced_urls = []
    if state.user_id:
        emit_event_once(state.user_id, "first_message_sent", channel=channel)

    pending_prompt = state.confirmation_gate.next_prompt()
    if pending_prompt is not None:
        await _ws_send_json(ws, {
            "type": "confirmation_prompt",
            "event_ids": list(pending_prompt.all_event_ids),
            "topic": pending_prompt.topic,
            "key": pending_prompt.key,
            "message": pending_prompt.question,
        })

    task_type = _detect_task_type(user_text)
    if task_type == "general":
        _pending_email = state.session_store.get_pending_email() or {}
        if _pending_email.get("recipients") or _pending_email.get("subject") or _pending_email.get("content"):
            task_type = "email"

    state.memory_context = await _resolve_memory_context(state, task_type=task_type, user_text=user_text)
    turn_id = _new_turn_id(state)

    # --- streamed run + TTS ------------------------------------------------
    collector = StreamCollector()
    first_audio_sent = False
    chunks_sent = 0
    llm_start = time.time()

    async def _token_source():
        # Yields raw model text deltas; StreamCollector captures the finished run.
        # WP1.D2 (ledger 1a.4 part 4): this path BYPASSES the daily token
        # budget ENTIRELY, not merely "leaves it uncounted" — there is no
        # _reserve_daily_spend/_finalize_daily_spend call anywhere on this
        # path (stream_agent_text_with_fallbacks takes no `stats=`, so there
        # is no CascadeStats to reconcile against even if there were). A
        # user can exhaust the WHOLE intent of ledger 1a.4's spend ceiling by
        # speaking instead of typing, with no refusal ever, on this path.
        # This is the ledger's accepted, explicit gap ("streaming joins in
        # Phase 4") — not an oversight, but whoever picks up Phase 4 should
        # understand the size of it, not just its existence. Not fixed here
        # (usage_limits.total_tokens_limit still bounds one streamed turn's
        # OWN worst case via pydantic-ai, it just never touches
        # turtle:spend:{uid}:{yyyymmdd} at all).
        async for delta in stream_agent_text_with_fallbacks(
            agents_mgr.main_assistant,
            agents_mgr.main_assistant_fallbacks,
            user_text,
            deps=state,
            message_history=message_history,
            usage_limits=agents_mgr.usage_limits,
            collector=collector,
        ):
            yield delta

    from core.latency_budgets import budgets, check_sla
    from core.streaming_tts import stream_tts_from_token_stream

    await _ws_send_json(ws, {"type": "status", "status": "speaking"})
    tts_start = time.time()
    try:
        # One per-turn trace span, same record the batch path writes to disk, so
        # "why did Turtle answer X" stays answerable for streamed turns too.
        with trace_sink.span(
            "turtle.turn",
            user_id=state.user_id,
            session_id=state.session_store.session_id or "",
            turn_id=turn_id,
            intent=task_type,
            memory_context_chars=len(state.memory_context or ""),
            channel=channel,
            streamed=True,
        ):
            async for _sentence, audio_bytes in stream_tts_from_token_stream(
                _token_source(),
                speed=float(config.get("TURTLE_TTS_SPEED", 1.2)),
                tts_timeout_s=budgets.TOOL_S,
                clean_fn=clean_text_for_tts,
            ):
                if not first_audio_sent:
                    timings["tts_first_byte_ms"] = round((time.time() - tts_start) * 1000)
                    check_sla("tts_first_byte", tts_start, budgets.TTS_FIRST_BYTE_MAX_MS)
                    first_audio_sent = True
                await ws.send_bytes(audio_bytes)
                chunks_sent += 1
    except Exception as exc:
        # Nothing spoken yet → the batch path can still serve this turn cleanly.
        if not first_audio_sent:
            print(f"LOG: voice stream failed pre-audio ({channel}): {exc}")
            raise _StreamPreAudioError(str(exc)) from exc
        # Audio already went out; log and continue to persist what we have.
        print(f"LOG: voice stream error after first audio ({channel}): {exc}")
        traceback.print_exc()

    timings["llm_ms"] = round((time.time() - llm_start) * 1000)

    final_output = clean_text_for_model(collector.output or "")
    if not final_output:
        # No usable text produced and nothing spoken → fall back to batch.
        if not first_audio_sent:
            raise _StreamPreAudioError("stream produced no output")
        return TurnOutcome(message_history, None, "")

    # Send the full reply text for the transcript UI once synthesis is underway.
    # Display-cleaned (markdown links preserved) so the chat renders clickable links.
    await _ws_send_json(ws, {
        "type": "done",
        "content": clean_text_for_display(collector.output or ""),
        "tool_urls": list(state.tool_sourced_urls),
    })

    # --- post-run: identical bookkeeping to the batch path -----------------
    message_history = _persist_history(message_history, collector)
    await state.session_store.replace_messages(message_history)
    state.rag_system.add_conversation(user_text, final_output)
    _apply_explicit_facts_from_turn(
        state,
        session_id=state.session_store.session_id or "unknown_session",
        turn_id=turn_id,
        user_text=user_text,
        task_type=task_type,
    )
    _queue_confirmation_candidates_from_turn(
        state,
        session_id=state.session_store.session_id or "unknown_session",
        user_text=user_text,
    )
    cap_notice = pop_pending_storage_cap_notice(_storage_cap_key(state))
    if cap_notice:
        await _ws_send_json(ws, cap_notice)
    if state.reflector is not None:
        await state.reflector.on_turn(
            state,
            session_id=state.session_store.session_id or "",
            message_history=message_history or [],
        )

    timings["tts_ms"] = round((time.time() - tts_start) * 1000)
    timings["total_ms"] = round((time.time() - overall_start) * 1000)
    await _ws_send_json(ws, {"type": "timing", **timings})
    print(f"LOG: voice stream done — {chunks_sent} chunks, "
          f"first_audio={timings.get('tts_first_byte_ms')}ms, total={timings['total_ms']}ms")

    return TurnOutcome(message_history, final_output, final_output)


async def _handle_audio_message(
    ws: WebSocket,
    state: SharedState,
    audio_bytes: bytes,
    message_history: list[ModelMessage] | None,
    *,
    sample_rate: int = 16000,
) -> list[ModelMessage] | None:
    """Voice entrypoint: STT + transcription echo → canonical turn pipeline
    (channel="web_voice") → streaming TTS of the returned reply.

    The turn itself is delegated to _execute_turn; this handler owns only the
    audio-specific bookends (speech in, speech out) and their timings.
    """
    timings: dict[str, float] = {}
    overall_start = time.time()

    await _ws_send_json(ws, {"type": "status", "status": "transcribing"})

    try:
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(audio_array) < 1000:
            await _ws_send_json(ws, {"type": "error", "message": "Audio too short"})
            return message_history

        # STT
        stt_start = time.time()
        print(f"LOG: STT transcribing {len(audio_array)} samples @ {sample_rate}Hz")
        try:
            # STT is a blocking network+CPU call; running it inline would freeze
            # the whole event loop (every other session's pings/turns stall) for
            # its full duration. Offload to a thread so the loop stays responsive.
            loop = asyncio.get_event_loop()
            transcription = await loop.run_in_executor(
                None,
                agents_mgr.stt.transcribe_from_audio,
                (sample_rate, audio_array),
            )
            timings["stt_ms"] = round((time.time() - stt_start) * 1000)
            print(f"LOG: STT completed in {timings['stt_ms']}ms -> {repr(transcription[:80]) if transcription else 'empty'}")
        except Exception as stt_exc:
            timings["stt_ms"] = round((time.time() - stt_start) * 1000)
            print(f"LOG: STT failed after {timings['stt_ms']}ms: {type(stt_exc).__name__}: {stt_exc}")
            traceback.print_exc()
            await _ws_send_json(ws, {"type": "error", "message": f"STT error: {stt_exc}"})
            return message_history

        if not transcription or not transcription.strip():
            print("LOG: STT returned empty transcription")
            await _ws_send_json(ws, {"type": "error", "message": "No speech detected"})
            return message_history

        transcription = transcription.strip()

        # Send transcription to client
        await _ws_send_json(ws, {"type": "transcription", "text": transcription})

        return await _reply_and_speak(
            ws, state, transcription, message_history,
            timings=timings, overall_start=overall_start,
        )

    except Exception as e:
        print(f"LOG: Audio handler error: {e}")
        traceback.print_exc()
        await _ws_send_json(ws, {"type": "error", "message": str(e)})
        return message_history


async def _reply_and_speak(
    ws: WebSocket,
    state: SharedState,
    transcript: str,
    message_history: list[ModelMessage] | None,
    *,
    timings: dict[str, float],
    overall_start: float,
) -> list[ModelMessage] | None:
    """Run one voice turn for ``transcript`` and speak the reply.

    Shared by the batch audio handler (one utterance per WS binary frame) and the
    streaming-STT consumer (one call per Flux EndOfTurn), so both routes get the
    identical turn pipeline + TTS behaviour.
    """
    # Opt-in fast path: stream LLM tokens straight into sentence-chunked TTS so
    # speech starts at the first sentence boundary. Falls back to the canonical
    # batch turn below if streaming fails before any audio is spoken.
    if _voice_stream_llm_enabled():
        try:
            outcome = await _execute_turn_streaming(
                ws, state, transcript, message_history,
                channel="web_voice", timings=timings, overall_start=overall_start,
            )
            return outcome.new_history
        except _StreamPreAudioError as stream_exc:
            print(f"LOG: falling back to batch turn: {stream_exc}")

    # The full turn — routing, memory, persistence, extraction, trace span,
    # confirmation sidecar, classified errors — is owned by the one canonical
    # pipeline. Voice gains everything the text path has.
    outcome = await _execute_turn(
        ws, state, transcript, message_history, channel="web_voice",
    )
    message_history = outcome.new_history
    final_output = outcome.output_text

    # A pipeline error returns no speakable text (and has already emitted its own
    # error frame): stop here.
    if not final_output:
        return message_history

    # Streaming TTS with sentence-boundary chunking: each sentence is synthesised
    # as its boundary is detected and sent immediately — no waiting for the whole
    # audio file.
    await _ws_send_json(ws, {"type": "status", "status": "speaking"})
    tts_start = time.time()
    tts_text = clean_text_for_tts(final_output)
    tts_debug = settings.tts_debug

    from core.latency_budgets import budgets, check_sla
    from core.streaming_tts import stream_tts_from_text

    first_chunk_sent = False
    chunks_sent = 0

    try:
        async for sentence_text, audio_bytes in stream_tts_from_text(
            tts_text,
            speed=float(config.get("TURTLE_TTS_SPEED", 1.2)),
            tts_timeout_s=budgets.TOOL_S,
        ):
            if not first_chunk_sent:
                first_byte_ms = round((time.time() - tts_start) * 1000)
                timings["tts_first_byte_ms"] = first_byte_ms
                check_sla("tts_first_byte", tts_start, budgets.TTS_FIRST_BYTE_MAX_MS)
                print(f"LOG: TTS first chunk in {first_byte_ms}ms ({len(audio_bytes)} bytes)")
                first_chunk_sent = True
            await ws.send_bytes(audio_bytes)
            chunks_sent += 1

        timings["tts_ms"] = round((time.time() - tts_start) * 1000)
        if chunks_sent:
            print(f"LOG: TTS streaming done in {timings['tts_ms']}ms ({chunks_sent} chunks)")
        else:
            print("LOG: TTS produced no audio chunks (empty text?)")

    except Exception as e:
        timings["tts_ms"] = round((time.time() - tts_start) * 1000)
        print(f"LOG: TTS streaming error after {timings['tts_ms']}ms: {type(e).__name__}: {e}")
        if tts_debug:
            traceback.print_exc()
        await _ws_send_json(ws, {"type": "error", "message": f"TTS error: {e}"})

    timings["total_ms"] = round((time.time() - overall_start) * 1000)
    await _ws_send_json(ws, {"type": "timing", **timings})

    return message_history


# ---------------------------------------------------------------------------
# Streaming STT (Deepgram Flux) — continuous-frame voice input
# ---------------------------------------------------------------------------

def _voice_stream_stt_enabled() -> bool:
    """Whether the server accepts continuous mic frames into Deepgram Flux (opt-in).

    Off by default: streaming STT overlaps transcription with the user's speech
    and replaces the fixed client silence timer with model end-of-turn, but the
    continuous-frame protocol needs live validation. Enable via VOICE_STREAM_STT=1
    (env) or the same key in turtle_config.json.
    """
    raw = os.getenv("VOICE_STREAM_STT")
    if raw is None:
        raw = str(config.get("VOICE_STREAM_STT", "0"))
    return raw.strip().lower() in ("1", "true", "yes", "on")


class _MicStreamSession:
    """A live Flux streaming-STT session bound to one websocket connection.

    ``history`` is a one-key holder so the consumer task and the receive loop
    share the same conversation of record: the consumer writes each turn's new
    history into it, and the receive loop reads it back on mic_close.
    """

    __slots__ = ("stt", "consumer_task", "history", "sample_rate",
                 "finishing", "turn_started", "current_turn")

    def __init__(
        self,
        stt: Any,
        history: dict,
        *,
        sample_rate: int,
        finishing: "asyncio.Event",
        turn_started: "asyncio.Event",
    ) -> None:
        self.stt = stt
        self.history = history
        self.sample_rate = sample_rate
        # Set on mic_close; tells the consumer to stop after the final turn.
        self.finishing = finishing
        # Set by the consumer when it processes an EndOfTurn (a real turn began).
        self.turn_started = turn_started
        # The consumer task; assigned right after construction in _open_mic_stream.
        self.consumer_task: "asyncio.Task | None" = None
        # The in-flight reply task (LLM + TTS) for the current turn, or None.
        # The receive loop cancels this to interrupt the agent mid-reply.
        self.current_turn: "asyncio.Task | None" = None

    def interrupt(self) -> bool:
        """Cancel the in-flight reply, if any. Returns True if something was cancelled."""
        turn = self.current_turn
        if turn is not None and not turn.done():
            turn.cancel()
            return True
        return False


async def _run_streamed_turn(
    ws: WebSocket,
    state: SharedState,
    text: str,
    session: "_MicStreamSession",
) -> None:
    """Run one reply (LLM + TTS) for a streamed utterance, into the shared history.

    Executed as a cancellable task so the receive loop can interrupt it (or a
    barge-in can) mid-reply. On cancellation, playback stops on the client, we
    tell the client the reply was interrupted, and the holder keeps whatever the
    turn managed to persist.
    """
    timings: dict[str, float] = {}
    overall_start = time.time()
    try:
        session.history["messages"] = await _reply_and_speak(
            ws, state, text, session.history["messages"],
            timings=timings, overall_start=overall_start,
        )
    except asyncio.CancelledError:
        print("LOG: streamed reply interrupted")
        try:
            await _ws_send_json(ws, {"type": "interrupted"})
            await _ws_send_json(ws, {"type": "status", "status": "ready"})
        except Exception:
            pass
        raise
    except Exception as turn_exc:
        print(f"LOG: mic-stream turn error: {turn_exc}")
        traceback.print_exc()
        await _ws_send_json(ws, {"type": "error", "message": f"Turn error: {turn_exc}"})


async def _flux_mic_consumer(
    ws: WebSocket,
    state: SharedState,
    session: "_MicStreamSession",
) -> None:
    """Consume Flux turn events: stream interim captions, run a turn on EndOfTurn.

    Runs as its own task alongside the receive loop. It is the SOLE writer of
    ``session.history["messages"]`` while streaming is active, and it processes
    events one at a time, so turns never overlap and there is no history race.

    Each reply runs as ``session.current_turn`` — a cancellable task — so the
    receive loop (on an ``interrupt`` frame) or a barge-in (Flux ``StartOfTurn``
    while a reply is playing) can stop it mid-reply. ``finishing`` (set on
    mic_close) makes the consumer exit after the final utterance's turn.
    """
    stt = session.stt
    partials = 0
    turns = 0
    started = time.time()
    try:
        async for ev in stt.events():
            if ev.kind == "connected":
                print(f"LOG: Flux STT connected in {round((time.time()-started)*1000)}ms")
            elif ev.kind == "start_of_turn":
                # Barge-in: the user started speaking. If a reply is playing,
                # cancel it so the agent stops and listens.
                if session.interrupt():
                    print("LOG: barge-in — user spoke over the reply")
            elif ev.kind == "update":
                if ev.transcript:
                    partials += 1
                    if partials == 1:
                        print(f"LOG: Flux first partial in {round((time.time()-started)*1000)}ms")
                    await _ws_send_json(ws, {
                        "type": "transcription_partial", "text": ev.transcript,
                    })
            elif ev.kind == "end_of_turn":
                text = (ev.transcript or "").strip()
                print(f"LOG: Flux EndOfTurn ({partials} partials) -> "
                      f"{repr(text[:80]) if text else 'EMPTY (skipped)'}")
                if not text:
                    if session.finishing.is_set():
                        break
                    continue
                turns += 1
                session.turn_started.set()
                await _ws_send_json(ws, {"type": "transcription", "text": text})
                # Run the reply as a cancellable task and wait via asyncio.wait so
                # an interrupt (which cancels the turn task) does NOT look like a
                # cancellation of this consumer — the two must stay distinct.
                turn = asyncio.create_task(_run_streamed_turn(ws, state, text, session))
                session.current_turn = turn
                try:
                    await asyncio.wait({turn})
                finally:
                    if not turn.done():
                        turn.cancel()
                    session.current_turn = None
                if session.finishing.is_set():
                    break
                session.turn_started.clear()
            elif ev.kind == "error":
                print(f"LOG: Flux STT error event: {ev.raw!r}")
                await _ws_send_json(ws, {
                    "type": "error", "message": "Streaming STT error.",
                })
                break
            elif ev.kind == "closed":
                print(f"LOG: Flux stream closed ({partials} partials, {turns} turns this session)")
                break
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"LOG: mic consumer crashed: {exc}")
        traceback.print_exc()


def _build_stt_keyterms(user_id: str) -> list[str]:
    """Harvest the user's name, emails, and frequent contacts as Flux keyterms.

    Flux biases recognition toward keyterms, so feeding it the proper nouns it
    can't guess (a name, an email's local-part) stops the mangling we saw —
    "Shriyash Beohar" heard as "Shriish sharai", emails spelled out letter by
    letter. Names + email local-parts are the useful spoken forms.
    """
    if not user_id:
        return []
    import re
    try:
        profile = PersonalMemoryStore(user_id=user_id).load_profile_snapshot()
    except Exception:
        return []
    if not isinstance(profile, dict):
        return []

    terms: list[str] = []
    identity = profile.get("identity", {}) or {}
    name = identity.get("name")
    if name:
        terms.append(name)
        terms.extend(tok for tok in str(name).split() if len(tok) >= 2)
    for email in identity.get("emails", []) or []:
        if not email:
            continue
        terms.append(email)
        local = str(email).split("@", 1)[0]
        if local:
            terms.append(local)
            terms.extend(p for p in re.split(r"[._+\-]", local) if len(p) >= 2)
    workflow = profile.get("workflow", {}) or {}
    for rcpt in (workflow.get("common_recipients") or [])[:10]:
        if rcpt and "@" in str(rcpt):
            local = str(rcpt).split("@", 1)[0]
            if local:
                terms.append(local)

    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        cleaned = str(term).strip()
        key = cleaned.lower()
        if len(cleaned) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= 30:  # keep the biasing list bounded
            break
    return out


async def _open_mic_stream(
    ws: WebSocket,
    state: SharedState,
    message_history: list[ModelMessage] | None,
    *,
    sample_rate: int = 16000,
) -> "_MicStreamSession":
    """Open a Flux session + consumer task for a client that started streaming."""
    from core.stt_streaming import FluxStreamingSTT

    keyterms = _build_stt_keyterms(state.user_id)
    stt = FluxStreamingSTT(sample_rate=sample_rate, keyterms=keyterms)
    await stt.start()
    history = {"messages": message_history}
    session = _MicStreamSession(
        stt, history,
        sample_rate=sample_rate,
        finishing=asyncio.Event(),
        turn_started=asyncio.Event(),
    )
    session.consumer_task = asyncio.create_task(_flux_mic_consumer(ws, state, session))
    await _ws_send_json(ws, {"type": "status", "status": "listening"})
    print(f"LOG: streaming STT session opened (keyterms={len(keyterms)})")
    return session


async def _begin_mic_close(session: "_MicStreamSession") -> None:
    """Start finalising a streaming session WITHOUT blocking the receive loop.

    Feeds a tail of silence so Flux fires EndOfTurn for the final utterance
    (PTT/VAD stop provides no silence gap of its own), then hands off to a
    background task to run that last turn and tear down. Keeping the receive loop
    free is what lets an ``interrupt`` frame reach the server mid-reply.
    """
    session.finishing.set()
    silence = b"\x00\x00" * int(0.7 * session.sample_rate)
    try:
        await session.stt.send_audio(silence)
        await session.stt.finish()
    except Exception:
        pass
    asyncio.create_task(_finalize_mic_session(session))


async def _finalize_mic_session(session: "_MicStreamSession") -> None:
    """Background teardown: let the final turn drain, then shut the Flux socket."""
    try:
        # The consumer exits after the final turn (finishing is set). Cap it so a
        # stuck turn can't leak the session.
        try:
            await asyncio.wait_for(asyncio.shield(session.consumer_task), timeout=60)
        except asyncio.TimeoutError:
            pass
    finally:
        await session.stt.aclose()
        if session.consumer_task is not None and not session.consumer_task.done():
            session.consumer_task.cancel()
            try:
                await session.consumer_task
            except (asyncio.CancelledError, Exception):
                pass
        print("LOG: streaming STT session closed")


async def _reclaim_mic_session(session: "_MicStreamSession") -> list[ModelMessage] | None:
    """Force a session fully down and return its up-to-date conversation.

    Used when the receive loop needs the current history back (new mic_open, a
    text turn, or disconnect) — it guarantees teardown even if the background
    finaliser hasn't finished.
    """
    session.interrupt()
    session.finishing.set()
    await session.stt.aclose()
    if session.consumer_task is not None and not session.consumer_task.done():
        try:
            await asyncio.wait_for(session.consumer_task, timeout=10)
        except (asyncio.TimeoutError, Exception):
            session.consumer_task.cancel()
            try:
                await session.consumer_task
            except (asyncio.CancelledError, Exception):
                pass
    return session.history["messages"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    host = settings.host or str(config.get("SERVER_HOST", SERVER_HOST))
    port = settings.port or int(config.get("SERVER_PORT", SERVER_PORT))
    reload_enabled = settings.server_reload
    print(f"[Turtle AI] Web Server starting at http://{host}:{port}")
    if reload_enabled:
        uvicorn.run(
            "apps.turtle_server:app",
            host=host,
            port=port,
            log_level="info",
            reload=True,
            reload_dirs=[str(ROOT_DIR)],
        )
    else:
        uvicorn.run(app, host=host, port=port, log_level="info")
