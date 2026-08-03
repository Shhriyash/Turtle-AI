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
import base64
import hashlib
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
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import UsageLimits, RunUsage

from core.llm_client import (
    get_groq_model,
    get_google_models,
    get_openrouter_models,
    get_groq_fallback_model,
    run_agent_with_fallbacks,
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
    send_email_now,
    validate_recipients,
    validate_send_email_args,
)
from core.output_clean import clean_text_for_model, clean_text_for_tts
from core.confirmation_gate import ConfirmationGate
from core.guardrails import (
    StorageCapExceededError,
    WebSocketRateLimitExceeded,
    ws_rate_limiter,
)
from core.telemetry import emit as emit_event, emit_once as emit_event_once
from core.memory_journal import JournalStore, make_event
from core.memory_schema import decide_write_policy, statement_for
from core.memory_extractor import extract_memory_event_specs
from core.memory_replayer import replay
from core.observability import trace_sink
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
from tools.contracts import ToolResult, WebSearchArgs, UrlFetchArgs, EmailArgs, RecallArgs, CalendarCreateArgs, CalendarListArgs

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
    model always sees the CURRENT memory snapshot exactly once — never baked
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
    memory_block = (state.memory_context or "").strip() if state is not None else ""
    if memory_block:
        parts.append(
            "Current user memory (authoritative; this is the only current copy — "
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
    # Phase 1: the memory block for the current turn. Delivered to the model
    # via per-turn instructions (never inside the persisted user prompt).
    memory_context: str = ""


# ---------------------------------------------------------------------------
# Robust shutdown wiring (Phase 3)
# ---------------------------------------------------------------------------
_ACTIVE_STATES: dict[int, "SharedState"] = {}
_ACTIVE_STATES_BY_USER: dict[str, "SharedState"] = {}
_SHUTDOWN_LOCK = threading.Lock()
_SHUTDOWN_REQUESTED = False


def _register_shutdown_state(state: "SharedState") -> None:
    _ACTIVE_STATES[id(state)] = state
    if state.user_id:
        _ACTIVE_STATES_BY_USER[state.user_id] = state


def _unregister_shutdown_state(state: "SharedState") -> None:
    _ACTIVE_STATES.pop(id(state), None)
    if state.user_id:
        _ACTIVE_STATES_BY_USER.pop(state.user_id, None)


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


def _trim_history_for_context(history: list[ModelMessage]) -> list[ModelMessage]:
    if len(history) <= ACTIVE_HISTORY_MAX_MESSAGES:
        approx_tokens = sum(len(str(m)) // 4 for m in history)
        if approx_tokens <= ACTIVE_HISTORY_MAX_TOKENS:
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

    while trimmed and sum(len(str(m)) // 4 for m in trimmed) > ACTIVE_HISTORY_MAX_TOKENS:
        trimmed = trimmed[1:]

    while trimmed and isinstance(trimmed[0], ModelResponse):
        trimmed = trimmed[1:]

    # Pair-aware front trim: a leading ModelRequest whose only parts are
    # ToolReturnPart / tool-tied RetryPromptPart is necessarily an orphan —
    # the originating ToolCallPart precedes it in conversation order, and
    # anything before the front of `trimmed` has already been discarded. Drop
    # it. Gemini direct rejects such histories with INVALID_ARGUMENT, and
    # even on lenient providers it confuses the model.
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

    while len(trimmed) > 1 and _is_leading_orphan(trimmed[0]):
        trimmed = trimmed[1:]

    # Guardrail: ensure at least one real user prompt remains in context.
    # Without this, long tool-call loops can leave only tool request/return turns,
    # causing the model to respond as if no user question was asked.
    if trimmed and not any(_is_user_turn_request(m) for m in trimmed):
        latest_user_index = -1
        for index in range(len(history) - 1, -1, -1):
            if _is_user_turn_request(history[index]):
                latest_user_index = index
                break
        if latest_user_index >= 0:
            candidate = history[latest_user_index:]
            if len(candidate) > ACTIVE_HISTORY_MAX_MESSAGES:
                candidate = candidate[-ACTIVE_HISTORY_MAX_MESSAGES:]
            while (
                len(candidate) > 1
                and sum(len(str(m)) // 4 for m in candidate) > ACTIVE_HISTORY_MAX_TOKENS
            ):
                candidate = candidate[1:]
            while candidate and not _is_user_turn_request(candidate[0]):
                candidate = candidate[1:]
            if candidate:
                return candidate

    return trimmed or history[-ACTIVE_HISTORY_MAX_MESSAGES:]


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

    loop.create_task(
        _queue_confirmation_candidates_async(
            state, session_id=session_id, user_text=user_text
        )
    )
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
                    _register_user_routines_safe(state)
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


def _register_user_routines_safe(state: "SharedState") -> None:
    """Phase 4 / E1: re-scan + register a user's routines after a write.

    Idempotent — APScheduler replaces existing job ids on re-registration.
    """
    if not state.user_id:
        return
    try:
        sched = get_routine_scheduler()
        if sched is None:
            return
        n = sched.register_for_user(state.user_id)
        if n:
            print(f"LOG: Re-registered {n} routine(s) for {state.user_id}")
    except Exception as e:
        print(f"LOG: routine registration failed for {state.user_id}: {e}")


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
) -> str:
    """Persist an explicit user-stated fact; return the agent-facing result string.

    Extracted from the `remember` tool (a closure, so otherwise untestable) so
    the storage-cap failure path can be unit tested. On a cap breach the fact is
    NOT saved: the user is notified and an honest failure string is returned
    instead of a fabricated "Stored".
    """
    from core.memory_journal import generate_event_id

    try:
        event = make_event(
            event_id=generate_event_id(),
            kind="fact",
            topic=topic,
            key=f"{topic}.{key_slug}",
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
            f"Noted: {topic}.{key_slug} = {value_text}. But memory storage is at "
            "its cap, so my memory files couldn't refresh — ask me to forget "
            "things to free up space."
        )
    except Exception as e:
        print(f"LOG: remember-tool replay failed after journal append: {e}")

    return ToolResult.ok(f"Stored: {topic}.{key_slug} = {value_text}").to_agent_string()


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
                _register_user_routines_safe(state)
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
        self.usage_limits = UsageLimits(request_limit=30)
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

        # Model pools — one per provider/key. Each pool is ordered so the first
        # element is the preferred entry-point for that provider.
        openrouter_models = get_openrouter_models(
            model_name=cfg.get("OPEN_ROUTER_MODEL"), settings=settings,
        )
        gemini_models = get_google_models(
            model_name=cfg.get("GEMINI_MODEL"), settings=settings,
        )
        gpt_oss = get_groq_model(
            model_name="openai/gpt-oss-120b", settings=settings,
        )
        groq_llama = get_groq_model(
            model_name=cfg.get("GROQ_PRIMARY_MODEL"), settings=settings,
        )
        groq_llama_small = get_groq_fallback_model(
            model_name=cfg.get("GROQ_FALLBACK_MODEL"), settings=settings,
        )

        if not (openrouter_models or gemini_models or gpt_oss or groq_llama):
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

        # main_assistant: Gemini direct → Gemini via OpenRouter → gpt-oss → Llama.
        # a1: Gemini leads. It's a strong tool-caller, and on the free tier its
        # token limits dwarf Groq's 8k TPM — which gpt-oss-120b blows on Turtle's
        # tool surface (see problems/2026-05-30-groq-tpm-and-gemini-thinking.md).
        # gpt-oss is demoted to a deep fallback: still there if Google is fully
        # down, but no longer the rung that 413s on every turn. The override goes
        # through get_google_models, so it inherits thinking-disabled settings;
        # the _override_redundant guard collapses it onto gemini_models[0].
        main_override = _build_model_from_str(cfg.get("MAIN_AGENT_MODEL", ""), settings)
        if _override_redundant(main_override, gemini_models):
            main_override = None
        main_head: Any = main_override or (gemini_models[0] if gemini_models else gpt_oss)
        # Groq llama is the final rescue: if every Gemini variant 400s on
        # function-call adjacency (Google strict, OR often proxies to the same
        # backend), llama-3.3-70b on Groq is lenient about message order.
        main_chain = build_chain(
            main_head, gemini_models, openrouter_models, gpt_oss, groq_llama, groq_llama_small,
        )

        # email_agent: Gemini direct → Gemini via OpenRouter → Llama (Groq).
        # Email composition is structure-heavy, so leading with Gemini's
        # stronger instruction-following pays off. Llama is the final rung in
        # case Google + OpenRouter are both unavailable.
        email_override = _build_model_from_str(cfg.get("EMAIL_AGENT_MODEL", ""), settings)
        if _override_redundant(email_override, gemini_models):
            email_override = None
        email_head: Any = email_override or (gemini_models[0] if gemini_models else None)
        email_chain = build_chain(
            email_head,
            gemini_models, openrouter_models, groq_llama, groq_llama_small,
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
            f"main={cfg.get('MAIN_AGENT_MODEL') or cfg.get('GROQ_PRIMARY_MODEL', 'default')}, "
            f"email={cfg.get('EMAIL_AGENT_MODEL') or 'same'}, "
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

            known_contacts: dict[str, Any] = {}
            try:
                _snapshot = ctx.deps.personal_memory_store.load_profile_snapshot()
                known_contacts = {
                    "contacts": _snapshot.get("contacts") or {},
                    "relations": _snapshot.get("relations") or {},
                }
            except Exception:
                known_contacts = {}

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
                await ctx.deps.session_store.set_pending_email(
                    recipients=merged["recipients"], cc_recipients=merged["cc_recipients"],
                    bcc_recipients=merged["bcc_recipients"], subject=merged["subject"], content=merged["content"],
                )
                return clean_text_for_model(format_missing_email_prompt(missing, merged))

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
                # B5: idempotency check — prevent double-sends within 60 s
                from tools.idempotency import build_email_idempotency_key, is_duplicate_invocation, record_invocation
                idem_key = build_email_idempotency_key(
                    merged["recipients"],
                    merged["subject"],
                    merged["content"],
                    cc=merged["cc_recipients"],
                    bcc=merged["bcc_recipients"],
                )
                cached_result = is_duplicate_invocation(idem_key)
                if cached_result is not None:
                    print(f"LOG: Email idempotency hit — skipping duplicate send ({idem_key[:12]}...)")
                    return clean_text_for_model(cached_result)

                send_result = send_email_now(merged)
                record_invocation(idem_key, send_result)
            except _ModelRetry:
                # pydantic_ai's retry protocol — swallowing it hands the model
                # a prose failure instead of a structured retry.
                raise
            except Exception as e:
                await ctx.deps.session_store.set_pending_email(
                    recipients=merged["recipients"], cc_recipients=merged["cc_recipients"],
                    bcc_recipients=merged["bcc_recipients"], subject=merged["subject"], content=merged["content"],
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

        async def calendar_create(ctx: RunContext[SharedState], args: CalendarCreateArgs) -> str:
            """Create a Google Calendar event. See tool contract for full spec."""
            from tools.calendar_tool import create_calendar_event
            from tools.calendar_tool import CalendarCreateArgs as _CalendarCreateArgs
            inner = _CalendarCreateArgs(
                title=args.title,
                start_iso=args.start_iso,
                end_iso=args.end_iso,
                attendee_emails=args.attendee_emails,
                description=args.description,
                add_google_meet=args.add_google_meet,
            )
            result = await create_calendar_event(inner)
            return result.to_agent_string()

        async def calendar_list(ctx: RunContext[SharedState], args: CalendarListArgs) -> str:
            """List upcoming Google Calendar events. See tool contract for full spec."""
            from tools.calendar_tool import list_upcoming_events
            from tools.calendar_tool import CalendarListArgs as _CalendarListArgs
            inner = _CalendarListArgs(
                max_results=args.max_results,
                time_min_iso=args.time_min_iso or None,
            )
            result = await list_upcoming_events(inner)
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

            # Store + storage-cap handling lives in a module-level helper so the
            # cap-failure path is unit testable (this tool is a closure).
            return _store_remembered_fact(
                ctx.deps, topic=topic, key_slug=key_slug, value_text=value_text
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
            ("calendar_list", calendar_list),
            ("remember", remember),
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
async def _start_routine_scheduler() -> None:
    global _routine_scheduler, _APP_LOOP
    # Capture the running app loop so the scheduler thread can bridge routine
    # notices back onto it (run_coroutine_threadsafe). Must happen on the loop.
    try:
        _APP_LOOP = asyncio.get_running_loop()
    except RuntimeError:
        _APP_LOOP = None
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
app.include_router(_twilio_voice_router)
app.include_router(_discord_router)

from apps.onboarding_routes import router as _onboarding_router, verify_session_cookie
app.include_router(_onboarding_router)

from apps.admin_routes import router as _admin_router
app.include_router(_admin_router)


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
_CHANNEL_STATE_IDLE_TTL_S = 30 * 60


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
    confirmation_gate = ConfirmationGate(
        journal=journal_store,
        store=personal_memory_store,
        state_path=personal_memory_dir(user_id) / "confirmation_state.json",
        sqlite_index=sqlite_index,
    )
    personal_memory_prompt = PersonalMemoryPromptBuilder(
        personal_memory_store,
        config=PersonalMemoryPromptConfig(
            max_bytes=PERSONAL_MEMORY_MAX_BYTES,
            max_topic_files=PERSONAL_MEMORY_MAX_TOPIC_FILES,
        ),
    )
    task_history_store = TaskHistoryStore(TASK_HISTORY_FILE)
    rag_system = TurtleRAGSystem(user_id=user_id)

    from core.storage.local.faiss_store import FAISSVectorStore
    from core.retrieval_broker import RetrievalBroker
    vector_store = FAISSVectorStore()
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
    key = (event.user_id, event.channel)
    async with _channel_state_lock(key):
        cached = _CHANNEL_STATES.get(key)
        state = cached[0] if cached is not None else None
        if state is None:
            state = await _build_channel_state(event.user_id, event.channel)
        _CHANNEL_STATES[key] = (state, now)

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
    hit this.
    """
    return JSONResponse({"status": "ok"})


@app.get("/favicon.ico")
async def serve_favicon():
    favicon_svg = STATIC_DIR / "favicon.svg"
    if favicon_svg.exists():
        return FileResponse(favicon_svg, media_type="image/svg+xml")
    return RedirectResponse(url="/static/favicon.svg")


# ---------------------------------------------------------------------------
# REST: Dev-mode config endpoints
# ---------------------------------------------------------------------------
@app.get("/api/config")
async def get_config():
    """Return current config for the dev-mode panel."""
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
    preserving the current zero-config developer flow. GET /api/config is left
    open on purpose — the dev panel reads config to render, and it exposes no
    secrets.
    """
    global config
    expected = (
        settings.admin_token.get_secret_value()
        if settings.admin_token is not None
        else None
    )
    if not expected and settings.is_cloud:
        # Cloud with no admin token: fail CLOSED like /admin/* does — an open
        # model-hot-swap endpoint on a public deployment is not acceptable
        # (Codex P6 #1). Local dev (not cloud, no token) stays open.
        return JSONResponse(
            {"error": "Config updates are disabled (TURTLE_ADMIN_TOKEN not set)."},
            status_code=503,
        )
    if expected and x_admin_token != expected:
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
    """Extract user_id from Bearer token, or fall back to local-dev identity."""
    from apps.auth import verify_token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
        try:
            payload = verify_token(token)
            return payload.get("sub")
        except Exception:
            return None
    if not settings.is_cloud:
        return "local_dev_user"
    return None


@app.get("/api/memory/pending")
async def get_pending_memory(request: Request):
    """Return all queued memory candidates awaiting user confirmation."""
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    state = _ACTIVE_STATES_BY_USER.get(user_id)
    if state is None:
        return JSONResponse({"pending": []})

    from core.confirmation_gate import _render_question  # noqa: PLC0415
    gate = state.confirmation_gate
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

    state = _ACTIVE_STATES_BY_USER.get(user_id)
    if state is None:
        return JSONResponse({"error": "No active session for user"}, status_code=404)

    result = state.confirmation_gate.record_response(event_id, accepted=accepted)
    if result is None:
        return JSONResponse({"error": "event_id not found in pending queue"}, status_code=404)
    # This panel is now the ONLY confirmation surface (chat-text confirmation
    # is gone entirely), so the side effects the old chat handler owned live
    # here: routine registration on a workflow accept, and the first-confirm
    # telemetry event.
    if accepted:
        # getattr-defensive: endpoint tests drive this with partial state stubs.
        confirm_user = getattr(state, "user_id", "")
        if confirm_user:
            emit_event_once(confirm_user, "memory_first_confirmed", topic=result.topic)
        if result.topic == "workflow":
            _register_user_routines_safe(state)
    return JSONResponse({"status": "ok", "applied": accepted})


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
        confirmation_gate = ConfirmationGate(
            journal=journal_store,
            store=personal_memory_store,
            state_path=personal_memory_dir(user_id) / "confirmation_state.json",
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
        task_history_store = TaskHistoryStore(TASK_HISTORY_FILE)
        rag_system = TurtleRAGSystem(user_id=user_id)

        # D4: construct RetrievalBroker for 4-tier memory context retrieval
        from core.storage.local.faiss_store import FAISSVectorStore
        from core.retrieval_broker import RetrievalBroker
        vector_store = FAISSVectorStore()
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

        if restore_result.restored:
            await _ws_send_json(ws, {
                "type": "status",
                "status": "restored",
                "session_id": restore_result.session_id,
                "message_count": restore_result.message_count,
            })

        await _ws_send_json(ws, {"type": "status", "status": "ready"})

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
                        ws_rate_limiter.check_and_record(user_id)
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
                    if not await _check_user_message_rate():
                        break
                    audio_bytes = raw["bytes"]
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
                    state.sqlite_index.checkpoint()
                except Exception:
                    pass
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


def deliver_routine_notice(user_id: str, frame: dict[str, Any]) -> bool:
    """Push a routine frame to the user's live socket(s), or queue it.

    Called from the routine scheduler's worker thread when a routine fires. The
    ONLY safe cross-thread bridge is asyncio.run_coroutine_threadsafe against the
    captured app loop (_APP_LOOP) — we never create_task/get_running_loop from
    this thread, and never block on the returned future's .result(). The bridged
    coroutine verifies each send and stashes the frame itself when no socket
    accepted it, so a stale registry entry cannot lose a fire.

    Returns True when the delivery attempt was scheduled onto the app loop;
    False when it was stashed directly (no live socket / no captured loop /
    closed loop). Never raises — a delivery failure must never affect the
    journal write that already happened upstream.
    """
    try:
        loop = _APP_LOOP
        with _LIVE_SOCKETS_LOCK:
            has_sockets = bool(_LIVE_SOCKETS.get(user_id))
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

    if state.user_id:
        emit_event_once(state.user_id, "first_message_sent", channel=channel)

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
        with trace_sink.span(
            "turtle.turn",
            user_id=state.user_id,
            session_id=state.session_store.session_id or "",
            turn_id=turn_id,
            intent=task_type,
            memory_context_chars=len(state.memory_context or ""),
            channel=channel,
        ):
            # Preserve the rich Logfire span the deleted graph layer used to
            # emit, so a turn's model spans still nest under one logical unit.
            if _logfire_loaded:
                import logfire as _lf_turn
                _turn_span_cm = _lf_turn.span("turtle.turn", intent=task_type, channel=channel)
            else:
                from contextlib import nullcontext
                _turn_span_cm = nullcontext()
            with _turn_span_cm:
                # ONE agent call with fallbacks. The 60s timeout is the one the
                # deleted graph layer used to own.
                response = await asyncio.wait_for(
                    run_agent_with_fallbacks(
                        agents_mgr.main_assistant,
                        agents_mgr.main_assistant_fallbacks,
                        prompt_input,
                        deps=state,
                        message_history=message_history,
                        usage=RunUsage(),
                        usage_limits=agents_mgr.usage_limits,
                    ),
                    timeout=60.0,
                )
        timings["llm_ms"] = round((time.time() - llm_start) * 1000)

        final_output = clean_text_for_model(response.output)

        # Send complete response
        await _emit(ws, {"type": "done", "content": final_output})

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


async def _handle_text_message(
    ws: WebSocket,
    state: SharedState,
    user_text: str,
    message_history: list[ModelMessage] | None,
) -> list[ModelMessage] | None:
    """Web text entrypoint — a thin wrapper over the canonical turn pipeline."""
    outcome = await _execute_turn(ws, state, user_text, message_history, channel="web")
    return outcome.new_history


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
            transcription = agents_mgr.stt.transcribe_from_audio((sample_rate, audio_array))
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

        # The full turn — routing, memory, graph, persistence, extraction,
        # trace span, confirmation sidecar, classified errors — is owned by the
        # one canonical pipeline. Voice thereby gains everything the text path
        # had that this handler used to omit.
        outcome = await _execute_turn(
            ws, state, transcription, message_history, channel="web_voice",
        )
        message_history = outcome.new_history
        final_output = outcome.output_text

        # A pipeline error returns no speakable text (and has already emitted its
        # own error frame): stop here.
        if not final_output:
            return message_history

        # E3: Streaming TTS with sentence-boundary chunking.
        # Each sentence is synthesised as soon as its boundary is detected and
        # sent to the client immediately — no waiting for the full audio file.
        await _ws_send_json(ws, {"type": "status", "status": "speaking"})
        tts_start = time.time()
        tts_text = clean_text_for_tts(final_output)
        tts_debug = settings.tts_debug

        # E4: latency budget
        from core.latency_budgets import budgets, check_sla
        from core.streaming_tts import stream_tts_from_text

        first_chunk_sent = False
        chunks_sent = 0
        tts_errors = 0

        try:
            async for sentence_text, audio_bytes in stream_tts_from_text(
                tts_text,
                speed=float(config.get("TURTLE_TTS_SPEED", 1.2)),
                tts_timeout_s=budgets.TOOL_S,
            ):
                if not first_chunk_sent:
                    first_byte_ms = round((time.time() - tts_start) * 1000)
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

    except Exception as e:
        print(f"LOG: Audio handler error: {e}")
        traceback.print_exc()
        await _ws_send_json(ws, {"type": "error", "message": str(e)})
        return message_history


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
