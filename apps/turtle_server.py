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
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
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
from core.graph import select_graph as _select_graph
from core.memory_store import MemoryStore
from core.confirmation_gate import ConfirmationGate
from core.guardrails import (
    StorageCapExceededError,
    WebSocketRateLimitExceeded,
    ws_rate_limiter,
)
from core.telemetry import emit as emit_event, emit_once as emit_event_once
from core.dream_pass import DreamPass
from core.memory_journal import JournalStore, make_event
from core.memory_extractor import extract_memory_event_specs
from core.memory_replayer import replay
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
from tools.contracts import ToolResult, WebSearchArgs, UrlFetchArgs, EmailArgs, HistoryArgs, RecallArgs, CalendarCreateArgs, CalendarListArgs

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
        "DREAM_PASS_AGENT_MODEL": "",
        "PERSONAL_MEMORY_DREAM_PASS_ENABLED": settings.personal_memory_dream_pass_enabled,
        "ROUTER_AGENT_MODEL": "groq:llama-3.1-8b-instant",
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
    main_assistant.txt are replaced with live values so prompt caching still
    works on the static parts of the block. Dynamic content is isolated to
    these small substitutions.
    """
    import datetime
    now_utc = datetime.datetime.now(datetime.UTC).strftime("%A, %d %B %Y, %H:%M UTC")
    runtime_lines = [
        f"Current date and time: {now_utc}",
        f"User timezone: {timezone}",
        f"Active channel: {channel}",
    ]
    runtime_context = "\n".join(runtime_lines)
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


def _dream_pass_enabled() -> bool:
    """Read the dream-pass flag dynamically so the dev panel can hot-toggle it."""
    raw = config.get("PERSONAL_MEMORY_DREAM_PASS_ENABLED")
    if raw is None:
        return bool(settings.personal_memory_dream_pass_enabled)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
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
    memory_store: MemoryStore | None
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
    # 3b: the routed intent for the current turn. Read by _scope_tools_by_intent
    # to expose only the tools relevant to this turn (a chitchat turn ships no
    # tool contracts at all). Set per-turn by the handlers before graph.run.
    intent: str = ""


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

    Falls back to PersonalMemoryPromptBuilder, then to MemoryStore.get_context_lines.
    The bypass path (_compose_prompt_with_memory calling build_memory_block directly)
    is replaced by this function.
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

    # Tier 3: Raw MemoryStore lines (legacy; only populated for non-web entrypoints)
    if state.memory_store is None:
        return ""
    fallback_lines = state.memory_store.get_context_lines(task_type=task_type, query=user_text)
    return "\n".join(fallback_lines).strip()


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


def _parse_confirmation_answer(user_text: str) -> bool | None:
    text = " ".join((user_text or "").strip().lower().split())
    if not text:
        return None

    yes_prefixes = (
        "yes",
        "y",
        "yeah",
        "yep",
        "sure",
        "ok",
        "okay",
        "please do",
        "go ahead",
        "affirmative",
    )
    no_prefixes = (
        "no",
        "n",
        "nope",
        "nah",
        "not now",
        "don't",
        "do not",
        "skip",
        "negative",
    )

    if any(text == token or text.startswith(f"{token} ") for token in yes_prefixes):
        return True
    if any(text == token or text.startswith(f"{token} ") for token in no_prefixes):
        return False
    return None


def _wants_preview(user_text: str) -> bool:
    text = " ".join(str(user_text or "").strip().lower().split())
    if not text:
        return False
    triggers = (
        "show me", "show it", "show the", "show what",
        "what would you save", "what do you want to save",
        "what is it", "what's it", "what pattern",
        "preview", "details", "more info", "more context",
        "which memory", "the exact",
    )
    return any(trigger in text for trigger in triggers)


def _maybe_handle_confirmation_turn(state: SharedState, user_text: str) -> str | None:
    prompt = state.confirmation_gate.next_prompt()
    if prompt is None:
        return None

    if _wants_preview(user_text):
        preview = state.confirmation_gate.preview_pending(list(prompt.all_event_ids))
        if preview:
            return preview

    accepted = _parse_confirmation_answer(user_text)
    if accepted is None:
        # C2+C3: don't hard-intercept; the turn proceeds normally and the
        # caller is expected to surface the prompt as a sidecar message so
        # the user can answer it next turn while the current question is
        # answered now.
        return None

    for event_id in prompt.all_event_ids:
        state.confirmation_gate.record_response(event_id, accepted=accepted)
    if accepted:
        if state.user_id:
            emit_event_once(state.user_id, "memory_first_confirmed", topic=prompt.topic)
        if prompt.topic == "workflow":
            _register_user_routines_safe(state)
        if len(prompt.all_event_ids) > 1:
            return "Got it. I will remember those."
        return "Got it. I will remember that."
    if len(prompt.all_event_ids) > 1:
        return "Understood. I will not store those."
    return "Understood. I will not store that preference."


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

    pending_events = []
    for idx, candidate in enumerate(candidates):
        event = _candidate_to_journal_event(
            candidate=candidate,
            session_id=session_id,
            ordinal=idx,
        )
        if event is None:
            continue
        if event.applied:
            continue
        if event.source == "explicit":
            continue
        pending_events.append(event)

    if not pending_events:
        return 0

    state.journal_store.append_many(pending_events)
    queued = 0
    for event in pending_events:
        if state.confirmation_gate.queue_candidate(event):
            queued += 1

    if queued:
        print(f"LOG: Queued {queued} confirmation candidate(s) for {session_id}")
    return queued


async def _run_dream_pass_if_needed(
    state: SharedState,
    *,
    session_id: str,
) -> None:
    """Run Stage C dream pass for pending memory candidates when trigger conditions are met."""
    if not PERSONAL_MEMORY_ENABLED or not _dream_pass_enabled():
        return
    if not session_id:
        return

    dream_pass = DreamPass(
        journal=state.journal_store,
        store=state.personal_memory_store,
        confirmation_gate=state.confirmation_gate,
        state_path=personal_memory_dir(state.user_id) / "dream_pass_state.json",
        snapshots_dir=personal_memory_dir(state.user_id) / "snapshots",
    )

    if not dream_pass.should_run():
        print(f"LOG: Dream pass skipped for {session_id} (trigger not met)")
        return

    dream_model = _build_model_from_str(
        str(config.get("DREAM_PASS_AGENT_MODEL", "") or ""),
        agents_mgr.model_settings,
    )
    if dream_model is None:
        dream_model = get_groq_model(
            model_name="openai/gpt-oss-120b",
            settings=agents_mgr.model_settings,
        )

    if dream_model is None:
        print(f"LOG: Dream pass skipped for {session_id} (no dream model available)")
        return

    result = await dream_pass.run(session_id=session_id, model=dream_model)
    if result.skipped_reason:
        print(f"LOG: Dream pass skipped for {session_id}: {result.skipped_reason}")
    if result.rolled_back:
        print(f"LOG: Dream pass rolled back for {session_id}: {result.sanity_failures}")


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


def _should_auto_apply_event(
    *,
    kind: str,
    source: str,
    confidence: float,
    topic: str = "",
) -> bool:
    # Explicit, high-confidence facts/preferences always auto-apply.
    if source == "explicit" and confidence >= 0.9 and kind in {"fact", "preference"}:
        return True
    # Phase 1: confidence-tiered auto-apply for non-identity topics.
    # Inferred preferences/workflow/projects with conf>=0.85 bypass the gate.
    if (
        topic in {"preferences", "workflow", "projects"}
        and confidence >= 0.85
        and kind in {"fact", "preference", "behavior"}
    ):
        return True
    return False


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
        return None

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
        applied=_should_auto_apply_event(kind=kind, source=source, confidence=confidence, topic=topic),
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
    try:
        await _run_dream_pass_if_needed(state, session_id=session_id)
    except Exception as e:
        print(f"LOG: Dream pass failed for {session_id}: {e}")


def _auto_promote_pending_workflow_on_confirm(
    state: "SharedState",
    *,
    user_text: str,
) -> int:
    """B3: promote workflow.* pending candidates when user_text reads as confirm.

    If the gate's next prompt belongs to topic='workflow' and the current turn's
    user_text parses as a yes, accept the whole batch so the gate writes the
    superseding `source=explicit, applied=True` event and the replayer picks
    it up on the next memory load. Returns count of events promoted.

    Note: when `_maybe_handle_confirmation_turn` already intercepts a pure
    yes/no, this function never runs (intercepted before the agent reply).
    This path matters when a turn carries BOTH new content AND a confirmation,
    and the gate handler didn't intercept.
    """
    accepted = _parse_confirmation_answer(user_text)
    if accepted is not True:
        return 0

    prompt = state.confirmation_gate.next_prompt()
    if prompt is None or prompt.topic != "workflow":
        return 0

    promoted = 0
    for event_id in prompt.all_event_ids:
        result = state.confirmation_gate.record_response(event_id, accepted=True)
        if result is not None:
            promoted += 1
    if promoted:
        print(f"LOG: Auto-promoted {promoted} pending workflow event(s) on confirmation")
        _register_user_routines_safe(state)
    return promoted


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

    Phase 2 / B3: also auto-promote any pending workflow.* candidate when the
    current user text reads as confirmation. This catches the multi-turn flow
    where the candidate was queued by B1+B2's async extractor on an earlier
    turn and the user's current message is a yes/save-it without going through
    the dedicated confirmation_turn handler (e.g. when the message also
    carries new content).
    """
    if not PERSONAL_MEMORY_ENABLED:
        return
    try:
        _auto_promote_pending_workflow_on_confirm(state, user_text=user_text)
    except Exception as e:
        print(f"LOG: Workflow auto-promote failed for {session_id}: {e}")
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
    except Exception as e:
        print(f"LOG: Per-turn fact extraction failed for {session_id}: {e}")


def _runtime_agent_registry() -> list[dict[str, Any]]:
    main_model = str(config.get("MAIN_AGENT_MODEL") or f"groq:{config.get('GROQ_PRIMARY_MODEL', 'llama-3.3-70b-versatile')}")
    email_model = str(config.get("EMAIL_AGENT_MODEL") or main_model)
    dream_model = str(config.get("DREAM_PASS_AGENT_MODEL") or "auto (groq:openai/gpt-oss-120b)")
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
            "id": "dream_pass_reviewer",
            "label": "Dream Pass Reviewer",
            "model": dream_model,
            "editable": True,
            "config_key": "DREAM_PASS_AGENT_MODEL",
            "status": "active" if _dream_pass_enabled() else "disabled",
        },
        {
            "id": "router",
            "label": "Intent Router",
            "model": str(config.get("ROUTER_AGENT_MODEL") or "groq:llama-3.1-8b-instant"),
            "editable": True,
            "config_key": "ROUTER_AGENT_MODEL",
            "status": "active",
        },
        {
            "id": "planner",
            "label": "Multi-step Planner",
            "model": "groq:llama-3.1-8b-instant",
            "editable": False,
            "status": "hardcoded",
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
# 3b: intent-scoped tools
# ---------------------------------------------------------------------------
# The router classifies every turn; we expose only the tools that intent can
# use, instead of shipping all 7 tool contracts (~3k tokens) on every request.
# A "hey turtle" chitchat turn then carries ZERO tool contracts — which is what
# pushed gpt-oss over Groq's 8k TPM cap (see problems/2026-05-30-*). An intent
# absent from this map (incl. multi_step, where the planner may dispatch
# anything, and any unknown/router-failure value) gets the full toolset.
_TOOL_NAMES_BY_INTENT: dict[str, set[str]] = {
    "chitchat": set(),
    "web": {"search_web", "search_url"},
    "url": {"search_url", "search_web"},
    "email": {"send_email_assistant"},
    "calendar": {"calendar_create", "calendar_list"},
    "memory_recall": {"recall", "history_tool"},
}


async def _scope_tools_by_intent(ctx: RunContext[SharedState], tool_defs: list) -> list:
    """pydantic-ai prepare_tools hook: filter the toolset to the routed intent.

    Returns the unfiltered list for unknown/multi_step intents (safe default),
    or only the intent's allowed tools (possibly empty, e.g. chitchat).
    """
    try:
        intent = (ctx.deps.intent or "").strip() if ctx.deps is not None else ""
    except Exception:
        intent = ""
    allowed = _TOOL_NAMES_BY_INTENT.get(intent)
    if allowed is None:
        return tool_defs
    return [td for td in tool_defs if getattr(td, "name", None) in allowed]


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

        # Main assistant. prepare_tools (3b) scopes the toolset to the routed
        # intent per turn — applied to every rung so the cascade keeps the same
        # tool-scoping behaviour on fallback.
        self.main_assistant = Agent(
            main_chain[0],
            deps_type=SharedState,
            output_type=str,
            output_retries=OUTPUT_RETRIES,
            instructions=MAIN_ASSISTANT_PROMPT,
            history_processors=[_trim_history_for_context, _sanitize_tool_pairs],
            prepare_tools=_scope_tools_by_intent,
        )
        self.main_assistant_fallbacks = [
            Agent(m, deps_type=SharedState, output_type=str,
                  output_retries=OUTPUT_RETRIES, instructions=MAIN_ASSISTANT_PROMPT,
                  history_processors=[_trim_history_for_context, _sanitize_tool_pairs],
                  prepare_tools=_scope_tools_by_intent)
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
        """Register all tools on self.main_assistant with typed args + rich contracts."""
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
            async def _user_greeting(ctx: RunContext[SharedState]) -> str:
                try:
                    uid = ctx.deps.user_id if ctx.deps is not None else ""
                except Exception:
                    uid = ""
                return _build_user_greeting_block(uid)

        _attach_user_greeting(agent)
        for fb in self.main_assistant_fallbacks:
            _attach_user_greeting(fb)

        @agent.tool(description=_load_tool_contract("search_web"))
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

        @agent.tool(description=_load_tool_contract("search_url"))
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

        @agent.tool(description=_load_tool_contract("send_email_assistant"))
        async def send_email_assistant(ctx: RunContext[SharedState], args: EmailArgs) -> str:
            """Send emails on behalf of the user. See tool contract for full spec."""
            query = args.query.strip()
            if not query:
                return ToolResult.invalid("query describing email request must not be empty").to_agent_string()
            print(f"\nEMAIL: Delegating to email specialist")
            pending_email = ctx.deps.session_store.get_pending_email()
            deterministic = extract_deterministic_email_details(query)

            extraction_prompt = (
                "Extract only email send fields from the latest user request.\n"
                "Rules:\n"
                "- Do not invent values that are not present in latest message or clear context.\n"
                "- Return recipients as a list of email strings.\n"
                "- Return cc_recipients as a list of email strings when user specifies cc.\n"
                "- Return bcc_recipients as a list of email strings when user specifies bcc.\n"
                "- Return empty strings for missing subject/content.\n"
                "- send_intent should be true only when user asks to send now.\n\n"
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
            if authored_this_turn and prefers_draft:
                await ctx.deps.session_store.set_pending_email(
                    recipients=merged["recipients"], cc_recipients=merged["cc_recipients"],
                    bcc_recipients=merged["bcc_recipients"], subject=merged["subject"], content=merged["content"],
                )
                return clean_text_for_model(
                    "Here's the draft:\n\n"
                    + format_email_draft(merged)
                    + "\n\nReply \"send\" to send it, or tell me what to change."
                )

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
            return clean_text_for_model(send_result)


        @agent.tool(description=_load_tool_contract("history_tool"))
        async def history_tool(ctx: RunContext[SharedState], args: HistoryArgs) -> str:
            """Search conversation history for past discussions. See tool contract for full spec."""
            query = args.query.strip()
            if not query:
                return ToolResult.invalid("query must not be empty").to_agent_string()
            try:
                broker = ctx.deps.retrieval_broker
                if broker is None:
                    return ToolResult.empty("Recall is not available.").to_agent_string()
                recall_text = await broker.recall(
                    query=query,
                    scope="episodic",
                    message_history=ctx.deps.session_store.message_history,
                    trim_fn=_trim_history_for_context,
                )
                if not recall_text:
                    return ToolResult.empty("No relevant information found in previous conversations.").to_agent_string()
                return ToolResult.ok(recall_text).to_agent_string()
            except Exception as e:
                return ToolResult.upstream_error(f"History lookup failed: {e}").to_agent_string()

        @agent.tool(description=_load_tool_contract("recall"))
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

        @agent.tool(description=_load_tool_contract("calendar_create"))
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

        @agent.tool(description=_load_tool_contract("calendar_list"))
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


@app.on_event("startup")
async def _start_routine_scheduler() -> None:
    global _routine_scheduler
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

app.include_router(_whatsapp_router)
app.include_router(_imessage_router)
app.include_router(_slack_router)
app.include_router(_twilio_voice_router)

from apps.onboarding_routes import router as _onboarding_router, verify_session_cookie
app.include_router(_onboarding_router)

from apps.admin_routes import router as _admin_router
app.include_router(_admin_router)


# Per-(user_id, channel) message history — keyed by (user_id, channel).
# Capped at 40 messages (same window as the WebSocket path).
_channel_histories: dict[tuple[str, str], list] = {}
_CHANNEL_HISTORY_LIMIT = 40


async def _channel_dispatch_handler(event: TurtleEvent) -> TurtleResponse:
    """
    Channel-agnostic dispatch: router → graph → main LLM → response text.

    Uses the global agents_mgr (same models as the WS handler).
    Conversation history is maintained per (user_id, channel) in
    _channel_histories so each user retains context across turns.
    """
    from core.output_clean import clean_text_for_model
    from pydantic_ai.usage import RunUsage

    history_key = (event.user_id, event.channel)
    message_history = _channel_histories.get(history_key) or None

    graph = _select_graph("chitchat")  # router not wired for channels yet; default graph
    prompt = event.content
    response = await graph.run(
        agents_mgr.main_assistant,
        prompt,
        fallback_agents=agents_mgr.main_assistant_fallbacks,
        deps=None,
        message_history=message_history,
        usage=RunUsage(),
        usage_limits=agents_mgr.usage_limits,
    )

    # Persist history, capped to the last N messages
    updated = response.all_messages()
    _channel_histories[history_key] = updated[-_CHANNEL_HISTORY_LIMIT:]

    text = clean_text_for_model(response.output)
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
async def update_config(body: dict[str, Any] | None = None):
    """Update config and hot-reload agent chain."""
    global config
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
    return JSONResponse({"status": "ok", "applied": accepted})


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
        session_store = SessionStore()
        restore_result = await session_store.start_or_restore(mode=SESSION_RESTORE_MODE)
        # Legacy single-tenant MemoryStore intentionally not constructed here —
        # personal memory now lives under personal_memory_dir(user_id).
        memory_store = None
        personal_memory_store = PersonalMemoryStore(user_id=user_id)
        from core.memory_sqlite import MemorySQLiteIndex
        sqlite_index = MemorySQLiteIndex(user_id=user_id)
        journal_store = JournalStore(user_id=user_id, on_append=sqlite_index.index_event)
        # Backfill the FTS5 index from the journal (idempotent; no-op on restart).
        try:
            sqlite_index.backfill_from_journal(journal_store)
        except Exception as exc:
            print(f"LOG: SQLite memory index backfill failed for {user_id}: {exc}")
        confirmation_gate = ConfirmationGate(
            journal=journal_store,
            store=personal_memory_store,
            state_path=personal_memory_dir(user_id) / "confirmation_state.json",
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
            memory_store=memory_store,
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
                    await _run_dream_pass_if_needed(state, session_id=pending_sid)
                except Exception as _e:
                    print(f"LOG: Dream pass error for pending session {pending_sid}: {_e}")

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
            # Session cleanup
            if state.session_store.session_id and state.memory_store is not None:
                state.memory_store.force_checkpoint(
                    session_id=state.session_store.session_id,
                    reason="session_end",
                )
            session_id = state.session_store.session_id
            # Capture messages before archive_active() clears them.
            final_messages = list(state.session_store.message_history)
            await state.session_store.archive_active(status="pending_finalization")
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
                try:
                    await _run_dream_pass_if_needed(state, session_id=session_id)
                except Exception as _e:
                    print(f"LOG: Dream pass error on session end for {session_id}: {_e}")
            print("LOG: Session archived and cleaned up")
            _unregister_shutdown_state(state)


# ---------------------------------------------------------------------------
# Message handlers
# ---------------------------------------------------------------------------

async def _ws_send_json(ws: WebSocket, data: dict[str, Any]) -> None:
    """Send a JSON message to the WebSocket client."""
    try:
        await ws.send_json(data)
    except Exception:
        pass


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


async def _handle_text_message(
    ws: WebSocket,
    state: SharedState,
    user_text: str,
    message_history: list[ModelMessage] | None,
) -> list[ModelMessage] | None:
    """Process a text chat message and stream the response."""
    timings: dict[str, float] = {}
    overall_start = time.time()

    if state.user_id:
        emit_event_once(state.user_id, "first_message_sent", channel="web")

    await _ws_send_json(ws, {"type": "status", "status": "thinking"})

    try:
        confirmation_reply = _maybe_handle_confirmation_turn(state, user_text)
        if confirmation_reply is not None:
            await _ws_send_json(ws, {"type": "done", "content": confirmation_reply})
            timings["total_ms"] = round((time.time() - overall_start) * 1000)
            await _ws_send_json(ws, {"type": "timing", **timings})
            return message_history

        # C2+C3: if a memory-confirmation prompt is pending and the user is
        # not answering it right now, surface the question as a sidecar
        # message before the agent reply. Their next yes/no will be picked
        # up by _maybe_handle_confirmation_turn on the following turn.
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

        # A1: Router stage — runs concurrently with memory resolution.
        # RouterDecision drives graph selection in Tier 1 (A2); here it feeds logs + timings.
        from core.router import route_turn as _route_turn
        router_start = time.time()
        router_task = asyncio.create_task(_route_turn(user_text, model_name=str(config.get("ROUTER_AGENT_MODEL") or "")))

        memory_context = await _resolve_memory_context(state, task_type=task_type, user_text=user_text)
        prompt_input = _compose_prompt_with_memory(user_text, memory_context)
        turn_id = _new_turn_id(state)

        # Await router (likely already done by now)
        try:
            router_decision = await router_task
            timings["router_ms"] = round((time.time() - router_start) * 1000)
            # Update task_type from router for richer context
            task_type = router_decision.intent if router_decision.intent != "clarify" else task_type
        except Exception as _re:
            print(f"LOG: Router await failed: {_re}")
            timings["router_ms"] = -1

        # Consume RouterDecision intent — route into the appropriate graph
        graph = _select_graph(task_type)
        state.intent = task_type  # 3b: scope tools to this turn's intent

        llm_start = time.time()
        response = await graph.run(
            agents_mgr.main_assistant,
            prompt_input,
            fallback_agents=agents_mgr.main_assistant_fallbacks,
            deps=state,
            message_history=message_history,
            usage=RunUsage(),
            usage_limits=agents_mgr.usage_limits,
        )
        timings["llm_ms"] = round((time.time() - llm_start) * 1000)

        final_output = clean_text_for_model(response.output)

        # Send complete response
        await _ws_send_json(ws, {"type": "done", "content": final_output})

        # Update session
        message_history = response.all_messages()
        await state.session_store.replace_messages(message_history)
        state.rag_system.add_conversation(user_text, final_output)
        if state.memory_store is not None:
            state.memory_store.record_turn(
                session_id=state.session_store.session_id or "unknown_session",
                turn_id=turn_id,
                user_text=user_text,
                assistant_text=final_output,
                task_type=task_type,
            )
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
        if state.reflector is not None:
            await state.reflector.on_turn(
                state,
                session_id=state.session_store.session_id or "",
                message_history=message_history or [],
                dream_pass_runner=_run_dream_pass_if_needed,
            )

        timings["total_ms"] = round((time.time() - overall_start) * 1000)
        await _ws_send_json(ws, {"type": "timing", **timings})

        return message_history

    except Exception as e:
        print(f"LOG: Text handler error: {e}")
        traceback.print_exc()
        code, friendly = _classify_handler_error(e)
        if _logfire_loaded:
            try:
                import logfire as _lf
                _lf.error(
                    "turtle.turn_failed",
                    error_class=e.__class__.__name__,
                    error_code=code,
                    error_message=str(e),
                )
            except Exception:
                pass
        await _ws_send_json(ws, {"type": "error", "code": code, "message": friendly})
        return message_history


async def _handle_audio_message(
    ws: WebSocket,
    state: SharedState,
    audio_bytes: bytes,
    message_history: list[ModelMessage] | None,
    *,
    sample_rate: int = 16000,
) -> list[ModelMessage] | None:
    """Process voice input: STT → LLM → TTS."""
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

        confirmation_reply = _maybe_handle_confirmation_turn(state, transcription)
        if confirmation_reply is not None:
            await _ws_send_json(ws, {"type": "done", "content": confirmation_reply})
            timings["total_ms"] = round((time.time() - overall_start) * 1000)
            await _ws_send_json(ws, {"type": "timing", **timings})
            return message_history

        # Process as text
        await _ws_send_json(ws, {"type": "status", "status": "thinking"})

        task_type = _detect_task_type(transcription)

        # A1: Router stage — same as text path, concurrent with memory resolution.
        from core.router import route_turn as _route_turn
        router_start = time.time()
        router_task = asyncio.create_task(_route_turn(transcription, model_name=str(config.get("ROUTER_AGENT_MODEL") or "")))

        memory_context = await _resolve_memory_context(state, task_type=task_type, user_text=transcription)
        prompt_input = _compose_prompt_with_memory(transcription, memory_context)
        turn_id = _new_turn_id(state)

        try:
            router_decision = await router_task
            timings["router_ms"] = round((time.time() - router_start) * 1000)
            task_type = router_decision.intent if router_decision.intent != "clarify" else task_type
        except Exception as _re:
            print(f"LOG: Router await failed: {_re}")
            timings["router_ms"] = -1

        graph = _select_graph(task_type)
        state.intent = task_type  # 3b: scope tools to this turn's intent

        llm_start = time.time()
        response = await graph.run(
            agents_mgr.main_assistant,
            prompt_input,
            fallback_agents=agents_mgr.main_assistant_fallbacks,
            deps=state,
            message_history=message_history,
            usage=RunUsage(),
            usage_limits=agents_mgr.usage_limits,
        )
        timings["llm_ms"] = round((time.time() - llm_start) * 1000)

        final_output = clean_text_for_model(response.output)
        await _ws_send_json(ws, {"type": "done", "content": final_output})

        # Update session
        message_history = response.all_messages()
        await state.session_store.replace_messages(message_history)
        state.rag_system.add_conversation(transcription, final_output)
        if state.memory_store is not None:
            state.memory_store.record_turn(
                session_id=state.session_store.session_id or "unknown_session",
                turn_id=turn_id,
                user_text=transcription,
                assistant_text=final_output,
                task_type=task_type,
            )
        # Immediately apply explicit facts (email, name) so next turn sees them
        _apply_explicit_facts_from_turn(
            state,
            session_id=state.session_store.session_id or "unknown_session",
            turn_id=turn_id,
            user_text=transcription,
            task_type=task_type,
        )
        # D3: candidates queued silently — no in-turn confirmation interrupt.
        # The web UI /api/memory/pending endpoint will expose them for batch review.
        _queue_confirmation_candidates_from_turn(
            state,
            session_id=state.session_store.session_id or "unknown_session",
            user_text=transcription,
        )
        if state.reflector is not None:
            await state.reflector.on_turn(
                state,
                session_id=state.session_store.session_id or "",
                message_history=message_history or [],
                dream_pass_runner=_run_dream_pass_if_needed,
            )

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
