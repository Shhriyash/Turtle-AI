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
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.usage import UsageLimits, RunUsage

from core.llm_client import (
    get_groq_model,
    get_openrouter_models,
    get_groq_fallback_model,
    run_agent_with_fallbacks,
)
from core.email_flow import (
    combine_extracted_email_details,
    extract_deterministic_email_details,
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
from core.graph_store import GraphStore
from core.memory_store import MemoryStore
from core.confirmation_gate import ConfirmationGate
from core.dream_pass import DreamPass
from core.memory_journal import JournalStore, make_event
from core.memory_extractor import extract_memory_event_specs
from core.memory_replayer import replay
from core.personal_memory_extract import (
    PersonalMemoryCandidate,
    extract_memory_candidates_from_messages,
    run_stage_b_session_extractor,
)
from core.periodic_reflector import PeriodicReflector
from core.personal_memory_prompt import PersonalMemoryPromptBuilder, PersonalMemoryPromptConfig
from core.personal_memory_store import PersonalMemoryStore
from core.task_history import TaskHistoryStore
from core.paths import (
    MEMORY_EPISODES_FILE,
    MEMORY_EVENTS_FILE,
    MEMORY_GRAPH_FILE,
    MEMORY_PROFILE_FILE,
    MEMORY_STATE_FILE,
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
        "TURTLE_HISTORY_MAX_TOKENS": 12000,
        "TURTLE_MEMORY_FLUSH_TURNS": 20,
        "TURTLE_MEMORY_FLUSH_TOKENS": 20000,
        "TURTLE_MEMORY_PROFILE_MAX_LINES": 6,
        "TTS_DEBUG": False,
        "STT_MODEL": "whisper-large-v3-turbo",
        "MAIN_AGENT_MODEL": "openrouter:openai/gpt-oss-120b",
        "EMAIL_AGENT_MODEL": "groq:llama-3.3-70b-versatile",
        "DREAM_PASS_AGENT_MODEL": "",
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
EMAIL_PROMPT = load_prompt("email_agent")
_MAIN_ASSISTANT_PROMPT_TEMPLATE = load_prompt("main_assistant")


def _build_main_assistant_prompt(*, timezone: str = "UTC", channel: str = "web") -> str:
    """Inject runtime context into the main assistant system prompt (C2).

    The {runtime_context} placeholder in main_assistant.txt is replaced with
    live values so prompt caching still works on the static parts of the block.
    Dynamic content is isolated to this small substitution.
    """
    import datetime
    now_utc = datetime.datetime.now(datetime.UTC).strftime("%A, %d %B %Y, %H:%M UTC")
    runtime_lines = [
        f"Current date and time: {now_utc}",
        f"User timezone: {timezone}",
        f"Active channel: {channel}",
    ]
    runtime_context = "\n".join(runtime_lines)
    return _MAIN_ASSISTANT_PROMPT_TEMPLATE.replace("{runtime_context}", runtime_context)


# Static fallback used by AgentManager.rebuild (agents are built at startup);
# per-request prompts are built via _build_main_assistant_prompt() per-turn.
MAIN_ASSISTANT_PROMPT = _build_main_assistant_prompt()

# Retries: 1 for plain-string output (chitchat — no validator, extra retries are
# a pure latency tax).  Kept at 2 for typed outputs (router/extractor/email).
# See H3 in arch_improve.md.
OUTPUT_RETRIES = 1
SESSION_RESTORE_MODE = os.getenv("SESSION_RESTORE_MODE", "strict_new")
ACTIVE_HISTORY_MAX_TURNS = int(config.get("TURTLE_HISTORY_MAX_TURNS", 12))
ACTIVE_HISTORY_MAX_MESSAGES = int(config.get("ACTIVE_HISTORY_MAX_MESSAGES", 40))
ACTIVE_HISTORY_MAX_TOKENS = int(config.get("TURTLE_HISTORY_MAX_TOKENS", 12000))
MEMORY_FLUSH_TURNS = int(config.get("TURTLE_MEMORY_FLUSH_TURNS", 20))
MEMORY_FLUSH_TOKENS = int(config.get("TURTLE_MEMORY_FLUSH_TOKENS", 20000))
MEMORY_PROFILE_MAX_LINES = int(config.get("TURTLE_MEMORY_PROFILE_MAX_LINES", 6))
PERSONAL_MEMORY_ENABLED = settings.personal_memory_enabled
PERSONAL_MEMORY_DREAM_PASS_ENABLED = settings.personal_memory_dream_pass_enabled
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
    memory_store: MemoryStore
    personal_memory_store: PersonalMemoryStore
    personal_memory_prompt: PersonalMemoryPromptBuilder
    journal_store: JournalStore
    confirmation_gate: ConfirmationGate
    task_history_store: TaskHistoryStore
    rag_system: TurtleRAGSystem
    retrieval_broker: Any | None = None   # D4: wired in setup_shared_state
    reflector: PeriodicReflector | None = None
    search_cache: dict[str, str] = field(default_factory=dict)
    turn_counter: int = 0
    user_id: str = ""


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

    # Tier 3: Raw MemoryStore lines
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
        preview = state.confirmation_gate.preview_pending(prompt.event_id)
        if preview:
            return preview

    accepted = _parse_confirmation_answer(user_text)
    if accepted is None:
        return f"Quick check: {prompt.question} Please answer yes or no, or say 'show me' to see what I'd save."

    state.confirmation_gate.record_response(prompt.event_id, accepted=accepted)
    if accepted:
        return "Got it. I will remember that."
    return "Understood. I will not store that preference."


def _queue_confirmation_candidates_from_turn(
    state: SharedState,
    *,
    session_id: str,
    user_text: str,
) -> int:
    """Queue non-explicit memory candidates for yes/no confirmation in web mode."""
    if not PERSONAL_MEMORY_ENABLED:
        return 0

    try:
        profile = state.personal_memory_store.load_profile_snapshot()
        fake_msg = ModelRequest(parts=[UserPromptPart(content=user_text)])
        candidates = extract_memory_candidates_from_messages(
            message_history=[fake_msg],
            session_id=session_id,
            profile=profile,
        )
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
    except Exception as e:
        print(f"LOG: Confirmation candidate queue failed for {session_id}: {e}")
        return 0


async def _run_dream_pass_if_needed(
    state: SharedState,
    *,
    session_id: str,
) -> None:
    """Run Stage C dream pass for pending memory candidates when trigger conditions are met."""
    if not PERSONAL_MEMORY_ENABLED or not PERSONAL_MEMORY_DREAM_PASS_ENABLED:
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
            )
        ]
        if not events:
            return

        state.journal_store.append_many(events)
        result = replay(state.journal_store.load_all(), store=state.personal_memory_store)
        if result.written_topics:
            print(f"LOG: Per-turn memory applied for {session_id}: {result.written_topics}")
    except Exception as e:
        print(f"LOG: Per-turn fact extraction failed for {session_id}: {e}")


def _runtime_agent_registry() -> list[dict[str, Any]]:
    main_model = str(config.get("MAIN_AGENT_MODEL") or f"groq:{config.get('GROQ_PRIMARY_MODEL', 'llama-3.3-70b-versatile')}")
    email_model = str(config.get("EMAIL_AGENT_MODEL") or main_model)
    dream_model = str(config.get("DREAM_PASS_AGENT_MODEL") or "auto (groq:openai/gpt-oss-120b)")

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
            "status": "active" if PERSONAL_MEMORY_DREAM_PASS_ENABLED else "disabled",
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

def _build_model_from_str(model_str: str, settings: Any) -> Any | None:
    """Parse 'provider:model_name' and return a pydantic-ai model object."""
    if not model_str:
        return None
    if model_str.startswith("groq:"):
        return get_groq_model(model_name=model_str[5:], settings=settings)
    if model_str.startswith("openrouter:"):
        models = get_openrouter_models(model_name=model_str[11:], settings=settings)
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

        openrouter_models = get_openrouter_models(
            model_name=cfg.get("OPEN_ROUTER_MODEL"),
            settings=settings,
        )
        if not openrouter_models:
            raise RuntimeError("No OpenRouter API keys found.")

        primary_groq = get_groq_model(
            model_name=cfg.get("GROQ_PRIMARY_MODEL"),
            settings=settings,
        )
        main_model = primary_groq or openrouter_models[0]

        openrouter_fallbacks = openrouter_models[1:] or openrouter_models
        delegator_fallbacks = openrouter_models if primary_groq else openrouter_fallbacks
        groq_fallback = get_groq_fallback_model(
            model_name=cfg.get("GROQ_FALLBACK_MODEL"),
            settings=settings,
        )

        # Per-agent model overrides
        actual_main_model = (
            _build_model_from_str(cfg.get("MAIN_AGENT_MODEL", ""), settings)
            or main_model
        )
        actual_email_model = (
            _build_model_from_str(cfg.get("EMAIL_AGENT_MODEL", ""), settings)
            or main_model
        )

        # Main assistant
        self.main_assistant = Agent(
            actual_main_model,
            deps_type=SharedState,
            output_type=str,
            output_retries=OUTPUT_RETRIES,
            instructions=MAIN_ASSISTANT_PROMPT,
            history_processors=[_trim_history_for_context],
        )
        self.main_assistant_fallbacks = []
        for fb in delegator_fallbacks:
            self.main_assistant_fallbacks.append(
                Agent(fb, deps_type=SharedState, output_type=str,
                      output_retries=OUTPUT_RETRIES, instructions=MAIN_ASSISTANT_PROMPT,
                      history_processors=[_trim_history_for_context])
            )
        if groq_fallback:
            self.main_assistant_fallbacks.append(
                Agent(groq_fallback, deps_type=SharedState, output_type=str,
                      output_retries=OUTPUT_RETRIES, instructions=MAIN_ASSISTANT_PROMPT,
                      history_processors=[_trim_history_for_context])
            )

        # Email agent
        self.email_agent = Agent(
            actual_email_model,
            deps_type=SharedState,
            output_type=str,
            output_retries=OUTPUT_RETRIES,
            instructions=EMAIL_PROMPT,
            history_processors=[_trim_history_for_context],
        )
        self.email_agent_fallbacks = []
        for fb in delegator_fallbacks:
            self.email_agent_fallbacks.append(
                Agent(fb, deps_type=SharedState, output_type=str,
                      output_retries=OUTPUT_RETRIES, instructions=EMAIL_PROMPT,
                      history_processors=[_trim_history_for_context])
            )
        if groq_fallback:
            self.email_agent_fallbacks.append(
                Agent(groq_fallback, deps_type=SharedState, output_type=str,
                      output_retries=OUTPUT_RETRIES, instructions=EMAIL_PROMPT,
                      history_processors=[_trim_history_for_context])
            )

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

            missing = missing_email_fields(merged)
            if missing:
                await ctx.deps.session_store.set_pending_email(
                    recipients=merged["recipients"], cc_recipients=merged["cc_recipients"],
                    bcc_recipients=merged["bcc_recipients"], subject=merged["subject"], content=merged["content"],
                )
                return clean_text_for_model(format_missing_email_prompt(missing, merged))

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
async def serve_index():
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
        # OpenAI via OpenRouter
        "openai/gpt-oss-120b",
        "openai/gpt-4o-mini",
        # Llama 4
        "meta-llama/llama-4-scout:free",
        "meta-llama/llama-4-maverick:free",
        # Llama 3.3 / 3.1
        "meta-llama/llama-3.3-70b-instruct:free",
        "meta-llama/llama-3.1-70b-instruct:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        # DeepSeek
        "deepseek/deepseek-r1:free",
        "deepseek/deepseek-chat-v3-0324:free",
        # Qwen 3
        "qwen/qwen3-30b-a3b:free",
        "qwen/qwen3-8b:free",
        "qwen/qwen-2.5-72b-instruct:free",
        # Google Gemma 3
        "google/gemma-3-27b-it:free",
        "google/gemma-3-12b-it:free",
        "google/gemma-2-9b-it:free",
        # Nvidia / Mistral / Microsoft
        "nvidia/llama-3.1-nemotron-70b-instruct:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "mistralai/mistral-nemo:free",
        "mistralai/mistral-7b-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "microsoft/phi-3-medium-128k-instruct:free",
    ]
    groq_models = [
        # Llama 3.3 / 3.1
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        # Llama 3.2 vision
        "llama-3.2-90b-vision-preview",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview",
        # Llama 3 legacy
        "llama3-70b-8192",
        "llama3-8b-8192",
        # Deepseek
        "deepseek-r1-distill-llama-70b",
        # Qwen
        "qwen-qwq-32b",
        # Mixtral / Gemma
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "gemma-7b-it",
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
        graph_store = GraphStore(graph_path=MEMORY_GRAPH_FILE)
        memory_store = MemoryStore(
            profile_path=MEMORY_PROFILE_FILE,
            events_path=MEMORY_EVENTS_FILE,
            episodes_path=MEMORY_EPISODES_FILE,
            state_path=MEMORY_STATE_FILE,
            graph_store=graph_store,
            flush_turns=MEMORY_FLUSH_TURNS,
            flush_tokens=MEMORY_FLUSH_TOKENS,
            profile_max_lines=MEMORY_PROFILE_MAX_LINES,
            write_enabled=False,
        )
        personal_memory_store = PersonalMemoryStore(user_id=user_id)
        journal_store = JournalStore(user_id=user_id)
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
        rag_system = TurtleRAGSystem()

        # D4: construct RetrievalBroker for 4-tier memory context retrieval
        from core.storage.local.faiss_store import FAISSVectorStore
        from core.retrieval_broker import RetrievalBroker
        vector_store = FAISSVectorStore()
        retrieval_broker = RetrievalBroker(
            store=personal_memory_store,
            task_store=task_history_store,
            journal_store=journal_store,
            session_store=session_store,
            rag_system=rag_system,
            vector_store=vector_store,
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

                # Binary frame = audio data
                if "bytes" in raw and raw["bytes"]:
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
                            message_history = await _handle_text_message(
                                ws, state, content, message_history
                            )

                    elif msg_type == "audio":
                        # Base64-encoded audio fallback
                        audio_b64 = msg.get("data", "")
                        sample_rate = int(msg.get("sample_rate", 16000))
                        if audio_b64:
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
            if state.session_store.session_id:
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


async def _handle_text_message(
    ws: WebSocket,
    state: SharedState,
    user_text: str,
    message_history: list[ModelMessage] | None,
) -> list[ModelMessage] | None:
    """Process a text chat message and stream the response."""
    timings: dict[str, float] = {}
    overall_start = time.time()

    await _ws_send_json(ws, {"type": "status", "status": "thinking"})

    try:
        confirmation_reply = _maybe_handle_confirmation_turn(state, user_text)
        if confirmation_reply is not None:
            await _ws_send_json(ws, {"type": "done", "content": confirmation_reply})
            timings["total_ms"] = round((time.time() - overall_start) * 1000)
            await _ws_send_json(ws, {"type": "timing", **timings})
            return message_history

        task_type = _detect_task_type(user_text)

        # A1: Router stage — runs concurrently with memory resolution.
        # RouterDecision drives graph selection in Tier 1 (A2); here it feeds logs + timings.
        from core.router import route_turn as _route_turn
        router_start = time.time()
        router_task = asyncio.create_task(_route_turn(user_text))

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
        await _ws_send_json(ws, {"type": "error", "message": str(e)})
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
        router_task = asyncio.create_task(_route_turn(transcription))

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
