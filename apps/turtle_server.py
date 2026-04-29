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

import asyncio
import base64
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Path bootstrap (same as turtle_voice.py)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
)
from core.personal_memory_prompt import PersonalMemoryPromptBuilder, PersonalMemoryPromptConfig
from core.personal_memory_store import PersonalMemoryStore
from core.task_history import TaskHistoryStore
from core.paths import (
    MEMORY_EPISODES_FILE,
    MEMORY_EVENTS_FILE,
    MEMORY_GRAPH_FILE,
    MEMORY_PROFILE_FILE,
    MEMORY_STATE_FILE,
    PERSONAL_MEMORY_DIR,
    PERSONAL_MEMORY_SNAPSHOTS_DIR,
    TASK_HISTORY_FILE,
    TEMP_AUDIO_DIR,
    ensure_dirs,
)
from core.session_store import SessionStore
from core.system_prompts import load_prompt
from core.openrouter_tts import synthesize_speech
from core.stt_fastrtc import FastRTCSTT
from core.web_search import format_search_results, search_duckduckgo
from rag.system.complete_rag import TurtleRAGSystem
from tools.url_tools import fetch_url_content_async

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
MAIN_ASSISTANT_PROMPT = load_prompt("main_assistant")

OUTPUT_RETRIES = 3
SESSION_RESTORE_MODE = os.getenv("SESSION_RESTORE_MODE", "strict_new")
ACTIVE_HISTORY_MAX_TURNS = int(config.get("TURTLE_HISTORY_MAX_TURNS", 12))
ACTIVE_HISTORY_MAX_MESSAGES = int(config.get("ACTIVE_HISTORY_MAX_MESSAGES", 40))
ACTIVE_HISTORY_MAX_TOKENS = int(config.get("TURTLE_HISTORY_MAX_TOKENS", 12000))
MEMORY_FLUSH_TURNS = int(config.get("TURTLE_MEMORY_FLUSH_TURNS", 20))
MEMORY_FLUSH_TOKENS = int(config.get("TURTLE_MEMORY_FLUSH_TOKENS", 20000))
MEMORY_PROFILE_MAX_LINES = int(config.get("TURTLE_MEMORY_PROFILE_MAX_LINES", 6))
PERSONAL_MEMORY_ENABLED = os.getenv("TURTLE_PERSONAL_MEMORY_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
PERSONAL_MEMORY_DREAM_PASS_ENABLED = os.getenv("TURTLE_PERSONAL_MEMORY_DREAM_PASS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
PERSONAL_MEMORY_MAX_BYTES = int(os.getenv("TURTLE_PERSONAL_MEMORY_MAX_BYTES", "2048"))
PERSONAL_MEMORY_MAX_TOPIC_FILES = int(os.getenv("TURTLE_PERSONAL_MEMORY_MAX_TOPIC_FILES", "2"))
TOOL_OUTPUT_MAX_CHARS = int(os.getenv("TURTLE_TOOL_OUTPUT_MAX_CHARS", "3500"))

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2"))


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
    search_cache: dict[str, str] = field(default_factory=dict)
    turn_counter: int = 0


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


def _resolve_memory_context(state: SharedState, *, task_type: str, user_text: str) -> str:
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


def _maybe_handle_confirmation_turn(state: SharedState, user_text: str) -> str | None:
    prompt = state.confirmation_gate.next_prompt()
    if prompt is None:
        return None

    accepted = _parse_confirmation_answer(user_text)
    if accepted is None:
        return f"Quick check: {prompt.question} Please answer yes or no."

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
        state_path=PERSONAL_MEMORY_DIR / "dream_pass_state.json",
        snapshots_dir=PERSONAL_MEMORY_SNAPSHOTS_DIR,
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


def _should_auto_apply_event(*, kind: str, source: str, confidence: float) -> bool:
    if source != "explicit":
        return False
    if confidence < 0.9:
        return False
    return kind in {"fact", "preference"}


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
        applied=_should_auto_apply_event(kind=kind, source=source, confidence=confidence),
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
        """Register all tools on self.main_assistant."""
        agent = self.main_assistant

        @agent.tool
        async def search_web(ctx: RunContext[SharedState], query: str) -> str:
            """Search the web for current information."""
            print("\nSEARCHING: Web search for current information")
            normalized_query = " ".join(query.split())
            cache_key = f"web::{normalized_query}"
            cached = ctx.deps.search_cache.get(cache_key)
            if cached:
                return cached
            try:
                results = await search_duckduckgo(ctx.deps.http_client, normalized_query, max_results=10)

                # DuckDuckGo can underperform with strict site: filters for jobs.
                # Retry once with a relaxed query to avoid empty tool outputs.
                if not results and "site:" in normalized_query.lower():
                    relaxed_query = " ".join(
                        token for token in normalized_query.split()
                        if not token.lower().startswith("site:")
                    ).strip()
                    if relaxed_query:
                        results = await search_duckduckgo(
                            ctx.deps.http_client,
                            relaxed_query,
                            max_results=10,
                        )
                        if results:
                            normalized_query = relaxed_query

                formatted = format_search_results(normalized_query, results)
            except Exception as e:
                formatted = f"Web search failed for query: {normalized_query}\nError: {e}"
            cleaned = clean_text_for_model(formatted)
            trimmed = _truncate_tool_output(cleaned, label="web search results")
            ctx.deps.search_cache[cache_key] = trimmed
            return trimmed

        @agent.tool
        async def search_url(ctx: RunContext[SharedState], url: str) -> str:
            """Analyze and extract detailed content from a URL."""
            print(f"\nANALYZING: URL content extraction from {url}")
            normalized_url = _normalize_url_for_cache(url)
            cache_key = f"url::{normalized_url}"
            cached = ctx.deps.search_cache.get(cache_key)
            if cached:
                return cached
            result = await fetch_url_content_async(ctx.deps.http_client, normalized_url)
            cleaned = clean_text_for_model(result.to_formatted_string())
            trimmed = _truncate_tool_output(cleaned, label="url analysis")
            ctx.deps.search_cache[cache_key] = trimmed
            return trimmed

        @agent.tool
        async def send_email_assistant(ctx: RunContext[SharedState], query: str) -> str:
            """Send emails using the email specialist agent."""
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
                ctx.deps.session_store.set_pending_email(
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
                ctx.deps.session_store.set_pending_email(
                    recipients=merged["recipients"], cc_recipients=merged["cc_recipients"],
                    bcc_recipients=merged["bcc_recipients"], subject=merged["subject"], content=merged["content"],
                )
                return clean_text_for_model(format_missing_email_prompt(missing, merged))

            try:
                validate_send_email_args(
                    merged["recipients"], merged["subject"], merged["content"],
                    merged["cc_recipients"], merged["bcc_recipients"],
                )
                send_result = send_email_now(merged)
            except Exception as e:
                ctx.deps.session_store.set_pending_email(
                    recipients=merged["recipients"], cc_recipients=merged["cc_recipients"],
                    bcc_recipients=merged["bcc_recipients"], subject=merged["subject"], content=merged["content"],
                )
                return clean_text_for_model(f"Failed to send email: {e}")

            if send_result.startswith("Email sent successfully!"):
                ctx.deps.session_store.clear_pending_email()
            else:
                ctx.deps.session_store.set_pending_email(
                    recipients=merged["recipients"], cc_recipients=merged["cc_recipients"],
                    bcc_recipients=merged["bcc_recipients"], subject=merged["subject"], content=merged["content"],
                )
            return clean_text_for_model(send_result)

        @agent.tool
        async def history_tool(ctx: RunContext[SharedState], query: str) -> str:
            """Search conversation history for past discussions and information."""
            try:
                task_history = ctx.deps.task_history_store.format_search_results(query, max_results=5)
                result = await ctx.deps.rag_system.query_history(query)
                sections: list[str] = []
                if task_history:
                    sections.append(task_history)
                if result != "cannot find in history":
                    sections.append(result)
                if not sections:
                    return "No relevant information found in task history or previous conversations."
                return "\n\n".join(sections)
            except Exception:
                return "Unable to access conversation history at the moment."


# ---------------------------------------------------------------------------
# Global agent manager
# ---------------------------------------------------------------------------
agents_mgr = AgentManager()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Turtle AI", docs_url=None, redoc_url=None)

if _logfire_loaded:
    import logfire as _lf
    _lf.instrument_fastapi(app)

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
# WebSocket: Main chat interface
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("LOG: WebSocket client connected")

    # Build SharedState for this connection
    async with httpx.AsyncClient() as client:
        session_store = SessionStore()
        restore_result = session_store.start_or_restore(mode=SESSION_RESTORE_MODE)
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
        personal_memory_store = PersonalMemoryStore()
        journal_store = JournalStore()
        confirmation_gate = ConfirmationGate(
            journal=journal_store,
            store=personal_memory_store,
            state_path=PERSONAL_MEMORY_DIR / "confirmation_state.json",
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
        )

        # Process pending sessions from previous runs (personal memory + RAG finalization)
        for pending_sid, pending_archive_path in session_store.list_pending_finalization_archives():
            print(f"LOG: Finalizing pending session {pending_sid}")
            await _sync_personal_memory_from_archive(
                state, session_id=pending_sid, archive_path=pending_archive_path,
            )
            try:
                rag_finalized = await rag_system.finalize_archived_session(
                    session_id=pending_sid, archive_path=pending_archive_path,
                )
                if rag_finalized:
                    pending_manifest = pending_archive_path / "session.json"
                    if pending_manifest.exists():
                        manifest = json.loads(pending_manifest.read_text(encoding="utf-8"))
                        manifest["status"] = "completed"
                        pending_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    print(f"LOG: Pending session {pending_sid} finalized")
                else:
                    print(f"LOG: Pending session {pending_sid} RAG finalization still pending")
            except Exception as _e:
                print(f"LOG: RAG finalization error for {pending_sid}: {_e}")

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
            archive_path = state.session_store.archive_active(status="pending_finalization")
            if session_id and archive_path:
                # Personal memory sync — must run before RAG finalization
                await _sync_personal_memory_from_archive(
                    state, session_id=session_id, archive_path=archive_path,
                )
                try:
                    rag_finalized = await rag_system.finalize_archived_session(
                        session_id=session_id, archive_path=archive_path,
                    )
                    if rag_finalized:
                        session_manifest = archive_path / "session.json"
                        if session_manifest.exists():
                            manifest = json.loads(session_manifest.read_text(encoding="utf-8"))
                            manifest["status"] = "completed"
                            session_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                except Exception as _e:
                    print(f"LOG: RAG finalization error for {session_id}: {_e}")
            print("LOG: Session archived and cleaned up")


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
        memory_context = _resolve_memory_context(state, task_type=task_type, user_text=user_text)
        prompt_input = _compose_prompt_with_memory(user_text, memory_context)
        turn_id = _new_turn_id(state)

        llm_start = time.time()
        response = await run_agent_with_fallbacks(
            agents_mgr.main_assistant,
            agents_mgr.main_assistant_fallbacks,
            prompt_input,
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
        state.session_store.replace_messages(message_history)
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
        queued = _queue_confirmation_candidates_from_turn(
            state,
            session_id=state.session_store.session_id or "unknown_session",
            user_text=user_text,
        )
        if queued:
            prompt = state.confirmation_gate.next_prompt()
            if prompt is not None:
                await _ws_send_json(
                    ws,
                    {
                        "type": "done",
                        "content": f"Quick check: {prompt.question} Please answer yes or no.",
                    },
                )

        if state.turn_counter >= 8:
            try:
                await _run_dream_pass_if_needed(state, session_id=state.session_store.session_id or "unknown_session")
            except Exception as e:
                print(f"LOG: Dream pass (per-turn) failed: {e}")

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
        memory_context = _resolve_memory_context(state, task_type=task_type, user_text=transcription)
        prompt_input = _compose_prompt_with_memory(transcription, memory_context)
        turn_id = _new_turn_id(state)

        llm_start = time.time()
        response = await run_agent_with_fallbacks(
            agents_mgr.main_assistant,
            agents_mgr.main_assistant_fallbacks,
            prompt_input,
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
        state.session_store.replace_messages(message_history)
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
        queued = _queue_confirmation_candidates_from_turn(
            state,
            session_id=state.session_store.session_id or "unknown_session",
            user_text=transcription,
        )
        if queued:
            prompt = state.confirmation_gate.next_prompt()
            if prompt is not None:
                await _ws_send_json(
                    ws,
                    {
                        "type": "done",
                        "content": f"Quick check: {prompt.question} Please answer yes or no.",
                    },
                )

        if state.turn_counter >= 8:
            try:
                await _run_dream_pass_if_needed(state, session_id=state.session_store.session_id or "unknown_session")
            except Exception as e:
                print(f"LOG: Dream pass (per-turn) failed: {e}")

        # TTS
        await _ws_send_json(ws, {"type": "status", "status": "speaking"})
        tts_start = time.time()
        tts_text = clean_text_for_tts(final_output)
        audio_filename = f"tts_{int(time.time() * 1000)}.wav"
        speech_path = TEMP_AUDIO_DIR / audio_filename
        tts_debug = os.getenv("TTS_DEBUG", "0") == "1"

        try:
            print(f"LOG: TTS attempting synthesis ({len(tts_text)} chars, path={speech_path})")
            result_path = synthesize_speech(
                tts_text,
                speech_path,
                speed=float(config.get("TURTLE_TTS_SPEED", 1.2)),
            )
            timings["tts_ms"] = round((time.time() - tts_start) * 1000)
            print(f"LOG: TTS synthesis succeeded in {timings['tts_ms']}ms -> {result_path}")

            if result_path and result_path.exists():
                audio_data = result_path.read_bytes()
                print(f"LOG: Sending TTS audio ({len(audio_data)} bytes)")
                # Send audio as binary WebSocket frame
                await ws.send_bytes(audio_data)
                # Clean up temp file
                result_path.unlink(missing_ok=True)
            else:
                print("LOG: TTS returned path but file does not exist")
                await _ws_send_json(ws, {"type": "error", "message": "TTS generation failed: output file missing"})
        except Exception as e:
            timings["tts_ms"] = round((time.time() - tts_start) * 1000)
            print(f"LOG: TTS error after {timings['tts_ms']}ms: {type(e).__name__}: {e}")
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

    host = str(config.get("SERVER_HOST", SERVER_HOST))
    port = int(config.get("SERVER_PORT", SERVER_PORT))
    reload_enabled = os.getenv("TURTLE_SERVER_RELOAD", "1").strip().lower() not in {"0", "false", "no", "off"}
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
