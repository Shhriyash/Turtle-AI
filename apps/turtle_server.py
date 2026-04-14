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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
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
except Exception:
    pass

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
        "OPEN_ROUTER_MODEL": "nvidia/nemotron-3-nano-30b-a3b:free",
        "GROQ_PRIMARY_MODEL": "openai/gpt-oss-120b",
        "GROQ_FALLBACK_MODEL": "llama-3.1-8b-instant",
        "DEEPGRAM_TTS_MODEL": "aura-2-orion-en",
        "DEEPGRAM_TTS_ENCODING": "linear16",
        "DEEPGRAM_TTS_CONTAINER": "wav",
        "DEEPGRAM_TTS_SAMPLE_RATE": 24000,
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
        self.rebuild(config)

    def rebuild(self, cfg: dict[str, Any]) -> None:
        """Rebuild all agents from the given config dict."""
        self.model_settings = {
            "temperature": float(cfg.get("temperature", 0.2)),
            "max_tokens": int(cfg.get("max_tokens", 1024)),
        }
        settings = self.model_settings

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

        # Main assistant
        self.main_assistant = Agent(
            main_model,
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
            main_model,
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
        print(f"LOG: Agent chain rebuilt — model={cfg.get('OPEN_ROUTER_MODEL', 'default')}, "
              f"temp={settings.get('temperature')}, max_tokens={settings.get('max_tokens')}")

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
                results = await search_duckduckgo(ctx.deps.http_client, normalized_query, max_results=5)
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

# Serve static files from web/ directory
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"error": "Frontend not built yet"}, status_code=404)
    return FileResponse(index_path, media_type="text/html")


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
    return JSONResponse({
        "openrouter_models": [
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "meta-llama/llama-4-scout:free",
            "meta-llama/llama-4-maverick:free",
            "google/gemma-3-27b-it:free",
            "qwen/qwen3-235b-a22b:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "microsoft/mai-ds-r1:free",
        ],
        "groq_models": [
            "openai/gpt-oss-120b",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "gemma2-9b-it",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ],
        "deepgram_tts_models": [
            "aura-2-orion-en",
            "aura-2-asteria-en",
            "aura-2-luna-en",
            "aura-2-stella-en",
            "aura-2-athena-en",
            "aura-2-hera-en",
            "aura-2-arcas-en",
            "aura-2-perseus-en",
            "aura-2-angus-en",
            "aura-2-orpheus-en",
            "aura-2-helios-en",
            "aura-2-zeus-en",
        ],
        "groq_tts_voices": [
            "orion",
            "atlas",
            "vale",
            "celeste",
            "nova",
        ],
    })


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
                except Exception:
                    pass
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
        transcription = agents_mgr.stt.transcribe_from_audio((sample_rate, audio_array))
        timings["stt_ms"] = round((time.time() - stt_start) * 1000)

        if not transcription or not transcription.strip():
            await _ws_send_json(ws, {"type": "error", "message": "No speech detected"})
            return message_history

        # Send transcription to client
        await _ws_send_json(ws, {"type": "transcription", "text": transcription})

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

        # TTS
        await _ws_send_json(ws, {"type": "status", "status": "speaking"})
        tts_start = time.time()
        tts_text = clean_text_for_tts(final_output)
        audio_filename = f"tts_{int(time.time() * 1000)}.wav"
        speech_path = TEMP_AUDIO_DIR / audio_filename

        try:
            result_path = synthesize_speech(tts_text, speech_path)
            timings["tts_ms"] = round((time.time() - tts_start) * 1000)

            if result_path and result_path.exists():
                audio_data = result_path.read_bytes()
                # Send audio as binary WebSocket frame
                await ws.send_bytes(audio_data)
                # Clean up temp file
                result_path.unlink(missing_ok=True)
            else:
                await _ws_send_json(ws, {"type": "error", "message": "TTS generation failed"})
        except Exception as e:
            print(f"LOG: TTS error: {e}")
            timings["tts_ms"] = round((time.time() - tts_start) * 1000)
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
    print(f"[Turtle AI] Web Server starting at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
