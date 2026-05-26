"""
Turtle - Personal Assistant with Web Search and URL Context Capabilities

Enhanced assistant with real-time web search, URL analysis, and conversation memory.
"""

import atexit
import asyncio
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
import hashlib
import re
from typing import Optional, Generator, Tuple
from urllib.parse import urlsplit, urlunsplit
import httpx
import time
from pathlib import Path
import json
import numpy as np
from pydantic_ai import Agent, RunContext, ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.usage import UsageLimits, RunUsage
import logfire

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.env import load_env
# Basic VAD with STT

# Groq for STT
from groq import Groq

# custom URL tools package
from tools.url_tools import fetch_url_content_async
# Email tools
# RAG system for conversation memory
from rag.system.complete_rag import TurtleRAGSystem
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
from core.retrieval_broker import RetrievalBroker
from core.memory_extractor import extract_memory_event_specs
from core.memory_replayer import replay
from core.personal_memory_extract import PersonalMemoryCandidate, extract_memory_candidates_from_messages
from core.personal_memory_prompt import PersonalMemoryPromptBuilder, PersonalMemoryPromptConfig
from core.personal_memory_store import PersonalMemoryStore
from core.periodic_reflector import PeriodicReflector
from core.task_history import TaskHistoryStore
from core.paths import (
    MEMORY_EPISODES_FILE,
    MEMORY_EVENTS_FILE,
    MEMORY_GRAPH_FILE,
    PERSONAL_MEMORY_DIR,
    PERSONAL_MEMORY_SNAPSHOTS_DIR,
    MEMORY_PROFILE_FILE,
    MEMORY_STATE_FILE,
    TASK_HISTORY_FILE,
    TEMP_AUDIO_DIR,
    ensure_dirs,
)
from core.session_store import SessionStore
from core.system_prompts import load_prompt
from core.openrouter_tts import synthesize_speech
from core.stt_fastrtc import FastRTCSTT
from core.web_search import format_search_results, search_duckduckgo
from tools.contracts import ToolResult



load_env(override=True)
try:
    logfire.configure(send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)
except Exception as e:
    print(f"LOG: logfire disabled ({e})")
ensure_dirs()

EMAIL_PROMPT = load_prompt("email_agent")
MAIN_ASSISTANT_PROMPT = load_prompt("main_assistant")

# Initialize Groq client for STT
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2"))


@dataclass
class SharedState:
    """Shared state across all agents - now using UrlState for URL operations"""
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
    retrieval_broker: RetrievalBroker | None = field(default=None)
    reflector: PeriodicReflector | None = None


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


class TurtleVoiceProcessor:
    """Voice processing class for Turtle assistant with STT"""
    
    def __init__(self, shared_state: SharedState):
        self.shared_state = shared_state
        self.temp_dir = TEMP_AUDIO_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.stt = FastRTCSTT(groq_client=groq_client)
        
    async def process_voice_input(self, audio: Tuple[int, np.ndarray]) -> Optional[str]:
        """Process recorded audio and return transcription"""
        sample_rate, audio_array = audio
        
        print(f"LOG: Processing audio - {len(audio_array)} samples at {sample_rate}Hz")
        
        try:
            # Check if we have valid audio data
            if len(audio_array) < 1000:  # Skip very short audio snippets
                print("LOG: Audio too short, skipping")
                return None
            
            # Transcribe using Groq Whisper
            stt_start_time = time.time()
            transcription = self.stt.transcribe_from_audio(audio)
            stt_time = time.time() - stt_start_time
            
            if not transcription or not transcription.strip():
                print("LOG: No speech detected in transcription")
                return None
                
            print(f"LOG: STT completed in {stt_time:.2f}s")
            print(f"User: {transcription}")
            
            return transcription
                
        except Exception as e:
            print(f"LOG: Voice processing error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """Clean up resources"""
        print("LOG: Cleaning up temporary files")
        for file in self.temp_dir.glob("*.wav"):
            file.unlink()
        print("LOG: Cleanup completed")

    def text_to_speech(self, text):
        """Convert text to speech using Deepgram TTS (Groq fallback)"""
        try:
            audio_filename = f"tts_{int(time.time() * 1000)}.wav"
            speech_path = self.temp_dir / audio_filename
            return synthesize_speech(text, speech_path)
        except Exception as e:
            print(f"TTS error: {e}")
            return None

    def play_audio(self, audio_path):
        """Play audio file, preferring WAV playback without ffmpeg."""
        try:
            if audio_path.suffix.lower() == ".wav":
                import sounddevice as sd
                import scipy.io.wavfile as wavfile

                with open(audio_path, "rb") as audio_file:
                    header = audio_file.read(12)
                if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                    rate, data = wavfile.read(str(audio_path))
                else:
                    # Handle raw linear16 audio if WAV header is missing.
                    sample_rate = int(os.getenv("DEEPGRAM_TTS_SAMPLE_RATE", "24000"))
                    raw = audio_path.read_bytes()
                    data = np.frombuffer(raw, dtype=np.int16)
                    rate = sample_rate

                pad_len = int(rate * 0.12)
                if data.ndim == 1:
                    pad = np.zeros(pad_len, dtype=data.dtype)
                    data = np.concatenate([pad, data, pad])
                else:
                    pad = np.zeros((pad_len, data.shape[1]), dtype=data.dtype)
                    data = np.concatenate([pad, data, pad])
                sd.play(data, rate)
                sd.wait()
            else:
                from pydub import AudioSegment
                from pydub.playback import play

                audio = AudioSegment.from_file(str(audio_path))
                # Add short silence padding to avoid clipped start/end
                pad = AudioSegment.silent(duration=120)
                audio = pad + audio + pad
                audio = audio.fade_in(20).fade_out(40)
                play(audio)

            if audio_path.exists():
                audio_path.unlink()

            return True
        except Exception as e:
            print(f"Playback error: {e}")
            return False

# Prefer GROQ_API_KEY, fall back to GROQ_API_KEY2 if needed
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY", os.getenv("GROQ_API_KEY2", ""))

model_settings = {
    "temperature": 0.2,
    "max_tokens": 1024,
}
OUTPUT_RETRIES = 3
SESSION_RESTORE_MODE = os.getenv("SESSION_RESTORE_MODE", "strict_new")
ACTIVE_HISTORY_MAX_TURNS = int(os.getenv("TURTLE_HISTORY_MAX_TURNS", os.getenv("ACTIVE_HISTORY_MAX_TURNS", "12")))
ACTIVE_HISTORY_MAX_MESSAGES = int(os.getenv("ACTIVE_HISTORY_MAX_MESSAGES", "40"))
ACTIVE_HISTORY_MAX_TOKENS = int(os.getenv("TURTLE_HISTORY_MAX_TOKENS", "4000"))
MEMORY_FLUSH_TURNS = int(os.getenv("TURTLE_MEMORY_FLUSH_TURNS", "8"))
MEMORY_FLUSH_TOKENS = int(os.getenv("TURTLE_MEMORY_FLUSH_TOKENS", "6000"))
MEMORY_PROFILE_MAX_LINES = int(os.getenv("TURTLE_MEMORY_PROFILE_MAX_LINES", "6"))
PERSONAL_MEMORY_ENABLED = os.getenv("TURTLE_PERSONAL_MEMORY_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
PERSONAL_MEMORY_MAX_BYTES = int(os.getenv("TURTLE_PERSONAL_MEMORY_MAX_BYTES", "2048"))
PERSONAL_MEMORY_MAX_TOPIC_FILES = int(os.getenv("TURTLE_PERSONAL_MEMORY_MAX_TOPIC_FILES", "2"))
PERSONAL_MEMORY_STAGE_B_ENABLED = os.getenv("TURTLE_PERSONAL_MEMORY_STAGE_B_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
PERSONAL_MEMORY_STAGE_B_MAX_TURNS = int(os.getenv("TURTLE_PERSONAL_MEMORY_STAGE_B_MAX_TURNS", "60"))
PERSONAL_MEMORY_STAGE_B_MAX_CANDIDATES = int(os.getenv("TURTLE_PERSONAL_MEMORY_STAGE_B_MAX_CANDIDATES", "8"))
PERSONAL_MEMORY_DREAM_PASS_ENABLED = os.getenv("TURTLE_PERSONAL_MEMORY_DREAM_PASS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
INTERACTION_MODE = os.getenv("TURTLE_INTERACTION_MODE", "ask").strip().lower()
TOOL_OUTPUT_MAX_CHARS = int(os.getenv("TURTLE_TOOL_OUTPUT_MAX_CHARS", "3500"))


def _is_user_turn_request(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) for part in message.parts
    )


def _trim_history_for_context(history: list[ModelMessage]) -> list[ModelMessage]:
    if len(history) <= ACTIVE_HISTORY_MAX_MESSAGES:
        approx_tokens = sum(len(str(message)) // 4 for message in history)
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

    while trimmed and sum(len(str(message)) // 4 for message in trimmed) > ACTIVE_HISTORY_MAX_TOKENS:
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


def _parse_confirmation_answer(user_text: str) -> bool | None:
    text = " ".join(str(user_text).strip().lower().split())
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
        return f"Quick check: {prompt.question} Please answer yes or no, or say 'show me' to see what I'd save."

    for event_id in prompt.all_event_ids:
        state.confirmation_gate.record_response(event_id, accepted=accepted)
    if accepted:
        if len(prompt.all_event_ids) > 1:
            return "Got it. I will remember those."
        return "Got it. I will remember that."
    if len(prompt.all_event_ids) > 1:
        return "Understood. I will not store those."
    return "Understood. I will not store that preference."


def _normalize_behavior_event_for_gate(event: dict[str, object]) -> tuple[str, str, dict[str, object]] | None:
    key = str(event.get("key", "")).strip()
    value = event.get("value", {})
    if not key or not isinstance(value, dict):
        return None

    if key == "workflow.email_interaction":
        try:
            count = int(value.get("count", 1) or 1)
        except Exception:
            count = 1
        return ("workflow", "workflow.email_interactions_recorded", {"count": count})

    if key == "workflow.common_recipient":
        recipient = str(value.get("recipient", "")).strip().lower()
        if not recipient:
            return None
        try:
            count = int(value.get("count", 1) or 1)
        except Exception:
            count = 1
        return (
            "contacts",
            f"contacts.frequent_recipient.{recipient}",
            {"email": recipient, "count": count},
        )

    return None


def _queue_non_explicit_behavior_candidates(
    state: SharedState,
    *,
    session_id: str,
    turn_id: str,
    user_text: str,
    task_type: str,
) -> int:
    profile = state.personal_memory_store.load_profile_snapshot()
    event_specs = extract_memory_event_specs(
        user_text=user_text,
        task_type=task_type,
        profile=profile,
        mode="deterministic",
    )

    queued = 0
    for index, event in enumerate(event_specs):
        kind = str(event.get("kind", "")).strip().lower()
        source = str(event.get("source", "")).strip().lower()
        if kind != "behavior" or source == "explicit":
            continue

        try:
            confidence = float(event.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        # Keep deterministic noise out of the queue; Stage B/C candidates are expected to be stronger.
        if confidence < 0.7:
            continue

        normalized = _normalize_behavior_event_for_gate(event)
        if normalized is None:
            continue
        topic, key, value = normalized

        stable_input = (
            f"{session_id}|{turn_id}|{key}|"
            f"{json.dumps(value, ensure_ascii=False, sort_keys=True)}"
        )
        event_id = f"cand_{hashlib.sha1(stable_input.encode('utf-8')).hexdigest()[:20]}"
        source_value = source if source in {"inferred", "synthesized"} else "inferred"

        candidate = make_event(
            event_id=event_id,
            kind="behavior",
            topic=topic,
            key=key,
            value=value,
            confidence=confidence,
            source=source_value,
            extractor="deterministic",
            session_id=session_id,
            turn_id=f"{turn_id}_mem_{index}",
            evidence={
                "user_text": user_text,
                "observation_count": 1,
            },
            applied=False,
        )

        if state.confirmation_gate.queue_candidate(candidate):
            queued += 1

    return queued


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
        event_specs = extract_memory_event_specs(
            user_text=user_text,
            task_type=task_type,
            profile=profile,
            mode="deterministic",
        )
        # Build a minimal single-message history so we can reuse
        # extract_memory_candidates_from_messages with its dedup logic.
        fake_msg = ModelRequest(parts=[UserPromptPart(content=user_text)])
        candidates = extract_memory_candidates_from_messages(
            message_history=[fake_msg],
            session_id=session_id,
            profile=profile,
        )
        # Only keep explicit high-confidence facts/preferences — behaviors go to the gate
        explicit_candidates = [
            c for c in candidates
            if c.source == "explicit" and c.topic in {
                "identity", "preferences", "workflow", "contacts", "projects", "corrections"
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
            print(f"LOG: Per-turn memory applied: {result.written_topics}")
    except Exception as e:
        print(f"LOG: Per-turn fact extraction failed for {session_id}: {e}")


def _candidate_to_journal_event(
    *,
    candidate: PersonalMemoryCandidate,
    session_id: str,
    ordinal: int,
) -> object | None:
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
    elif topic == "identity" and key == "primary_email":
        event_key = "identity.primary_email"
        event_value = {"primary_email": value_lower}
    elif topic == "identity" and key.startswith("known_email:"):
        email = key.split(":", 1)[1].strip().lower() or value_lower
        if not email:
            return None
        event_key = f"identity.known_email.{email}"
        event_value = {"email": email}
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
        event_value = {
            "prefers_draft_before_send": value_lower in {"true", "1", "yes", "y"}
        }
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
    # Primary path: retrieval broker (Step 7 — token-budgeted, tier-ordered).
    if PERSONAL_MEMORY_ENABLED and state.retrieval_broker is not None:
        try:
            block = await state.retrieval_broker.build_context(
                task_type=task_type,
                query=user_text,
            )
            if block:
                return block
        except Exception as e:
            print(f"LOG: Retrieval broker failed, using prompt-builder fallback: {e}")

    # First fallback: existing PersonalMemoryPromptBuilder (no episodic/task tiers).
    if PERSONAL_MEMORY_ENABLED:
        try:
            personal_block = state.personal_memory_prompt.build_memory_block(
                task_type=task_type,
                query=user_text,
            )
            if personal_block:
                return personal_block
        except Exception as e:
            print(f"LOG: Personal memory prompt build failed, using compatibility fallback: {e}")

    # Deprecated fallback: JSON profile + graph-derived context.
    fallback_lines = state.memory_store.get_context_lines(task_type=task_type, query=user_text)
    return "\n".join(fallback_lines).strip()


def _sync_personal_memory_from_messages(
    state: SharedState,
    *,
    session_id: str | None,
    message_history: list[ModelMessage],
) -> None:
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
            state.personal_memory_store.append_daily_log(
                f"Replayed personal memory topics: {topics}",
                session_id=session_id,
            )
            print(
                f"LOG: Personal memory updated for {session_id} "
                f"({len(events)} events -> {replay_result.resolved_event_count} resolved entries across {topics})"
            )
            if queued_candidates:
                print(
                    f"LOG: Queued {queued_candidates} inferred memory candidate(s) for confirmation"
                )
    except Exception as e:
        print(f"LOG: Personal memory sync failed for {session_id}: {e}")


from core.personal_memory_extract import (
    run_stage_b_session_extractor as _shared_run_stage_b_session_extractor,
)


async def _run_stage_b_session_extractor(
    state: SharedState,
    *,
    session_id: str,
    message_history: list[ModelMessage],
) -> int:
    return await _shared_run_stage_b_session_extractor(
        state,
        session_id=session_id,
        message_history=message_history,
        model_settings=model_settings,
    )



async def _run_dream_pass_if_needed(
    state: SharedState,
    *,
    session_id: str,
) -> None:
    """Run Stage C (dream pass) at session end if trigger conditions are met.

    Guards: PERSONAL_MEMORY_ENABLED + PERSONAL_MEMORY_DREAM_PASS_ENABLED flags.
    If Groq is unavailable the pass is skipped silently.
    """
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

    groq_model = get_groq_model(model_name="openai/gpt-oss-120b", settings=model_settings)
    if groq_model is None:
        print(f"LOG: Dream pass skipped for {session_id} (Groq unavailable)")
        return

    result = await dream_pass.run(session_id=session_id, model=groq_model)
    if result.skipped_reason:
        print(f"LOG: Dream pass skipped for {session_id}: {result.skipped_reason}")
    if result.rolled_back:
        print(
            f"LOG: Dream pass rolled back for {session_id}: "
            f"{result.sanity_failures}"
        )


async def _sync_personal_memory_from_archive(
    state: SharedState,
    *,
    session_id: str | None,
    archive_path: Path,
) -> None:
    if not PERSONAL_MEMORY_ENABLED or not session_id:
        return

    messages_path = archive_path / "messages.json"
    if not messages_path.exists():
        return

    try:
        message_history = ModelMessagesTypeAdapter.validate_json(messages_path.read_bytes())
    except Exception as e:
        print(f"LOG: Unable to read archived messages for personal memory sync {session_id}: {e}")
        return

    _sync_personal_memory_from_messages(
        state,
        session_id=session_id,
        message_history=message_history,
    )
    try:
        await _run_stage_b_session_extractor(
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


def _record_task_history(
    state: SharedState,
    *,
    turn_id: str,
    task_type: str,
    status: str,
    query: str = "",
    tool_used: str = "",
    outcome: str = "",
    payload: dict[str, object] | None = None,
) -> None:
    session_id = state.session_store.session_id or "unknown_session"
    try:
        state.task_history_store.record(
            session_id=session_id,
            turn_id=turn_id,
            task_type=task_type,
            status=status,
            query=query,
            tool_used=tool_used,
            outcome=outcome,
            payload=payload,
        )
    except Exception as e:
        print(f"LOG: Task history write failed for {session_id}: {e}")


def _new_turn_id(state: SharedState) -> str:
    state.turn_counter += 1
    return f"{state.session_store.session_id or 'session'}_turn_{state.turn_counter}"


def _truncate_tool_output(text: str, *, label: str) -> str:
    if len(text) <= TOOL_OUTPUT_MAX_CHARS:
        return text
    return (
        f"{text[:TOOL_OUTPUT_MAX_CHARS]}\n\n"
        f"[Output truncated: {label} was too long. Ask follow-up questions for specific details.]"
    )


def _normalize_url_for_cache(url: str) -> str:
    raw = " ".join(url.split())
    try:
        parsed = urlsplit(raw)
        normalized_path = parsed.path or "/"
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), normalized_path, parsed.query, ""))
    except Exception:
        return raw


def _resolve_interaction_mode() -> str:
    if INTERACTION_MODE in {"voice", "text"}:
        return INTERACTION_MODE

    if INTERACTION_MODE not in {"", "ask", "prompt"}:
        print(
            "LOG: Invalid TURTLE_INTERACTION_MODE value. "
            "Use 'voice', 'text', or 'ask'. Falling back to prompt."
        )

    while True:
        print("\nChoose interaction mode:")
        print("1. Voice")
        print("2. Text")
        choice = input("Enter 1/2 (or voice/text): ").strip().lower()
        if choice in {"1", "voice", "v"}:
            return "voice"
        if choice in {"2", "text", "t"}:
            return "text"
        print("Invalid choice. Please enter 1, 2, voice, or text.")

openrouter_models = get_openrouter_models(settings=model_settings)
if not openrouter_models:
    raise RuntimeError("No OpenRouter API keys found. Set OPEN_ROUTER_API_KEY_1/2/3 or OPENROUTER_API_KEY in .env.")

primary_groq_model = get_groq_model(settings=model_settings)
model = primary_groq_model or openrouter_models[0]  # Main assistant
model2 = primary_groq_model or openrouter_models[0]  # Email agent

openrouter_fallback_models = openrouter_models[1:]
if not openrouter_fallback_models:
    emergency_fallback_model = os.getenv("OPENROUTER_TOOL_FALLBACK_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free").strip()
    if emergency_fallback_model:
        emergency_models = get_openrouter_models(model_name=emergency_fallback_model, settings=model_settings)
        # Use emergency models only when no key-index fallbacks exist.
        if emergency_models:
            openrouter_fallback_models = emergency_models

delegator_fallback_models = openrouter_models if primary_groq_model else openrouter_fallback_models
groq_fallback_model = get_groq_fallback_model(settings=model_settings)
usage_limits = UsageLimits(request_limit=30)

# Global voice processor instance
voice_processor = None

# Email specialist agent for handling email operations
email_agent = Agent(
    model2,
    deps_type=SharedState,
    output_type=str,
    output_retries=OUTPUT_RETRIES,
    instructions=EMAIL_PROMPT,
    history_processors=[_trim_history_for_context],
)
email_agent_fallbacks: list[Agent] = []
for fallback_model in delegator_fallback_models:
    email_agent_fallbacks.append(
        Agent(
            fallback_model,
            deps_type=SharedState,
            output_type=str,
            output_retries=OUTPUT_RETRIES,
            instructions=EMAIL_PROMPT,
            history_processors=[_trim_history_for_context],
        )
    )
if groq_fallback_model:
    email_agent_fallbacks.append(
        Agent(
            groq_fallback_model,
            deps_type=SharedState,
            output_type=str,
            output_retries=OUTPUT_RETRIES,
            instructions=EMAIL_PROMPT,
            history_processors=[_trim_history_for_context],
        )
    )

# Main assistant agent with enhanced delegation
main_assistant = Agent(
    model,
    deps_type=SharedState,
    output_type=str,
    output_retries=OUTPUT_RETRIES,
    instructions=MAIN_ASSISTANT_PROMPT,
    history_processors=[_trim_history_for_context],
)
main_assistant_fallbacks: list[Agent] = []
for fallback_model in delegator_fallback_models:
    main_assistant_fallbacks.append(
        Agent(
            fallback_model,
            deps_type=SharedState,
            output_type=str,
            output_retries=OUTPUT_RETRIES,
            instructions=MAIN_ASSISTANT_PROMPT,
            history_processors=[_trim_history_for_context],
        )
    )
if groq_fallback_model:
    main_assistant_fallbacks.append(
        Agent(
            groq_fallback_model,
            deps_type=SharedState,
            output_type=str,
            output_retries=OUTPUT_RETRIES,
            instructions=MAIN_ASSISTANT_PROMPT,
            history_processors=[_trim_history_for_context],
        )
    )

@main_assistant.tool
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
        formatted = (
            f"Web search failed for query: {normalized_query}\n"
            f"Error: {e}"
        )

    cleaned = clean_text_for_model(formatted)
    trimmed = _truncate_tool_output(cleaned, label="web search results")
    ctx.deps.search_cache[cache_key] = trimmed
    return trimmed

@main_assistant.tool
async def search_url(ctx: RunContext[SharedState], url: str) -> str:
    """Analyze and extract detailed content from a URL using custom extraction tool"""
    print(f"\nANALYZING: URL content extraction from {url}")
    normalized_url = _normalize_url_for_cache(url)
    cache_key = f"url::{normalized_url}"
    cached = ctx.deps.search_cache.get(cache_key)
    if cached:
        return cached
    
    # Use our custom URL extraction tool
    result = await fetch_url_content_async(ctx.deps.http_client, normalized_url)
    
    # Return formatted string representation
    cleaned = clean_text_for_model(result.to_formatted_string())
    trimmed = _truncate_tool_output(cleaned, label="url analysis")
    ctx.deps.search_cache[cache_key] = trimmed
    return trimmed

@main_assistant.tool
async def send_email_assistant(ctx: RunContext[SharedState], query: str) -> str:
    """Send emails using the email specialist agent. Pass the complete user request about sending emails."""
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
        email_agent,
        email_agent_fallbacks,
        extraction_prompt,
        deps=ctx.deps,
        usage=ctx.usage,
    )
    llm_extraction = parse_email_extraction_response(extraction_result.output).model_dump()
    latest_fields = combine_extracted_email_details(deterministic, llm_extraction)
    merged = merge_email_details(pending_email, latest_fields)

    valid_recipients, invalid_recipients = validate_recipients(merged["recipients"])
    valid_cc_recipients, invalid_cc_recipients = validate_recipients(merged.get("cc_recipients", []))
    valid_bcc_recipients, invalid_bcc_recipients = validate_recipients(merged.get("bcc_recipients", []))
    merged["recipients"] = valid_recipients
    merged["cc_recipients"] = valid_cc_recipients
    merged["bcc_recipients"] = valid_bcc_recipients

    if invalid_recipients or invalid_cc_recipients or invalid_bcc_recipients:
        ctx.deps.session_store.set_pending_email(
            recipients=valid_recipients,
            cc_recipients=valid_cc_recipients,
            bcc_recipients=valid_bcc_recipients,
            subject=merged["subject"],
            content=merged["content"],
        )
        invalid_parts: list[str] = []
        if invalid_recipients:
            invalid_parts.append(f"to: {', '.join(invalid_recipients)}")
        if invalid_cc_recipients:
            invalid_parts.append(f"cc: {', '.join(invalid_cc_recipients)}")
        if invalid_bcc_recipients:
            invalid_parts.append(f"bcc: {', '.join(invalid_bcc_recipients)}")
        invalid_text = "; ".join(invalid_parts)
        return clean_text_for_model(
            (
            f"I found invalid email format: {invalid_text}. "
            "Please provide the address again."
            )
        )

    missing = missing_email_fields(merged)
    if missing:
        ctx.deps.session_store.set_pending_email(
            recipients=merged["recipients"],
            cc_recipients=merged["cc_recipients"],
            bcc_recipients=merged["bcc_recipients"],
            subject=merged["subject"],
            content=merged["content"],
        )
        return clean_text_for_model(format_missing_email_prompt(missing, merged))

    try:
        validate_send_email_args(
            merged["recipients"],
            merged["subject"],
            merged["content"],
            merged["cc_recipients"],
            merged["bcc_recipients"],
        )
        send_result = send_email_now(merged)
    except Exception as e:
        tool_turn_id = f"{ctx.deps.session_store.session_id or 'session'}_tool_{ctx.deps.turn_counter}"
        _record_task_history(
            ctx.deps,
            turn_id=tool_turn_id,
            task_type="email",
            status="failed",
            query=query,
            tool_used="send_email_now",
            outcome=str(e),
            payload={
                "recipients": merged["recipients"],
                "subject": merged["subject"],
            },
        )
        ctx.deps.session_store.set_pending_email(
            recipients=merged["recipients"],
            cc_recipients=merged["cc_recipients"],
            bcc_recipients=merged["bcc_recipients"],
            subject=merged["subject"],
            content=merged["content"],
        )
        return clean_text_for_model(f"Failed to send email: {e}")
    if send_result.startswith("Email sent successfully!"):
        tool_turn_id = f"{ctx.deps.session_store.session_id or 'session'}_tool_{ctx.deps.turn_counter}"
        _record_task_history(
            ctx.deps,
            turn_id=tool_turn_id,
            task_type="email",
            status="completed",
            query=query,
            tool_used="send_email_now",
            outcome=f"Sent email to {', '.join(merged['recipients'])} with subject {merged['subject']}",
            payload={
                "recipients": merged["recipients"],
                "subject": merged["subject"],
                "content_length": len(merged["content"]),
            },
        )
        ctx.deps.session_store.clear_pending_email()
    else:
        ctx.deps.session_store.set_pending_email(
            recipients=merged["recipients"],
            cc_recipients=merged["cc_recipients"],
            bcc_recipients=merged["bcc_recipients"],
            subject=merged["subject"],
            content=merged["content"],
        )
    return clean_text_for_model(send_result)

@main_assistant.tool
async def history_tool(ctx: RunContext[SharedState], query: str) -> str:
    """Search conversation history for past discussions and information"""
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

    except Exception:
        return ToolResult.upstream_error("History lookup failed.").to_agent_string()


@main_assistant.tool
async def recall(ctx: RunContext[SharedState], query: str, scope: str) -> str:
    """Recall personal, episodic, task, or working context."""
    query_text = str(query or "").strip()
    scope_text = str(scope or "").strip().lower()
    if not query_text:
        return ToolResult.invalid("query must not be empty").to_agent_string()
    if scope_text not in {"personal", "episodic", "tasks", "working"}:
        return ToolResult.invalid("scope must be personal, episodic, tasks, or working").to_agent_string()
    broker = ctx.deps.retrieval_broker
    if broker is None:
        return ToolResult.empty("Recall is not available.").to_agent_string()
    try:
        recall_text = await broker.recall(
            query=query_text,
            scope=scope_text,
            message_history=ctx.deps.session_store.message_history,
            trim_fn=_trim_history_for_context,
        )
    except Exception:
        return ToolResult.upstream_error("Recall failed.").to_agent_string()
    if not recall_text:
        return ToolResult.empty("No relevant information found.").to_agent_string()
    return ToolResult.ok(recall_text).to_agent_string()


async def text_chat(state: SharedState, return_to_voice: bool = True):
    """Text chat mode for interactive typing-based conversations"""
    print("\n" + "="*50)
    print("SWITCHED TO TEXT MODE")
    print("Type your messages and press Enter")
    print("Type 'quit', 'exit', 'bye', or 'voice mode' to exit")
    print("="*50 + "\n")
    
    try:
        message_history: list[ModelMessage] | None = state.session_store.message_history or None
        usage = RunUsage()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    if return_to_voice:
                        print("Turtle: Goodbye! Returning to voice mode...")
                    else:
                        print("Turtle: Goodbye!")
                    break
                elif user_input.lower() in ['voice mode', 'switch to voice mode', 'run voice mode']:
                    if return_to_voice:
                        print("Turtle: Switching back to voice mode...")
                        break
                    print("Turtle: Voice mode is not active in this session.")
                    continue
                
                if not user_input:
                    continue

                confirmation_reply = _maybe_handle_confirmation_turn(state, user_input)
                if confirmation_reply:
                    print(f"Turtle: {confirmation_reply}")
                    continue

                task_type = _detect_task_type(user_input)
                memory_context = await _resolve_memory_context(state, task_type=task_type, user_text=user_input)
                prompt_input = _compose_prompt_with_memory(user_input, memory_context)
                turn_id = _new_turn_id(state)
                
                # Run with complete message history for conversation continuity
                response = await run_agent_with_fallbacks(
                    main_assistant,
                    main_assistant_fallbacks,
                    prompt_input,
                    deps=state,
                    message_history=message_history,
                    usage=usage,
                    usage_limits=usage_limits
                )
                final_output = clean_text_for_model(response.output)

                print(f"Turtle: {final_output}")
                
                # Update message history with complete conversation for continuity
                message_history = response.all_messages()
                state.session_store.replace_messages(message_history)
                
                # Add conversation to RAG system for long-term memory
                state.rag_system.add_conversation(user_input, final_output)
                state.memory_store.record_turn(
                    session_id=state.session_store.session_id or "unknown_session",
                    turn_id=turn_id,
                    user_text=user_input,
                    assistant_text=final_output,
                    task_type=task_type,
                )
                queued = _queue_non_explicit_behavior_candidates(
                    state,
                    session_id=state.session_store.session_id or "unknown_session",
                    turn_id=turn_id,
                    user_text=user_input,
                    task_type=task_type,
                )
                if queued:
                    print(f"LOG: Queued {queued} memory confirmation candidate(s)")
                _apply_explicit_facts_from_turn(
                    state,
                    session_id=state.session_store.session_id or "unknown_session",
                    turn_id=turn_id,
                    user_text=user_input,
                    task_type=task_type,
                )
                if state.reflector is not None:
                    await state.reflector.on_turn(
                        state,
                        session_id=state.session_store.session_id or "",
                        message_history=message_history or [],
                        dream_pass_runner=_run_dream_pass_if_needed,
                    )

                # Show usage information periodically
                if usage.requests % 5 == 0 and usage.requests > 0:
                    print(f"\n[Usage: {usage.requests} requests, {usage.total_tokens} tokens]")
                
            except KeyboardInterrupt:
                if return_to_voice:
                    print("\nTurtle: Returning to voice mode...")
                else:
                    print("\nTurtle: Exiting text mode...")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Let's try again...")
                # Message history is preserved even after errors
                
    except Exception as e:
        print(f"Text mode error: {e}")


async def voice_response_handler(audio: Tuple[int, np.ndarray], state: SharedState) -> bool:
    """Handle voice input and generate response using main assistant
    
    Returns:
        bool: True if mode switch to text is requested, False otherwise
    """
    global voice_processor
    
    # Process voice input to get transcription
    transcription = await voice_processor.process_voice_input(audio)
    
    if not transcription:
        return False
    
    # Check for mode switch commands
    switch_phrases = [
        'switch to text mode', 'run text mode', 'text mode', 
        'switch to text', 'run text', 'text chat',
        'use text mode', 'start text mode'
    ]
    
    transcription_lower = transcription.lower().strip()
    for phrase in switch_phrases:
        if phrase in transcription_lower:
            print(f"Turtle: Sure! Switching to text mode now...")
            return True
    
    try:
        confirmation_reply = _maybe_handle_confirmation_turn(state, transcription)
        if confirmation_reply:
            print(f"Turtle: {confirmation_reply}")
            speech_file = voice_processor.text_to_speech(clean_text_for_tts(confirmation_reply))
            if speech_file and speech_file.exists():
                voice_processor.play_audio(speech_file)
            return False

        # Get response from main assistant
        task_type = _detect_task_type(transcription)
        memory_context = await _resolve_memory_context(state, task_type=task_type, user_text=transcription)
        prompt_input = _compose_prompt_with_memory(transcription, memory_context)
        turn_id = _new_turn_id(state)

        llm_start_time = time.time()
        response = await run_agent_with_fallbacks(
            main_assistant,
            main_assistant_fallbacks,
            prompt_input,
            deps=state,
            message_history=state.session_store.message_history or None,
            usage=RunUsage()
        )
        llm_time = time.time() - llm_start_time
        final_output = clean_text_for_model(response.output)
        
        print(f"LOG: LLM response generated in {llm_time:.2f}s")
        print(f"Turtle: {final_output}")
        state.session_store.replace_messages(response.all_messages())

        # Add conversation to RAG system for memory
        state.rag_system.add_conversation(transcription, final_output)
        state.memory_store.record_turn(
            session_id=state.session_store.session_id or "unknown_session",
            turn_id=turn_id,
            user_text=transcription,
            assistant_text=final_output,
            task_type=task_type,
        )
        queued = _queue_non_explicit_behavior_candidates(
            state,
            session_id=state.session_store.session_id or "unknown_session",
            turn_id=turn_id,
            user_text=transcription,
            task_type=task_type,
        )
        if queued:
            print(f"LOG: Queued {queued} memory confirmation candidate(s)")
        _apply_explicit_facts_from_turn(
            state,
            session_id=state.session_store.session_id or "unknown_session",
            turn_id=turn_id,
            user_text=transcription,
            task_type=task_type,
        )
        if state.reflector is not None:
            await state.reflector.on_turn(
                state,
                session_id=state.session_store.session_id or "",
                message_history=state.session_store.message_history or [],
                dream_pass_runner=_run_dream_pass_if_needed,
            )

        # Generate and play TTS
        tts_start_time = time.time()
        speech_file = voice_processor.text_to_speech(clean_text_for_tts(final_output))
        tts_generation_time = time.time() - tts_start_time
        if speech_file and speech_file.exists():
            print(f"LOG: TTS generation completed in {tts_generation_time:.2f}s")
            playback_start = time.time()
            playback_success = voice_processor.play_audio(speech_file)
            playback_time = time.time() - playback_start
            if not playback_success:
                print("LOG: Audio playback failed")
            else:
                print(f"LOG: Audio playback completed in {playback_time:.2f}s")
        else:
            print("LOG: TTS generation failed")

        return False
        
    except Exception as e:
        print(f"LOG: LLM processing error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def voice_chat():
    """Voice chat mode using FastRTC VAD and STT with main assistant"""
    global voice_processor
    
    async with httpx.AsyncClient() as client:
        session_store = SessionStore()
        restore_result = session_store.start_or_restore(mode=SESSION_RESTORE_MODE)
        # Deprecated graph dependency for legacy MemoryStore fallback only.
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
        rag_system = TurtleRAGSystem(user_id="local_voice_user")
        retrieval_broker = RetrievalBroker(
            store=personal_memory_store,
            task_store=task_history_store,
            journal_store=journal_store,
            session_store=session_store,
            rag_system=rag_system,
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
        )
        _register_shutdown_state(state)
        voice_processor = TurtleVoiceProcessor(state)
        if restore_result.had_corrupt_active:
            print("LOG: Corrupt active session files were quarantined before starting a new session")

        for pending_session_id, pending_archive_path in session_store.list_pending_finalization_archives():
            print(f"LOG: Finalizing archived session {pending_session_id}")
            await _sync_personal_memory_from_archive(
                state,
                session_id=pending_session_id,
                archive_path=pending_archive_path,
            )
            finalized_pending = await rag_system.finalize_archived_session(
                session_id=pending_session_id,
                archive_path=pending_archive_path,
            )
            if finalized_pending:
                session_manifest = pending_archive_path / "session.json"
                if session_manifest.exists():
                    try:
                        manifest = json.loads(session_manifest.read_text(encoding="utf-8"))
                        manifest["status"] = "completed"
                        session_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                print(f"LOG: Archived session {pending_session_id} finalized")
            else:
                print(
                    f"LOG: Archived session {pending_session_id} still pending finalization "
                    f"at {pending_archive_path}"
                )

        await rag_system.start_session(session_id=restore_result.session_id)
        if restore_result.restored:
            print(
                f"LOG: Restored session {restore_result.session_id} "
                f"with {restore_result.message_count} messages"
            )
        print("Manual audio recording with VAD processing")
        print("LOG: Setting up manual audio recording with VAD")
        
        import pyaudio
        import keyboard
        
        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        print("Hold SPACE to record, release to stop (Press Ctrl+C to quit)")
        print("LOG: Ready for voice input")
        
        try:
            while True:
                try:
                    print("\nWaiting for SPACE key...")
                    
                    # Wait for space key press
                    keyboard.wait('space')
                    
                    # Hold SPACE to record, release to stop
                    print("Recording... (release SPACE to stop)")
                    frames = []
                    
                    # Start recording
                    stream = p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK)
                    
                    recording_start = time.time()
                    
                    # Record while space is held
                    while keyboard.is_pressed('space'):
                        data = stream.read(CHUNK)
                        frames.append(data)
                        
                        # Safety limit - stop after 15 seconds max
                        if time.time() - recording_start > 15:
                            print("Maximum recording time reached")
                            break
                    
                    stream.stop_stream()
                    stream.close()
                    
                    recording_time = time.time() - recording_start
                    print(f"LOG: Recorded {recording_time:.2f}s of audio")
                    
                    # Process audio
                    if len(frames) > 0:
                        # Convert audio to numpy array for processing
                        audio_data = b''.join(frames)
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        
                        # Process through voice handler
                        print("LOG: Processing audio through VAD")
                        
                        audio_tuple = (RATE, audio_array)
                        
                        # Process the audio through our voice handler
                        should_switch_to_text = await voice_response_handler(audio_tuple, state)
                        
                        # If mode switch is requested, enter text mode
                        if should_switch_to_text:
                            # Cleanup current voice session
                            stream.stop_stream()
                            stream.close()
                            p.terminate()
                            print("LOG: Audio system paused for text mode")
                            
                            # Enter text mode
                            await text_chat(state)
                            
                            # Restart voice mode after text chat ends
                            print("\nReturning to voice mode...")
                            print("Hold SPACE to record, release to stop (Press Ctrl+C to quit)")
                            print("LOG: Ready for voice input")
                            
                            # Reinitialize PyAudio
                            p = pyaudio.PyAudio()
                            continue
                    
                except KeyboardInterrupt:
                    print("\nLOG: Received Ctrl+C")
                    break
                except Exception as e:
                    print(f"LOG: Error during recording: {e}")
                    continue
        
        finally:
            try:
                p.terminate()
                print("LOG: Audio system terminated")
            except:
                pass
            
            # Cleanup
            if voice_processor:
                voice_processor.cleanup()

            if state.session_store.session_id:
                state.memory_store.force_checkpoint(
                    session_id=state.session_store.session_id,
                    reason="session_end",
                )

            session_id = state.session_store.session_id
            archive_path = state.session_store.archive_active(status="pending_finalization")
            if session_id and archive_path:
                await _sync_personal_memory_from_archive(
                    state,
                    session_id=session_id,
                    archive_path=archive_path,
                )
                rag_finalized = await rag_system.finalize_archived_session(
                    session_id=session_id,
                    archive_path=archive_path,
                )
                if rag_finalized:
                    session_manifest = archive_path / "session.json"
                    if session_manifest.exists():
                        try:
                            manifest = json.loads(session_manifest.read_text(encoding="utf-8"))
                            manifest["status"] = "completed"
                            session_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                else:
                    print(
                        f"LOG: Session {session_id} archived but still pending finalization "
                        f"at {archive_path}"
                    )
            _unregister_shutdown_state(state)



async def text_mode_chat():
    """Text-only startup mode that skips voice stack initialization."""
    async with httpx.AsyncClient() as client:
        session_store = SessionStore()
        restore_result = session_store.start_or_restore(mode=SESSION_RESTORE_MODE)
        # Deprecated graph dependency for legacy MemoryStore fallback only.
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
        rag_system = TurtleRAGSystem(user_id="local_voice_user")
        retrieval_broker = RetrievalBroker(
            store=personal_memory_store,
            task_store=task_history_store,
            journal_store=journal_store,
            session_store=session_store,
            rag_system=rag_system,
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
        )
        _register_shutdown_state(state)
        if restore_result.had_corrupt_active:
            print("LOG: Corrupt active session files were quarantined before starting a new session")

        for pending_session_id, pending_archive_path in session_store.list_pending_finalization_archives():
            print(f"LOG: Finalizing archived session {pending_session_id}")
            await _sync_personal_memory_from_archive(
                state,
                session_id=pending_session_id,
                archive_path=pending_archive_path,
            )
            finalized_pending = await rag_system.finalize_archived_session(
                session_id=pending_session_id,
                archive_path=pending_archive_path,
            )
            if finalized_pending:
                session_manifest = pending_archive_path / "session.json"
                if session_manifest.exists():
                    try:
                        manifest = json.loads(session_manifest.read_text(encoding="utf-8"))
                        manifest["status"] = "completed"
                        session_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                print(f"LOG: Archived session {pending_session_id} finalized")
            else:
                print(
                    f"LOG: Archived session {pending_session_id} still pending finalization "
                    f"at {pending_archive_path}"
                )

        await rag_system.start_session(session_id=restore_result.session_id)
        if restore_result.restored:
            print(
                f"LOG: Restored session {restore_result.session_id} "
                f"with {restore_result.message_count} messages"
            )

        try:
            await text_chat(state, return_to_voice=False)
        finally:
            if state.session_store.session_id:
                state.memory_store.force_checkpoint(
                    session_id=state.session_store.session_id,
                    reason="session_end",
                )
            session_id = state.session_store.session_id
            archive_path = state.session_store.archive_active(status="pending_finalization")
            if session_id and archive_path:
                await _sync_personal_memory_from_archive(
                    state,
                    session_id=session_id,
                    archive_path=archive_path,
                )
                rag_finalized = await rag_system.finalize_archived_session(
                    session_id=session_id,
                    archive_path=archive_path,
                )
                if rag_finalized:
                    session_manifest = archive_path / "session.json"
                    if session_manifest.exists():
                        try:
                            manifest = json.loads(session_manifest.read_text(encoding="utf-8"))
                            manifest["status"] = "completed"
                            session_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                else:
                    print(
                        f"LOG: Session {session_id} archived but still pending finalization "
                        f"at {archive_path}"
                    )
            _unregister_shutdown_state(state)


async def main():
    print("Welcome to Turtle Assistant!")
    print("Say 'switch to text mode' or 'run text mode' to switch to typing mode")
    mode = _resolve_interaction_mode()
    print(f"LOG: Starting Turtle in {mode} mode")

    try:
        if mode == "text":
            await text_mode_chat()
        else:
            await voice_chat()
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
        # RAG session cleanup is handled in voice_chat function
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(main())
