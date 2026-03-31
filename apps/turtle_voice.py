"""
Turtle - Personal Assistant with Web Search and URL Context Capabilities

Enhanced assistant with real-time web search, URL analysis, and conversation memory.
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Generator, Tuple
import httpx
import time
from pathlib import Path
import json
import numpy as np
from pydantic_ai import Agent, RunContext
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
from rag.system.complete_rag import get_rag_system
from core.llm_client import (
    get_groq_model,
    get_openrouter_models,
    get_groq_fallback_model,
    run_agent_with_fallbacks,
)
from core.email_flow import (
    extract_deterministic_email_details,
    format_missing_email_prompt,
    merge_email_details,
    missing_email_fields,
    parse_email_extraction_response,
    sanitize_email_details,
    send_email_now,
    validate_recipients,
    validate_send_email_args,
)
from core.output_clean import clean_text_for_model, clean_text_for_tts
from core.graph_store import GraphStore
from core.memory_store import MemoryStore
from core.paths import (
    MEMORY_EPISODES_FILE,
    MEMORY_EVENTS_FILE,
    MEMORY_GRAPH_FILE,
    MEMORY_PROFILE_FILE,
    MEMORY_STATE_FILE,
    TEMP_AUDIO_DIR,
    ensure_dirs,
)
from core.session_store import SessionStore
from core.system_prompts import load_prompt
from core.openrouter_tts import synthesize_speech
from core.stt_fastrtc import FastRTCSTT
from core.web_search import format_search_results, search_duckduckgo



load_env(override=True)
try:
    logfire.configure(send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()
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
    search_cache: dict[str, str] = field(default_factory=dict)
    turn_counter: int = 0


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
ACTIVE_HISTORY_MAX_TOKENS = int(os.getenv("TURTLE_HISTORY_MAX_TOKENS", "12000"))
MEMORY_FLUSH_TURNS = int(os.getenv("TURTLE_MEMORY_FLUSH_TURNS", "20"))
MEMORY_FLUSH_TOKENS = int(os.getenv("TURTLE_MEMORY_FLUSH_TOKENS", "20000"))
MEMORY_PROFILE_MAX_LINES = int(os.getenv("TURTLE_MEMORY_PROFILE_MAX_LINES", "6"))
INTERACTION_MODE = os.getenv("TURTLE_INTERACTION_MODE", "ask").strip().lower()


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


def _compose_prompt_with_memory(user_text: str, memory_lines: list[str]) -> str:
    if not memory_lines:
        return user_text
    context = "\n".join(memory_lines)
    return (
        "Relevant user memory:\n"
        f"{context}\n\n"
        "User request:\n"
        f"{user_text}"
    )


def _new_turn_id(state: SharedState) -> str:
    state.turn_counter += 1
    return f"{state.session_store.session_id or 'session'}_turn_{state.turn_counter}"


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
    cached = ctx.deps.search_cache.get(normalized_query)
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
    ctx.deps.search_cache[normalized_query] = cleaned
    return cleaned

@main_assistant.tool
async def search_url(ctx: RunContext[SharedState], url: str) -> str:
    """Analyze and extract detailed content from a URL using custom extraction tool"""
    print(f"\nANALYZING: URL content extraction from {url}")
    
    # Use our custom URL extraction tool
    result = await fetch_url_content_async(ctx.deps.http_client, url)
    
    # Return formatted string representation
    return clean_text_for_model(result.to_formatted_string())

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
    latest_fields = sanitize_email_details(
        {
            "recipients": deterministic.get("recipients") or llm_extraction.get("recipients", []),
            "subject": deterministic.get("subject") or llm_extraction.get("subject", ""),
            "content": deterministic.get("content") or llm_extraction.get("content", ""),
            "send_intent": bool(deterministic.get("send_intent") or llm_extraction.get("send_intent")),
        }
    )
    merged = merge_email_details(pending_email, latest_fields)

    valid_recipients, invalid_recipients = validate_recipients(merged["recipients"])
    merged["recipients"] = valid_recipients

    if invalid_recipients:
        ctx.deps.session_store.set_pending_email(
            recipients=valid_recipients,
            subject=merged["subject"],
            content=merged["content"],
        )
        invalid_text = ", ".join(invalid_recipients)
        return clean_text_for_model(
            (
            f"I found invalid recipient email format: {invalid_text}. "
            "Please provide the recipient address again."
            )
        )

    missing = missing_email_fields(merged)
    if missing:
        ctx.deps.session_store.set_pending_email(
            recipients=merged["recipients"],
            subject=merged["subject"],
            content=merged["content"],
        )
        return clean_text_for_model(format_missing_email_prompt(missing, merged))

    try:
        validate_send_email_args(
            merged["recipients"],
            merged["subject"],
            merged["content"],
        )
        send_result = send_email_now(merged)
    except Exception as e:
        tool_turn_id = f"{ctx.deps.session_store.session_id or 'session'}_tool_{ctx.deps.turn_counter}"
        ctx.deps.memory_store.record_task_outcome(
            session_id=ctx.deps.session_store.session_id or "unknown_session",
            turn_id=tool_turn_id,
            task_type="email",
            summary=str(e),
            success=False,
        )
        ctx.deps.session_store.set_pending_email(
            recipients=merged["recipients"],
            subject=merged["subject"],
            content=merged["content"],
        )
        return clean_text_for_model(f"Failed to send email: {e}")
    if send_result.startswith("Email sent successfully!"):
        tool_turn_id = f"{ctx.deps.session_store.session_id or 'session'}_tool_{ctx.deps.turn_counter}"
        ctx.deps.memory_store.record_common_recipients(
            session_id=ctx.deps.session_store.session_id or "unknown_session",
            turn_id=tool_turn_id,
            recipients=merged["recipients"],
        )
        ctx.deps.memory_store.record_task_outcome(
            session_id=ctx.deps.session_store.session_id or "unknown_session",
            turn_id=tool_turn_id,
            task_type="email",
            summary=f"Sent email to {', '.join(merged['recipients'])} with subject {merged['subject']}",
            success=True,
        )
        ctx.deps.session_store.clear_pending_email()
    else:
        ctx.deps.session_store.set_pending_email(
            recipients=merged["recipients"],
            subject=merged["subject"],
            content=merged["content"],
        )
    return clean_text_for_model(send_result)

@main_assistant.tool
async def history_tool(ctx: RunContext[SharedState], query: str) -> str:
    """Search conversation history for past discussions and information"""
    try:
        rag_system = get_rag_system()
        result = await rag_system.query_history(query)
        
        if result == "cannot find in history":
            return "No relevant information found in our previous conversations."
        else:
     â€¦73 tokens truncatedâ€¦ng-based conversations"""
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

                task_type = _detect_task_type(user_input)
                memory_lines = state.memory_store.get_context_lines(task_type=task_type, query=user_input)
                prompt_input = _compose_prompt_with_memory(user_input, memory_lines)
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
                rag_system.add_conversation(user_input, final_output)
                state.memory_store.record_turn(
                    session_id=state.session_store.session_id or "unknown_session",
                    turn_id=turn_id,
                    user_text=user_input,
                    assistant_text=final_output,
                    task_type=task_type,
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


async def voice_response_handler(audio: Tuple[int, np.ndarray], state: SharedState, rag_system) -> bool:
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
        # Get response from main assistant
        task_type = _detect_task_type(transcription)
        memory_lines = state.memory_store.get_context_lines(task_type=task_type, query=transcription)
        prompt_input = _compose_prompt_with_memory(transcription, memory_lines)
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
        rag_system.add_conversation(transcription, final_output)
        state.memory_store.record_turn(
            session_id=state.session_store.session_id or "unknown_session",
            turn_id=turn_id,
            user_text=transcription,
            assistant_text=final_output,
            task_type=task_type,
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
        )
        state = SharedState(http_client=client, session_store=session_store, memory_store=memory_store)
        voice_processor = TurtleVoiceProcessor(state)
        
        # Initialize RAG system
        rag_system = get_rag_system()
        if restore_result.had_corrupt_active:
            print("LOG: Corrupt active session files were quarantined before starting a new session")

        for pending_session_id, pending_archive_path in session_store.list_pending_finalization_archives():
            print(f"LOG: Finalizing archived session {pending_session_id}")
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
                        should_switch_to_text = await voice_response_handler(audio_tuple, state, rag_system)
                        
                        # If mode switch is requested, enter text mode
                        if should_switch_to_text:
                            # Cleanup current voice session
                            stream.stop_stream()
                            stream.close()
                            p.terminate()
                            print("LOG: Audio system paused for text mode")
                            
                            # Enter text mode
                            await text_chat(state, rag_system)
                            
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



async def text_mode_chat():
    """Text-only startup mode that skips voice stack initialization."""
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
        )
        state = SharedState(http_client=client, session_store=session_store, memory_store=memory_store)

        rag_system = get_rag_system()
        if restore_result.had_corrupt_active:
            print("LOG: Corrupt active session files were quarantined before starting a new session")

        for pending_session_id, pending_archive_path in session_store.list_pending_finalization_archives():
            print(f"LOG: Finalizing archived session {pending_session_id}")
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
            await text_chat(state, rag_system, return_to_voice=False)
        finally:
            if state.session_store.session_id:
                state.memory_store.force_checkpoint(
                    session_id=state.session_store.session_id,
                    reason="session_end",
                )
            session_id = state.session_store.session_id
            archive_path = state.session_store.archive_active(status="pending_finalization")
            if session_id and archive_path:
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
