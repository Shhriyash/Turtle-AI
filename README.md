# Turtle Personal Assistant

A multi-agent personal assistant with persistent personal memory, web search, URL analysis, email capabilities, and a web chat UI - built with Pydantic AI.

## Features

- **Personal Memory**: 3-stage pipeline (extraction → confirmation gate → Dream Pass) with journal-backed topic files
- **Conversation Memory**: RAG-based system for cross-session conversation history
- **Web Chat UI**: Browser-based chat with WebSocket backend, voice input/output, and dev mode
- **Web Search**: Real-time information retrieval using DuckDuckGo
- **URL Analysis**: Custom content extraction from web pages
- **Email Integration**: Automated email sending with professional formatting
- **Multi-Agent Architecture**: Specialized agents for different tasks
- **LLM Fallback**: OpenRouter key rotation plus optional Groq primary/fallback models

---

## Architecture

### Core Components

```
Main Assistant (Pydantic AI + core/llm_client model selection)
|-- Web Search Tool (DuckDuckGo HTML + parser)
|-- Email Specialist Agent
|-- Custom URL Tools (BeautifulSoup + httpx)
|-- RAG Memory System (FAISS + Cohere embeddings)
`-- Personal Memory System (journal + topic markdown files)
```

---

## Personal Memory System

Three-stage pipeline: **deterministic extraction → user confirmation gate → batch LLM review (Dream Pass)**. All state is stored as append-only journal events replayed into human-readable markdown topic files.

The journal + replay design gives a full audit trail, rollback safety, and idempotent writes. The confirmation gate prevents hallucinated facts from landing silently.

### Components

| Component | File | Role |
|-----------|------|------|
| `PersonalMemoryStore` | `core/personal_memory_store.py` | Markdown-backed topic files (identity, preferences, workflow, contacts, projects, corrections) |
| `JournalStore` | `core/memory_journal.py` | Append-only JSONL event log, sharded by month |
| `ConfirmationGate` | `core/confirmation_gate.py` | Queues candidates, tracks user yes/no, enforces 14-day silence windows |
| `DreamPass` | `core/dream_pass.py` | Batch LLM review of pending candidates (Stage C) |
| `replay()` | `core/memory_replayer.py` | Projects journal events → markdown topic files deterministically |
| `PersonalMemoryPromptBuilder` | `core/personal_memory_prompt.py` | Selects up to 2 relevant topic files and injects into model context |
| `extract_memory_candidates_from_messages()` | `core/personal_memory_extract.py` | Regex + pattern extraction from message history |

### Three-Stage Pipeline

**Stage A - Per-Turn Extraction (Deterministic)**
- Trigger: Every message, after agent response
- Regex/pattern extraction → `PersonalMemoryCandidate` objects
- Writes `MemoryEvent(applied=False)` to journal, queues in `ConfirmationGate`
- If not silenced: user shown "I noticed X - want me to remember that?"

**Stage B - Session-End Extraction (Deterministic)**
- Trigger: Session finalization / archive sync
- Full message history re-extracted; unconfirmed candidates queued
- All `applied=True` events replayed → topic markdown updated

**Stage C - Dream Pass (Batch LLM, optional)**
- Trigger: Session end + per-turn after turn 8+
- Run conditions: `pending_count >= 3` OR `>= 24h` since last pass with ≥1 pending
- Process: snapshot → collect unapplied candidates → LLM promotes or drops each → `replay()` → sanity checks
- Failure: rollback from snapshot
- Enable: `TURTLE_PERSONAL_MEMORY_DREAM_PASS_ENABLED=1` (default OFF)

### Confirmation Gate Flow

```
Extract → MemoryEvent(applied=False) → ConfirmationGate.queue_candidate()
         ↓
Gate: not silenced (14-day window) + not explicit?
         ↓ Yes
Show prompt: "I noticed X - remember that?"
         ↓
User accepts → MemoryEvent(applied=True) supersedes candidate → replay() → topic updated
User rejects → contradiction event → 14-day silence window
```


### Data Files

```
data/memory/personal/
  journal/{YYYY-MM}/events.jsonl     ← Append-only event log
  identity.md                         ← Replayed from journal
  preferences.md
  workflow.md
  contacts.md
  projects.md
  corrections.md
  MEMORY.md                           ← Topic index
  confirmation_state.json
  dream_pass_state.json
  snapshots/{YYYYMMDDTHHMMSSZ}/       ← Pre-dream-pass rollback points
  logs/{YYYY}/{MM}/{YYYY-MM-DD}.md

data/sessions/active/{session_id}/
  session.json
  messages.json                       ← Full snapshot (every 20 messages)
  messages.delta.jsonl                ← Incremental delta

data/sessions/archive/{session_id}/   ← Moved here at session end
```

### MemoryEvent Schema

```python
@dataclass
class MemoryEvent:
    event_id: str          # ULID-style (48-bit timestamp + 80-bit random)
    session_id: str
    turn_id: str
    observed_at: str       # ISO-8601
    kind: str              # fact | preference | behavior | correction | contradiction
    topic: str             # identity | preferences | workflow | contacts | projects | corrections
    key: str               # hierarchical, e.g. "preferences.response_style"
    value: dict
    confidence: float      # 0.0-1.0
    source: str            # explicit | inferred | synthesized | migration
    extractor: str         # deterministic | llm_turn | dream_pass | migration
    evidence: dict
    supersedes: str | None
    applied: bool          # False = candidate; True = confirmed/promoted
    rejected: bool
```

Replay rules: only `applied=True` events projected; `supersedes` chains resolved; events >30 days old dropped (except `topic=identity` or `source=migration`).

### Personal Memory Config

**Environment variables:**
```
TURTLE_PERSONAL_MEMORY_ENABLED=1             # Default ON
TURTLE_PERSONAL_MEMORY_DREAM_PASS_ENABLED=0  # Default OFF; set 1 to enable Stage C
TURTLE_PERSONAL_MEMORY_MAX_BYTES=2048
TURTLE_PERSONAL_MEMORY_MAX_TOPIC_FILES=2
```

**`config/turtle_config.json`:**
```json
"DREAM_PASS_AGENT_MODEL": "groq:llama-3.1-70b-versatile"
```

**Module constants:**
```python
DREAM_PASS_MIN_CANDIDATES = 3   # dream_pass.py
DREAM_PASS_MIN_HOURS = 24       # dream_pass.py
DEFAULT_SILENCE_DAYS = 14       # confirmation_gate.py
DECAY_DAYS = 30                 # memory_replayer.py
```

---

## Conversation Memory (RAG)

FAISS-based vector memory provides semantic recall of past conversations across sessions.

- **Session Storage**: JSON files during active conversations; archived at session end
- **Vector Database**: FAISS with cosine similarity search
- **Embeddings**: Cohere embed-english-v3.0 (1024 dimensions)
- **Chunking**: LangChain RecursiveTextSplitter (~300 tokens, 100 overlap)
- **Retrieval**: Top-5 most similar chunks per query (threshold 0.3)

---

## Web Chat UI

### Architecture

```
Browser ──WebSocket──► FastAPI (turtle_server.py)
  │                         │
  │  Text / Audio blob      ├──► Pydantic AI Agent
  │                         │      ├── search_web tool
  │                         │      ├── search_url tool
  │                         │      ├── send_email_assistant tool
  │                         │      └── history_tool (RAG)
  │                         │
  │  ◄── JSON frames ───────┘   STT: Groq Whisper
  │  ◄── Binary audio            TTS: Deepgram (Groq fallback)
  ▼
AudioContext playback
```

---

## File Structure

```
turtle/
|-- apps/
|   |-- turtle_voice.py          # CLI voice assistant entry point
|   |-- turtle_server.py         # FastAPI web server + WebSocket backend
|   `-- websearch_cli.py         # Web search CLI
|-- core/
|   |-- env.py                   # .env loader (shared)
|   |-- llm_client.py            # Model selection + fallback chain
|   |-- paths.py                 # Standard data/output paths
|   |-- personal_memory_store.py # Markdown-backed topic file store
|   |-- memory_journal.py        # Append-only JSONL event log
|   |-- memory_replayer.py       # Journal → topic markdown projection
|   |-- personal_memory_extract.py  # Regex candidate extraction
|   |-- personal_memory_prompt.py   # Context injection builder
|   |-- personal_memory_schema.py   # MemoryEvent dataclass
|   |-- confirmation_gate.py     # User confirmation queue + silence
|   |-- dream_pass.py            # Batch LLM review (Stage C)
|   |-- session_store.py         # Session file lifecycle
|   |-- retrieval_broker.py      # Routes queries to RAG or personal memory
|   |-- memory_extractor.py      # Legacy extraction utilities
|   |-- memory_store.py          # Legacy JSON/JSONL shim
|   |-- openrouter_tts.py        # OpenRouter TTS client
|   |-- stt_fastrtc.py           # FastRTC STT integration
|   |-- output_clean.py          # Response text cleanup
|   |-- task_history.py          # Task history logging
|   |-- web_search.py            # DuckDuckGo search
|   `-- system_prompts/
|       |-- main_assistant.txt
|       |-- email_agent.txt
|       `-- rag_agent.txt
|-- rag/
|   |-- embedder/embedding_model.py   # Cohere embedding wrapper
|   |-- chunking/json_chunking.py     # Conversation chunking
|   |-- storage/vector_storage.py     # FAISS vector database
|   `-- system/complete_rag.py        # RAG orchestration
|-- web/
|   |-- index.html
|   |-- FRONTEND.md              # Frontend architecture runbook
|   |-- css/                     # Modular stylesheets
|   `-- js/                      # Modular JS modules
|-- tools/
|   |-- url_tools/               # Custom URL extraction
|   |-- email_tools/             # Email config + SMTP
|   `-- tts/                     # TTS utilities (Deepgram + Groq fallback)
|-- rtc_vad/
|   |-- vad_simple.py            # Energy-based VAD (console)
|   |-- vad_fastrtc.py           # FastRTC + Silero VAD (console)
|   `-- fastrtc_real.py          # FastRTC with manual + VAD modes
|-- data/
|   |-- memory/personal/         # Personal memory journal + topic files
|   `-- rag/                     # FAISS vector storage
|-- config/
|   `-- turtle_config.json       # Runtime config (dream pass model, flush intervals)
|-- docs/
|   |-- vad-comparison.md        # VAD approach comparison notes
|   |-- websearch-root-cause.md  # Web search failure analysis
|   `-- email-agent.md           # Email agent design notes
|-- test/
|   `-- *.py                     # Unit/integration tests
|-- requirements.txt
`-- README.md
```

---

## Environment Setup

### Required API Keys

Create a `.env` file in the repo root:

```bash
# Core LLM (OpenRouter) - key rotation: key1 → key2 → key3 on rate limit
OPEN_ROUTER_MODEL="nvidia/nemotron-3-nano-30b-a3b:free"
OPEN_ROUTER_API_KEY_1="your_openrouter_api_key_1"
OPEN_ROUTER_API_KEY_2="your_openrouter_api_key_2"
OPEN_ROUTER_API_KEY_3="your_openrouter_api_key_3"
OPENROUTER_APP_URL="your_app_url"
OPENROUTER_APP_TITLE="your_app_title"

# Fallback LLM + STT (Groq Whisper) - leave empty to disable Groq
GROQ_PRIMARY_MODEL=openai/gpt-oss-120b
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
GROQ_API_KEY="your_groq_api_key"
GROQ_API_KEY2="your_groq_api_key_2"

# RAG conversation memory
COHERE_API_KEY=your_cohere_api_key

# TTS - Deepgram primary, Groq Orpheus fallback
DEEPGRAM_API_KEY=your_deepgram_api_key
DEEPGRAM_TTS_MODEL=aura-2-orion-en
DEEPGRAM_TTS_ENCODING=linear16
DEEPGRAM_TTS_CONTAINER=wav
DEEPGRAM_TTS_SAMPLE_RATE=24000
GROQ_TTS_MODEL=canopylabs/orpheus-v1-english
GROQ_TTS_VOICE=orion
GROQ_TTS_FORMAT=wav

# Email
TURTLE_EMAIL_NAME="Your Name"
TURTLE_EMAIL_ADDRESS="your_email@gmail.com"
TURTLE_EMAIL_PASSKEY="your_app_password"

# Personal memory
TURTLE_PERSONAL_MEMORY_ENABLED=1
TURTLE_PERSONAL_MEMORY_DREAM_PASS_ENABLED=0

# Optional monitoring
LOGFIRE_TOKEN=your_logfire_token
```

### Installation

```bash
python -m pip install -r requirements.txt
```

---

## Usage

### Web Chat UI (recommended)

```bash
python apps/turtle_server.py
# Open http://localhost:8765 in a browser
```

The server runs with autoreload by default in development. Set `TURTLE_SERVER_RELOAD=0` to disable.

### CLI Voice Assistant

```bash
python apps/turtle_voice.py
```

---

## Dependencies

### Core Framework
```
pydantic-ai        # Agent framework
python-dotenv      # Environment management
logfire            # Monitoring
fastapi + uvicorn  # Web server
```

### RAG System
```
cohere             # Embedding generation
faiss-cpu          # Vector database
langchain-text-splitters
numpy
```

### Tools
```
httpx              # HTTP client
beautifulsoup4     # HTML parsing
lxml               # XML parser
```

### Model Providers
```
openai             # OpenRouter client dependency
groq               # Optional primary/fallback LLM + Whisper STT
deepgram-sdk       # Primary TTS provider
fastrtc            # RTC + VAD
```

### Voice Stack
```
pyaudio            # Microphone input
pydub              # Audio playback
sounddevice        # Streaming audio
scipy              # Audio I/O
keyboard           # Hotkey recording control
```

---

## Example Interactions

**Web search**
```
You: What happened at Google I/O this week?
Turtle: Searching... Google announced Gemini 2.5 Pro, a new NotebookLM app, and Android 16 features including adaptive refresh...
```

**URL analysis**
```
You: Summarise https://pydantic-ai.readthedocs.io/en/latest/agents/
Turtle: The page covers Pydantic AI's Agent class - how to define tools, system prompts, result validators, and dependency injection...
```

**Email**
```
You: Email Alex at alex@example.com and tell him the Thursday standup is moved to 3pm
Turtle: Drafting email to Alex... Subject: Standup time change - Thursday 3pm. Ready to send?
```

**Memory recall**
```
You: What did we decide about the database schema last session?
Turtle: From our conversation on Monday: you settled on a single-table design with a JSONB column for metadata rather than normalising into subtables.
```

**Personal memory confirmation**
```
Turtle: I noticed you prefer bullet-point summaries over prose. Want me to remember that?
You: Yes
Turtle: Got it - I'll default to bullet points going forward.
```

---

## Known Limitations

- **URL Extraction**: No JavaScript execution (SPAs fail), no auth-gated content, subject to anti-bot measures
- **RAG Memory**: Requires internet for Cohere embeddings; English-optimized; storage grows with volume
- **Personal Memory Stage C**: Dream Pass is OFF by default; requires a capable LLM (70B+) for quality results
- **Model Dependencies**: OpenRouter rate limits, Groq model availability, API costs for production use

---

## Troubleshooting

### RAG Memory Not Working
- Check `COHERE_API_KEY` environment variable
- Verify internet connection for embeddings
- Check `data/rag/` directory permissions

### Personal Memory Not Persisting
- Check `TURTLE_PERSONAL_MEMORY_ENABLED=1`
- Inspect `data/memory/personal/journal/` for written events
- Run `python -c "from core.memory_replayer import replay; replay()"` to force a replay

### URL Extraction Fails
- JavaScript-heavy site: use alternative sources
- Site blocking bots: try different URLs

### Email Not Sending
- Verify credentials in environment
- Check Gmail app password setup

### Debug Commands
```bash
# Test RAG system
python -c "from rag.system.complete_rag import get_rag_system; print(get_rag_system().get_system_stats())"

# Test URL tools
python -c "from tools.url_tools import fetch_url_content_sync; print('URL tools working')"

# Inspect personal memory journal
python -c "from core.memory_journal import JournalStore; js = JournalStore(); print(js.recent_events(10))"
```

---

*Turtle Personal Assistant - Built with modern AI frameworks for practical daily assistance*
