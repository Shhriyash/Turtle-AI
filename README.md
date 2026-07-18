# Turtle Personal Assistant

A personal assistant with persistent personal memory, web search, URL analysis, email, calendar, a web chat UI, and multi-channel messaging support — built with Pydantic AI. Each turn runs a single tool-equipped agent over a provider fallback cascade; the model's own tool choice is the routing.

## Features

- **Personal Memory**: journal-backed topic files fed by deterministic per-turn extraction, an LLM session reflector, and a user confirmation gate surfaced in the web UI
- **Conversation Memory**: RAG-based system for cross-session conversation history
- **Web Chat UI**: Browser-based chat with WebSocket backend, voice input/output, and dev mode
- **Web Search**: Real-time information retrieval using DuckDuckGo
- **URL Analysis**: Custom content extraction from web pages
- **Email Integration**: Automated email sending with professional formatting
- **Calendar**: Create and list events via the Google Calendar API
- **Multi-Channel Messaging**: WhatsApp (Twilio), iMessage (SendBlue), Slack (Events API), and Twilio Voice
- **Single-Agent, Tool-Routed Turns**: one Pydantic AI agent per turn with every tool always available — no separate intent router or graph executor
- **Provider Fallback Cascade**: an ordered list of model rungs (Gemini / OpenRouter / Groq) with per-rung health cooldowns; a failing rung is skipped on the next call
- **Identity Management**: Per-channel user identity mapping to a canonical internal user ID
- **Trace Spans**: one on-disk span per turn (`data/traces/traces.jsonl`) replayable via `scripts/trace_replay.py`

---

## Architecture

### Core Components

Every entrypoint funnels through one canonical turn pipeline (`_execute_turn` in
`apps/turtle_server.py`). There is no intent router and no graph executor: the
model decides what to do by choosing tools.

```
Turn pipeline (apps/turtle_server.py :: _execute_turn)
  |
  |-- pre-step (deterministic)
  |     |-- pending-email bypass (continue a half-finished draft)
  |     |-- memory-context injection via per-turn instructions
  |     `-- confirmation prompt surfaced to the web UI panel (ws only)
  |
  |-- ONE Pydantic AI agent call over the provider fallback cascade
  |     run_agent_with_fallbacks (core/llm_client.py)
  |       ordered rungs (Gemini / OpenRouter / Groq) + per-rung health cooldowns
  |     tools always available — the model's tool choice IS the routing:
  |       search_web · search_url · send_email_assistant · recall
  |       calendar_create · calendar_list · remember
  |
  |-- post-step (deterministic)
  |     |-- output cleaning + session persistence
  |     |-- extraction → write policy → journal + read model
  |     |-- explicit-fact apply / silent candidate queuing (confirmation gate)
  |     `-- periodic reflector (mid-session Stage B + episodic summary)
  |
  `-- one trace span per turn → data/traces/traces.jsonl

Shared backends
|-- RAG Memory System (FAISS + Cohere embeddings)
|-- Personal Memory System (journal + topic markdown files)
`-- Identity Manager (core/identity.py) — SQLite (channel, channel_user_id) → user_id

Channel Adapters (apps/channels/) — all funnel into the same _execute_turn
|-- WhatsApp  → Twilio Cloud API (POST /channels/whatsapp)
|-- iMessage  → SendBlue API    (POST /channels/imessage)
|-- Slack     → Events API      (POST /channels/slack/events)
`-- Voice     → Twilio Media Streams WebSocket (/channels/twilio/voice/stream)
      STT: Groq Whisper · TTS: Deepgram μ-law 8 kHz
```

---

## Personal Memory System

Pipeline: **deterministic per-turn extraction → write policy → journal + read model**, with a **user confirmation gate** for low-confidence candidates and a **mid-session reflector** that runs an LLM session-level extractor (Stage B) plus episodic summarisation. All state is stored as append-only journal events replayed into human-readable markdown topic files.

The journal + replay design gives a full audit trail, rollback safety, and idempotent writes. The confirmation gate prevents hallucinated facts from landing silently; it is surfaced in the web UI's confirm panel (`/api/memory/confirm`).

### Components

| Component | File | Role |
|-----------|------|------|
| `PersonalMemoryStore` | `core/personal_memory_store.py` | Markdown-backed topic files (identity, preferences, workflow, contacts, projects, corrections) |
| `JournalStore` | `core/memory_journal.py` | Append-only JSONL event log, sharded by month |
| `ConfirmationGate` | `core/confirmation_gate.py` | Queues candidates, tracks user yes/no, enforces 14-day silence windows |
| `PeriodicReflector` | `core/periodic_reflector.py` | Mid-session Stage B session extractor + episodic/rolling summaries |
| `replay()` | `core/memory_replayer.py` | Projects journal events → markdown topic files deterministically |
| `RetrievalBroker` | `core/retrieval_broker.py` | Budget-aware retrieval; selects memory to inject into the turn |
| `PersonalMemoryPromptBuilder` | `core/personal_memory_prompt.py` | Fallback builder — selects relevant topic files for model context |
| `extract_memory_candidates_from_messages()` | `core/personal_memory_extract.py` | Deterministic + LLM extraction from message history |

### Pipeline Stages

**Stage A — Per-Turn Extraction (Deterministic)**
- Trigger: Every message, in the turn's post-step
- Deterministic/pattern extraction → `PersonalMemoryCandidate` objects
- Explicit facts are applied straight to the journal; low-confidence candidates are queued in `ConfirmationGate` as `MemoryEvent(applied=False)`
- If not silenced: user shown "I noticed X — want me to remember that?" via the web UI confirm panel

**Stage B — Session-Level Extraction (LLM, mid-session)**
- Trigger: `PeriodicReflector` fires every N turns or after an idle gap (and again on the archive sweep at startup for `pending_finalization` sessions)
- Windowed message history re-extracted with an LLM; candidates queued or applied
- Also produces episodic summaries and a rolling-window session summary
- All `applied=True` events replayed → topic markdown updated

### Confirmation Gate Flow

```
Extract → MemoryEvent(applied=False) → ConfirmationGate.queue_candidate()
         ↓
Gate: not silenced (14-day window) + not explicit?
         ↓ Yes
Show prompt: "I noticed X — remember that?"
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
  snapshots/{YYYYMMDDTHHMMSSZ}/       ← Pre-write rollback points
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
    confidence: float      # 0.0–1.0
    source: str            # explicit | inferred | synthesized | migration
    extractor: str         # deterministic | llm_turn | migration | scheduler
    evidence: dict
    supersedes: str | None
    applied: bool          # False = candidate; True = confirmed/promoted
    rejected: bool
```

Replay rules: only `applied=True` events projected; `supersedes` chains resolved; events >30 days old dropped (except `topic=identity` or `source=migration`).

### Personal Memory Config

**Environment variables:**
```
TURTLE_PERSONAL_MEMORY_ENABLED=1     # Default ON
TURTLE_PERSONAL_MEMORY_MAX_BYTES=2048
TURTLE_PERSONAL_MEMORY_MAX_TOPIC_FILES=2
TURTLE_REFLECT_ENABLED=1             # Mid-session Stage B reflector, default ON
TURTLE_REFLECT_EVERY_TURNS=15        # Reflect every N turns
TURTLE_REFLECT_IDLE_SECONDS=1800     # ...or after this idle gap
```

**Module constants:**
```python
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
  │  Text / Audio blob      ├──► _execute_turn → one Pydantic AI agent
  │                         │      (fallback cascade) with tools:
  │                         │      search_web · search_url · send_email_assistant
  │                         │      recall · calendar_create · calendar_list · remember
  │                         │
  │  ◄── JSON frames ───────┘   STT: Groq Whisper
  │  ◄── Binary audio            TTS: Deepgram (Groq fallback)
  │  ◄── confirmation_prompt     memory confirm panel (/api/memory/confirm)
  ▼
AudioContext playback
```

---

## File Structure

```
turtle/
|-- apps/
|   |-- turtle_server.py         # FastAPI web server + WebSocket backend; owns _execute_turn
|   |-- auth.py                  # WebSocket / HTTP authentication
|   |-- websearch_cli.py         # Web search CLI
|   `-- channels/
|       |-- __init__.py          # TurtleEvent / TurtleResponse types + dispatch wiring
|       |-- whatsapp.py          # WhatsApp adapter — Twilio Cloud API webhook
|       |-- imessage.py          # iMessage adapter — SendBlue webhook
|       |-- slack.py             # Slack adapter — Events API (app_mention + DM)
|       `-- twilio_voice.py      # Voice adapter — Twilio Media Streams WebSocket
|-- core/
|   |-- env.py                   # .env loader (shared)
|   |-- config.py                # Centralised pydantic-settings config (TurtleSettings)
|   |-- llm_client.py            # Model selection + run_agent_with_fallbacks cascade
|   |-- health_tracker.py        # Per-rung cooldown tracking for the cascade
|   |-- paths.py                 # Standard data/output paths
|   |-- identity.py              # F5: channel → canonical user_id (SQLite)
|   |-- personal_memory_store.py # Markdown-backed topic file store
|   |-- memory_journal.py        # Append-only JSONL event log
|   |-- memory_replayer.py       # Journal → topic markdown projection
|   |-- personal_memory_extract.py  # Deterministic + LLM candidate extraction
|   |-- personal_memory_prompt.py   # Fallback context-injection builder
|   |-- personal_memory_schema.py   # MemoryEvent dataclass
|   |-- confirmation_gate.py     # User confirmation queue + silence
|   |-- periodic_reflector.py    # Mid-session Stage B + episodic summary runner
|   |-- episodic_summarizer.py   # Episodic conversation summarisation
|   |-- email_flow.py            # Email extraction + SMTP send helpers
|   |-- session_store.py         # Session file lifecycle
|   |-- retrieval_broker.py      # Budget-aware memory retrieval for the turn
|   |-- memory_extractor.py      # Legacy extraction utilities
|   |-- openrouter_tts.py        # OpenRouter TTS client
|   |-- streaming_tts.py         # Streaming TTS helpers
|   |-- stt_fastrtc.py           # FastRTC STT integration
|   |-- output_clean.py          # Response text cleanup
|   |-- task_history.py          # Task history logging
|   |-- web_search.py            # DuckDuckGo search
|   |-- background_tasks.py      # App-level background task registration
|   |-- observability.py         # Logfire + on-disk trace-span sink
|   |-- latency_budgets.py       # Per-task latency targets
|   |-- worker.py                # Background worker utilities
|   |-- storage/
|   |   |-- local/sqlite_store.py
|   |   `-- local/faiss_store.py
|   `-- system_prompts/
|       |-- main_assistant.txt
|       |-- email_agent.txt
|       |-- memory_extractor.txt
|       `-- tools/               # Per-tool contract descriptions (search_web, recall, ...)
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
|   `-- turtle_config.json       # Runtime config (model names, flush intervals)
|-- scripts/
|   `-- trace_replay.py          # Replays data/traces/traces.jsonl turn spans
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
# Core LLM (OpenRouter) — key rotation: key1 → key2 → key3 on rate limit
OPEN_ROUTER_MODEL="nvidia/nemotron-3-nano-30b-a3b:free"
OPEN_ROUTER_API_KEY_1="your_openrouter_api_key_1"
OPEN_ROUTER_API_KEY_2="your_openrouter_api_key_2"
OPEN_ROUTER_API_KEY_3="your_openrouter_api_key_3"
OPENROUTER_APP_URL="your_app_url"
OPENROUTER_APP_TITLE="your_app_title"

# Fallback LLM + STT (Groq Whisper) — leave empty to disable Groq
GROQ_PRIMARY_MODEL=openai/gpt-oss-120b
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
GROQ_API_KEY="your_groq_api_key"
GROQ_API_KEY2="your_groq_api_key_2"

# RAG conversation memory
COHERE_API_KEY=your_cohere_api_key

# TTS — Deepgram primary, Groq Orpheus fallback
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

# Channel adapters — Twilio (WhatsApp + Voice)
TWILIO_ACCOUNT_SID="your_twilio_account_sid"
TWILIO_AUTH_TOKEN="your_twilio_auth_token"
TWILIO_WHATSAPP_NUMBER="whatsapp:+14155238886"   # Twilio sandbox or purchased number
TWILIO_VOICE_NUMBER="+15005550006"               # Twilio voice number

# Channel adapters — iMessage via SendBlue
SENDBLUE_API_KEY="your_sendblue_api_key"
SENDBLUE_API_SECRET="your_sendblue_api_secret"

# Channel adapters — Slack
SLACK_BOT_TOKEN="xoxb-your-slack-bot-token"
SLACK_SIGNING_SECRET="your_slack_signing_secret"

# Channel adapters — Google Calendar (optional)
GOOGLE_CALENDAR_CREDENTIALS_JSON='{"installed":{"client_id":"..."}}'
GOOGLE_CALENDAR_TOKEN_JSON='{"token":"..."}'

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

The server runs with autoreload by default in development. Set `TURTLE_SERVER_RELOAD=0` to disable. Voice input/output is available directly in the web UI.

### Inspecting a turn

```bash
python scripts/trace_replay.py list             # recent turn spans from data/traces/traces.jsonl
python scripts/trace_replay.py show <turn_id>   # full span for one turn
```

### Channel Adapters (webhook setup)

All channel routes are mounted on `turtle_server.py`. Expose the server publicly (e.g. via ngrok) and configure each platform with the corresponding webhook URL:

| Channel | Webhook URL | Platform setup |
|---------|-------------|---------------|
| WhatsApp | `POST /channels/whatsapp` | Twilio Console → Messaging → WhatsApp sandbox |
| iMessage | `POST /channels/imessage` | SendBlue dashboard → Webhook URL |
| Slack | `POST /channels/slack/events` | api.slack.com/apps → Event Subscriptions |
| Voice | `POST /channels/twilio/voice/incoming` | Twilio Console → Phone Numbers → Voice |

Adapters operate without credentials in dev mode (signature verification is skipped), so they can be tested locally without real API keys.

---

## Turn Pipeline

There is no separate intent router and no graph executor. Every user turn —
from the web UI, voice, or any channel adapter — runs the same canonical
pipeline (`_execute_turn` in `apps/turtle_server.py`). The model does the
routing by choosing tools.

### 1. Deterministic pre-step

- **Pending-email bypass** — if a half-finished email draft exists in the
  session, the turn is treated as a continuation of it (memory context and trace
  label are forced to the email flow) even when the words don't say "email".
- **Memory-context injection** — `RetrievalBroker` (falling back to
  `PersonalMemoryPromptBuilder`) builds a budget-bounded memory block, injected
  via per-turn instructions. The persisted user turn stays the user's bare words.
- **Confirmation surfacing** — on the WebSocket path, any pending memory
  confirmation prompt is emitted as a sidecar frame so the web UI can render it.

### 2. One agent call over the fallback cascade

`run_agent_with_fallbacks` (`core/llm_client.py`) runs a single Pydantic AI
agent against an ordered list of model rungs (Gemini / OpenRouter / Groq). Rungs
in cooldown after a recent failure are skipped (`core/health_tracker.py`); if
every rung is cooling, the tracker is bypassed rather than failing outright. All
tools are offered on every turn — **the model's tool choice is the routing**:

| Tool | Purpose |
|------|---------|
| `search_web` | DuckDuckGo web search |
| `search_url` | Fetch + extract a specific URL |
| `send_email_assistant` | Author and send email (SMTP) |
| `recall` | Retrieve from conversation / personal memory |
| `calendar_create` | Create a Google Calendar event |
| `calendar_list` | List upcoming events |
| `remember` | Persist an explicit user fact |

Tool contracts live in `core/system_prompts/tools/*.md`.

### 3. Deterministic post-step

- Output cleaning + session persistence
- Extraction → write policy → journal + read model
- Explicit facts applied immediately; low-confidence candidates queued in the
  confirmation gate
- `PeriodicReflector` fires the mid-session Stage B extractor + episodic summary
- One trace span per turn written to `data/traces/traces.jsonl`
  (replay via `scripts/trace_replay.py`)

---

## Channel Adapters

All channel adapters normalise inbound payloads to a `TurtleEvent` and call the shared `dispatch_event()` handler. Replies are sent back asynchronously via the respective platform API.

### WhatsApp (`apps/channels/whatsapp.py`)

- Transport: Twilio Cloud API webhook — `POST /channels/whatsapp`
- Auth: HMAC-SHA1 `X-Twilio-Signature` verified on every request (403 on failure)
- Idempotency: `MessageSid` cached for 60 s to deduplicate Twilio retries
- Reply: Twilio Messages REST API (`From: whatsapp:<TWILIO_WHATSAPP_NUMBER>`)

### iMessage (`apps/channels/imessage.py`)

- Transport: SendBlue webhook — `POST /channels/imessage`
- Auth: HMAC-SHA256 `X-SendBlue-Signature`
- Reply: `POST https://api.sendblue.co/api/send-message`

### Slack (`apps/channels/slack.py`)

- Transport: Slack Events API — `POST /channels/slack/events`
- Auth: HMAC-SHA256 `X-Slack-Signature` with 5-minute replay protection
- Events handled: `app_mention`, `message.im` (direct messages)
- Reply: `chat.postMessage` as a threaded reply; response sent as a background task to satisfy Slack's 3-second acknowledgement requirement

### Twilio Voice (`apps/channels/twilio_voice.py`)

- Transport: Twilio Media Streams over WebSocket (`/channels/twilio/voice/stream`)
- Call entry: `POST /channels/twilio/voice/incoming` returns TwiML `<Connect><Stream>`
- Audio format: PCMU G.711 μ-law, 8 kHz, 20 ms frames
- VAD: energy-based silence detection (800 ms threshold)
- STT: Groq Whisper (`whisper-large-v3-turbo`)
- TTS: Deepgram Aura (linear16 → transcoded to μ-law 8 kHz)

### Identity Manager (`core/identity.py`)

Maps `(channel, channel_user_id)` to a stable internal `user_id` stored in `data/users.sqlite`. A new canonical ID is auto-created on first contact from any channel.

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

### Channel Adapters
```
twilio             # WhatsApp + Voice (optional)
aiosqlite          # Identity manager (users.sqlite)
pydantic-settings  # Centralised TurtleSettings config
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
Turtle: The page covers Pydantic AI's Agent class — how to define tools, system prompts, result validators, and dependency injection...
```

**Email**
```
You: Email Alex at alex@example.com and tell him the Thursday standup is moved to 3pm
Turtle: Drafting email to Alex... Subject: Standup time change — Thursday 3pm. Ready to send?
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
Turtle: Got it — I'll default to bullet points going forward.
```

---

## Known Limitations

- **URL Extraction**: No JavaScript execution (SPAs fail), no auth-gated content, subject to anti-bot measures
- **RAG Memory**: Requires internet for Cohere embeddings; English-optimized; storage grows with volume
- **Personal Memory Stage B**: The mid-session LLM reflector needs a capable model for quality extraction; it runs best-effort and never blocks a turn
- **Model Dependencies**: OpenRouter rate limits, Groq model availability, API costs for production use
- **Twilio Voice VAD**: Uses simple energy-based silence detection (800 ms threshold); full Silero VAD is not yet wired
- **Channel Adapters**: All adapters require the server to be publicly reachable over HTTPS/WSS; not suitable for local-only setups without a tunnel (e.g. ngrok)
- **iMessage via SendBlue**: Requires a US phone number and SendBlue account; Apple-native iMessage delivery is not guaranteed for non-Apple hardware

---

## Troubleshooting

### RAG Memory Not Working
- Check `COHERE_API_KEY` environment variable
- Verify internet connection for embeddings
- Check `data/rag/` directory permissions

### Personal Memory Not Persisting
- Check `TURTLE_PERSONAL_MEMORY_ENABLED=1`
- Inspect `data/memory/personal/journal/` for written events (the journal is the source of truth; topic markdown files are a replayed projection)
- The journal is replayed into topic files after every turn — if a fact is in the journal but not in a topic file, check server logs for a replay/storage-cap error

### URL Extraction Fails
- JavaScript-heavy site: use alternative sources
- Site blocking bots: try different URLs

### Email Not Sending
- Verify credentials in environment
- Check Gmail app password setup

### WhatsApp / iMessage / Slack Not Responding
- Confirm `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_WHATSAPP_NUMBER` are set
- For iMessage: confirm `SENDBLUE_API_KEY` and `SENDBLUE_API_SECRET`
- For Slack: confirm `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET`; ensure bot is invited to the channel
- Check server logs for `403 Invalid * signature` — mismatch between configured secret and platform secret

### Twilio Voice Call Gets No Audio
- Confirm `DEEPGRAM_API_KEY` (TTS) and `GROQ_API_KEY` (STT Whisper)
- Ensure the server is reachable over HTTPS/WSS (Twilio requires TLS for Media Streams)
- Check logs for `[TwilioVoice] STT failed` or `[TwilioVoice] TTS failed`

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

*Turtle Personal Assistant — Built with modern AI frameworks for practical daily assistance*
