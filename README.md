# Turtle Personal Assistant

A multi-agent personal assistant with conversation memory, web search, URL analysis, and email capabilities built with Pydantic AI.

## Features

- **Conversation Memory**: RAG-based system for cross-session conversation history
- **Web Search**: Real-time information retrieval using specialized search agents
- **URL Analysis**: Custom content extraction from web pages
- **Email Integration**: Automated email sending with professional formatting
- **Multi-Agent Architecture**: Specialized agents for different tasks
- **LLM Fallback**: OpenRouter key rotation plus optional Groq primary/fallback models

## Architecture

### Core Components

```
Main Assistant (Pydantic AI + core/llm_client model selection)
|-- Web Search Tool (DuckDuckGo HTML + parser)
|-- Email Specialist Agent
|-- Custom URL Tools (BeautifulSoup + httpx)
`-- RAG Memory System (FAISS + Cohere embeddings)
```

### Agent Specialization

- **Main Assistant**: Handles routing, tools, and conversation flow
- **Email Specialist Agent**: Extracts/merges email intent and sends via SMTP tool
- **RAG System**: Provides conversation memory across sessions
- **Tool Layer**: Web search and URL extraction are implemented as tools, not separate LLM agents

## Conversation Memory System

### Overview
The RAG (Retrieval-Augmented Generation) system provides persistent conversation memory across sessions using vector similarity search.

### Architecture
- **Session Storage**: JSON files during active conversations
- **Vector Database**: FAISS with cosine similarity search
- **Embeddings**: Cohere embed-english-v3.0 model
- **Chunking**: LangChain RecursiveTextSplitter for optimal chunk sizes

### How It Works
1. **During Session**: Conversations stored in temporary JSON files
2. **Session End**: Conversations chunked and converted to embeddings
3. **Vector Storage**: Chunks stored in FAISS database with metadata
4. **Retrieval**: Similarity search finds relevant past conversations
5. **Context**: Retrieved conversations provided to main agent

### Technical Implementation
- **Embedding Dimension**: 1024 (Cohere v3.0)
- **Similarity Threshold**: 0.3 (adjustable)
- **Chunk Size**: ~300 tokens with 100 token overlap
- **Top-K Retrieval**: 5 most similar chunks per query

## Tool Evolution

### Why Built-in Tools Had Problems

#### 1. URL Context Tool Issues
- **Limited Customization**: Built-in tools offered minimal configuration
- **Poor Error Handling**: Generic error messages without actionable feedback
- **Content Limitations**: Restricted content extraction capabilities
- **No Dynamic Content Support**: Failed on JavaScript-heavy sites

#### 2. Search Tool Limitations
- **Model Compatibility**: Some models had inconsistent function calling
- **Response Quality**: Generic search responses without context
- **Rate Limiting**: Built-in tools didn't handle API limits gracefully

### Our Solutions

#### 1. Custom URL Tools Package
**Created modular extraction system:**
- **Enhanced Error Handling**: Specific messages for different failure types
- **Extended Content Support**: 8000+ character extraction vs 3000 limit
- **Smart Content Detection**: CSS selectors for main content areas
- **Dynamic Site Detection**: Identifies and explains JavaScript requirements

**Package Structure:**
```
tools/url_tools/
|-- models.py      # Data structures (UrlAnalysisResult, UrlState)
|-- extractor.py   # Core extraction with async/sync support
`-- __init__.py    # Clean interface
```

#### 2. Specialized Delegation
**Current delegation model:**
- **Email Agent**: Specialized for email composition and sending
- **Search/URL Tools**: Implemented directly as tools in main assistant
- **Context Preservation**: Main assistant and delegated calls share session context

#### 3. RAG Memory System
**Replaced limited built-in memory with comprehensive solution:**
- **Cross-Session Persistence**: Memory survives restarts
- **Semantic Search**: Find conversations by meaning, not keywords
- **Scalable Storage**: FAISS handles thousands of conversations efficiently
- **Session Management**: Automatic lifecycle handling

## Dependencies

### Core Framework
```
pydantic-ai        # Agent framework
python-dotenv      # Environment management
logfire           # Monitoring
```

### RAG System
```
cohere            # Embedding generation
faiss-cpu         # Vector database
langchain-text-splitters  # Text chunking
numpy             # Numerical operations
```

### Tools
```
httpx             # HTTP client
beautifulsoup4    # HTML parsing
lxml              # XML parser
```

### Model Providers
```
openai            # OpenRouter client dependency (installed via pydantic-ai)
groq              # Optional primary/fallback LLM + Whisper STT
deepgram-sdk      # Primary TTS provider
fastrtc           # RTC + VAD experiments
```

### Voice Stack
```
pyaudio           # Microphone input
pydub             # Audio playback
sounddevice       # Streaming audio playback
scipy             # Audio I/O utilities
keyboard          # Hotkey recording control
groq TTS          # fallback voice synthesis
```

## Environment Setup

### Required API Keys
Create a `.env` file in the repo root and set:
```bash
# Core LLM (OpenRouter)
OPEN_ROUTER_MODEL="nvidia/nemotron-3-nano-30b-a3b:free"
OPENROUTER_MODEL="nvidia/nemotron-3-nano-30b-a3b:free"  # optional alternate name
OPEN_ROUTER_API_KEY_1="your_openrouter_api_key_1"
OPEN_ROUTER_API_KEY_2="your_openrouter_api_key_2"
OPEN_ROUTER_API_KEY_3="your_openrouter_api_key_3"
OPENROUTER_API_KEY="your_openrouter_api_key"   # optional single-key fallback
OPENROUTER_APP_URL="your_app_url"   # optional
OPENROUTER_APP_TITLE="your_app_title"  # optional

# OpenRouter key rotation is used for LLM (key 1 -> 2 -> 3 on rate limit).
# Current runtime expects at least one OpenRouter key to initialize the model chain.

# Fallback LLM + STT (Groq Whisper)
# GROQ_PRIMARY_MODEL is used when GROQ_API_KEY/GROQ_API_KEY2 is available.
# Leave GROQ_API_KEY* empty to disable Groq model usage.
GROQ_PRIMARY_MODEL=openai/gpt-oss-120b
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
GROQ_API_KEY2="your_groq_api_key"
GROQ_API_KEY="your_groq_api_key"

# RAG system
COHERE_API_KEY=your_cohere_api_key

# Voice (TTS)
# Deepgram TTS (default) with Groq fallback
DEEPGRAM_API_KEY=your_deepgram_api_key
DEEPGRAM_TTS_MODEL=aura-2-orion-en
DEEPGRAM_TTS_ENCODING=linear16
DEEPGRAM_TTS_CONTAINER=wav
DEEPGRAM_TTS_SAMPLE_RATE=24000
# Groq TTS fallback
GROQ_TTS_MODEL=canopylabs/orpheus-v1-english
GROQ_TTS_VOICE=orion
GROQ_TTS_FORMAT=wav

# Email functionality
TURTLE_EMAIL_NAME="Your Name"
TURTLE_EMAIL_ADDRESS="your_email@gmail.com"
TURTLE_EMAIL_PASSKEY="your_app_password"

# Optional monitoring
LOGFIRE_TOKEN=your_logfire_token
```

### Installation
```bash
python -m pip install -r requirements.txt
```

## Usage

### Starting Turtle
```bash
python apps/turtle_voice.py
```

### Other Entry Points
```bash
python apps/websearch_cli.py
python rtc_vad/vad_simple.py
python rtc_vad/vad_fastrtc.py
python rtc_vad/fastrtc_real.py
python tools/tts/tts.py
```

### Example Conversations

#### Web Search
```
You: What's the latest news about AI?
Turtle: Let me search for current AI news...
```

#### URL Analysis
```
You: What's this page about https://docs.python.org/3/library/asyncio.html
Turtle: I'll analyze that URL for you...
```

#### Email Sending
```
You: Send email to john@example.com about the meeting tomorrow
Turtle: I'll help you send that email...
```

#### Memory Recall
```
You: Do you remember what we discussed about Python yesterday?
Turtle: Let me check our conversation history...
```

## File Structure

```
turtle/
|-- apps/
|   |-- turtle_voice.py          # Main assistant entry point
|   `-- websearch_cli.py         # Web search CLI
|-- rtc_vad/
|   |-- vad_simple.py            # Energy-based VAD (console)
|   |-- vad_fastrtc.py           # FastRTC + streaming TTS (console)
|   `-- fastrtc_real.py          # FastRTC architecture (console)
|-- rag/
|   |-- embedder/
|   |   `-- embedding_model.py   # Cohere embedding wrapper
|   |-- chunking/
|   |   `-- json_chunking.py     # Conversation chunking
|   |-- storage/
|   |   `-- vector_storage.py    # FAISS vector database
|   `-- system/
|       `-- complete_rag.py      # RAG system orchestration
|-- tools/
|   |-- url_tools/               # Custom URL extraction
|   |-- email_tools/             # Email configuration + SMTP
|   `-- tts/                     # TTS utilities (Deepgram + Groq fallback)
|       |-- client.py
|       `-- tts.py
|-- core/
|   |-- env.py                   # .env loader (shared)
|   |-- llm_client.py            # Model selection + fallback chain
|   |-- system_prompts/          # System prompt references
|   |   |-- main_assistant.txt
|   |   |-- email_agent.txt
|   |   `-- rag_agent.txt
|   `-- paths.py                 # Standard data/output paths
|-- data/
|   `-- rag/                     # Vector storage directory
|       |-- current_session.json
|       `-- vector/
|           |-- faiss_index.bin
|           `-- chunk_metadata.json
|-- output/                      # Runtime outputs (audio, logs, exports)
|-- test/
|   `-- *.py                     # Unit/integration test modules
|-- VAD_COMPARISON.md
|-- requirements.txt
`-- README.md
```

## Voice Scripts Comparison

This repo includes three voice/VAD scripts in `rtc_vad/`. They share the same stack:
- STT: Groq Whisper `whisper-large-v3-turbo`
- LLM: OpenRouter (with optional Groq fallback)
- TTS: Deepgram Aura (`aura-2-orion-en`) with Groq Orpheus fallback

### Quick Comparison

| Script | Primary VAD Technique | Recording Style | VAD Control | Latency Potential | Notes |
| --- | --- | --- | --- | --- | --- |
| `rtc_vad/vad_simple.py` | Energy/RMS threshold in Python loop | Record-then-transcribe | Low | Highest | Most stable, least moving parts |
| `rtc_vad/vad_fastrtc.py` | FastRTC `ReplyOnPause` (Silero VAD) | Console mode still record-then-transcribe | Medium | Medium | FastRTC VAD path exists but console mode is still batch |
| `rtc_vad/fastrtc_real.py` | FastRTC `ReplyOnPause` (Silero VAD) + manual modes | Record-then-transcribe + VAD-driven stop | High | Best potential | Best foundation for low latency once fully streamed |

### Technique Details
- `vad_simple.py` uses a simple RMS energy threshold and stops on silence. It is easy to tune but not robust to noisy environments.
- `vad_fastrtc.py` integrates FastRTC's Silero VAD in the handler, but the current console mode still records fixed chunks before STT.
- `fastrtc_real.py` exposes more VAD control and timing logs, and is the most suitable base for true low-latency streaming.

### Notes on Latency
- All three scripts currently write audio to a WAV file, then run STT, then call the LLM, then do TTS.
- To reduce latency further, the next step is streaming audio into STT and streaming TTS output as chunks instead of full files.

For deeper measurements and experiments, see `VAD_COMPARISON.md`.

## Performance

### Typical Response Times
- **Simple Questions**: 1-2 seconds
- **Web Search**: 3-5 seconds  
- **URL Analysis**: 4-8 seconds
- **Memory Search**: 0.3-0.5 seconds
- **Email Sending**: 2-4 seconds

### Memory System Stats
- **Embedding Generation**: ~200ms per conversation
- **Vector Search**: ~50ms for 1000+ chunks
- **Storage Efficiency**: ~2KB per conversation chunk

## Known Limitations

### URL Extraction
- Cannot execute JavaScript (affects SPAs)
- No authentication-required content
- Limited by anti-bot measures

### Memory System
- Requires internet for embeddings (Cohere API)
- English-optimized (embed-english-v3.0)
- Storage grows with conversation volume

### Model Dependencies
- OpenRouter rate limits
- Groq model availability
- API costs for production use

## Troubleshooting

### Common Issues

#### Memory Not Working
- Check COHERE_API_KEY environment variable
- Verify internet connection for embeddings
- Check data/rag/ directory permissions

#### URL Extraction Fails
- Page requires JavaScript: Use alternative sources
- Site blocking bots: Try different URLs
- Network issues: Check connectivity

#### Email Not Sending  
- Verify email credentials in environment
- Check Gmail app password setup
- Confirm SMTP settings

### Debug Commands
```bash
# Test memory system
python -c "from rag.system.complete_rag import get_rag_system; print(get_rag_system().get_system_stats())"

# Test URL tools
python -c "from tools.url_tools import fetch_url_content_sync; print('URL tools working')"
```

## Future Enhancements

### Planned Improvements
- Browser automation for JavaScript sites
- Multi-language embedding support
- Enhanced email templates
- Conversation export/import

### Scalability
- Distributed vector storage
- Conversation summarization
- Hierarchical memory organization

---

*Turtle Personal Assistant - Built with modern AI frameworks for practical daily assistance*








