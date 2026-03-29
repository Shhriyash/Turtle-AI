from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

RAG_DATA_DIR = DATA_DIR / "rag"
RAG_VECTOR_DIR = RAG_DATA_DIR / "vector"
RAG_SESSION_FILE = RAG_DATA_DIR / "current_session.json"
SESSIONS_DIR = DATA_DIR / "sessions"
ACTIVE_SESSION_DIR = SESSIONS_DIR / "active"
ACTIVE_SESSION_MANIFEST = ACTIVE_SESSION_DIR / "session.json"
ACTIVE_SESSION_MESSAGES = ACTIVE_SESSION_DIR / "messages.json"
SESSION_ARCHIVE_DIR = SESSIONS_DIR / "archive"

TEMP_AUDIO_DIR = OUTPUT_DIR / "audio"


def ensure_dirs() -> None:
    """Create standard runtime directories if they do not exist."""
    for path in [
        DATA_DIR,
        OUTPUT_DIR,
        RAG_DATA_DIR,
        RAG_VECTOR_DIR,
        SESSIONS_DIR,
        ACTIVE_SESSION_DIR,
        SESSION_ARCHIVE_DIR,
        TEMP_AUDIO_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
