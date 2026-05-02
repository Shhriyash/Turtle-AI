from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

MEMORY_DIR = DATA_DIR / "memory"
PERSONAL_MEMORY_DIR = MEMORY_DIR / "personal"
PERSONAL_MEMORY_SNAPSHOTS_DIR = PERSONAL_MEMORY_DIR / "snapshots"
# Deprecated globals (will migrate to dynamic methods)
MEMORY_PROFILE_FILE = MEMORY_DIR / "profile.json"
MEMORY_EVENTS_FILE = MEMORY_DIR / "events.jsonl"
MEMORY_EPISODES_FILE = MEMORY_DIR / "episodes.jsonl"
MEMORY_STATE_FILE = MEMORY_DIR / "state.json"
MEMORY_GRAPH_FILE = MEMORY_DIR / "graph.json"

TASK_HISTORY_DIR = DATA_DIR / "tasks"
TASK_HISTORY_FILE = TASK_HISTORY_DIR / "history.jsonl"

RAG_DATA_DIR = DATA_DIR / "rag"
SESSIONS_DIR = DATA_DIR / "sessions"
ACTIVE_SESSION_DIR = SESSIONS_DIR / "active"
SESSION_ARCHIVE_DIR = SESSIONS_DIR / "archive"

def personal_memory_dir(user_id: str) -> Path:
    path = PERSONAL_MEMORY_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path

def personal_memory_file(user_id: str, filename: str) -> Path:
    return personal_memory_dir(user_id) / filename

def personal_journal_dir(user_id: str) -> Path:
    path = personal_memory_dir(user_id) / "journal"
    path.mkdir(parents=True, exist_ok=True)
    return path

def rag_vector_dir(user_id: str) -> Path:
    path = RAG_DATA_DIR / user_id / "vector"
    path.mkdir(parents=True, exist_ok=True)
    return path
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
        MEMORY_DIR,
        PERSONAL_MEMORY_DIR,
        PERSONAL_MEMORY_SNAPSHOTS_DIR,
        TASK_HISTORY_DIR,
        RAG_DATA_DIR,
        SESSIONS_DIR,
        ACTIVE_SESSION_DIR,
        SESSION_ARCHIVE_DIR,
        TEMP_AUDIO_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
