from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

MEMORY_DIR = DATA_DIR / "memory"
MEMORY_PROFILE_FILE = MEMORY_DIR / "profile.json"
MEMORY_EVENTS_FILE = MEMORY_DIR / "events.jsonl"
MEMORY_EPISODES_FILE = MEMORY_DIR / "episodes.jsonl"
MEMORY_STATE_FILE = MEMORY_DIR / "state.json"
# Deprecated: legacy graph-memory artifact kept for JSON fallback compatibility.
MEMORY_GRAPH_FILE = MEMORY_DIR / "graph.json"
TASK_HISTORY_DIR = DATA_DIR / "tasks"
TASK_HISTORY_FILE = TASK_HISTORY_DIR / "history.jsonl"
PERSONAL_MEMORY_DIR = MEMORY_DIR / "personal"
PERSONAL_MEMORY_INDEX_FILE = PERSONAL_MEMORY_DIR / "MEMORY.md"
PERSONAL_MEMORY_IDENTITY_FILE = PERSONAL_MEMORY_DIR / "identity.md"
PERSONAL_MEMORY_PREFERENCES_FILE = PERSONAL_MEMORY_DIR / "preferences.md"
PERSONAL_MEMORY_WORKFLOW_FILE = PERSONAL_MEMORY_DIR / "workflow.md"
PERSONAL_MEMORY_CONTACTS_FILE = PERSONAL_MEMORY_DIR / "contacts.md"
PERSONAL_MEMORY_PROJECTS_FILE = PERSONAL_MEMORY_DIR / "projects.md"
PERSONAL_MEMORY_CORRECTIONS_FILE = PERSONAL_MEMORY_DIR / "corrections.md"
PERSONAL_MEMORY_LOGS_DIR = PERSONAL_MEMORY_DIR / "logs"
PERSONAL_MEMORY_JOURNAL_DIR = PERSONAL_MEMORY_DIR / "journal"
PERSONAL_MEMORY_SNAPSHOTS_DIR = PERSONAL_MEMORY_DIR / "snapshots"

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
        MEMORY_DIR,
        TASK_HISTORY_DIR,
        PERSONAL_MEMORY_DIR,
        PERSONAL_MEMORY_LOGS_DIR,
        PERSONAL_MEMORY_JOURNAL_DIR,
        PERSONAL_MEMORY_SNAPSHOTS_DIR,
        RAG_DATA_DIR,
        RAG_VECTOR_DIR,
        SESSIONS_DIR,
        ACTIVE_SESSION_DIR,
        SESSION_ARCHIVE_DIR,
        TEMP_AUDIO_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
