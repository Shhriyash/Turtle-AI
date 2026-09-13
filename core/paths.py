from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_data_dir() -> Path:
    """Where all persistent state lives.

    Mirrors ``TurtleSettings.data_dir`` in core/config.py, including its
    anchoring rule, so identity (users.sqlite, via settings) and memory
    (journals and topic files, via these helpers) can never disagree about
    the location. The two used to disagree: settings honoured TURTLE_DATA_DIR
    while this module hardcoded the repo root, so pointing the env var at a
    temp dir moved only half the writers. That is why the test suite kept
    minting synthetic tenants into production memory (ISSUE-010).

    Resolved at import, not per call, so the value cannot change underneath a
    running process. Tests set the variable in conftest.py before importing
    anything from core.
    """
    raw = os.environ.get("TURTLE_DATA_DIR", "").strip()
    if not raw:
        return ROOT_DIR / "data"
    path = Path(raw)
    # A relative override (TURTLE_DATA_DIR=data) would be CWD-relative and
    # reintroduce the orphaned-memory hazard the absolute default fixes, so
    # anchor it to the repo root exactly as config.py does.
    return path if path.is_absolute() else ROOT_DIR / path


DATA_DIR = _resolve_data_dir()
OUTPUT_DIR = ROOT_DIR / "output"

MEMORY_DIR = DATA_DIR / "memory"
PERSONAL_MEMORY_DIR = MEMORY_DIR / "personal"
PERSONAL_MEMORY_SNAPSHOTS_DIR = PERSONAL_MEMORY_DIR / "snapshots"
# Deprecated single-tenant memory paths. MEMORY_PROFILE_FILE is retained only
# as the default source for scripts/migrate_profile_to_markdown.py; the others
# are legacy. No live code path instantiates a single-tenant store on these
# paths — per-user state lives under personal_memory_dir(user_id).
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

def _mkdir_unless_cloud(path: Path) -> None:
    """mkdir(parents=True, exist_ok=True), skipped in cloud mode.

    Mirrors ensure_dirs()'s own guard below: Vercel's function filesystem is
    read-only outside /tmp, and in cloud mode nothing should be writing to
    these local paths anyway (core/storage/cloud/* replaces every one of
    them with Postgres/Redis). Without this guard, callers like
    PersonalMemoryStore.__init__ that compute a default path via
    personal_memory_dir()/personal_memory_file() before checking
    settings.is_cloud themselves would mkdir unconditionally and crash with
    "OSError: Read-only file system" even when the cloud backend was about
    to be selected anyway (found 2026-09-13 on the first real onboarding
    claim against a cloud deploy).

    core.config is imported lazily for the same acyclic-import reason
    ensure_dirs() documents.
    """
    from core.config import settings

    if settings.is_cloud:
        return
    path.mkdir(parents=True, exist_ok=True)


def personal_memory_dir(user_id: str) -> Path:
    if not user_id:
        raise ValueError("user_id is required")
    path = PERSONAL_MEMORY_DIR / user_id
    _mkdir_unless_cloud(path)
    return path

def personal_memory_file(user_id: str, filename: str) -> Path:
    return personal_memory_dir(user_id) / filename

def personal_journal_dir(user_id: str) -> Path:
    path = personal_memory_dir(user_id) / "journal"
    _mkdir_unless_cloud(path)
    return path

def rag_vector_dir(user_id: str) -> Path:
    path = RAG_DATA_DIR / user_id / "vector"
    _mkdir_unless_cloud(path)
    return path
RAG_SESSION_FILE = RAG_DATA_DIR / "current_session.json"
SESSIONS_DIR = DATA_DIR / "sessions"
ACTIVE_SESSION_DIR = SESSIONS_DIR / "active"
ACTIVE_SESSION_MANIFEST = ACTIVE_SESSION_DIR / "session.json"
ACTIVE_SESSION_MESSAGES = ACTIVE_SESSION_DIR / "messages.json"
SESSION_ARCHIVE_DIR = SESSIONS_DIR / "archive"

TEMP_AUDIO_DIR = OUTPUT_DIR / "audio"


def ensure_dirs() -> None:
    """Create standard runtime directories if they do not exist.

    No-op in cloud mode (TURTLE_DEPLOY=cloud): Vercel's function filesystem
    is READ-ONLY outside /tmp, so an unconditional mkdir() here crashed the
    server at import time (apps/turtle_server.py calls this at module load)
    with "OSError: [Errno 30] Read-only file system: '/var/task/data'" —
    found on the first real deploy. Nothing in cloud mode should be writing
    to these local paths anyway (core/storage/cloud/* replaces every one of
    them with Postgres/Redis), so skipping the mkdir entirely is correct,
    not just a crash workaround.

    core.config is imported lazily (not at module level) so this leaf
    module's own import graph stays acyclic — same rationale
    tools/idempotency.py documents for the identical lazy import.
    """
    from core.config import settings

    if settings.is_cloud:
        return
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
