"""
core/routine_outbox.py
----------------------
Phase 6 (W1) — durable write-through outbox for routine-delivery notices.

When a routine fires but the user has no live socket, its frame is queued for
delivery on the user's next connect. That queue lives in-memory in
apps.turtle_server (``_PENDING_ROUTINE_NOTICES``) as the hot cache, but a
process restart between a fire and its delivery would drop the notice. This
module persists each user's queue to a small JSON file under the user's
personal-memory directory so the queue survives restarts and is loaded lazily
on the user's next connect — there is no startup scan; the connect-time drain
consults disk via ``pop_pending_routine_notices``.

The file is ``<personal_memory_dir(user_id)>/routine_outbox.json`` holding a
list of frame dicts (small: ``{"type","code","message","routine_key",
"fired_at"}``). All I/O is best-effort: any error — including a
``StorageCapExceededError`` bubbling from the storage layer — is LOGged and
swallowed so delivery falls back to memory-only rather than raising into the
scheduler worker thread or the app loop. The persisted list is capped to the
same most-recent-N as the in-memory queue. A corrupt/unparseable file is logged
and treated as empty (it is overwritten on the next save).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.io_atomic import atomic_write_json
from core.paths import personal_memory_dir

_OUTBOX_FILENAME = "routine_outbox.json"
# Keep in sync with apps.turtle_server._PENDING_ROUTINE_MAX_PER_USER.
_MAX_FRAMES = 5


def _outbox_path(user_id: str) -> Path:
    # personal_memory_dir mkdirs the user dir and reads PERSONAL_MEMORY_DIR at
    # call time, so test monkeypatching of the path root is honored.
    return personal_memory_dir(user_id) / _OUTBOX_FILENAME


def load_outbox(user_id: str) -> list[dict[str, Any]] | None:
    """Return the user's persisted routine frames, or ``None`` on I/O failure.

    The distinction matters (Codex P6 #4): a MISSING file or a CORRUPT file is a
    definitive empty state (``[]`` — safe for the caller to overwrite/clear),
    but a transient I/O failure (permissions, AV lock) returns ``None`` so the
    caller knows NOT to clear a file it never actually read. Never raises.
    """
    try:
        path = _outbox_path(user_id)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        # Definitively corrupt: treat as empty; the next save overwrites it.
        print(f"LOG: routine_outbox corrupt user={user_id}: {e}; treating as empty")
        return []
    except Exception as e:
        # Transient read failure: the file may still hold frames — signal it.
        print(f"LOG: routine_outbox load failed user={user_id}: {e}")
        return None
    if not isinstance(data, list):
        print(
            f"LOG: routine_outbox corrupt (non-list) user={user_id}; treating as empty"
        )
        return []
    return [frame for frame in data if isinstance(frame, dict)]


def save_outbox(user_id: str, frames: list[dict[str, Any]]) -> None:
    """Persist the user's routine frames atomically (best-effort; never raises).

    Empty ``frames`` deletes the file so an empty outbox leaves no residue. The
    list is capped to the most-recent ``_MAX_FRAMES``. Any I/O error — including
    a ``StorageCapExceededError`` bubbling from the storage layer — is LOGged and
    swallowed; delivery then falls back to the in-memory queue only.
    """
    try:
        path = _outbox_path(user_id)
        if not frames:
            path.unlink(missing_ok=True)
            return
        atomic_write_json(path, list(frames)[-_MAX_FRAMES:])
    except Exception as e:
        print(f"LOG: routine_outbox save failed user={user_id}: {e}")
