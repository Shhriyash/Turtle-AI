"""
core/guardrails.py
------------------
Phase 6 — per-user production guardrails.

Two primitives:

* :class:`StorageCapExceededError` + :func:`enforce_storage_cap` — checked from
  PersonalMemoryStore.write_topic and JournalStore.append so a runaway writer
  (huge paste, infinite extraction loop) cannot fill the disk for everyone
  else.
* :class:`WebSocketRateLimiter` — token-bucket-ish in-process per-user
  hourly + daily counters for inbound WebSocket messages. Swap for Redis
  once Turtle runs on more than one server.
"""
from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

from core.config import settings


class StorageCapExceededError(RuntimeError):
    """Raised when a write would push a user's memory dir past the cap."""

    def __init__(self, user_id: str, used_bytes: int, cap_bytes: int) -> None:
        self.user_id = user_id
        self.used_bytes = used_bytes
        self.cap_bytes = cap_bytes
        super().__init__(
            f"Storage cap exceeded for user {user_id}: "
            f"{used_bytes} bytes used, cap {cap_bytes} bytes."
        )


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def enforce_storage_cap(user_id: str, base_dir: Path, incoming_bytes: int = 0) -> None:
    """Raise StorageCapExceededError if base_dir+incoming would exceed the cap."""
    cap_mb = int(settings.user_storage_cap_mb)
    if cap_mb <= 0:
        return
    cap_bytes = cap_mb * 1024 * 1024
    used = _dir_size_bytes(base_dir) + max(0, int(incoming_bytes))
    if used > cap_bytes:
        raise StorageCapExceededError(user_id, used, cap_bytes)


class WebSocketRateLimitExceeded(RuntimeError):
    """Raised when a user has exceeded their inbound WebSocket message budget."""

    def __init__(self, user_id: str, window: str, limit: int) -> None:
        self.user_id = user_id
        self.window = window
        self.limit = limit
        super().__init__(
            f"WebSocket rate limit exceeded for user {user_id}: {limit}/{window}."
        )


class WebSocketRateLimiter:
    """Per-user-id sliding-window message counter.

    Maintains two parallel deques of receive timestamps per user (hourly +
    daily). check_and_record() prunes expired timestamps and either records
    a new event or raises WebSocketRateLimitExceeded. Backed by an in-process
    dict + lock; swap for Redis when running multi-process.
    """

    def __init__(
        self,
        *,
        per_hour: int | None = None,
        per_day: int | None = None,
    ) -> None:
        self.per_hour = int(per_hour if per_hour is not None else settings.ws_messages_per_hour)
        self.per_day = int(per_day if per_day is not None else settings.ws_messages_per_day)
        self._events: dict[str, list[float]] = {}
        self._lock = Lock()

    def check_and_record(self, user_id: str) -> None:
        if not user_id:
            return
        now = time.time()
        hour_cutoff = now - 3600
        day_cutoff = now - 86400
        with self._lock:
            history = [t for t in self._events.get(user_id, []) if t > day_cutoff]
            if self.per_day > 0 and len(history) >= self.per_day:
                self._events[user_id] = history
                raise WebSocketRateLimitExceeded(user_id, "day", self.per_day)
            recent_hour = sum(1 for t in history if t > hour_cutoff)
            if self.per_hour > 0 and recent_hour >= self.per_hour:
                self._events[user_id] = history
                raise WebSocketRateLimitExceeded(user_id, "hour", self.per_hour)
            history.append(now)
            self._events[user_id] = history


# Module-level singleton used by the WebSocket endpoint.
ws_rate_limiter = WebSocketRateLimiter()
