"""
core/session_store.py
---------------------
G2: High-level SessionStore wrapper around the new storage abstraction.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage

from core.storage import Session, SessionStoreProtocol
from core.storage.local.sqlite_store import SQLiteSessionStore


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _strip_legacy_memory_wrappers(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Old builds persisted 'Relevant user memory:\\n…\\nUser request:\\n…' inside
    user turns; unwrap so restored history carries only what the user said."""
    import re as _re
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    pattern = _re.compile(r"^Relevant user memory:.*?\nUser request:\n", flags=_re.DOTALL)
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                new_content = pattern.sub("", part.content)
                if new_content != part.content:
                    part.content = new_content
    return messages


class SessionRestoreResult:
    def __init__(
        self,
        session_id: str,
        restored: bool,
        message_count: int,
        previous_session_id: str | None = None,
        previous_archive_path: str | None = None,
    ):
        self.session_id = session_id
        self.restored = restored
        self.message_count = message_count
        self.previous_session_id = previous_session_id
        self.previous_archive_path = previous_archive_path


class SessionStore:
    PENDING_EMAIL_TTL_SECONDS = 3600

    def __init__(
        self, backend: SessionStoreProtocol | None = None, *, user_id: str = ""
    ) -> None:
        self.backend = backend or SQLiteSessionStore()
        # Sessions are tenant-scoped; empty string = legacy/unowned.
        self.user_id = user_id
        self.session_id: str | None = None
        self.message_history: list[ModelMessage] = []
        self.pending_email: dict[str, Any] = self._default_pending_email()
        self._pending_email_updated_at: str = ""
        self.current_status: str | None = None
        self.rolling_summary: list[dict[str, Any]] = []

    async def init_backend(self) -> None:
        if hasattr(self.backend, "init_db"):
            await getattr(self.backend, "init_db")()

    @staticmethod
    def _default_pending_email() -> dict[str, Any]:
        return {
            "recipients": [],
            "cc_recipients": [],
            "bcc_recipients": [],
            "subject": "",
            "content": "",
        }

    async def _sync_to_backend(self) -> None:
        if not self.session_id:
            return
        
        msgs_json = ModelMessagesTypeAdapter.dump_python(self.message_history, mode="json")
        
        data = {
            "status": self.current_status or "active",
            "user_id": self.user_id,
            "messages": msgs_json,
            "pending_email": self.pending_email,
            "pending_email_updated_at": self._pending_email_updated_at,
            "summary": self.rolling_summary,
            "updated_at": _utc_now()
        }
        await self.backend.put(Session(session_id=self.session_id, data=data))

    def _restore_from_session(self, session: Session) -> SessionRestoreResult:
        self.session_id = session.session_id
        self.current_status = "active"
        self.pending_email = session.data.get("pending_email", self._default_pending_email())
        self._pending_email_updated_at = session.data.get("pending_email_updated_at", "")
        summary = session.data.get("summary", [])
        self.rolling_summary = summary if isinstance(summary, list) else []
        raw_messages = session.data.get("messages", [])
        try:
            self.message_history = ModelMessagesTypeAdapter.validate_python(raw_messages)
            self.message_history = _strip_legacy_memory_wrappers(self.message_history)
        except Exception:
            self.message_history = []
        return SessionRestoreResult(
            session_id=self.session_id,
            restored=True,
            message_count=len(self.message_history),
        )

    @staticmethod
    def _seconds_since(updated_at: str) -> float:
        """Age in seconds of an ISO-8601 ``updated_at`` value; +inf if unparseable."""
        if not updated_at:
            return float("inf")
        try:
            ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except ValueError:
            return float("inf")
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return (datetime.now(UTC) - ts).total_seconds()

    async def _list_sessions_for_user(self, status_filter: str) -> list[Session]:
        sessions = await getattr(self.backend, "list_sessions")(status_filter=status_filter)
        # Custom backends may not accept user_id, so enforce tenant filtering here.
        return [s for s in sessions if s.data.get("user_id", "") == self.user_id]

    async def start_or_restore(
        self, mode: str = "strict_new", resume_window_seconds: int = 1800
    ) -> SessionRestoreResult:
        await self.init_backend()

        if mode == "resume_if_active":
            if hasattr(self.backend, "list_sessions"):
                # 1) A still-active session (e.g. a second concurrent tab, or a
                #    crash that skipped the disconnect finalizer). Resumable
                #    only within the recency window.
                active_sessions = await self._list_sessions_for_user("active")
                if active_sessions:
                    active_sessions.sort(key=lambda s: s.data.get("updated_at", ""), reverse=True)
                    latest = active_sessions[0]
                    age = self._seconds_since(latest.data.get("updated_at", ""))
                    if age <= resume_window_seconds:
                        return self._restore_from_session(latest)

                    for session in active_sessions:
                        if self._seconds_since(session.data.get("updated_at", "")) > resume_window_seconds:
                            # A crash-orphaned "active" from weeks ago must never be
                            # resumed as today's conversation; production had a
                            # 47-day-old one waiting.
                            session.data["status"] = "pending_finalization"
                            await self.backend.put(session)

                # 2) A recently-disconnected session. The WS finalizer archives
                #    every session as "pending_finalization" on disconnect, so a
                #    reconnect (drop, refresh, watchdog) finds nothing "active".
                #    Resume the most recent one within the window and flip it
                #    back to active so the connect-time finalizer skips it.
                pending = await self._list_sessions_for_user("pending_finalization")
                if pending:
                    pending.sort(key=lambda s: s.data.get("updated_at", ""), reverse=True)
                    latest = pending[0]
                    age = self._seconds_since(latest.data.get("updated_at", ""))
                    if age <= resume_window_seconds:
                        result = self._restore_from_session(latest)
                        # Persist the active flip so the finalization loop and
                        # any other connection no longer treat it as pending.
                        await self._sync_to_backend()
                        return result

        previous_session_id = None
        if hasattr(self.backend, "list_sessions"):
            active_sessions = await self._list_sessions_for_user("active")
            for session in active_sessions:
                session.data["status"] = "pending_finalization"
                await self.backend.put(session)
                previous_session_id = session.session_id

        self.session_id = f"turtle_session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.message_history = []
        self.pending_email = self._default_pending_email()
        self.rolling_summary = []
        self.current_status = "active"
        await self._sync_to_backend()
        
        return SessionRestoreResult(
            session_id=self.session_id,
            restored=False,
            message_count=0,
            previous_session_id=previous_session_id
        )

    async def replace_messages(self, messages: list[ModelMessage]) -> None:
        self.message_history = list(messages)
        await self._sync_to_backend()

    async def archive_active(self, status: str = "completed") -> str | None:
        if not self.session_id:
            return None
        self.current_status = status
        await self._sync_to_backend()
        archived_id = self.session_id
        self.session_id = None
        self.message_history = []
        return archived_id

    async def list_pending_finalization_archives(self) -> list[tuple[str, list]]:
        """Return (session_id, message_history) pairs for all pending-finalization sessions.

        Previously returned (session_id, archive_path) for a file-based store.
        In SQLite mode messages live in session.data["messages"], so we deserialise
        them here and return them directly — callers must use
        _sync_personal_memory_from_messages instead of _sync_personal_memory_from_archive.
        """
        if not hasattr(self.backend, "list_sessions"):
            return []
        pending = await getattr(self.backend, "list_sessions")(status_filter="pending_finalization")
        result = []
        allowed_user_ids = {self.user_id, ""}
        for s in pending:
            if not (hasattr(s, "data") and s.data):
                continue
            # The sweep extracts into the CONNECTING user's journal; processing
            # another user's transcript would cross-contaminate memory. Legacy
            # unowned rows (no user_id) stay eligible for one-time finalization.
            if s.data.get("user_id", "") not in allowed_user_ids:
                continue
            raw_messages = s.data.get("messages", [])
            try:
                messages = ModelMessagesTypeAdapter.validate_python(raw_messages)
            except Exception:
                messages = []
            result.append((s.session_id, messages))
        return result

    async def mark_finalized(self, session_id: str) -> None:
        """Flip a session to completed exactly once so the connect-time sweep
        stops re-running LLM extraction over the same transcript forever."""
        if not hasattr(self.backend, "get"):
            return
        session = await self.backend.get(session_id)
        if session is None:
            return
        session.data["status"] = "completed"
        session.data["updated_at"] = _utc_now()
        await self.backend.put(session)

    def get_pending_email(self) -> dict[str, Any]:
        # A draft abandoned an hour ago must not gap-fill recipients/subject
        # into a brand-new email request (stale-merge production bug).
        if self._pending_email_updated_at:
            if self._seconds_since(self._pending_email_updated_at) > self.PENDING_EMAIL_TTL_SECONDS:
                self.pending_email = self._default_pending_email()
                self._pending_email_updated_at = ""
        return self.pending_email

    async def set_pending_email(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if v is not None:
                self.pending_email[k] = v if isinstance(v, str) else list(v)
        self._pending_email_updated_at = _utc_now()
        await self._sync_to_backend()

    async def clear_pending_email(self) -> None:
        self.pending_email = self._default_pending_email()
        self._pending_email_updated_at = ""
        await self._sync_to_backend()

    def get_summary_tail(self, max_entries: int = 20) -> list[dict[str, Any]]:
        if not self.rolling_summary:
            return []
        if max_entries <= 0:
            return []
        return list(self.rolling_summary[-max_entries:])

    async def append_summary(
        self,
        *,
        bullets: list[str],
        turn_id_range: tuple[int, int] | None = None,
        timestamp: str | None = None,
        max_entries: int = 20,
    ) -> None:
        if not bullets:
            return
        entry = {
            "timestamp": timestamp or _utc_now(),
            "turn_id_range": turn_id_range,
            "bullets": [str(item).strip() for item in bullets if str(item).strip()],
        }
        if not entry["bullets"]:
            return
        self.rolling_summary.append(entry)
        if max_entries > 0 and len(self.rolling_summary) > max_entries:
            self.rolling_summary = self.rolling_summary[-max_entries:]
        await self._sync_to_backend()
