from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage

from core.paths import (
    ACTIVE_SESSION_MANIFEST,
    ACTIVE_SESSION_MESSAGES,
    SESSION_ARCHIVE_DIR,
    ensure_dirs,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class SessionRestoreResult:
    session_id: str
    restored: bool
    message_count: int
    previous_session_id: str | None = None
    previous_message_count: int = 0
    previous_archive_path: Path | None = None
    had_corrupt_active: bool = False
    previous_session_already_indexed: bool = False


class SessionStore:
    """Persist native PydanticAI message history for the active session."""

    def __init__(
        self,
        manifest_path: Path = ACTIVE_SESSION_MANIFEST,
        messages_path: Path = ACTIVE_SESSION_MESSAGES,
        archive_dir: Path = SESSION_ARCHIVE_DIR,
    ) -> None:
        ensure_dirs()
        self.manifest_path = manifest_path
        self.messages_path = messages_path
        self.archive_dir = archive_dir
        self.session_id: str | None = None
        self.created_at: str | None = None
        self.updated_at: str | None = None
        self.current_status: str | None = None
        self.message_history: list[ModelMessage] = []
        self.pending_email: dict[str, Any] = self._default_pending_email()

    def start_or_restore(self, mode: str = "strict_new") -> SessionRestoreResult:
        if mode not in {"strict_new", "resume_if_active"}:
            raise ValueError(f"Unsupported session restore mode: {mode}")

        previous_session_id: str | None = None
        previous_message_count = 0
        previous_archive_path: Path | None = None
        had_corrupt_active = False
        previous_session_already_indexed = False

        active_loaded = self._load_active_session()
        if active_loaded:
            if mode == "resume_if_active":
                self._persist_manifest(status="active")
                return SessionRestoreResult(
                    session_id=self.session_id or "",
                    restored=True,
                    message_count=len(self.message_history),
                )

            previous_session_id = self.session_id
            previous_message_count = len(self.message_history)
            previous_session_already_indexed = self.current_status == "indexed"
            previous_archive_path = self.archive_active(
                status="completed" if previous_session_already_indexed else "pending_finalization"
            )
            self._reset_state()
        elif self._has_partial_active_files():
            had_corrupt_active = True
            self._quarantine_corrupt_active()

        self._start_new_session()
        return SessionRestoreResult(
            session_id=self.session_id or "",
            restored=False,
            message_count=0,
            previous_session_id=previous_session_id,
            previous_message_count=previous_message_count,
            previous_archive_path=previous_archive_path,
            had_corrupt_active=had_corrupt_active,
            previous_session_already_indexed=previous_session_already_indexed,
        )

    def replace_messages(self, messages: list[ModelMessage]) -> None:
        self._require_active_session()
        self.message_history = list(messages)
        self.updated_at = _utc_now()
        self._persist_messages()
        self._persist_manifest(status="active")

    def archive_active(self, status: str = "completed") -> Path | None:
        if not self.session_id:
            return None

        archive_path = self.archive_dir / self.session_id
        archive_path.mkdir(parents=True, exist_ok=True)
        self._persist_messages(target=archive_path / "messages.json")
        self._persist_manifest(status=status, target=archive_path / "session.json")

        if self.manifest_path.exists():
            self.manifest_path.unlink()
        if self.messages_path.exists():
            self.messages_path.unlink()
        return archive_path

    def get_pending_email(self) -> dict[str, Any]:
        return {
            "recipients": list(self.pending_email.get("recipients", [])),
            "subject": self.pending_email.get("subject", ""),
            "content": self.pending_email.get("content", ""),
        }

    def set_pending_email(
        self,
        *,
        recipients: list[str] | None = None,
        subject: str | None = None,
        content: str | None = None,
    ) -> None:
        self._require_active_session()
        if recipients is not None:
            self.pending_email["recipients"] = list(recipients)
        if subject is not None:
            self.pending_email["subject"] = subject
        if content is not None:
            self.pending_email["content"] = content
        self.updated_at = _utc_now()
        self._persist_manifest(status="active")

    def clear_pending_email(self) -> None:
        self._require_active_session()
        self.pending_email = self._default_pending_email()
        self.updated_at = _utc_now()
        self._persist_manifest(status="active")

    def mark_active_indexed(self) -> None:
        self._require_active_session()
        self.updated_at = _utc_now()
        self._persist_manifest(status="indexed")

    def list_pending_finalization_archives(self) -> list[tuple[str, Path]]:
        pending_archives: list[tuple[str, Path]] = []
        if not self.archive_dir.exists():
            return pending_archives

        for archive_path in sorted(self.archive_dir.iterdir()):
            if not archive_path.is_dir():
                continue
            manifest_path = archive_path / "session.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if manifest.get("status") != "pending_finalization":
                continue
            session_id = str(manifest.get("session_id", "")).strip()
            if session_id:
                pending_archives.append((session_id, archive_path))

        return pending_archives

    def _persist_messages(self, target: Path | None = None) -> None:
        target_path = target or self.messages_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(ModelMessagesTypeAdapter.dump_json(self.message_history))

    def _persist_manifest(self, status: str, target: Path | None = None) -> None:
        self._require_active_session()
        target_path = target or self.manifest_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at or _utc_now(),
            "status": status,
            "message_count": len(self.message_history),
            "pending_email": self.get_pending_email(),
        }
        self.current_status = status
        target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_active_session(self) -> bool:
        if not self.manifest_path.exists() or not self.messages_path.exists():
            return False

        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.session_id = manifest["session_id"]
            self.created_at = manifest.get("created_at")
            self.updated_at = manifest.get("updated_at")
            self.current_status = manifest.get("status")
            self.pending_email = self._normalize_pending_email(manifest.get("pending_email"))
            self.message_history = ModelMessagesTypeAdapter.validate_json(
                self.messages_path.read_bytes()
            )
            return True
        except Exception:
            return False

    def _start_new_session(self) -> None:
        self.session_id = f"turtle_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.created_at = _utc_now()
        self.updated_at = self.created_at
        self.current_status = "active"
        self.message_history = []
        self.pending_email = self._default_pending_email()
        self._persist_manifest(status="active")
        self._persist_messages()

    def _reset_state(self) -> None:
        self.session_id = None
        self.created_at = None
        self.updated_at = None
        self.current_status = None
        self.message_history = []
        self.pending_email = self._default_pending_email()

    def _has_partial_active_files(self) -> bool:
        return self.manifest_path.exists() or self.messages_path.exists()

    def _quarantine_corrupt_active(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = self.archive_dir / f"corrupt_active_{timestamp}"
        target.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            shutil.move(str(self.manifest_path), str(target / "session.json"))
        if self.messages_path.exists():
            shutil.move(str(self.messages_path), str(target / "messages.json"))

    def _require_active_session(self) -> None:
        if not self.session_id:
            raise RuntimeError("Session store has not been initialized")

    @staticmethod
    def _default_pending_email() -> dict[str, Any]:
        return {
            "recipients": [],
            "subject": "",
            "content": "",
        }

    def _normalize_pending_email(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return self._default_pending_email()
        recipients = payload.get("recipients", [])
        if not isinstance(recipients, list):
            recipients = []
        normalized_recipients = [str(item).strip() for item in recipients if str(item).strip()]
        subject = str(payload.get("subject", "")).strip()
        content = str(payload.get("content", "")).strip()
        return {
            "recipients": normalized_recipients,
            "subject": subject,
            "content": content,
        }
