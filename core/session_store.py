from __future__ import annotations

import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage

from core.paths import (
    ACTIVE_SESSION_DIR,
    SESSION_ARCHIVE_DIR,
    ensure_dirs,
)
from core.io_atomic import atomic_write_bytes, atomic_write_json


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_iso_or_epoch(value: str | None, fallback: float) -> float:
    if not value:
        return fallback
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return fallback


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
    """Persist native PydanticAI message history for the active session.

    Default layout:
    - Active session files are namespaced by session id:
      data/sessions/active/<session_id>/session.json
      data/sessions/active/<session_id>/messages.json

    Legacy mode:
    - If manifest/messages paths are explicitly provided, SessionStore keeps the
      previous flat-file behavior for backward compatibility.
    """

    def __init__(
        self,
        manifest_path: Path | None = None,
        messages_path: Path | None = None,
        archive_dir: Path = SESSION_ARCHIVE_DIR,
        active_dir: Path = ACTIVE_SESSION_DIR,
    ) -> None:
        ensure_dirs()
        if (manifest_path is None) != (messages_path is None):
            raise ValueError("manifest_path and messages_path must be provided together")

        self._legacy_layout = manifest_path is not None and messages_path is not None
        self.active_dir = active_dir
        self.archive_dir = archive_dir
        self._active_session_dir: Path | None = None

        if self._legacy_layout:
            self.manifest_path = Path(manifest_path)  # type: ignore[arg-type]
            self.messages_path = Path(messages_path)  # type: ignore[arg-type]
            self.active_dir = self.manifest_path.parent
        else:
            self.manifest_path: Path | None = None
            self.messages_path: Path | None = None

        self._lock_path = self.active_dir / ".session.lock"
        self._thread_lock = threading.RLock()

        self.session_id: str | None = None
        self.created_at: str | None = None
        self.updated_at: str | None = None
        self.current_status: str | None = None
        self.message_history: list[ModelMessage] = []
        self.pending_email: dict[str, Any] = self._default_pending_email()
        self._messages_delta_path: Path | None = None
        self._delta_count: int = 0
        self._snapshot_interval = max(1, int(os.getenv("TURTLE_SESSION_SNAPSHOT_INTERVAL", "20")))

    def start_or_restore(self, mode: str = "strict_new") -> SessionRestoreResult:
        if mode not in {"strict_new", "resume_if_active"}:
            raise ValueError(f"Unsupported session restore mode: {mode}")

        with self._store_lock():
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
                previous_archive_path = self._archive_active_unlocked(
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
        with self._store_lock():
            self._require_active_session()
            previous_messages = list(self.message_history)
            next_messages = list(messages)
            self.message_history = next_messages
            self.updated_at = _utc_now()
            if (
                not self._legacy_layout
                and self._messages_delta_path is not None
                and self._is_append_only(previous_messages, next_messages)
            ):
                if not previous_messages:
                    self._persist_messages()
                    self._reset_delta_log()
                else:
                    delta_messages = next_messages[len(previous_messages):]
                    if delta_messages:
                        self._append_delta_messages(delta_messages)
                        self._delta_count += len(delta_messages)
                    if self._delta_count >= self._snapshot_interval:
                        self._persist_messages()
                        self._reset_delta_log()
            else:
                self._persist_messages()
                self._reset_delta_log()
            self._persist_manifest(status="active")

    def archive_active(self, status: str = "completed") -> Path | None:
        with self._store_lock():
            return self._archive_active_unlocked(status=status)

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
        with self._store_lock():
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
        with self._store_lock():
            self._require_active_session()
            self.pending_email = self._default_pending_email()
            self.updated_at = _utc_now()
            self._persist_manifest(status="active")

    def mark_active_indexed(self) -> None:
        with self._store_lock():
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

    def _archive_active_unlocked(self, status: str = "completed") -> Path | None:
        if not self.session_id:
            return None

        archive_path = self.archive_dir / self.session_id
        archive_path.mkdir(parents=True, exist_ok=True)
        # Always materialize the latest full message snapshot before archive.
        self._persist_messages()
        self._persist_messages(target=archive_path / "messages.json")
        self._persist_manifest(status=status, target=archive_path / "session.json")

        if self.manifest_path and self.manifest_path.exists():
            self.manifest_path.unlink()
        if self.messages_path and self.messages_path.exists():
            self.messages_path.unlink()
        if self._messages_delta_path and self._messages_delta_path.exists():
            self._messages_delta_path.unlink(missing_ok=True)

        if not self._legacy_layout and self._active_session_dir and self._active_session_dir.exists():
            try:
                shutil.rmtree(self._active_session_dir, ignore_errors=True)
            except Exception:
                pass

        return archive_path

    def _persist_messages(self, target: Path | None = None) -> None:
        target_path = target or self.messages_path
        if target_path is None:
            raise RuntimeError("Session message path is not initialized")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(target_path, ModelMessagesTypeAdapter.dump_json(self.message_history))

    def _persist_manifest(self, status: str, target: Path | None = None) -> None:
        self._require_active_session()
        target_path = target or self.manifest_path
        if target_path is None:
            raise RuntimeError("Session manifest path is not initialized")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at or _utc_now(),
            "status": status,
            "message_count": len(self.message_history),
            "pending_email": self.get_pending_email(),
            "storage_mode": "snapshot_delta" if (not self._legacy_layout and target is None) else "snapshot",
            "delta_count": self._delta_count if target is None else 0,
        }
        self.current_status = status
        atomic_write_json(target_path, payload, indent=2, ensure_ascii=False)

    def _load_active_session(self) -> bool:
        if self._legacy_layout:
            return self._load_session_from_paths(
                manifest_path=self.manifest_path,
                messages_path=self.messages_path,
                session_dir=self.active_dir,
            )

        candidates: list[tuple[Path, Path, Path, float]] = []
        if self.active_dir.exists():
            for entry in self.active_dir.iterdir():
                if not entry.is_dir():
                    continue
                manifest_path = entry / "session.json"
                messages_path = entry / "messages.json"
                if not manifest_path.exists() or not messages_path.exists():
                    continue
                ts = manifest_path.stat().st_mtime
                candidates.append((manifest_path, messages_path, entry, ts))

        # Backward compatibility: old flat active/session.json + active/messages.json.
        legacy_manifest = self.active_dir / "session.json"
        legacy_messages = self.active_dir / "messages.json"
        if legacy_manifest.exists() and legacy_messages.exists():
            candidates.append(
                (
                    legacy_manifest,
                    legacy_messages,
                    legacy_manifest.parent,
                    legacy_manifest.stat().st_mtime,
                )
            )

        for manifest_path, messages_path, session_dir, ts in sorted(candidates, key=lambda item: item[3], reverse=True):
            if self._load_session_from_paths(
                manifest_path=manifest_path,
                messages_path=messages_path,
                session_dir=session_dir,
            ):
                manifest_ts = _parse_iso_or_epoch(self.updated_at, ts)
                self.updated_at = datetime.fromtimestamp(manifest_ts, tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
                return True

        return False

    def _load_session_from_paths(
        self,
        *,
        manifest_path: Path | None,
        messages_path: Path | None,
        session_dir: Path,
    ) -> bool:
        if manifest_path is None or messages_path is None:
            return False
        if not manifest_path.exists() or not messages_path.exists():
            return False

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.session_id = str(manifest["session_id"])
            self.created_at = manifest.get("created_at")
            self.updated_at = manifest.get("updated_at")
            self.current_status = manifest.get("status")
            self.pending_email = self._normalize_pending_email(manifest.get("pending_email"))
            delta_path = session_dir / "messages.delta.jsonl"
            self.message_history = self._load_messages_with_optional_delta(messages_path, delta_path)
            self.manifest_path = manifest_path
            self.messages_path = messages_path
            self._active_session_dir = session_dir
            self._messages_delta_path = delta_path
            self._delta_count = int(manifest.get("delta_count", 0) or 0)
            return True
        except Exception:
            return False

    def _start_new_session(self) -> None:
        self.session_id = f"turtle_session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.created_at = _utc_now()
        self.updated_at = self.created_at
        self.current_status = "active"
        self.message_history = []
        self.pending_email = self._default_pending_email()

        if self._legacy_layout:
            if self.manifest_path is None or self.messages_path is None:
                raise RuntimeError("Legacy session paths are not initialized")
            self._messages_delta_path = self.messages_path.parent / "messages.delta.jsonl"
        else:
            self._active_session_dir = self.active_dir / self.session_id
            self._active_session_dir.mkdir(parents=True, exist_ok=True)
            self.manifest_path = self._active_session_dir / "session.json"
            self.messages_path = self._active_session_dir / "messages.json"
            self._messages_delta_path = self._active_session_dir / "messages.delta.jsonl"

        self._delta_count = 0
        self._persist_manifest(status="active")
        self._persist_messages()
        self._reset_delta_log()

    def _reset_state(self) -> None:
        self.session_id = None
        self.created_at = None
        self.updated_at = None
        self.current_status = None
        self.message_history = []
        self.pending_email = self._default_pending_email()
        self._delta_count = 0
        self._messages_delta_path = None
        if not self._legacy_layout:
            self.manifest_path = None
            self.messages_path = None
            self._active_session_dir = None

    def _has_partial_active_files(self) -> bool:
        if self._legacy_layout:
            return bool(self.manifest_path and self.manifest_path.exists()) or bool(self.messages_path and self.messages_path.exists())
        if not self.active_dir.exists():
            return False
        for entry in self.active_dir.iterdir():
            if entry.name == self._lock_path.name:
                continue
            return True
        return False

    def _quarantine_corrupt_active(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = self.archive_dir / f"corrupt_active_{timestamp}"
        target.mkdir(parents=True, exist_ok=True)

        if self._legacy_layout:
            if self.manifest_path and self.manifest_path.exists():
                shutil.move(str(self.manifest_path), str(target / "session.json"))
            if self.messages_path and self.messages_path.exists():
                shutil.move(str(self.messages_path), str(target / "messages.json"))
            return

        if not self.active_dir.exists():
            return
        for entry in self.active_dir.iterdir():
            if entry.name == self._lock_path.name:
                continue
            shutil.move(str(entry), str(target / entry.name))

    def _load_messages_with_optional_delta(self, messages_path: Path, delta_path: Path) -> list[ModelMessage]:
        base_messages = ModelMessagesTypeAdapter.validate_json(messages_path.read_bytes())
        if not delta_path.exists():
            return base_messages

        delta_payload: list[dict[str, Any]] = []
        with delta_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    if isinstance(payload, dict):
                        delta_payload.append(payload)
                except Exception:
                    continue

        if not delta_payload:
            return base_messages

        try:
            delta_messages = ModelMessagesTypeAdapter.validate_python(delta_payload)
        except Exception:
            return base_messages
        return list(base_messages) + list(delta_messages)

    @staticmethod
    def _is_append_only(previous: list[ModelMessage], current: list[ModelMessage]) -> bool:
        if len(current) < len(previous):
            return False
        if not previous:
            return True
        prev_payload = ModelMessagesTypeAdapter.dump_python(previous, mode="json")
        cur_prefix_payload = ModelMessagesTypeAdapter.dump_python(current[: len(previous)], mode="json")
        return prev_payload == cur_prefix_payload

    def _append_delta_messages(self, delta_messages: list[ModelMessage]) -> None:
        if not delta_messages or self._messages_delta_path is None:
            return
        self._messages_delta_path.parent.mkdir(parents=True, exist_ok=True)
        raw_payload = ModelMessagesTypeAdapter.dump_python(delta_messages, mode="json")
        with self._messages_delta_path.open("a", encoding="utf-8") as f:
            for message_payload in raw_payload:
                f.write(json.dumps(message_payload, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _reset_delta_log(self) -> None:
        self._delta_count = 0
        if self._messages_delta_path is None:
            return
        self._messages_delta_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(self._messages_delta_path, b"")

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

    @contextmanager
    def _store_lock(self) -> Iterator[None]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            lock_file = open(self._lock_path, "a+b")
            try:
                self._acquire_file_lock(lock_file)
                yield
            finally:
                self._release_file_lock(lock_file)
                lock_file.close()

    def _acquire_file_lock(self, lock_file: Any) -> None:
        lock_file.seek(0)
        if lock_file.tell() == 0:
            lock_file.write(b"0")
            lock_file.flush()

        deadline = time.time() + 5.0
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    return
                except OSError:
                    if time.time() >= deadline:
                        raise TimeoutError("Timed out acquiring session lock")
                    time.sleep(0.05)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return
                except OSError:
                    if time.time() >= deadline:
                        raise TimeoutError("Timed out acquiring session lock")
                    time.sleep(0.05)

    @staticmethod
    def _release_file_lock(lock_file: Any) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
