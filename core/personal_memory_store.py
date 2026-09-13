from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config import settings
from core.io_atomic import atomic_write_text
from core.guardrails import StorageCapExceededError, enforce_storage_cap
from core.paths import personal_memory_dir, personal_memory_file
from core.personal_memory_schema import (
    MarkdownMemoryDocument,
    parse_markdown_memory,
    serialize_markdown_memory,
    validate_memory_metadata,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _body_of(document: str) -> str:
    """Everything after the YAML frontmatter, for change detection.

    Used to elide no-op topic writes: the frontmatter carries a fresh
    `updated_at` on every render, so comparing whole documents would always
    report a difference and force a needless fsync.
    """
    text = document or ""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            newline = text.find("\n", end + 1)
            return text[newline + 1:] if newline != -1 else ""
    return text


@dataclass(frozen=True)
class PersonalMemoryIndexEntry:
    title: str
    file_name: str
    summary: str


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------
# Extracted so a cloud counterpart (core/storage/cloud/personal_memory_store.py
# ::PostgresPersonalMemoryBackend) can drop in behind the same 7-method
# surface without PersonalMemoryStore's business logic (frontmatter
# serialization, no-op write elision, storage-cap enforcement, the embed-job
# enqueue) needing to know which one it's talking to. This is what closes a
# real gap found in a post-migration audit: the local file backend has ZERO
# is_cloud awareness in its constructor path (personal_memory_dir/
# personal_memory_file), so the topic markdown that
# core/personal_memory_prompt.py actually renders into every chat turn's
# prompt was silently vanishing on every serverless cold start.


class _LocalPersonalMemoryBackend:
    """Original file-based implementation, unchanged in behavior."""

    def __init__(self, base_dir: Path, index_path: Path, logs_dir: Path) -> None:
        self.base_dir = base_dir
        self.index_path = index_path
        self.logs_dir = logs_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            atomic_write_text(self.index_path, "")

    def read_topic(self, topic_path: Path) -> str | None:
        if not topic_path.exists():
            return None
        return topic_path.read_text(encoding="utf-8")

    def write_topic(self, topic_path: Path, content: str) -> None:
        topic_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(topic_path, content)

    def delete_topic(self, topic_path: Path) -> bool:
        if topic_path.exists():
            topic_path.unlink()
            return True
        return False

    def topic_exists(self, topic_path: Path) -> bool:
        return topic_path.exists()

    def topic_size_bytes(self, topic_path: Path) -> int:
        try:
            return topic_path.stat().st_size if topic_path.exists() else 0
        except OSError:
            return 0

    def read_index(self) -> str | None:
        if not self.index_path.exists():
            return None
        return self.index_path.read_text(encoding="utf-8")

    def write_index(self, content: str) -> None:
        atomic_write_text(self.index_path, content)

    def read_daily_log(self, log_path: Path) -> str | None:
        if not log_path.exists():
            return None
        return log_path.read_text(encoding="utf-8")

    def write_daily_log(self, log_path: Path, content: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(log_path, content)

    def enforce_cap(self, user_id: str, *, incoming_bytes: int) -> None:
        enforce_storage_cap(user_id, self.base_dir, incoming_bytes=incoming_bytes)


class _CloudPersonalMemoryBackend:
    """Adapts core.storage.cloud.personal_memory_store.PostgresPersonalMemoryBackend
    (which is keyed by plain topic-name strings) to the Path-shaped interface
    _LocalPersonalMemoryBackend uses, so PersonalMemoryStore's methods don't
    need an is_cloud branch of their own at every call site — only in
    __init__, choosing which backend to build.
    """

    def __init__(self, user_id: str) -> None:
        from core.storage.cloud.personal_memory_store import PostgresPersonalMemoryBackend

        self._pg = PostgresPersonalMemoryBackend(user_id)
        self.user_id = user_id

    @staticmethod
    def _topic_key(topic_path: Path) -> str:
        # Paths here are always base_dir/<topic>.md — stem is the topic name.
        return topic_path.stem

    def read_topic(self, topic_path: Path) -> str | None:
        return self._pg.read_topic(self._topic_key(topic_path))

    def write_topic(self, topic_path: Path, content: str) -> None:
        self._pg.write_topic(self._topic_key(topic_path), content)

    def delete_topic(self, topic_path: Path) -> bool:
        return self._pg.delete_topic(self._topic_key(topic_path))

    def topic_exists(self, topic_path: Path) -> bool:
        return self._pg.read_topic(self._topic_key(topic_path)) is not None

    def topic_size_bytes(self, topic_path: Path) -> int:
        content = self._pg.read_topic(self._topic_key(topic_path))
        return len(content.encode("utf-8")) if content else 0

    def read_index(self) -> str | None:
        return self._pg.read_index()

    def write_index(self, content: str) -> None:
        self._pg.write_index(content)

    def read_daily_log(self, log_path: Path) -> str | None:
        return self._pg.read_daily_log(self._log_key(log_path))

    def write_daily_log(self, log_path: Path, content: str) -> None:
        self._pg.write_daily_log(self._log_key(log_path), content)

    @staticmethod
    def _log_key(log_path: Path) -> str:
        # Local layout is logs_dir/YYYY/MM/YYYY-MM-DD.md; the date stem alone
        # is already a unique, sortable key without the directory nesting.
        return log_path.stem

    def enforce_cap(self, user_id: str, *, incoming_bytes: int) -> None:
        cap_mb = int(settings.user_storage_cap_mb)
        if cap_mb <= 0:
            return
        cap_bytes = cap_mb * 1024 * 1024
        used = self._pg.total_bytes() + max(0, incoming_bytes)
        if used > cap_bytes:
            raise StorageCapExceededError(user_id, used, cap_bytes)


class PersonalMemoryStore:
    def __init__(
        self,
        user_id: str = "default",
        *,
        base_dir: Path | None = None,
        index_path: Path | None = None,
        logs_dir: Path | None = None,
        topic_paths: dict[str, Path] | None = None,
    ) -> None:
        self.user_id = user_id
        self.base_dir = base_dir or personal_memory_dir(user_id)
        self.index_path = index_path or personal_memory_file(user_id, "MEMORY.md")
        self.logs_dir = logs_dir or (self.base_dir / "logs")

        self.DEFAULT_TOPICS = {
            "identity": personal_memory_file(user_id, "identity.md"),
            "preferences": personal_memory_file(user_id, "preferences.md"),
            "workflow": personal_memory_file(user_id, "workflow.md"),
            "contacts": personal_memory_file(user_id, "contacts.md"),
            "projects": personal_memory_file(user_id, "projects.md"),
            "corrections": personal_memory_file(user_id, "corrections.md"),
            "relations": personal_memory_file(user_id, "relations.md"),
            "working_style": personal_memory_file(user_id, "working_style.md"),
            "communication_style": personal_memory_file(user_id, "communication_style.md"),
            "tool_preferences": personal_memory_file(user_id, "tool_preferences.md"),
            "decision_style": personal_memory_file(user_id, "decision_style.md"),
        }
        self.topic_paths = dict(self.DEFAULT_TOPICS)
        if topic_paths:
            self.topic_paths.update({self._normalize_topic_name(key): value for key, value in topic_paths.items()})

        # An explicit base_dir/index_path/etc. always forces the local
        # backend (test isolation must survive cloud mode unchanged, same
        # rule JournalStore/ConfirmationGate use elsewhere in this
        # migration); otherwise settings.is_cloud picks Postgres vs local.
        explicit_paths = base_dir is not None or index_path is not None or logs_dir is not None
        if not explicit_paths and settings.is_cloud and user_id and user_id not in {"", "default"}:
            self._backend = _CloudPersonalMemoryBackend(user_id)
        else:
            self._backend = _LocalPersonalMemoryBackend(self.base_dir, self.index_path, self.logs_dir)

    def load_index(self) -> list[PersonalMemoryIndexEntry]:
        raw = self._backend.read_index()
        if not raw:
            return []
        entries: list[PersonalMemoryIndexEntry] = []
        for raw_line in raw.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            entry = self._parse_index_line(line)
            if entry:
                entries.append(entry)
        return entries

    def save_index(self, entries: list[PersonalMemoryIndexEntry]) -> None:
        unique_by_file: dict[str, PersonalMemoryIndexEntry] = {}
        for entry in entries:
            file_name = entry.file_name.strip()
            if not file_name:
                continue
            unique_by_file[file_name] = PersonalMemoryIndexEntry(
                title=entry.title.strip() or self._derive_title_from_path(Path(file_name)),
                file_name=file_name,
                summary=self._normalize_summary(entry.summary),
            )

        ordered = sorted(unique_by_file.values(), key=self._index_sort_key)
        lines = [
            f"- [{entry.title}]({entry.file_name}) - {entry.summary}"
            for entry in ordered
        ]
        self._backend.write_index("\n".join(lines).rstrip() + ("\n" if lines else ""))

    def load_topic(self, name: str) -> MarkdownMemoryDocument:
        topic_name = self._normalize_topic_name(name)
        path = self.get_topic_path(topic_name)
        raw = self._backend.read_topic(path)
        if raw is None:
            return MarkdownMemoryDocument(
                metadata={"topic": self._topic_to_schema_type(topic_name), "updated_at": _utc_now()},
                lines=[],
            )
        return parse_markdown_memory(raw, default_topic=self._topic_to_schema_type(topic_name))

    def write_topic(
        self,
        name: str,
        content: str | list[str] | tuple[str, ...],
        metadata: dict[str, object] | None = None,
    ) -> MarkdownMemoryDocument:
        topic_name = self._normalize_topic_name(name)
        path = self.get_topic_path(topic_name)

        merged_metadata = {"topic": self._topic_to_schema_type(topic_name), "updated_at": _utc_now()}
        if metadata:
            merged_metadata.update(metadata)
        normalized_metadata = validate_memory_metadata(merged_metadata)

        lines = content if isinstance(content, str) else list(content)
        serialized = serialize_markdown_memory(normalized_metadata, lines)

        # NO-OP WRITE ELISION. `updated_at` is stamped fresh on every call, so a
        # byte comparison always differs and every replay rewrote every topic
        # file — a write per topic, per replay. replay() runs on every
        # fact-storing turn, so this was pure write latency on the hot path
        # for content that had not changed. Compare the BODY (everything
        # after the frontmatter); if it is identical, keep the existing
        # content and skip the write.
        existing_raw = self._backend.read_topic(path)
        if existing_raw is not None:
            try:
                if _body_of(existing_raw) == _body_of(serialized):
                    return parse_markdown_memory(existing_raw)
            except Exception:
                pass  # unreadable/corrupt — fall through and rewrite

        # Phase 6: enforce per-user storage cap before writing.
        existing_size = self._backend.topic_size_bytes(path)
        delta = max(0, len(serialized.encode("utf-8")) - existing_size)
        self._backend.enforce_cap(self.user_id, incoming_bytes=delta)
        self._backend.write_topic(path, serialized)

        # D5/G3: Enqueue embedding job.
        # Skip the enqueue entirely for the un-scoped default/empty tenant:
        # single-tenant/"default" stores are test/legacy constructs (real
        # tenants are usr_*), and embedding for them would land in the SHARED
        # data/memory/personal/default/vector index — cross-tenant collapse.
        if self.user_id and self.user_id not in {"", "default"}:
            from core.worker import dispatch_embed_personal_memory_job
            # Never hand the job a bare string: it would iterate characters.
            embed_lines = lines.splitlines() if isinstance(lines, str) else lines
            dispatch_embed_personal_memory_job(self.user_id, topic_name, embed_lines)

        return parse_markdown_memory(serialized, default_topic=normalized_metadata["topic"])

    def delete_topic(self, name: str) -> bool:
        """Remove a topic entirely (used when replay() finds no live facts
        left for it). Returns True if something was actually deleted.
        Previously callers (core/memory_replayer.py) did this by reaching
        into store.get_topic_path(...) and calling Path.unlink() directly —
        a no-op in cloud mode, silently leaving stale content behind forever
        since there is no local file to unlink there."""
        topic_name = self._normalize_topic_name(name)
        path = self.get_topic_path(topic_name)
        return self._backend.delete_topic(path)

    def topic_exists(self, name: str) -> bool:
        topic_name = self._normalize_topic_name(name)
        path = self.get_topic_path(topic_name)
        return self._backend.topic_exists(path)

    def update_index_entry(self, name: str, summary_line: str, *, title: str | None = None) -> list[PersonalMemoryIndexEntry]:
        topic_name = self._normalize_topic_name(name)
        path = self.get_topic_path(topic_name)
        if not self._backend.topic_exists(path):
            raise FileNotFoundError(f"Cannot index missing topic file: {path}")

        file_name = path.name
        next_entry = PersonalMemoryIndexEntry(
            title=(title or self._derive_title_from_path(path)).strip(),
            file_name=file_name,
            summary=self._normalize_summary(summary_line),
        )
        entries = [entry for entry in self.load_index() if entry.file_name != file_name]
        entries.append(next_entry)
        self.save_index(entries)
        return self.load_index()

    def append_daily_log(self, entry: str, *, session_id: str | None = None, timestamp: str | None = None) -> Path:
        line = str(entry).strip()
        if not line:
            raise ValueError("Daily log entry cannot be empty")

        resolved_timestamp = timestamp or _utc_now()
        dt = datetime.fromisoformat(resolved_timestamp.replace("Z", "+00:00"))
        log_path = self.logs_dir / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}.md"

        prefix = f"- {resolved_timestamp}"
        if session_id:
            prefix = f"{prefix} [session:{session_id}]"
        new_line = f"{prefix} {line}"

        existing_raw = self._backend.read_daily_log(log_path)
        existing = existing_raw.rstrip() if existing_raw else ""
        combined = f"{existing}\n{new_line}\n" if existing else f"{new_line}\n"
        self._backend.write_daily_log(log_path, combined)
        return log_path

    def get_topic_path(self, name: str) -> Path:
        topic_name = self._normalize_topic_name(name)
        return self.topic_paths.get(topic_name, self.base_dir / f"{topic_name}.md")

    def load_profile_snapshot(self) -> dict[str, Any]:
        profile: dict[str, Any] = {
            "identity": {"name": None, "emails": [], "timezone": None},
            "preferences": {"response_style": None, "humor_level": None, "email_tone": None},
            "workflow": {"prefers_draft_before_send": None, "common_recipients": [], "email_interactions": 0},
            "tool_preferences": {"primary_llm": None, "tools": []},
            "working_style": {"notes": []},
            "communication_style": {"notes": []},
            "decision_style": {"notes": []},
        }

        identity = self.load_topic("identity")
        for line in identity.lines:
            content = self._strip_bullet(line)
            lowered = content.lower()
            if lowered.startswith("name:"):
                profile["identity"]["name"] = content.split(":", 1)[1].strip() or None
            elif lowered.startswith("primary email:"):
                value = content.split(":", 1)[1].strip().lower()
                if value and value not in profile["identity"]["emails"]:
                    profile["identity"]["emails"].insert(0, value)
            elif lowered.startswith("known email:"):
                value = content.split(":", 1)[1].strip().lower()
                if value and value not in profile["identity"]["emails"]:
                    profile["identity"]["emails"].append(value)
            elif lowered.startswith("timezone:"):
                profile["identity"]["timezone"] = content.split(":", 1)[1].strip() or None

        preferences = self.load_topic("preferences")
        for line in preferences.lines:
            content = self._strip_bullet(line)
            lowered = content.lower()
            if lowered.startswith("response style:"):
                profile["preferences"]["response_style"] = content.split(":", 1)[1].strip() or None
            elif lowered.startswith("humor level:"):
                profile["preferences"]["humor_level"] = content.split(":", 1)[1].strip() or None
            elif lowered.startswith("email tone:"):
                profile["preferences"]["email_tone"] = content.split(":", 1)[1].strip() or None

        workflow = self.load_topic("workflow")
        routines: list[dict[str, Any]] = []
        for line in workflow.lines:
            content = self._strip_bullet(line)
            lowered = content.lower()
            if lowered.startswith("prefers draft before send:"):
                value = content.split(":", 1)[1].strip().lower()
                if value in {"true", "false"}:
                    profile["workflow"]["prefers_draft_before_send"] = value == "true"
            elif lowered.startswith("email interactions recorded:"):
                try:
                    profile["workflow"]["email_interactions"] = int(content.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif lowered.startswith("preferred primary model:"):
                profile["tool_preferences"]["primary_llm"] = content.split(":", 1)[1].strip() or None
            elif lowered.startswith("routine:"):
                parsed = self._parse_routine_line(content)
                if parsed is not None:
                    routines.append(parsed)
        profile["workflow"]["routines"] = routines

        contacts = self.load_topic("contacts")
        recipients: list[str] = []
        for line in contacts.lines:
            content = self._strip_bullet(line)
            lowered = content.lower()
            if not lowered.startswith("frequent recipient:"):
                continue
            value = content.split(":", 1)[1].strip()
            if " (count:" in value:
                value = value.split(" (count:", 1)[0].strip()
            normalized = value.lower()
            if normalized and normalized not in recipients:
                recipients.append(normalized)
        profile["workflow"]["common_recipients"] = recipients

        for topic_key in ("working_style", "communication_style", "decision_style"):
            doc = self.load_topic(topic_key)
            notes: list[str] = []
            for line in doc.lines:
                content = self._strip_bullet(line).strip()
                if content:
                    notes.append(content)
            profile[topic_key]["notes"] = notes

        tool_prefs = self.load_topic("tool_preferences")
        tools: list[str] = []
        for line in tool_prefs.lines:
            content = self._strip_bullet(line)
            lowered = content.lower()
            if lowered.startswith("preferred primary model:"):
                profile["tool_preferences"]["primary_llm"] = content.split(":", 1)[1].strip() or None
            elif lowered.startswith("tool:"):
                value = content.split(":", 1)[1].strip()
                if value and value not in tools:
                    tools.append(value)
        profile["tool_preferences"]["tools"] = tools

        return profile

    @staticmethod
    def _parse_routine_line(content: str) -> dict[str, Any] | None:
        """D4: parse a `Routine: <name> | <schedule> | items: a, b` line.

        Schedule is `<cadence>[ <HH:MM>][ <timezone>]`. Items section is optional.
        Returns a dict with keys: routine, cadence, time?, timezone?, items?.
        """
        body = content.split(":", 1)[1].strip() if ":" in content else ""
        if not body:
            return None
        parts = [p.strip() for p in body.split("|")]
        if not parts or not parts[0]:
            return None
        routine = parts[0]
        out: dict[str, Any] = {"routine": routine}
        if len(parts) >= 2 and parts[1]:
            tokens = parts[1].split()
            if tokens:
                out["cadence"] = tokens[0].lower()
            for tok in tokens[1:]:
                if re.match(r"^([01]?\d|2[0-3]):[0-5]\d$", tok):
                    hh, mm = tok.split(":")
                    out["time"] = f"{int(hh):02d}:{mm}"
                elif "/" in tok or tok == "UTC":
                    out["timezone"] = tok
        for seg in parts[2:]:
            low = seg.lower()
            if low.startswith("items:"):
                items_raw = seg.split(":", 1)[1]
                items = [i.strip() for i in items_raw.split(",") if i.strip()]
                if items:
                    out["items"] = items
        return out

    @staticmethod
    def _strip_bullet(line: str) -> str:
        stripped = str(line).strip()
        if stripped.startswith("- "):
            return stripped[2:].strip()
        if stripped.startswith("-"):
            return stripped[1:].strip()
        return stripped

    @staticmethod
    def _normalize_topic_name(name: str) -> str:
        normalized = str(name).strip().lower()
        if normalized.endswith(".md"):
            normalized = normalized[:-3]
        return normalized.replace(" ", "_")

    @staticmethod
    def _topic_to_schema_type(topic_name: str) -> str:
        topic = topic_name
        if topic.endswith("s") and topic[:-1] in {"preference", "contact", "project", "correction", "relation"}:
            return topic[:-1]
        return topic

    @staticmethod
    def _normalize_summary(summary: str) -> str:
        normalized = " ".join(str(summary).split()).strip()
        if not normalized:
            raise ValueError("Index summary cannot be empty")
        return normalized

    @staticmethod
    def _derive_title_from_path(path: Path) -> str:
        stem = path.stem.replace("_", " ").strip()
        return stem.title() if stem else "Memory"

    def _index_sort_key(self, entry: PersonalMemoryIndexEntry) -> tuple[int, str]:
        order = {path.name: index for index, path in enumerate(self.DEFAULT_TOPICS.values())}
        return (order.get(entry.file_name, len(order)), entry.file_name)

    @staticmethod
    def _parse_index_line(line: str) -> PersonalMemoryIndexEntry | None:
        if not line.startswith("- ["):
            return None
        try:
            title_end = line.index("](")
            file_end = line.index(")", title_end + 2)
        except ValueError:
            return None

        title = line[3:title_end].strip()
        file_name = line[title_end + 2:file_end].strip()
        remainder = line[file_end + 1:].strip()
        if remainder.startswith("-"):
            remainder = remainder[1:].strip()
        if not title or not file_name or not remainder:
            return None
        return PersonalMemoryIndexEntry(title=title, file_name=file_name, summary=remainder)
