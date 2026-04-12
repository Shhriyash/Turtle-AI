from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.io_atomic import atomic_write_text
from core.paths import (
    PERSONAL_MEMORY_CONTACTS_FILE,
    PERSONAL_MEMORY_DIR,
    PERSONAL_MEMORY_IDENTITY_FILE,
    PERSONAL_MEMORY_INDEX_FILE,
    PERSONAL_MEMORY_LOGS_DIR,
    PERSONAL_MEMORY_PREFERENCES_FILE,
    PERSONAL_MEMORY_PROJECTS_FILE,
    PERSONAL_MEMORY_WORKFLOW_FILE,
)
from core.personal_memory_schema import (
    MarkdownMemoryDocument,
    parse_markdown_memory,
    serialize_markdown_memory,
    validate_memory_metadata,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class PersonalMemoryIndexEntry:
    title: str
    file_name: str
    summary: str


class PersonalMemoryStore:
    DEFAULT_TOPICS = {
        "identity": PERSONAL_MEMORY_IDENTITY_FILE,
        "preferences": PERSONAL_MEMORY_PREFERENCES_FILE,
        "workflow": PERSONAL_MEMORY_WORKFLOW_FILE,
        "contacts": PERSONAL_MEMORY_CONTACTS_FILE,
        "projects": PERSONAL_MEMORY_PROJECTS_FILE,
    }

    def __init__(
        self,
        *,
        base_dir: Path = PERSONAL_MEMORY_DIR,
        index_path: Path = PERSONAL_MEMORY_INDEX_FILE,
        logs_dir: Path = PERSONAL_MEMORY_LOGS_DIR,
        topic_paths: dict[str, Path] | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.index_path = index_path
        self.logs_dir = logs_dir
        self.topic_paths = dict(self.DEFAULT_TOPICS)
        if topic_paths:
            self.topic_paths.update({self._normalize_topic_name(key): value for key, value in topic_paths.items()})
        self._ensure_layout()

    def load_index(self) -> list[PersonalMemoryIndexEntry]:
        if not self.index_path.exists():
            return []
        entries: list[PersonalMemoryIndexEntry] = []
        for raw_line in self.index_path.read_text(encoding="utf-8").splitlines():
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
        atomic_write_text(self.index_path, "\n".join(lines).rstrip() + ("\n" if lines else ""))

    def load_topic(self, name: str) -> MarkdownMemoryDocument:
        topic_name = self._normalize_topic_name(name)
        path = self.get_topic_path(topic_name)
        if not path.exists():
            return MarkdownMemoryDocument(
                metadata={"topic": self._topic_to_schema_type(topic_name), "updated_at": _utc_now()},
                lines=[],
            )
        return parse_markdown_memory(path.read_text(encoding="utf-8"), default_topic=self._topic_to_schema_type(topic_name))

    def write_topic(
        self,
        name: str,
        content: str | list[str] | tuple[str, ...],
        metadata: dict[str, object] | None = None,
    ) -> MarkdownMemoryDocument:
        topic_name = self._normalize_topic_name(name)
        path = self.get_topic_path(topic_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        merged_metadata = {"topic": self._topic_to_schema_type(topic_name), "updated_at": _utc_now()}
        if metadata:
            merged_metadata.update(metadata)
        normalized_metadata = validate_memory_metadata(merged_metadata)

        lines = content if isinstance(content, str) else list(content)
        serialized = serialize_markdown_memory(normalized_metadata, lines)
        atomic_write_text(path, serialized)
        return parse_markdown_memory(serialized, default_topic=normalized_metadata["topic"])

    def update_index_entry(self, name: str, summary_line: str, *, title: str | None = None) -> list[PersonalMemoryIndexEntry]:
        topic_name = self._normalize_topic_name(name)
        path = self.get_topic_path(topic_name)
        if not path.exists():
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
        log_path.parent.mkdir(parents=True, exist_ok=True)

        prefix = f"- {resolved_timestamp}"
        if session_id:
            prefix = f"{prefix} [session:{session_id}]"
        new_line = f"{prefix} {line}"

        existing = ""
        if log_path.exists():
            existing = log_path.read_text(encoding="utf-8").rstrip()
        combined = f"{existing}\n{new_line}\n" if existing else f"{new_line}\n"
        atomic_write_text(log_path, combined)
        return log_path

    def get_topic_path(self, name: str) -> Path:
        topic_name = self._normalize_topic_name(name)
        return self.topic_paths.get(topic_name, self.base_dir / f"{topic_name}.md")

    def load_profile_snapshot(self) -> dict[str, Any]:
        profile: dict[str, Any] = {
            "identity": {"name": None, "emails": [], "timezone": None},
            "preferences": {"response_style": None, "humor_level": None, "email_tone": None},
            "workflow": {"prefers_draft_before_send": None, "common_recipients": [], "email_interactions": 0},
            "tool_preferences": {"primary_llm": None},
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
        return profile

    def _ensure_layout(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            atomic_write_text(self.index_path, "")

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
        if topic.endswith("s") and topic[:-1] in {"preference", "contact", "project"}:
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
