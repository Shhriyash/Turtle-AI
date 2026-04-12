from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.personal_memory_store import PersonalMemoryIndexEntry, PersonalMemoryStore


def _estimate_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


@dataclass(frozen=True)
class PersonalMemoryPromptConfig:
    max_bytes: int = 2048
    max_topic_files: int = 2
    max_index_entries: int = 8


class PersonalMemoryPromptBuilder:
    """Build a compact prompt-memory block from Markdown-backed personal memory."""

    def __init__(
        self,
        store: PersonalMemoryStore,
        *,
        config: PersonalMemoryPromptConfig | None = None,
    ) -> None:
        self.store = store
        self.config = config or PersonalMemoryPromptConfig()

    def build_memory_block(self, *, task_type: str, query: str) -> str:
        entries = self.store.load_index()
        selected_topics = self._select_topics(task_type=task_type, query=query, entries=entries)

        sections: list[str] = []
        for topic_name in selected_topics[: self.config.max_topic_files]:
            document = self.store.load_topic(topic_name)
            if document.lines:
                title = document.metadata.get("title") or topic_name.replace("_", " ").title()
                section = "\n".join([f"[{title}]", *document.lines])
                sections.append(section)

        if not sections and entries:
            fallback_lines = []
            for entry in entries[: self.config.max_index_entries]:
                fallback_lines.append(f"- {entry.title}: {entry.summary}")
            if fallback_lines:
                sections.append("[Known Memory]\n" + "\n".join(fallback_lines))

        if not sections:
            return ""

        return self._truncate_sections(sections)

    def _select_topics(
        self,
        *,
        task_type: str,
        query: str,
        entries: list[PersonalMemoryIndexEntry],
    ) -> list[str]:
        selected: list[str] = []
        available_topic_names = {
            self._normalize_topic_name(entry.file_name)
            for entry in entries
        }
        if not available_topic_names:
            available_topic_names = {
                self._normalize_topic_name(path.name)
                for path in self.store.topic_paths.values()
                if path.exists()
            }

        def add(topic_name: str) -> None:
            normalized = self._normalize_topic_name(topic_name)
            if normalized in selected:
                return
            if available_topic_names and normalized not in available_topic_names:
                return
            selected.append(normalized)

        lowered = (query or "").lower()

        if task_type == "email":
            for topic in ("identity", "contacts", "workflow", "preferences"):
                add(topic)
        elif task_type == "web":
            for topic in ("preferences", "projects", "workflow"):
                add(topic)
        elif task_type == "url":
            for topic in ("projects", "preferences", "workflow"):
                add(topic)
        else:
            for topic in ("preferences", "identity", "workflow"):
                add(topic)

        if any(token in lowered for token in ("name", "who am i", "call me", "timezone", "email address", "my email")):
            add("identity")
        if any(token in lowered for token in ("prefer", "usually", "style", "tone", "concise", "detailed", "default response")):
            add("preferences")
        if any(token in lowered for token in ("workflow", "habit", "normally", "default", "often", "send it", "recipient")):
            add("workflow")
        if any(token in lowered for token in ("recipient", "contact", "send to", "mail to", "cc ", "bcc ")):
            add("contacts")
        if any(token in lowered for token in ("project", "repo", "repository", "codebase", "working on")):
            add("projects")

        return selected

    def _truncate_sections(self, sections: Iterable[str]) -> str:
        assembled: list[str] = []
        total_bytes = 0
        for section in sections:
            if not section.strip():
                continue
            candidate = section.strip()
            separator_bytes = 2 if assembled else 0
            candidate_bytes = _estimate_bytes(candidate)
            if assembled and total_bytes + separator_bytes + candidate_bytes > self.config.max_bytes:
                break
            if not assembled and candidate_bytes > self.config.max_bytes:
                assembled.append(self._truncate_text(candidate, self.config.max_bytes))
                break
            if assembled:
                total_bytes += separator_bytes
            assembled.append(candidate)
            total_bytes += candidate_bytes

        return "\n\n".join(assembled).strip()

    @staticmethod
    def _truncate_text(text: str, max_bytes: int) -> str:
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text
        truncated = encoded[: max(0, max_bytes - 3)].decode("utf-8", "ignore").rstrip()
        return f"{truncated}..."

    @staticmethod
    def _normalize_topic_name(file_name: str) -> str:
        normalized = str(file_name).strip().lower()
        if normalized.endswith(".md"):
            normalized = normalized[:-3]
        return normalized.replace(" ", "_")
