from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ALLOWED_MEMORY_TYPES = frozenset(
    {
        "identity",
        "preference",
        "workflow",
        "contact",
        "project",
        "correction",
        "relation",
        "index",
        "log",
    }
)
ALLOWED_CONFIDENCE_LEVELS = frozenset({"confirmed", "inferred", "weak_signal"})
ALLOWED_METADATA_FIELDS = frozenset(
    {
        "topic",
        "type",
        "title",
        "key",
        "confidence",
        "updated_at",
        "last_confirmed_at",
        "source",
        "source_session_id",
        "version",
        "tags",
    }
)


@dataclass(frozen=True)
class MarkdownMemoryDocument:
    metadata: dict[str, str]
    lines: list[str]


def parse_markdown_memory(text: str, *, default_topic: str | None = None) -> MarkdownMemoryDocument:
    metadata: dict[str, str] = {}
    body = text or ""

    if body.startswith("---\n"):
        parts = body.split("\n---\n", 1)
        if len(parts) != 2:
            raise ValueError("Unterminated frontmatter block")
        frontmatter, body = parts
        metadata = _parse_frontmatter(frontmatter)

    normalized_metadata = validate_memory_metadata(metadata, default_topic=default_topic)
    normalized_lines = normalize_memory_lines(body.splitlines())
    return MarkdownMemoryDocument(metadata=normalized_metadata, lines=normalized_lines)


def serialize_markdown_memory(metadata: Mapping[str, object], lines: list[str] | tuple[str, ...]) -> str:
    normalized_metadata = validate_memory_metadata(metadata)
    normalized_lines = normalize_memory_lines(lines)

    frontmatter_lines = ["---"]
    for key in sorted(normalized_metadata):
        frontmatter_lines.append(f"{key}: {normalized_metadata[key]}")
    frontmatter_lines.append("---")

    text_parts: list[str] = ["\n".join(frontmatter_lines)]
    if normalized_lines:
        text_parts.append("\n".join(normalized_lines))
    return "\n\n".join(text_parts).rstrip() + "\n"


def validate_memory_metadata(
    metadata: Mapping[str, object] | None,
    *,
    default_topic: str | None = None,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_key, raw_value in (metadata or {}).items():
        key = str(raw_key).strip().lower()
        if not key:
            continue
        if key not in ALLOWED_METADATA_FIELDS:
            raise ValueError(f"Unsupported metadata field: {key}")
        value = str(raw_value).strip()
        if not value:
            continue
        normalized[key] = value

    if default_topic:
        normalized.setdefault("topic", default_topic.strip().lower())

    topic = normalized.get("topic") or normalized.get("type")
    if not topic:
        raise ValueError("Memory metadata requires a topic or type field")
    topic = topic.strip().lower()
    if topic not in ALLOWED_MEMORY_TYPES:
        raise ValueError(f"Unsupported memory topic: {topic}")
    normalized["topic"] = topic

    if "type" in normalized and normalized["type"].strip().lower() not in ALLOWED_MEMORY_TYPES:
        raise ValueError(f"Unsupported memory type: {normalized['type']}")
    if "confidence" in normalized:
        confidence = normalized["confidence"].strip().lower()
        if confidence not in ALLOWED_CONFIDENCE_LEVELS:
            raise ValueError(f"Unsupported confidence level: {confidence}")
        normalized["confidence"] = confidence

    return normalized


def normalize_memory_lines(lines: list[str] | tuple[str, ...] | str) -> list[str]:
    if isinstance(lines, str):
        raw_lines = lines.splitlines()
    else:
        raw_lines = list(lines)

    normalized: list[str] = []
    for raw_line in raw_lines:
        line = str(raw_line).strip()
        if not line:
            continue
        if line.startswith("- "):
            normalized.append(line)
        elif line.startswith("-"):
            normalized.append(f"- {line[1:].strip()}")
        else:
            normalized.append(f"- {line}")
    return normalized


def _parse_frontmatter(frontmatter: str) -> dict[str, str]:
    lines = frontmatter.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Frontmatter must start with ---")

    parsed: dict[str, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(f"Malformed frontmatter line: {line}")
        key, value = stripped.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed
