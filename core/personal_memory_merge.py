from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from core.personal_memory_extract import PersonalMemoryCandidate
from core.personal_memory_store import PersonalMemoryStore


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


TOPIC_TITLES = {
    "identity": "Identity",
    "preferences": "Preferences",
    "workflow": "Workflow",
    "contacts": "Contacts",
    "projects": "Projects",
}

TOPIC_SUMMARIES = {
    "identity": "Name, email, location, timezone, language, and role",
    "preferences": "Tone, response style, and delivery defaults",
    "workflow": "Recurring habits and operational defaults",
    "contacts": "Frequent recipients and confirmed aliases",
    "projects": "Project context and recurring work references",
}

REPLACEABLE_LABELS = {
    "name": "Name",
    "home_city": "Home city",
    "current_city": "Current city",
    "country": "Country",
    "primary_email": "Primary email",
    "timezone": "Timezone",
    "preferred_language": "Preferred language",
    "occupation": "Occupation",
    "company": "Company",
    "response_style": "Response style",
    "humor_level": "Humor level",
    "email_tone": "Email tone",
    "prefers_draft_before_send": "Prefers draft before send",
    "primary_llm": "Preferred primary model",
}

LINE_SORT_ORDER = [
    "Name",
    "Home city",
    "Current city",
    "Country",
    "Primary email",
    "Known email",
    "Timezone",
    "Preferred language",
    "Occupation",
    "Company",
    "Response style",
    "Humor level",
    "Email tone",
    "Prefers draft before send",
    "Preferred primary model",
    "Frequent recipient",
    "Project",
]


@dataclass(frozen=True)
class PersonalMemoryMergeResult:
    written_topics: list[str]
    written_candidate_count: int
    skipped_candidate_count: int


def merge_personal_memory_candidates(
    *,
    store: PersonalMemoryStore,
    candidates: list[PersonalMemoryCandidate],
) -> PersonalMemoryMergeResult:
    grouped: dict[str, list[PersonalMemoryCandidate]] = {}
    skipped = 0

    for candidate in candidates:
        if not _should_persist(candidate):
            skipped += 1
            continue
        grouped.setdefault(candidate.topic, []).append(candidate)

    written_topics: list[str] = []
    written_candidate_count = 0

    for topic, topic_candidates in grouped.items():
        document = store.load_topic(topic)
        lines = list(document.lines)
        candidate_count = 0

        for candidate in topic_candidates:
            updated_lines = _apply_candidate(lines, candidate)
            if updated_lines != lines:
                lines = updated_lines
                candidate_count += 1

        if candidate_count == 0:
            continue

        metadata = dict(document.metadata)
        metadata.update(
            {
                "topic": document.metadata.get("topic") or topic.rstrip("s"),
                "title": TOPIC_TITLES.get(topic, topic.replace("_", " ").title()),
                "updated_at": _utc_now(),
                "confidence": _topic_confidence(topic_candidates),
            }
        )
        source_session_id = _latest_source_session_id(topic_candidates)
        if source_session_id:
            metadata["source_session_id"] = source_session_id

        normalized_lines = _sort_lines(lines)
        store.write_topic(topic, normalized_lines, metadata)
        store.update_index_entry(topic, TOPIC_SUMMARIES.get(topic, "Durable memory topic"))
        written_topics.append(topic)
        written_candidate_count += candidate_count

    return PersonalMemoryMergeResult(
        written_topics=written_topics,
        written_candidate_count=written_candidate_count,
        skipped_candidate_count=skipped,
    )


def _should_persist(candidate: PersonalMemoryCandidate) -> bool:
    if candidate.sensitivity != "normal":
        return False
    if candidate.confidence == "weak_signal":
        return False
    return True


def _apply_candidate(lines: list[str], candidate: PersonalMemoryCandidate) -> list[str]:
    next_lines = list(lines)

    if candidate.overwrite_policy == "replace":
        label = REPLACEABLE_LABELS.get(candidate.key)
        if not label:
            return next_lines
        filtered = [
            line
            for line in next_lines
            if not line.lower().startswith(f"- {label.lower()}:")
        ]
        filtered.append(candidate.line)
        return filtered

    if candidate.overwrite_policy == "append_unique":
        if candidate.line not in next_lines:
            next_lines.append(candidate.line)
        return next_lines

    return next_lines


def _topic_confidence(candidates: list[PersonalMemoryCandidate]) -> str:
    if any(candidate.confidence == "confirmed" for candidate in candidates):
        return "confirmed"
    if any(candidate.confidence == "inferred" for candidate in candidates):
        return "inferred"
    return "weak_signal"


def _latest_source_session_id(candidates: list[PersonalMemoryCandidate]) -> str | None:
    for candidate in reversed(candidates):
        if candidate.source_session_id:
            return candidate.source_session_id
    return None


def _sort_lines(lines: list[str]) -> list[str]:
    def line_sort_key(line: str) -> tuple[int, str]:
        label = line[2:].split(":", 1)[0].strip() if line.startswith("- ") else line
        try:
            index = LINE_SORT_ORDER.index(label)
        except ValueError:
            index = len(LINE_SORT_ORDER)
        return (index, line.lower())

    return sorted({line.strip() for line in lines if line.strip()}, key=line_sort_key)
