from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from core.memory_journal import JournalStore, MemoryEvent, make_event
from core.memory_replayer import replay
from core.personal_memory_store import PersonalMemoryStore


MIGRATION_SESSION_ID = "migration"


@dataclass(frozen=True)
class MigrationResult:
    emitted_event_count: int
    written_topics: list[str]
    dry_run: bool


def migrate_existing_topics(
    *,
    store: PersonalMemoryStore,
    journal: JournalStore,
    dry_run: bool = False,
) -> MigrationResult:
    """Scan current topic markdown files and emit synthetic journal events.

    Migration events are stamped source=migration, confidence=1.0, with
    observed_at set to the topic file's mtime. They are decay-exempt
    per the persistence plan.
    """
    events: list[MemoryEvent] = []
    topic_order = ("identity", "preferences", "workflow", "contacts", "projects", "corrections")

    for index, topic in enumerate(topic_order):
        path = store.get_topic_path(topic)
        if not path.exists():
            continue
        observed_at = _mtime_iso(path)
        document = store.load_topic(topic)
        for line_index, line in enumerate(document.lines):
            event = _line_to_event(
                topic=topic,
                line=line,
                observed_at=observed_at,
                turn_id=f"migration-{topic}-{line_index}",
            )
            if event is not None:
                events.append(event)

    if not dry_run:
        journal.append_many(events)
        all_events = journal.load_all()
        result = replay(all_events, store=store)
        return MigrationResult(
            emitted_event_count=len(events),
            written_topics=result.written_topics,
            dry_run=False,
        )

    return MigrationResult(
        emitted_event_count=len(events),
        written_topics=[],
        dry_run=True,
    )


def _mtime_iso(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    except Exception:
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _line_to_event(
    *,
    topic: str,
    line: str,
    observed_at: str,
    turn_id: str,
) -> MemoryEvent | None:
    stripped = line.strip()
    if stripped.startswith("- "):
        stripped = stripped[2:].strip()
    if not stripped or ":" not in stripped:
        return None

    label, _, raw_value = stripped.partition(":")
    label = label.strip().lower()
    raw_value = raw_value.strip()
    if not raw_value:
        return None

    mapping = _LINE_MAPPING.get((topic, label))
    if mapping is None:
        mapping = _LINE_MAPPING.get((topic, label.split(" ", 1)[0]))
    if mapping is None:
        return None

    key, value = mapping(raw_value)
    if key is None or value is None:
        return None

    return make_event(
        kind=_kind_for_topic(topic),
        topic=topic,
        key=key,
        value=value,
        confidence=1.0,
        source="migration",
        extractor="migration",
        session_id=MIGRATION_SESSION_ID,
        turn_id=turn_id,
        observed_at=observed_at,
        evidence={"migrated_line": line},
        applied=True,
    )


def _kind_for_topic(topic: str) -> str:
    if topic == "identity":
        return "fact"
    if topic == "corrections":
        return "correction"
    if topic == "contacts":
        return "fact"
    if topic == "projects":
        return "fact"
    return "preference"


def _parse_identity_name(raw: str) -> tuple[str, dict]:
    return "identity.name", {"name": raw}


def _parse_primary_email(raw: str) -> tuple[str, dict]:
    email = raw.strip().lower()
    return "identity.primary_email", {"primary_email": email}


def _parse_known_email(raw: str) -> tuple[str, dict]:
    email = raw.strip().lower()
    key = f"identity.known_email.{email}"
    return key, {"email": email}


def _parse_timezone(raw: str) -> tuple[str, dict]:
    return "identity.timezone", {"timezone": raw}


def _parse_response_style(raw: str) -> tuple[str, dict]:
    return "preferences.response_style", {"response_style": raw.lower()}


def _parse_humor_level(raw: str) -> tuple[str, dict]:
    return "preferences.humor_level", {"humor_level": raw.lower()}


def _parse_email_tone(raw: str) -> tuple[str, dict]:
    return "preferences.email_tone", {"email_tone": raw.lower()}


def _parse_draft_flag(raw: str) -> tuple[str, dict]:
    flag = raw.strip().lower() == "true"
    return "workflow.prefers_draft_before_send", {"prefers_draft_before_send": flag}


def _parse_primary_llm(raw: str) -> tuple[str, dict]:
    return "workflow.primary_llm", {"primary_llm": raw}


def _parse_email_interactions(raw: str) -> tuple[str, dict]:
    try:
        count = int(raw.strip())
    except Exception:
        return "", {}
    return "workflow.email_interactions_recorded", {"count": count}


_FREQUENT_RECIPIENT_RE = re.compile(r"^([^\s(]+)(?:\s*\(count:\s*(\d+)\))?$", re.IGNORECASE)


def _parse_frequent_recipient(raw: str) -> tuple[str, dict]:
    match = _FREQUENT_RECIPIENT_RE.match(raw.strip())
    if not match:
        return "", {}
    email = match.group(1).strip().lower()
    count_raw = match.group(2)
    value: dict = {"email": email}
    if count_raw is not None:
        try:
            value["count"] = int(count_raw)
        except Exception:
            pass
    return f"contacts.frequent_recipient.{email}", value


def _parse_project(raw: str) -> tuple[str, dict]:
    name, _, summary = raw.partition("—")
    name = name.strip() or raw.strip()
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unnamed"
    value: dict = {"name": name}
    if summary.strip():
        value["summary"] = summary.strip()
    return f"projects.project.{slug}", value


def _parse_correction(raw: str) -> tuple[str, dict]:
    slug = re.sub(r"[^a-z0-9]+", "_", raw.lower())[:40].strip("_") or "note"
    return f"corrections.{slug}", {"summary": raw}


_LINE_MAPPING: dict[tuple[str, str], callable] = {
    ("identity", "name"): _parse_identity_name,
    ("identity", "primary email"): _parse_primary_email,
    ("identity", "known email"): _parse_known_email,
    ("identity", "timezone"): _parse_timezone,
    ("preferences", "response style"): _parse_response_style,
    ("preferences", "humor level"): _parse_humor_level,
    ("preferences", "email tone"): _parse_email_tone,
    ("workflow", "prefers draft before send"): _parse_draft_flag,
    ("workflow", "preferred primary model"): _parse_primary_llm,
    ("workflow", "email interactions recorded"): _parse_email_interactions,
    ("contacts", "frequent recipient"): _parse_frequent_recipient,
    ("projects", "project"): _parse_project,
    ("corrections", "correction"): _parse_correction,
}
