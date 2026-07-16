from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterable

from core.memory_journal import MemoryEvent
from core.personal_memory_store import PersonalMemoryStore


DECAY_DAYS = 30
# Topics whose events never expire (name, email, timezone are load-bearing identity facts).
_DECAY_EXEMPT_TOPICS = frozenset({"identity"})
# Events written by the migration pass are considered authoritative until explicitly corrected.
_DECAY_EXEMPT_SOURCE = "migration"


TOPIC_TITLES = {
    "identity": "Identity",
    "preferences": "Preferences",
    "workflow": "Workflow",
    "contacts": "Contacts",
    "projects": "Projects",
    "corrections": "Corrections",
    "relations": "Relations",
    "working_style": "Working Style",
    "communication_style": "Communication Style",
    "tool_preferences": "Tool Preferences",
    "decision_style": "Decision Style",
}

TOPIC_SUMMARIES = {
    "identity": "Name, email, location, timezone, language, and role",
    "preferences": "Tone, response style, and delivery defaults",
    "workflow": "Recurring habits and operational defaults",
    "contacts": "Frequent recipients and confirmed aliases",
    "projects": "Project context and recurring work references",
    "corrections": "User corrections and how to apply them",
    "relations": "People in the user's life and their roles",
    "working_style": "How the user approaches tasks",
    "communication_style": "How the user likes to receive information",
    "tool_preferences": "Tools, languages, and environments the user favours",
    "decision_style": "How the user makes decisions",
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
    "Email interactions recorded",
    "Preferred primary model",
    "Frequent recipient",
    "Project",
    "Correction",
]

ALL_TOPICS = (
    "identity",
    "preferences",
    "workflow",
    "contacts",
    "projects",
    "corrections",
    "relations",
    "working_style",
    "communication_style",
    "tool_preferences",
    "decision_style",
)


@dataclass(frozen=True)
class ReplayResult:
    written_topics: list[str]
    cleared_topics: list[str]
    resolved_event_count: int


def replay(
    events: Iterable[MemoryEvent],
    *,
    store: PersonalMemoryStore,
    reference_time: datetime | None = None,
) -> ReplayResult:
    """Project journal events into topic markdown files.

    Deterministic. Same input events -> same output files.

    Only ``applied=True`` events are rendered. Candidates written by Stage B
    (LLM per-session extractor) or Stage C (dream pass) live in the journal
    with ``applied=False`` and become visible only after the confirmation
    gate promotes them.

    Supersedes links are resolved against the full event list first so a
    ``supersedes`` pointer on a non-applied contradiction still drops the
    old event from the rendered projection.

    Events older than ``DECAY_DAYS`` days are excluded unless they are exempt
    (``topic=identity`` or ``source=migration``). Pass ``reference_time`` in
    tests to control the clock.
    """
    ref = reference_time or datetime.now(UTC)
    full_events = [event for event in events if not event.rejected]
    superseded_ids: set[str] = {
        event.supersedes for event in full_events if event.supersedes
    }
    active = [
        event
        for event in full_events
        if event.event_id not in superseded_ids and event.applied
    ]
    resolved = _resolve_latest_by_key(active)

    # Decay filter: drop the latest event for a key if it has aged out.
    resolved = [event for event in resolved if not _is_decayed(event, ref)]

    topic_lines: dict[str, list[str]] = {topic: [] for topic in ALL_TOPICS}
    latest_session_by_topic: dict[str, str] = {}

    for event in resolved:
        if event.topic not in topic_lines:
            continue
        for line in _render_event_lines(event):
            if line and line not in topic_lines[event.topic]:
                topic_lines[event.topic].append(line)
        if event.session_id:
            latest_session_by_topic[event.topic] = event.session_id

    written: list[str] = []
    cleared: list[str] = []

    for topic in ALL_TOPICS:
        lines = _sort_lines(topic_lines[topic])
        path = store.get_topic_path(topic)
        if not lines:
            if path.exists():
                path.unlink()
                cleared.append(topic)
            continue

        metadata = {
            "title": TOPIC_TITLES[topic],
        }
        source_session = latest_session_by_topic.get(topic)
        if source_session and source_session != "migration":
            metadata["source_session_id"] = source_session

        store.write_topic(topic, lines, metadata)
        store.update_index_entry(topic, TOPIC_SUMMARIES[topic])
        written.append(topic)

    _prune_stale_index_entries(store, written)

    return ReplayResult(
        written_topics=written,
        cleared_topics=cleared,
        resolved_event_count=len(resolved),
    )


def _resolve_latest_by_key(events: list[MemoryEvent]) -> list[MemoryEvent]:
    """Keep the latest event per (topic, key). Assumes caller has already
    filtered out rejected and superseded events."""
    latest: dict[tuple[str, str], MemoryEvent] = {}
    for event in events:
        composite = (event.topic, event.key)
        previous = latest.get(composite)
        if previous is None or _event_sort_key(event) > _event_sort_key(previous):
            latest[composite] = event

    return sorted(latest.values(), key=_event_sort_key)


def _event_sort_key(event: MemoryEvent) -> tuple[str, str]:
    return (event.observed_at, event.event_id)


def _render_event_lines(event: MemoryEvent) -> list[str]:
    value = event.value or {}
    key = event.key

    if key == "identity.name":
        name = _clean_text(value.get("name"))
        return [f"- Name: {name}"] if name else []

    if key == "identity.home_city":
        home_city = _clean_text(value.get("home_city"))
        return [f"- Home city: {home_city}"] if home_city else []

    if key == "identity.current_city":
        current_city = _clean_text(value.get("current_city"))
        return [f"- Current city: {current_city}"] if current_city else []

    if key == "identity.country":
        country = _clean_text(value.get("country"))
        return [f"- Country: {country}"] if country else []

    if key == "identity.primary_email":
        email = _clean_text(value.get("primary_email") or value.get("email"))
        return [f"- Primary email: {email.lower()}"] if email else []

    if key.startswith("identity.known_email."):
        email = _clean_text(value.get("email"))
        return [f"- Known email: {email.lower()}"] if email else []

    if key == "identity.timezone":
        tz = _clean_text(value.get("timezone"))
        return [f"- Timezone: {tz}"] if tz else []

    if key == "identity.preferred_language":
        language = _clean_text(value.get("preferred_language"))
        return [f"- Preferred language: {language}"] if language else []

    if key == "identity.occupation":
        occupation = _clean_text(value.get("occupation"))
        return [f"- Occupation: {occupation}"] if occupation else []

    if key == "identity.company":
        company = _clean_text(value.get("company"))
        return [f"- Company: {company}"] if company else []

    if key == "preferences.response_style":
        style = _clean_text(value.get("response_style"))
        return [f"- Response style: {style}"] if style else []

    if key == "preferences.humor_level":
        humor = _clean_text(value.get("humor_level"))
        return [f"- Humor level: {humor}"] if humor else []

    if key == "preferences.email_tone":
        tone = _clean_text(value.get("email_tone"))
        return [f"- Email tone: {tone}"] if tone else []

    if key == "workflow.prefers_draft_before_send":
        flag = value.get("prefers_draft_before_send")
        if flag is None:
            return []
        return [f"- Prefers draft before send: {'true' if bool(flag) else 'false'}"]

    if key == "workflow.primary_llm":
        model = _clean_text(value.get("primary_llm"))
        return [f"- Preferred primary model: {model}"] if model else []

    if key == "workflow.email_interactions_recorded":
        count = value.get("count")
        if count is None:
            return []
        try:
            return [f"- Email interactions recorded: {int(count)}"]
        except Exception:
            return []

    # D4: routines — emit a parseable single-line summary so the profile
    # snapshot can read them back without re-touching the journal.
    if key in {"workflow.morning_routine", "workflow.daily_briefing"} or key.startswith("workflow.recurring_request"):
        routine = _clean_text(value.get("routine") or value.get("name"))
        if not routine:
            return []
        cadence = _clean_text(value.get("cadence") or value.get("frequency") or "daily")
        clock = _clean_text(value.get("time"))
        tz = _clean_text(value.get("timezone"))
        items_field = value.get("items") or value.get("steps") or []
        if isinstance(items_field, str):
            items_field = [items_field]
        items = [_clean_text(i) for i in items_field if _clean_text(i)] if isinstance(items_field, list) else []

        schedule_parts = [cadence] if cadence else []
        if clock:
            schedule_parts.append(clock)
        if tz:
            schedule_parts.append(tz)
        schedule = " ".join(schedule_parts) or "daily"

        line = f"- Routine: {routine} | {schedule}"
        if items:
            line += f" | items: {', '.join(items)}"
        return [line]

    if key.startswith("contacts.frequent_recipient."):
        email = _clean_text(value.get("email"))
        if not email:
            return []
        count = value.get("count")
        if count is None:
            return [f"- Frequent recipient: {email.lower()}"]
        try:
            return [f"- Frequent recipient: {email.lower()} (count: {int(count)})"]
        except Exception:
            return [f"- Frequent recipient: {email.lower()}"]

    if key.startswith("projects.project."):
        name = _clean_text(value.get("name"))
        summary = _clean_text(value.get("summary"))
        if not name:
            return []
        if summary:
            return [f"- Project: {name} — {summary}"]
        return [f"- Project: {name}"]

    if event.topic == "relations" and key.startswith("relations."):
        role = _clean_text(value.get("role") or key.split(".", 1)[-1])
        name = _clean_text(value.get("name"))
        if not name:
            return []
        label = role.replace("_", " ").title() if role else "Person"
        return [f"- {label}: {name}"]

    if event.topic == "corrections":
        summary = _clean_text(value.get("summary") or value.get("text"))
        if summary:
            return [f"- Correction: {summary}"]
        # fall through to the generic renderer for corrections whose value
        # carries structured fields instead of a summary/text string

    # Generic fallback: an applied event must never be invisible just because
    # its key is missing from the whitelist above (that is how the confirmed
    # corrections.name_role and preferences.sport facts vanished).
    label_source = key.rsplit(".", 1)[-1] if "." in key else (key or event.topic)
    label = _clean_text(label_source.replace("_", " ")).title() or event.topic.title()

    parts: list[str] = []

    def _collect(node: object) -> None:
        if isinstance(node, dict):
            for item in node.values():
                _collect(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _collect(item)
        elif isinstance(node, bool):
            parts.append("true" if node else "false")
        elif isinstance(node, (str, int, float)):
            text = _clean_text(node)
            if text:
                parts.append(text)

    _collect(value)
    flattened = ", ".join(parts)
    return [f"- {label}: {flattened}"] if flattened else []


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _sort_lines(lines: list[str]) -> list[str]:
    def sort_key(line: str) -> tuple[int, str]:
        label = line[2:].split(":", 1)[0].strip() if line.startswith("- ") else line
        try:
            index = LINE_SORT_ORDER.index(label)
        except ValueError:
            index = len(LINE_SORT_ORDER)
        return (index, line.lower())

    return sorted({line.strip() for line in lines if line.strip()}, key=sort_key)


def _is_decayed(event: MemoryEvent, reference_time: datetime) -> bool:
    """Return True if the event has aged past DECAY_DAYS without reinforcement.

    "Reinforcement" is implicit: the replayer has already selected the *latest*
    event per (topic, key) via ``_resolve_latest_by_key``, so if a newer event
    exists for the same key it will be in ``resolved`` instead of this one.
    Therefore we only need to check whether this (latest) event is stale.

    Exemptions:
    - ``topic in _DECAY_EXEMPT_TOPICS`` (identity facts never expire)
    - ``source == _DECAY_EXEMPT_SOURCE`` (migration events are authoritative)
    - ``source == "explicit"`` (explicit user statements persist until
      superseded or corrected; decay is for inferred behavioral signals)
    """
    if event.topic in _DECAY_EXEMPT_TOPICS:
        return False
    if event.source == _DECAY_EXEMPT_SOURCE:
        return False
    if event.source == "explicit":
        return False
    if not event.observed_at:
        return False
    try:
        observed = datetime.fromisoformat(event.observed_at.replace("Z", "+00:00"))
    except Exception:
        return False
    return (reference_time - observed) > timedelta(days=DECAY_DAYS)


def _prune_stale_index_entries(store: PersonalMemoryStore, written_topics: list[str]) -> None:
    """Remove index entries whose topic file no longer exists."""
    entries = store.load_index()
    kept = []
    written_files = {store.get_topic_path(topic).name for topic in written_topics}
    for entry in entries:
        path = store.base_dir / entry.file_name
        if entry.file_name in written_files or path.exists():
            kept.append(entry)
    if len(kept) != len(entries):
        store.save_index(kept)
