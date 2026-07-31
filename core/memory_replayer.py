from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

from core.memory_journal import MemoryEvent
from core.memory_schema import DECAY_DAYS, TOPICS, is_decayed, render_statement
from core.personal_memory_store import PersonalMemoryStore


# Decay policy now lives in core.memory_schema so the markdown projection here
# and the SQLite read model apply the identical rule (brutal review H2).
# DECAY_DAYS is re-exported above for backwards-compat with existing importers.


# Topic titles/summaries and the topic tuple all derive from the single registry
# in core.memory_schema so the vocabularies can never drift apart again.
TOPIC_TITLES = {name: spec.title for name, spec in TOPICS.items()}
TOPIC_SUMMARIES = {name: spec.summary for name, spec in TOPICS.items()}

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

ALL_TOPICS = tuple(TOPICS)


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
    # The per-key templates and generic fallback now live in the single registry
    # (core.memory_schema.render_statement); the replayer only owns the "- "
    # bullet wrapping and the empty-line filter.
    statement = render_statement(event)
    return [f"- {statement}"] if statement else []


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
    """Thin adapter over the shared ``memory_schema.is_decayed`` predicate.

    The replayer has already selected the *latest* event per (topic, key) via
    ``_resolve_latest_by_key``, so a newer restatement would be here instead —
    we only check whether this (latest) event is stale. Identity topics,
    explicit statements, and migration events are exempt (see the predicate).
    """
    return is_decayed(
        event.topic, event.source, event.observed_at, reference_time=reference_time
    )


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
