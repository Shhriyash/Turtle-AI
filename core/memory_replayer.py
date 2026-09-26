from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable, Iterator

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

    # WP3.D (ledger 3.8, transaction half): every topic write/delete below
    # must commit together or not at all. `store` is the backend-agnostic
    # PersonalMemoryStore (core/personal_memory_store.py) — local/SQLite has
    # no notion of a cross-file transaction (see _topic_write_transaction's
    # docstring for why that's left as-is), but the cloud backend opened its
    # own Postgres connection PER topic call, so a crash between topic 3 and
    # topic 4 of ~11 left some topics reflecting the new journal state and
    # others stale, with nothing recording that it happened.
    with _topic_write_transaction(store):
        for topic in ALL_TOPICS:
            lines = _sort_lines(topic_lines[topic])
            if not lines:
                if store.delete_topic(topic):
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


@contextlib.contextmanager
def _topic_write_transaction(store: PersonalMemoryStore) -> Iterator[None]:
    """Wrap replay()'s per-topic write loop in the store's own transaction,
    if it has one, so a crash partway through leaves no topics changed
    (ledger 3.8, transaction half).

    PersonalMemoryStore is deliberately backend-agnostic (local/SQLite files
    vs. Postgres rows) and exposes no transaction primitive of its own on its
    public surface, so this only reaches as far as `store._backend._pg`
    (present only on the cloud path — see core.personal_memory_store's
    _CloudPersonalMemoryBackend) to find the one Postgres-specific hook that
    matters: `PostgresPersonalMemoryBackend.transaction()`
    (core/storage/cloud/personal_memory_store.py). This is duck-typed via
    getattr rather than an isinstance/import of the cloud module, so replay()
    never touches psycopg/asyncpg directly and importing this module stays
    dependency-free on a machine with no cloud extras installed.

    Local/SQLite path: deliberately NOT given the same atomicity here. Each
    local topic write is already a single atomic file replace
    (core.io_atomic.atomic_write_text) and there was never a per-topic
    "implicit commit" for a crash to catch mid-loop the way Postgres's
    per-call `pool.connection()` created — a partial local replay leaves N
    correct files and the rest simply not-yet-written-this-run, not a torn
    write. Retrofitting cross-file atomicity to local storage (e.g. staging
    all N files and renaming as a batch) is a bigger change than this
    ledger item's scope and isn't the crash mode being fixed here.
    """
    pg_backend = getattr(getattr(store, "_backend", None), "_pg", None)
    transaction_fn = getattr(pg_backend, "transaction", None)
    if transaction_fn is None:
        yield
        return
    with transaction_fn():
        yield


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
    """Remove index entries whose topic no longer exists."""
    entries = store.load_index()
    kept = []
    written_files = {store.get_topic_path(topic).name for topic in written_topics}
    for entry in entries:
        if entry.file_name in written_files or store.topic_exists(entry.file_name):
            kept.append(entry)
    if len(kept) != len(entries):
        store.save_index(kept)
