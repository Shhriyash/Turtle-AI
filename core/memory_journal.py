from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from core.config import settings
from core.guardrails import StorageCapExceededError, enforce_storage_cap
from core.memory_schema import ALLOWED_TOPICS
from core.paths import personal_journal_dir, personal_memory_dir


ALLOWED_KINDS = frozenset({"fact", "preference", "behavior", "correction", "contradiction"})
# The 11-topic vocabulary now has a single home in core.memory_schema; re-exported
# here so existing `from core.memory_journal import ALLOWED_TOPICS` importers work.
ALLOWED_SOURCES = frozenset({"explicit", "inferred", "synthesized", "migration"})
ALLOWED_EXTRACTORS = frozenset({"deterministic", "llm_turn", "dream_pass", "migration", "scheduler"})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def generate_event_id() -> str:
    """Crockford-base32 ULID-like id: 48-bit time + 80-bit randomness."""
    timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    randomness = secrets.randbits(80)
    value = (timestamp_ms << 80) | randomness
    chars: list[str] = []
    for _ in range(26):
        chars.append(_ULID_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


@dataclass
class MemoryEvent:
    event_id: str
    session_id: str
    turn_id: str
    observed_at: str
    kind: str
    topic: str
    key: str
    value: dict[str, Any]
    confidence: float
    source: str
    extractor: str
    evidence: dict[str, Any] = field(default_factory=dict)
    supersedes: str | None = None
    applied: bool = False
    rejected: bool = False
    # Pre-rendered one-line projection, snapshotted at extraction time so the
    # replayer can render verbatim (statement-based rendering). Empty on old
    # events and on events built without one; the replayer falls back to the
    # key templates in core.memory_schema in that case.
    statement: str = ""

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MemoryEvent":
        return cls(
            event_id=str(payload["event_id"]),
            session_id=str(payload.get("session_id", "")),
            turn_id=str(payload.get("turn_id", "")),
            observed_at=str(payload.get("observed_at", "")),
            kind=str(payload["kind"]),
            topic=str(payload["topic"]),
            key=str(payload["key"]),
            value=dict(payload.get("value", {})),
            confidence=float(payload.get("confidence", 0.0)),
            source=str(payload["source"]),
            extractor=str(payload["extractor"]),
            evidence=dict(payload.get("evidence", {})),
            supersedes=payload.get("supersedes"),
            applied=bool(payload.get("applied", False)),
            rejected=bool(payload.get("rejected", False)),
            statement=str(payload.get("statement", "")),
        )


def validate_event(event: MemoryEvent) -> None:
    if not event.event_id:
        raise ValueError("event_id is required")
    if event.kind not in ALLOWED_KINDS:
        raise ValueError(f"invalid kind: {event.kind}")
    if event.topic not in ALLOWED_TOPICS:
        raise ValueError(f"invalid topic: {event.topic}")
    if event.source not in ALLOWED_SOURCES:
        raise ValueError(f"invalid source: {event.source}")
    if event.extractor not in ALLOWED_EXTRACTORS:
        raise ValueError(f"invalid extractor: {event.extractor}")
    if not 0.0 <= event.confidence <= 1.0:
        raise ValueError(f"confidence out of range: {event.confidence}")
    if not event.key:
        raise ValueError("key is required")
    if not isinstance(event.value, dict):
        raise ValueError("value must be a dict")


class _LocalJournalBackend:
    """Append-only JSONL journal, sharded by month, idempotent by event_id.

    Extracted from JournalStore's original single-backend implementation so
    the cloud counterpart (core/storage/cloud/journal_store.PostgresJournalBackend)
    can drop in behind the same 4-method surface without JournalStore's
    business logic (validation, dedup, cap enforcement, the on_append hook)
    needing to know which one it's talking to.
    """

    def __init__(self, journal_dir: Path) -> None:
        self.journal_dir = journal_dir
        self.journal_dir.mkdir(parents=True, exist_ok=True)

    def _shard_path_for(self, observed_at: str) -> Path:
        try:
            dt = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.now(UTC)
        shard = self.journal_dir / f"{dt.year:04d}-{dt.month:02d}"
        shard.mkdir(parents=True, exist_ok=True)
        return shard / "events.jsonl"

    def event_exists(self, event_id: str) -> bool:
        for existing in self.iter_events():
            if existing.event_id == event_id:
                return True
        return False

    def append_line(self, event: MemoryEvent) -> None:
        path = self._shard_path_for(event.observed_at)
        line = json.dumps(event.to_payload(), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            try:
                os.fsync(file.fileno())
            except Exception:
                pass

    def iter_events(self) -> Iterator[MemoryEvent]:
        if not self.journal_dir.exists():
            return
        shard_files: list[Path] = []
        for shard_dir in sorted(self.journal_dir.iterdir()):
            if not shard_dir.is_dir():
                continue
            path = shard_dir / "events.jsonl"
            if path.exists():
                shard_files.append(path)
        for path in shard_files:
            with path.open("r", encoding="utf-8") as file:
                for raw in file:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    try:
                        yield MemoryEvent.from_payload(payload)
                    except Exception:
                        continue

    def total_bytes(self) -> int:
        total = 0
        try:
            for entry in self.journal_dir.rglob("*"):
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def created_at_timestamp(self) -> float | None:
        try:
            return self.journal_dir.stat().st_ctime
        except OSError:
            return None

    def flush(self) -> None:
        if not self.journal_dir.exists():
            return
        for shard_dir in sorted(self.journal_dir.iterdir()):
            if not shard_dir.is_dir():
                continue
            path = shard_dir / "events.jsonl"
            if not path.exists():
                continue
            try:
                with path.open("a", encoding="utf-8") as file:
                    file.flush()
                    try:
                        os.fsync(file.fileno())
                    except Exception:
                        pass
            except Exception:
                continue


class JournalStore:
    """Append-only journal, idempotent by event_id.

    Local mode: JSONL sharded by month (see _LocalJournalBackend). Cloud mode
    (TURTLE_DEPLOY=cloud): Postgres, one row per event (see
    core/storage/cloud/journal_store.PostgresJournalBackend) — the local
    JSONL files do not survive a serverless cold start.

    An explicit ``journal_dir`` always forces the local backend regardless of
    settings.is_cloud — this is how the test suite gets an isolated,
    disposable journal per test, and that isolation must keep working
    unchanged in a cloud-mode CI run.
    """

    def __init__(
        self,
        user_id: str = "default",
        journal_dir: Path | None = None,
        *,
        on_append: Callable[[MemoryEvent], None] | None = None,
    ) -> None:
        self.user_id = user_id
        # Optional write-through hook (e.g. SQLite FTS5 index). Wrapped in
        # try/except at the call site so a failing index never blocks the
        # journal write — the journal is the source of truth.
        self.on_append = on_append

        if journal_dir is not None:
            self._backend = _LocalJournalBackend(journal_dir)
        elif settings.is_cloud:
            from core.storage.cloud.journal_store import PostgresJournalBackend

            self._backend = PostgresJournalBackend(user_id)
        else:
            self._backend = _LocalJournalBackend(personal_journal_dir(user_id))

    def append(self, event: MemoryEvent) -> MemoryEvent:
        validate_event(event)
        if self._backend.event_exists(event.event_id):
            return event
        line_bytes = len(
            json.dumps(event.to_payload(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ) + 1
        # Phase 6: enforce per-user storage cap before appending.
        if self.user_id and self.user_id != "default":
            self._enforce_cap(incoming_bytes=line_bytes)
        self._backend.append_line(event)
        if self.on_append is not None:
            try:
                self.on_append(event)
            except Exception as exc:
                print(f"LOG: JournalStore on_append hook failed for {event.event_id}: {exc}")
        return event

    def _enforce_cap(self, *, incoming_bytes: int) -> None:
        """Same guardrail as core.guardrails.enforce_storage_cap, sourced from
        whichever backend is active: local disk usage under
        personal_memory_dir(user_id) locally, this user's total journal row
        size in Postgres in cloud mode (there is no shared local disk there
        for a runaway writer to fill, but an unbounded per-user table is its
        own cost/quota concern worth guarding the same way).
        """
        cap_mb = int(settings.user_storage_cap_mb)
        if cap_mb <= 0:
            return
        if isinstance(self._backend, _LocalJournalBackend):
            enforce_storage_cap(
                self.user_id, personal_memory_dir(self.user_id), incoming_bytes=incoming_bytes
            )
            return
        cap_bytes = cap_mb * 1024 * 1024
        used = self._backend.total_bytes() + max(0, incoming_bytes)
        if used > cap_bytes:
            raise StorageCapExceededError(self.user_id, used, cap_bytes)

    def append_many(self, events: Iterable[MemoryEvent]) -> list[MemoryEvent]:
        # Dedup-on-append: skip only TRUE no-ops — an incoming event whose
        # (kind, applied, value) matches the CURRENT latest event for its
        # (topic, key). Comparing against the current latest (not set-membership
        # over the last 50) is what fixes the flip-back bug: with the journal at
        # [humor=high, humor=low], a user restating "high" differs from the
        # current latest ("low") and MUST journal — the old signature-set logic
        # matched it against the earlier "high" and silently dropped it, leaving
        # the stale "low" served (brutal review H1). Key+applied still matter so
        # an applied=True restatement is never suppressed by a lingering
        # applied=False candidate the gate is holding.
        recent: list[MemoryEvent] = []
        try:
            recent = self.load_all()[-50:]
        except Exception:
            recent = []

        def _normalized(ev: MemoryEvent) -> str:
            try:
                return json.dumps(ev.value, ensure_ascii=False, sort_keys=True)
            except Exception:
                return str(ev.value)

        def _sort_key(ev: MemoryEvent) -> tuple[str, str]:
            return (str(ev.observed_at), str(ev.event_id))

        # Current latest event per (topic, key) among recent journal events.
        latest_by_key: dict[tuple[str, str], MemoryEvent] = {}
        for ev in recent:
            composite = (str(ev.topic), str(ev.key))
            current = latest_by_key.get(composite)
            if current is None or _sort_key(ev) > _sort_key(current):
                latest_by_key[composite] = ev

        def _is_noop(event: MemoryEvent) -> bool:
            current = latest_by_key.get((str(event.topic), str(event.key)))
            if current is None:
                return False
            return (
                str(current.kind) == str(event.kind)
                and bool(current.applied) == bool(event.applied)
                and _normalized(current) == _normalized(event)
            )

        results: list[MemoryEvent] = []
        for event in events:
            if _is_noop(event):
                continue
            results.append(self.append(event))
            # A freshly appended event is now the latest for its key, so a later
            # event in this same batch dedups against it (in-batch no-ops still
            # collapse, and a within-batch flip-back still lands).
            latest_by_key[(str(event.topic), str(event.key))] = event
        return results

    def append_rejection(self, original: "MemoryEvent") -> "MemoryEvent":
        """Append a tombstone that permanently rejects *original*.

        Rejection must live in the journal (source of truth), not only as an
        out-of-band index flag — otherwise any index rebuild resurrects the
        rejected fact. ``MemorySQLiteIndex.backfill_from_journal`` honors
        these tombstones.
        """
        tombstone = MemoryEvent(
            event_id=generate_event_id(),
            session_id=original.session_id,
            turn_id=f"{original.turn_id}_rejected",
            observed_at=_utc_now(),
            kind="contradiction",
            topic=original.topic,
            key=original.key,
            value={"rejected_event_id": original.event_id},
            confidence=1.0,
            source="explicit",
            extractor="deterministic",
            evidence={"note": "rejection tombstone"},
            supersedes=original.event_id,
            applied=False,
        )
        return self.append(tombstone)

    def iter_events(self) -> Iterator[MemoryEvent]:
        return self._backend.iter_events()

    def load_all(self) -> list[MemoryEvent]:
        return list(self.iter_events())

    def _event_exists(self, event_id: str) -> bool:
        return self._backend.event_exists(event_id)

    def get_created_at_timestamp(self) -> float | None:
        """Epoch seconds this journal was first created, or None when
        unavailable. Local mode: the journal directory's ctime. Cloud mode:
        the earliest journal event's observed_at (there is no filesystem
        ctime; a user's first journal write happens moments after their
        journal would have been created locally, so this is a faithful
        substitute). Used by core/confirmation_gate.py's first-session
        heuristic, which already treats an exception/None as "can't tell,
        fall through to the event-count check alone".
        """
        return self._backend.created_at_timestamp()

    def flush(self) -> None:
        """Best-effort fsync for existing journal shards (local mode) / no-op
        (cloud mode — Postgres commits are already durable)."""
        self._backend.flush()


def make_event(
    *,
    kind: str,
    topic: str,
    key: str,
    value: dict[str, Any],
    confidence: float,
    source: str,
    extractor: str,
    session_id: str,
    turn_id: str,
    observed_at: str | None = None,
    evidence: dict[str, Any] | None = None,
    supersedes: str | None = None,
    applied: bool = False,
    event_id: str | None = None,
    statement: str = "",
) -> MemoryEvent:
    event = MemoryEvent(
        event_id=event_id or generate_event_id(),
        session_id=session_id,
        turn_id=turn_id,
        observed_at=observed_at or _utc_now(),
        kind=kind,
        topic=topic,
        key=key,
        value=dict(value),
        confidence=float(confidence),
        source=source,
        extractor=extractor,
        evidence=dict(evidence or {}),
        supersedes=supersedes,
        applied=applied,
        statement=statement,
    )
    validate_event(event)
    return event
