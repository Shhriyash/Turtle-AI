from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from core.paths import PERSONAL_MEMORY_JOURNAL_DIR


ALLOWED_KINDS = frozenset({"fact", "preference", "behavior", "correction", "contradiction"})
ALLOWED_TOPICS = frozenset(
    {"identity", "preferences", "workflow", "contacts", "projects", "corrections"}
)
ALLOWED_SOURCES = frozenset({"explicit", "inferred", "synthesized", "migration"})
ALLOWED_EXTRACTORS = frozenset({"deterministic", "llm_turn", "dream_pass", "migration"})


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


class JournalStore:
    """Append-only JSONL journal, sharded by month, idempotent by event_id."""

    def __init__(self, journal_dir: Path = PERSONAL_MEMORY_JOURNAL_DIR) -> None:
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

    def append(self, event: MemoryEvent) -> MemoryEvent:
        validate_event(event)
        if self._event_exists(event.event_id):
            return event
        path = self._shard_path_for(event.observed_at)
        line = json.dumps(event.to_payload(), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            try:
                os.fsync(file.fileno())
            except Exception:
                pass
        return event

    def append_many(self, events: Iterable[MemoryEvent]) -> list[MemoryEvent]:
        return [self.append(event) for event in events]

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

    def load_all(self) -> list[MemoryEvent]:
        return list(self.iter_events())

    def _event_exists(self, event_id: str) -> bool:
        for existing in self.iter_events():
            if existing.event_id == event_id:
                return True
        return False


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
    )
    validate_event(event)
    return event
