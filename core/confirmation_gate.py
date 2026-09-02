from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.io_atomic import atomic_write_json
from core.memory_journal import JournalStore, MemoryEvent, make_event
from core.memory_replayer import replay
from core.personal_memory_store import PersonalMemoryStore

if TYPE_CHECKING:
    from core.memory_sqlite import MemoryEventRow, MemorySQLiteIndex


DEFAULT_SILENCE_DAYS = 14
# Phase 5: first-session users (sparse journals or freshly-created accounts)
# get a much shorter silence window so the memory pipeline feels responsive
# during onboarding. After they cross the event threshold or the account
# ages past 24h, the gate reverts to ``silence_days``.
DEFAULT_FIRST_SESSION_WINDOW_MINUTES = 5
DEFAULT_FIRST_SESSION_EVENT_THRESHOLD = 20
DEFAULT_FIRST_SESSION_ACCOUNT_AGE_HOURS = 24


@dataclass(frozen=True)
class GatePrompt:
    event_id: str
    question: str
    topic: str
    key: str
    # Additional candidate event_ids batched into this prompt (always
    # includes the primary `event_id` as the first element). When the user
    # answers yes/no, the caller should iterate over ``all_event_ids`` and
    # record_response on each — accepting or rejecting them as a group.
    event_ids: tuple[str, ...] = ()

    @property
    def all_event_ids(self) -> tuple[str, ...]:
        return self.event_ids or (self.event_id,)


# Maximum number of pending candidates to fold into a single gate prompt.
# Beyond this the user is asking too many yes/nos at once — overflow stays
# in the queue and surfaces on subsequent turns.
GATE_BATCH_MAX = 3


class ConfirmationGate:
    """Mediator between unconfirmed memory candidates and the user.

    Candidates emitted by Stage B (LLM per-session extractor) or Stage C
    (dream pass) should be written to the journal with ``applied=False``
    and then queued through this gate. The main agent calls
    :meth:`next_prompt` at turn boundaries; if it returns a ``GatePrompt``
    the agent surfaces the question to the user, then calls
    :meth:`record_response` with the user's answer.

    Acceptance emits a new, applied event with ``source=explicit`` that
    supersedes the candidate, so the topic files update. Rejection emits
    a contradiction event that supersedes the candidate so the candidate
    never renders, and suppresses future prompts for the same
    ``(topic, key)`` for ``silence_days`` days.

    This class is intentionally free of any voice- or LLM-specific code:
    it only touches the journal, the replayer, and its own small
    JSON state file, so it is unit-testable in isolation.
    """

    def __init__(
        self,
        *,
        journal: JournalStore,
        store: PersonalMemoryStore,
        state_path: Path,
        silence_days: int = DEFAULT_SILENCE_DAYS,
        first_session_window_minutes: int = DEFAULT_FIRST_SESSION_WINDOW_MINUTES,
        first_session_event_threshold: int = DEFAULT_FIRST_SESSION_EVENT_THRESHOLD,
        first_session_account_age_hours: int = DEFAULT_FIRST_SESSION_ACCOUNT_AGE_HOURS,
        sqlite_index: "MemorySQLiteIndex | None" = None,
    ) -> None:
        self.journal = journal
        self.store = store
        self.state_path = state_path
        # Optional read model (Phase 2 W3). When wired — it must be the same
        # index the journal write-throughs to via on_append — the hot-path
        # existence / silence / latest-by-key lookups route through indexed
        # queries instead of O(n) journal scans. Default None preserves the
        # exact journal-scan behavior, so callers that don't pass it are
        # unchanged. apps/turtle_server.py wires it in a one-line follow-up.
        self.sqlite_index = sqlite_index
        self.silence_days = int(silence_days)
        self.first_session_window_minutes = int(first_session_window_minutes)
        self.first_session_event_threshold = int(first_session_event_threshold)
        self.first_session_account_age_hours = int(first_session_account_age_hours)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    def queue_candidate(self, event: MemoryEvent) -> bool:
        """Queue a candidate event for user confirmation.

        Returns True if queued, False if silently dropped (already
        queued, already explicit, or within the silence window for the
        same key).
        """
        if event.source == "explicit":
            return False
        if event.applied:
            return False
        if event.event_id in self._state["pending"]:
            return False
        if self._is_silenced(event.topic, event.key):
            return False

        if not self._event_exists_in_journal(event.event_id):
            self.journal.append(event)

        self._state["pending"].append(event.event_id)
        self._save_state()
        return True

    def next_prompt(self) -> GatePrompt | None:
        """Peek at the next batch of pending candidates without removing them.

        Up to ``GATE_BATCH_MAX`` valid candidates are folded into a single
        prompt so the user isn't pelted with one yes/no per turn. The caller
        iterates ``GatePrompt.all_event_ids`` and calls :meth:`record_response`
        on each with the user's group answer.

        Silenced or missing candidates are dropped lazily from the front of
        the queue. Returns None when no valid candidates remain.
        """
        # Drain invalid entries from the front so the first peek is valid.
        while self._state["pending"]:
            event_id = self._state["pending"][0]
            event = self._load_event(event_id)
            if event is None:
                self._state["pending"].pop(0)
                self._save_state()
                continue
            if self._is_silenced(event.topic, event.key):
                self._state["pending"].pop(0)
                self._save_state()
                continue
            break
        if not self._state["pending"]:
            return None

        # Peek (without popping) at up to GATE_BATCH_MAX valid candidates.
        # Dedupe by (topic, key) — if the extractor queued two events for the
        # same key, we only ask once. The first event for a key wins.
        batch: list[MemoryEvent] = []
        seen_keys: set[tuple[str, str]] = set()
        for event_id in self._state["pending"]:
            event = self._load_event(event_id)
            if event is None:
                continue
            if self._is_silenced(event.topic, event.key):
                continue
            key_pair = (event.topic, event.key)
            if key_pair in seen_keys:
                continue
            seen_keys.add(key_pair)
            batch.append(event)
            if len(batch) >= GATE_BATCH_MAX:
                break

        if not batch:
            return None

        primary = batch[0]
        if len(batch) == 1:
            question = _render_question(primary)
        else:
            question = _render_batch_question(batch)

        return GatePrompt(
            event_id=primary.event_id,
            question=question,
            topic=primary.topic,
            key=primary.key,
            event_ids=tuple(e.event_id for e in batch),
        )

    def record_response(self, event_id: str, *, accepted: bool) -> MemoryEvent | None:
        """Resolve a pending candidate with the user's answer.

        On accept, emits a new ``source=explicit`` event with
        ``applied=True`` that supersedes the candidate and re-runs the
        replayer so topic files update immediately.

        On reject, emits a ``kind=contradiction`` event that supersedes
        the candidate so the candidate never renders, and starts the
        silence window for the same key.
        """
        candidate = self._load_event(event_id)
        if candidate is None:
            self._remove_from_pending(event_id)
            return None

        if accepted:
            response_event = make_event(
                kind=candidate.kind,
                topic=candidate.topic,
                key=candidate.key,
                value=candidate.value,
                confidence=1.0,
                source="explicit",
                extractor="deterministic",
                session_id=candidate.session_id,
                turn_id=candidate.turn_id,
                evidence={
                    "confirmed_from": candidate.event_id,
                    "confirmation": "accepted",
                },
                supersedes=candidate.event_id,
                applied=True,
            )
        else:
            response_event = make_event(
                kind="contradiction",
                topic=candidate.topic,
                key=candidate.key,
                value={"rejected": True, "original_key": candidate.key},
                confidence=1.0,
                source="explicit",
                extractor="deterministic",
                session_id=candidate.session_id,
                turn_id=candidate.turn_id,
                evidence={
                    "rejected_from": candidate.event_id,
                    "confirmation": "rejected",
                    "silence_days": self.silence_days,
                },
                supersedes=candidate.event_id,
                applied=False,
            )

        self.journal.append(response_event)
        self._remove_from_pending(event_id)
        replay(self.journal.load_all(), store=self.store)
        return response_event

    def preview_pending(self, event_id: str | list[str] | tuple[str, ...]) -> str | None:
        """Return a natural-language preview of pending candidate(s).

        Accepts a single event_id or a sequence (when a batched prompt is
        active). Renders each candidate as a short, human-readable phrase
        instead of dumping the raw topic / key / value / extractor /
        confidence / evidence — those are debugging metadata, not user copy.
        """
        if isinstance(event_id, (list, tuple)):
            ids = list(event_id)
        else:
            ids = [event_id]

        lines: list[str] = []
        for eid in ids:
            event = self._load_event(eid)
            if event is None:
                continue
            lines.append(_render_value_natural(event))

        if not lines:
            return None
        if len(lines) == 1:
            body = lines[0]
        else:
            body = "\n".join(f"  - {line}" for line in lines)
        return f"Proposed memory:\n{body}\nReply yes to save, no to reject."

    def pending_count(self) -> int:
        return len(self._state["pending"])

    def get_pending_ids(self) -> list[str]:
        """Return a copy of the pending event_id queue (order preserved)."""
        return list(self._state["pending"])

    def force_remove_pending(self, event_id: str) -> None:
        """Remove a candidate from the pending queue without recording a response.

        Used by the dream pass to clear processed candidates after batch decisions
        without going through the accept/reject flow.
        """
        self._remove_from_pending(event_id)

    def is_silenced(self, topic: str, key: str) -> bool:
        return self._is_silenced(topic, key)

    def _silence_cutoff(self, events: list[MemoryEvent]) -> datetime | None:
        return self._silence_cutoff_from_count(len(events))

    def _silence_cutoff_from_count(self, event_count: int) -> datetime | None:
        """Return the timestamp before which a rejection no longer silences.

        Returns None when silencing is disabled (silence_days <= 0). For
        first-session users (sparse journal or fresh account) the window
        collapses to ``first_session_window_minutes`` so onboarding feels
        responsive instead of muted for two weeks. Takes the event *count*
        rather than the list because that is all the first-session verdict
        needs — so the indexed path can pass ``sqlite_index.count()`` without
        materializing every event.
        """
        if self.silence_days <= 0:
            return None
        now = datetime.now(UTC)
        if self._is_first_session_from_count(event_count):
            return now - timedelta(minutes=max(0, self.first_session_window_minutes))
        return now - timedelta(days=self.silence_days)

    def _is_first_session(self, events: list[MemoryEvent]) -> bool:
        return self._is_first_session_from_count(len(events))

    def _is_first_session_from_count(self, event_count: int) -> bool:
        if event_count < self.first_session_event_threshold:
            return True
        try:
            ctime = self.journal.journal_dir.stat().st_ctime
            account_age_hours = (datetime.now(UTC).timestamp() - ctime) / 3600
            if account_age_hours < self.first_session_account_age_hours:
                return True
        except Exception:
            pass
        return False

    def _usable_index(self):
        """The sqlite index, or None when absent OR stale.

        A write-through failure after a successful journal append leaves the
        read model missing events; trusting it then would change gate decisions
        (missed contradiction tombstones stop silencing, real journal events
        look nonexistent). Stale means: fall back to the journal scan until a
        successful full backfill clears the flag.
        """
        index = self.sqlite_index
        if index is not None and not getattr(index, "is_stale", False):
            return index
        return None

    def _is_silenced(self, topic: str, key: str) -> bool:
        # Indexed path: reproduce the journal scan below exactly over the read
        # model. The first-session verdict only needs the total event count, and
        # the silence check itself only needs the (topic, key) contradictions —
        # both are indexed lookups, so the whole journal never has to be walked.
        index = self._usable_index()
        if index is not None:
            cutoff = self._silence_cutoff_from_count(index.count())
            if cutoff is None:
                return False
            for row in index.events_for_key(topic, key):
                if row.kind != "contradiction":
                    continue
                if not row.value.get("rejected"):
                    continue
                observed = _parse_iso(row.observed_at)
                if observed is None:
                    continue
                if observed >= cutoff:
                    return True
            return False

        events = list(self.journal.iter_events())
        cutoff = self._silence_cutoff_from_count(len(events))
        if cutoff is None:
            return False
        for event in events:
            if event.topic != topic or event.key != key:
                continue
            if event.kind != "contradiction":
                continue
            if not event.value.get("rejected"):
                continue
            observed = _parse_iso(event.observed_at)
            if observed is None:
                continue
            if observed >= cutoff:
                return True
        return False

    def _load_event(self, event_id: str) -> MemoryEvent | None:
        # Indexed path: primary-key lookup + reconstruct the event, instead of a
        # linear journal scan. The reconstruction is faithful for every field the
        # gate reads downstream (kind/topic/key/value/session_id/turn_id); fields
        # the gate never touches fall back to journal defaults.
        index = self._usable_index()
        if index is not None:
            row = index.get_event(event_id)
            if row is None:
                return None
            return self._row_to_event(row)
        for event in self.journal.iter_events():
            if event.event_id == event_id:
                return event
        return None

    def _event_exists_in_journal(self, event_id: str) -> bool:
        index = self._usable_index()
        if index is not None:
            return index.event_exists(event_id)
        return self._load_event(event_id) is not None

    @staticmethod
    def _row_to_event(row: "MemoryEventRow") -> MemoryEvent:
        """Reconstruct a MemoryEvent from an indexed read-model row."""
        return MemoryEvent(
            event_id=row.event_id,
            session_id=row.session_id,
            turn_id=row.turn_id,
            observed_at=row.observed_at,
            kind=row.kind,
            topic=row.topic,
            key=row.key,
            value=dict(row.value),
            confidence=float(row.confidence),
            source=row.source,
            extractor=row.extractor or "deterministic",
            applied=bool(row.applied),
            statement=row.statement,
        )

    def _remove_from_pending(self, event_id: str) -> None:
        pending = self._state["pending"]
        if event_id in pending:
            pending.remove(event_id)
            self._save_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"pending": []}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"pending": []}
        pending = payload.get("pending") if isinstance(payload, dict) else None
        if not isinstance(pending, list):
            pending = []
        return {"pending": [str(item) for item in pending if item]}

    def _save_state(self) -> None:
        atomic_write_json(self.state_path, {"pending": list(self._state["pending"])})


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _render_question(event: MemoryEvent) -> str:
    value = event.value or {}
    key = event.key

    if key == "identity.name":
        name = value.get("name") or value.get("value")
        if name:
            return f"You introduced yourself as {name} — want me to remember that?"
    if key == "identity.location.city":
        city = value.get("city") or value.get("value")
        if city:
            return f"You mentioned you're in {city} — want me to keep that on file?"
    if key.startswith("identity."):
        # Generic identity branch (e.g. identity.email, identity.role, etc.)
        # Surface the captured value if it's a simple scalar; otherwise fall
        # through to the all-purpose phrasing at the end.
        if isinstance(value, dict) and len(value) == 1:
            (only_val,) = value.values()
            if isinstance(only_val, (str, int, float)) and only_val:
                label = key.split(".", 1)[1].replace(".", " ").replace("_", " ")
                return f"I picked up your {label} as {only_val} — want me to remember that?"

    if key == "preferences.response_style":
        style = value.get("response_style")
        if style:
            return f"I've noticed you prefer {style} responses — want me to remember that?"
    if key == "preferences.humor_level":
        level = value.get("humor_level")
        if level:
            return f"It seems you prefer a {level} humor level — should I lock that in?"
    if key == "preferences.email_tone":
        tone = value.get("email_tone")
        if tone:
            return f"Your emails usually sound {tone} — want me to default to that tone?"
    if key == "workflow.prefers_draft_before_send":
        flag = value.get("prefers_draft_before_send")
        if flag is not None:
            if bool(flag):
                return "You keep asking to see a draft before sending — should I always draft first?"
            return "You keep sending emails without a draft step — want me to skip drafting by default?"
    if key == "workflow.primary_llm":
        model = value.get("primary_llm")
        if model:
            return f"You seem to prefer {model} as the primary model — want me to default to it?"
    if key.startswith("contacts.frequent_recipient."):
        email = value.get("email")
        if email:
            return f"I've seen {email} come up a lot — want me to save it as a frequent contact?"
    if key in {"workflow.morning_routine", "workflow.daily_briefing"} or key.startswith("workflow.recurring_request"):
        items = value.get("items") or value.get("steps") or []
        cadence = value.get("cadence") or value.get("frequency") or "daily"
        clock = value.get("time")
        tz = value.get("timezone")
        time_phrase = ""
        if isinstance(clock, str) and clock:
            time_phrase = f" {clock}" + (f" {tz}" if isinstance(tz, str) and tz else "")
        if isinstance(items, list) and items:
            items_str = ", ".join(str(i) for i in items)
            return f"Sounds like a {cadence}{time_phrase} routine ({items_str}) — want me to remember it?"
        routine = value.get("routine") or value.get("name")
        if routine:
            return f"Sounds like a {cadence}{time_phrase} routine ({routine}) — want me to remember it?"
    if key.startswith("projects.project."):
        name = value.get("name")
        if name:
            return f"You keep coming back to the {name} project — want me to track it as a recurring project?"

    return (
        "I've spotted a pattern worth remembering about "
        f"{event.topic} ({event.key}). Want me to save it?"
    )


def _render_short_phrase(event: MemoryEvent) -> str:
    """One-clause natural description of a single candidate for batched prompts.

    e.g. "your name (Shriyash)", "your city (Indore)", "your preferred
    response style (concise)". Falls back to the topic/key tuple when the
    value shape is unfamiliar.
    """
    value = event.value or {}
    key = event.key

    if key == "identity.name" and isinstance(value, dict):
        name = value.get("name") or value.get("value")
        if name:
            return f"your name ({name})"
    if key == "identity.location.city" and isinstance(value, dict):
        city = value.get("city") or value.get("value")
        if city:
            return f"your city ({city})"
    if key.startswith("identity.") and isinstance(value, dict) and len(value) == 1:
        (only_val,) = value.values()
        if isinstance(only_val, (str, int, float)) and only_val:
            label = key.split(".", 1)[1].replace(".", " ").replace("_", " ")
            return f"your {label} ({only_val})"
    if key == "preferences.response_style":
        style = value.get("response_style") if isinstance(value, dict) else None
        if style:
            return f"your preferred response style ({style})"
    if key == "preferences.humor_level":
        level = value.get("humor_level") if isinstance(value, dict) else None
        if level:
            return f"your humor level ({level})"
    if key == "preferences.email_tone":
        tone = value.get("email_tone") if isinstance(value, dict) else None
        if tone:
            return f"your email tone ({tone})"
    if key.startswith("contacts.frequent_recipient.") and isinstance(value, dict):
        email = value.get("email")
        if email:
            return f"a frequent contact ({email})"

    # Generic: humanize the key.
    label = key.replace(".", " ").replace("_", " ")
    return f"a {label} preference"


def _render_batch_question(events: list[MemoryEvent]) -> str:
    """Combine multiple candidates into one yes/no prompt.

    Example: "I noticed your name (Shriyash) and your city (Indore) — save
    both?" Three candidates use an Oxford-comma list. Beyond three the
    queue is capped (GATE_BATCH_MAX), so we don't render more than that.
    """
    phrases = [_render_short_phrase(e) for e in events]
    if len(phrases) == 2:
        joined = f"{phrases[0]} and {phrases[1]}"
        tail = "save both"
    elif len(phrases) >= 3:
        joined = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
        tail = "save all of these"
    else:
        # Defensive — caller already special-cased length 1.
        joined = phrases[0]
        tail = "save it"
    return f"I noticed {joined} — want me to {tail}?"


def _render_value_natural(event: MemoryEvent) -> str:
    """Single-line natural rendering of an event's value for preview_pending.

    No metadata (topic, source, extractor, confidence, evidence) — the user
    only needs to see WHAT you'd save, not the pipeline that produced it.
    """
    value = event.value or {}
    key = event.key

    if key == "identity.name" and isinstance(value, dict):
        name = value.get("name") or value.get("value")
        if name:
            return f"Name: {name}"
    if key == "identity.location.city" and isinstance(value, dict):
        city = value.get("city") or value.get("value")
        if city:
            return f"City: {city}"
    if key.startswith("identity.") and isinstance(value, dict):
        label = key.split(".", 1)[1].replace(".", " ").replace("_", " ").title()
        if len(value) == 1:
            (only_val,) = value.values()
            return f"{label}: {only_val}"
        return f"{label}: " + ", ".join(f"{k}={v}" for k, v in value.items())

    if isinstance(value, dict) and value:
        # Generic dict: humanize each entry.
        parts = []
        for k, v in value.items():
            label = str(k).replace("_", " ").capitalize()
            parts.append(f"{label}: {v}")
        return "; ".join(parts)

    # Scalar / unknown shape — show the key humanized, then the value.
    label = key.replace(".", " ").replace("_", " ").capitalize()
    return f"{label}: {value}"
