from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from core.io_atomic import atomic_write_json
from core.memory_journal import JournalStore, MemoryEvent, make_event
from core.memory_replayer import replay
from core.personal_memory_store import PersonalMemoryStore


DEFAULT_SILENCE_DAYS = 14


@dataclass(frozen=True)
class GatePrompt:
    event_id: str
    question: str
    topic: str
    key: str


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
    ) -> None:
        self.journal = journal
        self.store = store
        self.state_path = state_path
        self.silence_days = int(silence_days)
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
        """Peek at the next pending candidate without removing it.

        Drops silenced or missing candidates lazily. Returns None when
        the queue is empty. The caller is responsible for invoking
        :meth:`record_response` after the user answers.
        """
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
            return GatePrompt(
                event_id=event.event_id,
                question=_render_question(event),
                topic=event.topic,
                key=event.key,
            )
        return None

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

    def _is_silenced(self, topic: str, key: str) -> bool:
        if self.silence_days <= 0:
            return False
        cutoff = datetime.now(UTC) - timedelta(days=self.silence_days)
        for event in self.journal.iter_events():
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
        for event in self.journal.iter_events():
            if event.event_id == event_id:
                return event
        return None

    def _event_exists_in_journal(self, event_id: str) -> bool:
        return self._load_event(event_id) is not None

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
    if key.startswith("projects.project."):
        name = value.get("name")
        if name:
            return f"You keep coming back to the {name} project — want me to track it as a recurring project?"

    return (
        "I've spotted a pattern worth remembering about "
        f"{event.topic} ({event.key}). Want me to save it?"
    )
