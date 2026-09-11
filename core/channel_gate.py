"""
core/channel_gate.py
---------------------
Channel-native answering for ConfirmationGate (ISSUE-011).

The web UI answers a pending memory candidate through a dedicated REST call
(/api/memory/confirm) — deliberately NOT by parsing chat text, because a bare
"yes" floating in ordinary conversation could silently promote a stale
candidate that has nothing to do with what the user just said (see the
Phase 4 note in personal_memory_extract.py). Channels (Discord, Telegram,
WhatsApp, Slack, iMessage) have no such panel, so a candidate queued for a
channel user was previously never asked about — and never applied.

This module reopens chat-text answering, but only for channels, and only
inside guards narrow enough to avoid the original hazard:
  1. A prompt must have been surfaced to THIS (user_id, channel) moments ago.
  2. It must be answered within a short TTL of being surfaced.
  3. The reply must parse as a short, unambiguous yes/no token — not any
     message that merely contains the word "yes".
Any reply that fails a guard is not consumed and falls through to a normal
agent turn, so "yes, and also send the invoice" is treated as a fresh
message rather than a gate answer with a side order of invoice.

Deliberately free of any server/LLM/channel-adapter code so it is
unit-testable in isolation, matching ConfirmationGate's own design note.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


# Short, exact tokens only. Case/whitespace/trailing punctuation normalized
# before matching; a caller-supplied phrase like "yes please" (two words,
# first word in the accept set) is still allowed through, but "yes but only
# my name" is not — see parse_gate_answer's word-count guard.
_ACCEPT_WORDS = {
    "yes", "y", "yep", "yeah", "yup", "sure", "correct", "confirm", "confirmed",
    "ok", "okay",
}
_REJECT_WORDS = {
    "no", "n", "nope", "nah", "cancel", "skip", "negative", "reject",
}

# Long enough for a normal chat reply cadence; short enough that a much later,
# unrelated "yes" cannot resurrect a stale ask. Matches the order of magnitude
# of _CHANNEL_STATE_IDLE_TTL_S's spirit (bounded, not indefinite) without
# reusing that constant, since the two TTLs protect different things.
DEFAULT_TTL_SECONDS = 300


def parse_gate_answer(text: str) -> bool | None:
    """Classify a chat reply as accept (True) / reject (False) / neither (None).

    Strict on purpose. Returns None for anything longer than a short
    yes/no-shaped reply, including a sentence that merely CONTAINS "yes" —
    that ambiguity is exactly what closed off text-parsing in Phase 4.
    """
    normalized = (text or "").strip().lower().rstrip(".!?,;: ")
    if not normalized:
        return None
    words = normalized.split()
    if len(words) > 2:
        return None
    first = words[0]
    if normalized in _ACCEPT_WORDS or first in _ACCEPT_WORDS:
        return True
    if normalized in _REJECT_WORDS or first in _REJECT_WORDS:
        return False
    return None


@dataclass(frozen=True)
class _Outstanding:
    event_ids: tuple[str, ...]
    expires_at: float


class ChannelGateBuffer:
    """Tracks the ONE outstanding confirmation-gate prompt per (user_id, channel).

    In-memory and best-effort, matching the rest of the channel-state cache's
    posture (apps/turtle_server.py::_CHANNEL_STATES): a lost entry (process
    restart, missed TTL) is not a data-loss bug — the candidate stays queued
    in the journal and ConfirmationGate.next_prompt() simply re-asks on the
    user's next turn.

    Callers MUST only note/consume prompts for a DM/private context — the
    caller is responsible for that check (mirrors TurtleEvent.is_private's
    existing use for account-link claim codes: never surface or resolve a
    personal-fact prompt in a shared channel).
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS, max_entries: int = 256) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._outstanding: dict[tuple[str, str], _Outstanding] = {}

    def note_prompt(
        self,
        key: tuple[str, str],
        event_ids: tuple[str, ...],
        *,
        now: float | None = None,
    ) -> None:
        """Record that `event_ids` were just surfaced to (user_id, channel) `key`."""
        if not event_ids:
            return
        now = now if now is not None else time.monotonic()
        self._prune(now)
        if len(self._outstanding) >= self._max_entries and key not in self._outstanding:
            # Defensive cap — drop the entry closest to expiry rather than
            # grow unbounded under a pathological number of distinct senders.
            oldest_key = min(
                self._outstanding, key=lambda k: self._outstanding[k].expires_at, default=None
            )
            if oldest_key is not None:
                self._outstanding.pop(oldest_key, None)
        self._outstanding[key] = _Outstanding(event_ids=tuple(event_ids), expires_at=now + self._ttl_seconds)

    def try_consume_answer(
        self,
        key: tuple[str, str],
        text: str,
        *,
        now: float | None = None,
    ) -> tuple[bool, tuple[str, ...]] | None:
        """If `key` has a live outstanding prompt AND `text` parses as yes/no,
        pop and return (accepted, event_ids). Otherwise return None WITHOUT
        touching any outstanding prompt — a non-answer must not burn it, so
        the user can still answer correctly on a later turn within the TTL.
        """
        now = now if now is not None else time.monotonic()
        outstanding = self._outstanding.get(key)
        if outstanding is None:
            return None
        if now >= outstanding.expires_at:
            self._outstanding.pop(key, None)
            return None
        verdict = parse_gate_answer(text)
        if verdict is None:
            return None
        self._outstanding.pop(key, None)
        return verdict, outstanding.event_ids

    def has_outstanding(self, key: tuple[str, str], *, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        outstanding = self._outstanding.get(key)
        return outstanding is not None and now < outstanding.expires_at

    def clear(self, key: tuple[str, str]) -> None:
        self._outstanding.pop(key, None)

    def _prune(self, now: float) -> None:
        expired = [k for k, v in self._outstanding.items() if now >= v.expires_at]
        for k in expired:
            self._outstanding.pop(k, None)
