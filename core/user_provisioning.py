"""
core/user_provisioning.py
-------------------------
First-contact provisioning for users who arrive on a CHANNEL (Discord, Slack,
WhatsApp) rather than through web onboarding.

Web users get a seeded profile at /onboarding/start: identity.md plus matching
*applied* journal events, so Turtle knows their name from turn 1 and the
replayer doesn't unlink the seeded file. Channel users got none of that — a
Discord user was an empty shell with no name, no identity.md, and no timezone,
even though the platform hands us a display name for free on every message.

This module is the channel-side equivalent. It is deliberately:

* **idempotent** — safe to call on every turn; it no-ops once identity exists;
* **non-destructive** — it NEVER overwrites a name the user stated themselves.
  A platform display name ("Shang Tsung") is a weak, low-confidence signal, so
  it is journaled as `inferred` and is outranked by anything the user actually
  tells Turtle;
* **best-effort** — provisioning must never break a turn, so every failure is
  logged and swallowed.
"""
from __future__ import annotations

from typing import Any

# Platform display names that carry no information about the human.
_USELESS_NAMES = frozenset({"", "unknown", "user", "none", "null", "deleted user"})


def _looks_like_a_name(value: str) -> bool:
    cleaned = (value or "").strip()
    if len(cleaned) < 2 or len(cleaned) > 60:
        return False
    return cleaned.lower() not in _USELESS_NAMES


def has_identity(user_id: str) -> bool:
    """True when this user already has a stored, APPLIED name."""
    try:
        from core.personal_memory_store import PersonalMemoryStore

        doc = PersonalMemoryStore(user_id=user_id).load_topic("identity")
        lines = getattr(doc, "lines", None) or []
        return any("name:" in str(line).lower() for line in lines)
    except Exception:
        return False


def _has_pending_first_contact(user_id: str) -> bool:
    """True when we already journaled a first_contact candidate for this user.
    Used to make seeding idempotent now that we no longer render identity.md
    (which was the old idempotency signal)."""
    try:
        from core.memory_journal import JournalStore

        for ev in JournalStore(user_id=user_id).load_all():
            if ev.session_id == "first_contact" and ev.key == "identity.name":
                return True
    except Exception:
        pass
    return False


def seed_channel_profile(
    user_id: str,
    *,
    display_name: str = "",
    channel: str = "",
) -> bool:
    """Seed a first-contact profile for a channel user. Returns True if seeded.

    Writes identity.md AND a matching applied journal event — both are required.
    identity.md alone is not durable: the replayer rebuilds topic files from the
    journal, sees zero applied identity events, and unlinks the seeded file
    (the same trap web onboarding documents).
    """
    if not user_id:
        return False
    name = (display_name or "").strip()
    if not _looks_like_a_name(name):
        return False
    if has_identity(user_id):
        return False  # already known — never clobber
    if _has_pending_first_contact(user_id):
        return False  # already candidate-seeded on a prior turn (idempotent)

    try:
        from core.memory_journal import JournalStore, make_event

        # Codex adversarial review: a platform display name is UNTRUSTED input.
        # A guild moderator can rename someone; a Discord user can pick any
        # display name themselves. Journaling it as applied=True made that name
        # authoritative memory and produced a durable identity-spoof + prompt-
        # poisoning path through the display-name field.
        #
        # It now lands as applied=False (pending), so the replayer WILL NOT
        # render it into identity.md and the retrieval broker WILL NOT inject
        # it as authoritative memory. It still serves as a soft hint the
        # extractor can lift on a later turn once the user speaks in first
        # person ("call me X"), and the confirmation gate can promote it.
        # Notably: we no longer write identity.md ourselves either — the
        # replayer owns that projection, and it will skip a pending event.
        JournalStore(user_id=user_id).append(
            make_event(
                kind="fact",
                topic="identity",
                key="identity.name",
                value={"name": name},
                confidence=0.5,
                source="inferred",
                extractor="deterministic",
                session_id="first_contact",
                turn_id=f"first_contact_{channel or 'channel'}",
                evidence={"note": f"{channel or 'channel'} profile display name (untrusted)"},
                applied=False,
            )
        )
        print(
            f"LOG: seeded first-contact CANDIDATE for {user_id} ({channel}) "
            f"name={name!r} (pending, not applied)"
        )
        return True
    except Exception as exc:  # never break a turn over provisioning
        print(f"LOG: channel profile seeding failed for {user_id}: {exc}")
        return False


def provision_channel_user(event: Any) -> bool:
    """Convenience wrapper: seed from a TurtleEvent's sender metadata."""
    return seed_channel_profile(
        getattr(event, "user_id", "") or "",
        display_name=getattr(event, "sender_name", "") or "",
        channel=str(getattr(event, "channel", "") or ""),
    )
