"""
Phase 9 — first-contact provisioning for channel users.

Web users are seeded at /onboarding/start with identity.md + matching applied
journal events, so Turtle knows their name from turn 1. Channel users (Discord,
Slack) got none of that: an empty shell with no name and no identity.md, even
though Discord hands us a display name on every single message. Turtle greeted a
stranger forever and had nothing to personalise with.

core.user_provisioning closes that gap. It must be idempotent (called every
turn), non-destructive (a platform display name must NEVER overwrite a name the
user stated themselves), and durable (identity.md alone is unlinked by the
replayer — the journal event is what makes it survive).
"""
from __future__ import annotations

import pytest

import core.paths as core_paths
from core.personal_memory_store import PersonalMemoryStore
from core.user_provisioning import (
    has_identity,
    provision_channel_user,
    seed_channel_profile,
)


@pytest.fixture(autouse=True)
def pm_root(tmp_path, monkeypatch):
    root = tmp_path / "personal"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root, raising=False)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snap", raising=False)
    return root


def _identity_lines(user_id: str) -> list[str]:
    doc = PersonalMemoryStore(user_id=user_id).load_topic("identity")
    return [str(x) for x in (getattr(doc, "lines", None) or [])]


def _pending_name_events(user_id: str):
    from core.memory_journal import JournalStore
    return [
        e for e in JournalStore(user_id=user_id).load_all()
        if e.key == "identity.name" and not e.applied
    ]


def test_first_contact_seeds_a_pending_name_not_applied():
    """A platform display name is untrusted input (Codex review): it must land
    as PENDING, not applied — otherwise it becomes authoritative memory that
    the replayer renders and the retrieval broker injects."""
    assert seed_channel_profile("usr_new", display_name="Shang Tsung", channel="discord")
    pending = _pending_name_events("usr_new")
    assert pending, "no pending identity.name event was journaled"
    assert pending[0].value.get("name") == "Shang Tsung"
    assert pending[0].source == "inferred"
    # No applied event, no identity.md rendered.
    assert not has_identity("usr_new")
    assert not any("Shang Tsung" in line for line in _identity_lines("usr_new"))


def test_seeding_is_idempotent():
    """Called on every turn — the second call must be a no-op, not a duplicate."""
    assert seed_channel_profile("usr_idem", display_name="Ada", channel="discord") is True
    assert seed_channel_profile("usr_idem", display_name="Ada", channel="discord") is False
    # Only one pending candidate — no duplicates journaled either.
    assert len(_pending_name_events("usr_idem")) == 1


def test_platform_name_never_overwrites_a_user_stated_name():
    """The human's own words outrank a platform profile handle."""
    store = PersonalMemoryStore(user_id="usr_known")
    store.write_topic("identity", ["- Name: Shriyash"], {"title": "Identity"})

    assert seed_channel_profile("usr_known", display_name="xX_dragon_Xx", channel="discord") is False
    lines = _identity_lines("usr_known")
    assert any("Shriyash" in l for l in lines)
    assert not any("dragon" in l.lower() for l in lines)


def test_junk_display_names_are_rejected():
    for junk in ("", "  ", "user", "Unknown", "Deleted User", "x"):
        assert seed_channel_profile("usr_junk", display_name=junk, channel="discord") is False
    assert not has_identity("usr_junk")
    assert not _pending_name_events("usr_junk")


def test_seed_writes_a_journal_candidate():
    """The candidate is journaled (single source of truth) even though NOT
    applied — so a later confirmation can promote it deterministically."""
    from core.memory_journal import JournalStore

    seed_channel_profile("usr_journal", display_name="Grace", channel="discord")
    events = JournalStore(user_id="usr_journal").load_all()
    assert any(
        e.key == "identity.name" and e.value.get("name") == "Grace" and not e.applied
        for e in events
    ), "expected a pending (unapplied) identity.name candidate"


def test_provision_from_turtle_event():
    from apps.channels import TurtleEvent

    event = TurtleEvent(
        user_id="usr_evt",
        channel="discord",
        modality="text",
        content="hi",
        sender_name="Linus",
    )
    assert provision_channel_user(event) is True
    # Candidate journaled but NOT projected — Linus doesn't appear in identity.md.
    pending = _pending_name_events("usr_evt")
    assert pending and pending[0].value.get("name") == "Linus"
    assert not has_identity("usr_evt")


def test_turtle_event_carries_sender_name_by_default():
    from apps.channels import TurtleEvent

    event = TurtleEvent(user_id="u", channel="discord", modality="text", content="hi")
    assert event.sender_name == ""
