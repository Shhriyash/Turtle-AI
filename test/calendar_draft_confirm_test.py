"""
test/calendar_draft_confirm_test.py
------------------------------------
WP1.E1 (ledger 1b.1, solution S-7.5): calendar_create must draft instead of
creating, calendar_confirm executes the draft with a reservation.

Covers:
  - pending_calendar in core/session_store.py: default shape, TTL-on-read,
    set/clear, and round trip through _sync_to_backend / _restore_from_session
  - tools/idempotency.py: the email/calendar key discriminator, the
    generalised `success` param on record_invocation/send_with_reservation
  - apps/turtle_server.py's calendar_create/calendar_confirm closures, at the
    source level (building a full RunContext/SharedState to drive the tool
    end-to-end is out of scope for this WP's file ownership — same rationale
    test/phase1_tools_email_test.py already documents for send_email_assistant)
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.session_store import SessionStore
from core.storage import Session
from core.storage.local.sqlite_store import SQLiteSessionStore


def run(coro):
    return asyncio.run(coro)


def old_iso(hours: int = 0) -> str:
    value = datetime.now(timezone.utc) - timedelta(hours=hours)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# core/session_store.py — pending_calendar
# ---------------------------------------------------------------------------

def test_default_pending_calendar_shape(tmp_path):
    store = SessionStore(SQLiteSessionStore(db_path=tmp_path / "s.sqlite"), user_id="usr_a")
    assert store.pending_calendar == {
        "title": "",
        "start_iso": "",
        "end_iso": "",
        "attendee_emails": [],
        "description": "",
        "add_google_meet": True,
        "notify_attendees": False,
    }


def test_set_and_get_pending_calendar(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        store = SessionStore(SQLiteSessionStore(db_path=db_path), user_id="usr_a")
        await store.start_or_restore("strict_new")
        await store.set_pending_calendar(
            title="Board Sync",
            start_iso="2026-06-01T10:00:00+05:30",
            end_iso="2026-06-01T10:30:00+05:30",
            attendee_emails=["alice@example.com"],
            notify_attendees=True,
        )
        pending = store.get_pending_calendar()
        assert pending["title"] == "Board Sync"
        assert pending["attendee_emails"] == ["alice@example.com"]
        assert pending["notify_attendees"] is True

    run(scenario())


def test_pending_calendar_ttl_clears_stale_draft(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        store = SessionStore(SQLiteSessionStore(db_path=db_path), user_id="usr_a")
        await store.start_or_restore("strict_new")
        await store.set_pending_calendar(title="Stale meeting")

        store._pending_calendar_updated_at = old_iso(hours=2)

        assert store.get_pending_calendar() == store._default_pending_calendar()

    run(scenario())


def test_clear_pending_calendar(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        store = SessionStore(SQLiteSessionStore(db_path=db_path), user_id="usr_a")
        await store.start_or_restore("strict_new")
        await store.set_pending_calendar(title="Meeting")
        await store.clear_pending_calendar()
        assert store.get_pending_calendar() == store._default_pending_calendar()

    run(scenario())


def test_pending_calendar_survives_sync_and_restore_round_trip(tmp_path):
    """The draft must survive a session round trip through
    _sync_to_backend / _restore_from_session — a reconnect must not silently
    lose an in-flight calendar draft."""
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        store = SessionStore(backend, user_id="usr_a")
        await store.start_or_restore("strict_new")
        await store.set_pending_calendar(
            title="Quarterly Review",
            start_iso="2026-07-01T09:00:00+00:00",
            end_iso="2026-07-01T10:00:00+00:00",
            attendee_emails=["bob@example.com"],
            description="Q3 numbers",
            add_google_meet=False,
            notify_attendees=True,
        )
        session_id = store.session_id

        restored_store = SessionStore(SQLiteSessionStore(db_path=db_path), user_id="usr_a")
        session = await restored_store.backend.get(session_id)
        restored_store._restore_from_session(session)

        pending = restored_store.get_pending_calendar()
        assert pending["title"] == "Quarterly Review"
        assert pending["start_iso"] == "2026-07-01T09:00:00+00:00"
        assert pending["attendee_emails"] == ["bob@example.com"]
        assert pending["add_google_meet"] is False
        assert pending["notify_attendees"] is True

    run(scenario())


def test_mark_finalized_resets_pending_calendar(tmp_path):
    async def scenario():
        db_path = tmp_path / "s.sqlite"
        backend = SQLiteSessionStore(db_path=db_path)
        await backend.init_db()
        await backend.put(
            Session(
                session_id="pending",
                data={
                    "status": "pending_finalization",
                    "user_id": "usr_a",
                    "messages": [],
                    "pending_email": {},
                    "pending_calendar": {"title": "Leftover draft"},
                    "summary": [],
                    "updated_at": old_iso(),
                },
            )
        )
        store = SessionStore(backend, user_id="usr_a")
        await store.mark_finalized("pending")

        finalized = await backend.get("pending")
        assert finalized.data["pending_calendar"] == store._default_pending_calendar()

    run(scenario())


# ---------------------------------------------------------------------------
# tools/idempotency.py — discriminator + generalised success
# ---------------------------------------------------------------------------

def test_email_and_calendar_keys_never_collide_for_the_same_user():
    from tools.idempotency import build_calendar_idempotency_key, build_email_idempotency_key

    email_key = build_email_idempotency_key(
        "usr_a", recipients=["x@example.com"], subject="s", body="b", cc=[], bcc=[],
    )
    cal_key = build_calendar_idempotency_key(
        "usr_a", title="s", start_iso="b", end_iso="", attendee_emails=[],
    )
    assert email_key != cal_key
    assert ":email:" in email_key
    assert ":cal:" in cal_key


def test_email_idempotency_key_still_scoped_per_user_after_discriminator_change():
    """The email idempotency path must behave identically after the
    discriminator change: same user, correct namespace prefix."""
    from tools.idempotency import build_email_idempotency_key

    key_a = build_email_idempotency_key(
        "usr_a", recipients=["shared@example.com"], subject="Hi", body="Same body", cc=[], bcc=[],
    )
    key_b = build_email_idempotency_key(
        "usr_b", recipients=["shared@example.com"], subject="Hi", body="Same body", cc=[], bcc=[],
    )
    assert key_a != key_b
    assert key_a.startswith("usr_a:email:")
    assert key_b.startswith("usr_b:email:")

    # Deterministic: identical inputs -> identical key.
    key_a_again = build_email_idempotency_key(
        "usr_a", recipients=["shared@example.com"], subject="Hi", body="Same body", cc=[], bcc=[],
    )
    assert key_a == key_a_again


def test_calendar_idempotency_key_is_scoped_per_user_and_stable():
    from tools.idempotency import build_calendar_idempotency_key

    key_a = build_calendar_idempotency_key(
        "usr_a", "Sync", "2026-06-01T10:00:00+00:00", "2026-06-01T10:30:00+00:00", ["x@example.com"],
    )
    key_b = build_calendar_idempotency_key(
        "usr_b", "Sync", "2026-06-01T10:00:00+00:00", "2026-06-01T10:30:00+00:00", ["x@example.com"],
    )
    assert key_a != key_b
    assert key_a.startswith("usr_a:cal:")

    key_a_again = build_calendar_idempotency_key(
        "usr_a", "Sync", "2026-06-01T10:00:00+00:00", "2026-06-01T10:30:00+00:00", ["x@example.com"],
    )
    assert key_a == key_a_again


def test_record_invocation_email_path_unchanged_without_success_kwarg(tmp_path, monkeypatch):
    """Omitting `success` (the email call site's exact usage) must keep
    sniffing "Email sent successfully" — the generalisation must not change
    the email caller's behaviour."""
    import tools.idempotency as idem

    monkeypatch.setattr(idem, "_DB_PATH", tmp_path / "idempotency.sqlite3")
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)

    key_ok = "usr_a:email:ok"
    assert idem.is_duplicate_invocation(key_ok) is None
    idem.record_invocation(key_ok, "Email sent successfully! message id 1")
    assert idem.is_duplicate_invocation(key_ok) == "Email sent successfully! message id 1"

    key_fail = "usr_a:email:fail"
    assert idem.is_duplicate_invocation(key_fail) is None
    idem.record_invocation(key_fail, "Failed to send email: boom")
    # A failure deletes the reservation -> a fresh attempt is a new reservation.
    assert idem.is_duplicate_invocation(key_fail) is None


def test_record_invocation_explicit_success_true_caches_non_email_result(tmp_path, monkeypatch):
    """A calendar-shaped result string (never starting with "Email sent
    successfully") must still be cached when the caller explicitly says
    success=True — this is the whole point of generalising record_invocation."""
    import tools.idempotency as idem

    monkeypatch.setattr(idem, "_DB_PATH", tmp_path / "idempotency.sqlite3")
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)

    key = "usr_a:cal:evt"
    assert idem.is_duplicate_invocation(key) is None
    idem.record_invocation(key, "Event created: Board Sync", success=True)
    assert idem.is_duplicate_invocation(key) == "Event created: Board Sync"


def test_record_invocation_explicit_success_false_releases_reservation(tmp_path, monkeypatch):
    import tools.idempotency as idem

    monkeypatch.setattr(idem, "_DB_PATH", tmp_path / "idempotency.sqlite3")
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)

    key = "usr_a:cal:evt2"
    assert idem.is_duplicate_invocation(key) is None
    idem.record_invocation(key, "some upstream_error text", success=False)
    # Released -> a retry acquires a fresh reservation.
    assert idem.is_duplicate_invocation(key) is None


def test_send_with_reservation_is_success_callback_drives_caching(tmp_path, monkeypatch):
    """send_with_reservation's new `is_success` callback (used by
    calendar_confirm) must override the default email-string sniff."""
    import tools.idempotency as idem

    monkeypatch.setattr(idem, "_DB_PATH", tmp_path / "idempotency.sqlite3")
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)

    key = "usr_a:cal:evt3"
    assert idem.is_duplicate_invocation(key) is None

    async def _create() -> str:
        return "Event created successfully!"  # deliberately NOT the email sentinel

    async def scenario():
        result = await idem.send_with_reservation(key, _create, is_success=lambda _r: True)
        assert result == "Event created successfully!"

    run(scenario())

    # Cached: a duplicate confirm within the window returns the cached result
    # instead of creating a second event.
    assert idem.is_duplicate_invocation(key) == "Event created successfully!"


def test_send_with_reservation_failure_releases_reservation_for_retry(tmp_path, monkeypatch):
    """A failed create (raises) must release the reservation so the user can
    retry — not strand it for the full window (the wave-1 bug the brief
    warns about: a raise, not a return, stranding a reservation)."""
    import tools.idempotency as idem

    monkeypatch.setattr(idem, "_DB_PATH", tmp_path / "idempotency.sqlite3")
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)

    key = "usr_a:cal:evt4"
    assert idem.is_duplicate_invocation(key) is None

    async def _fails() -> str:
        raise RuntimeError("upstream calendar API boom")

    async def scenario():
        with pytest.raises(RuntimeError):
            await idem.send_with_reservation(key, _fails, is_success=lambda _r: True)

    run(scenario())

    # Released -> a retry acquires a fresh reservation, not blocked.
    assert idem.is_duplicate_invocation(key) is None


def test_send_with_reservation_catches_base_exception_not_just_exception(tmp_path, monkeypatch):
    """Must release the reservation on asyncio.CancelledError too (does not
    subclass Exception since Python 3.8) — the brief's "cover BaseException,
    not just Exception" instruction."""
    import tools.idempotency as idem

    monkeypatch.setattr(idem, "_DB_PATH", tmp_path / "idempotency.sqlite3")
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)

    key = "usr_a:cal:evt5"
    assert idem.is_duplicate_invocation(key) is None

    async def _cancelled() -> str:
        raise asyncio.CancelledError()

    async def scenario():
        with pytest.raises(asyncio.CancelledError):
            await idem.send_with_reservation(key, _cancelled, is_success=lambda _r: True)

    run(scenario())

    assert idem.is_duplicate_invocation(key) is None


# ---------------------------------------------------------------------------
# apps/turtle_server.py — source-level checks
#
# Building a full RunContext/SharedState to drive calendar_create/
# calendar_confirm end-to-end is out of scope for this WP's file ownership
# (same rationale test/phase1_tools_email_test.py documents for
# send_email_assistant). These assert the closures are wired the way the
# brief requires.
# ---------------------------------------------------------------------------

def _turtle_server_source() -> str:
    import apps.turtle_server as ts
    return inspect.getsource(ts)


def test_calendar_create_stores_draft_and_never_creates_directly():
    """calendar_create must not call create_calendar_event at all — it only
    stages a pending_calendar draft. Only calendar_confirm may call
    create_calendar_event."""
    source = _turtle_server_source()
    create_start = source.index("async def calendar_create(")
    confirm_start = source.index("async def calendar_confirm(")
    assert create_start < confirm_start
    calendar_create_body = source[create_start:confirm_start]

    assert "create_calendar_event" not in calendar_create_body
    assert "set_pending_calendar(" in calendar_create_body
    assert "render_calendar_draft(" in calendar_create_body


def test_calendar_confirm_uses_reservation_and_clears_draft_only_on_success():
    source = _turtle_server_source()
    confirm_start = source.index("async def calendar_confirm(")
    next_def = source.index("\n        async def ", confirm_start + 10)
    calendar_confirm_body = source[confirm_start:next_def]

    assert "create_calendar_event" in calendar_confirm_body
    assert "build_calendar_idempotency_key" in calendar_confirm_body
    assert "send_with_reservation(" in calendar_confirm_body
    assert "is_duplicate_invocation" in calendar_confirm_body
    assert "clear_pending_calendar()" in calendar_confirm_body
    # Clearing must be gated on success, not unconditional.
    assert 'if success_holder["ok"]:' in calendar_confirm_body


def test_calendar_confirm_handles_no_pending_draft():
    source = _turtle_server_source()
    confirm_start = source.index("async def calendar_confirm(")
    next_def = source.index("\n        async def ", confirm_start + 10)
    calendar_confirm_body = source[confirm_start:next_def]

    assert "get_pending_calendar()" in calendar_confirm_body
    assert "no pending calendar event to confirm" in calendar_confirm_body.lower()


def test_calendar_tools_registered_with_contracts():
    source = _turtle_server_source()
    assert '("calendar_create", calendar_create)' in source
    assert '("calendar_confirm", calendar_confirm)' in source

    repo_root = Path(_import_turtle_server_path()).resolve().parents[1]
    create_contract = repo_root / "core" / "system_prompts" / "tools" / "calendar_create.md"
    confirm_contract = repo_root / "core" / "system_prompts" / "tools" / "calendar_confirm.md"
    assert create_contract.exists()
    assert confirm_contract.exists()
    assert "calendar_confirm" in create_contract.read_text(encoding="utf-8")
    assert "draft" in confirm_contract.read_text(encoding="utf-8").lower()


def _import_turtle_server_path() -> str:
    import apps.turtle_server as ts
    return ts.__file__
