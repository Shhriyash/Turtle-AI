from pathlib import Path

import pytest


def test_spoken_email_normalizer_preserves_sentence_boundaries():
    from core.email_flow import extract_recipients, normalize_spoken_email_text

    text = "send it to a@b.com. In the mail say hi"
    normalized = normalize_spoken_email_text(text)

    assert "a@b.com" in normalized
    assert "a@b.com.In" not in normalized
    assert extract_recipients(normalized) == ["a@b.com"]

    spoken = normalize_spoken_email_text("john at the rate gmail dot com")
    assert "john@gmail.com" in spoken


def test_email_idempotency_key_is_time_independent(monkeypatch):
    import tools.idempotency as idem

    monkeypatch.setattr(idem.time, "time", lambda: 60.5)
    key1 = idem.build_email_idempotency_key(
        "usr_a",
        recipients=["USER@example.com"],
        subject="Hello",
        body="Same body",
        cc=[],
        bcc=[],
    )

    monkeypatch.setattr(idem.time, "time", lambda: 125.5)
    key2 = idem.build_email_idempotency_key(
        "usr_a",
        recipients=["USER@example.com"],
        subject="Hello",
        body="Same body",
        cc=[],
        bcc=[],
    )

    assert key1 == key2


def test_email_idempotency_key_is_scoped_per_user(monkeypatch):
    """The cross-tenant fix: two different users sending the byte-identical
    email must get DIFFERENT keys, so one tenant's send never dedups against
    another's."""
    import tools.idempotency as idem

    monkeypatch.setattr(idem.time, "time", lambda: 60.5)
    key_a = idem.build_email_idempotency_key(
        "usr_a", recipients=["shared@example.com"], subject="Hi", body="Same body", cc=[], bcc=[],
    )
    key_b = idem.build_email_idempotency_key(
        "usr_b", recipients=["shared@example.com"], subject="Hi", body="Same body", cc=[], bcc=[],
    )

    assert key_a != key_b
    assert key_a.startswith("usr_a:")
    assert key_b.startswith("usr_b:")


def test_idempotency_records_only_successes(monkeypatch, tmp_path):
    import tools.idempotency as idem

    monkeypatch.setattr(idem, "_DB_PATH", tmp_path / "idempotency.sqlite3")
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)

    idem.record_invocation("k", "Failed to send email: boom")
    assert idem.is_duplicate_invocation("k") is None

    idem.record_invocation("k2", "Email sent successfully! message id 123")
    assert idem.is_duplicate_invocation("k2") == "Email sent successfully! message id 123"


def test_local_reservation_same_key_twice_sends_once(monkeypatch, tmp_path):
    """Same user, identical email, twice within the window: the second call
    must be told it's a duplicate (either 'still in flight' or the cached
    completed result), never a fresh None telling the caller to send again."""
    import tools.idempotency as idem

    monkeypatch.setattr(idem, "_DB_PATH", tmp_path / "idempotency.sqlite3")
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)

    key = "usr_a:deadbeef"
    # First call: reservation acquired.
    assert idem.is_duplicate_invocation(key) is None
    # Second call before the send finished: reservation still pending.
    assert idem.is_duplicate_invocation(key) == idem._PENDING_MESSAGE

    idem.record_invocation(key, "Email sent successfully! ok")

    # Third call after the send completed: cached result, still a duplicate.
    assert idem.is_duplicate_invocation(key) == "Email sent successfully! ok"


def test_local_reservation_failed_send_deletes_reservation_allows_retry(monkeypatch, tmp_path):
    """A failed send must release the reservation so the user's retry is not
    blocked for the rest of the window."""
    import tools.idempotency as idem

    monkeypatch.setattr(idem, "_DB_PATH", tmp_path / "idempotency.sqlite3")
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)

    key = "usr_a:failcase"
    assert idem.is_duplicate_invocation(key) is None  # reserved
    idem.record_invocation(key, "Failed to send email: smtp boom")

    # Retry should be a fresh reservation, not a duplicate.
    assert idem.is_duplicate_invocation(key) is None


def test_local_reservation_unavailable_refuses_send(monkeypatch, tmp_path):
    """When the reservation store cannot be reached, is_duplicate_invocation
    must raise (fail closed) rather than silently returning None (fail open,
    the old behaviour)."""
    import tools.idempotency as idem

    monkeypatch.setattr(idem, "_DB_PATH", tmp_path / "idempotency.sqlite3")
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)

    def _boom(*_a, **_kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(idem, "_ensure_db", _boom)

    with pytest.raises(idem.IdempotencyReservationError):
        idem.is_duplicate_invocation("usr_a:boom")


def test_local_two_tenants_identical_email_both_send(monkeypatch, tmp_path):
    """Ledger acceptance criterion: two different user_ids sending the
    byte-identical email within the window both get a fresh reservation
    (both send) — the previous global key made the second tenant's send
    silently no-op against the first tenant's."""
    import tools.idempotency as idem

    monkeypatch.setattr(idem, "_DB_PATH", tmp_path / "idempotency.sqlite3")
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)

    key_a = idem.build_email_idempotency_key(
        "usr_a", recipients=["shared@example.com"], subject="Hi", body="Same body", cc=[], bcc=[],
    )
    key_b = idem.build_email_idempotency_key(
        "usr_b", recipients=["shared@example.com"], subject="Hi", body="Same body", cc=[], bcc=[],
    )

    # Both are fresh reservations -> both proceed to send.
    assert idem.is_duplicate_invocation(key_a) is None
    assert idem.is_duplicate_invocation(key_b) is None


def test_local_concurrent_reservation_exactly_one_proceeds(monkeypatch, tmp_path):
    """Two concurrent callers racing on the same key: exactly one gets the
    reservation (None), the other gets told it's a duplicate."""
    import threading

    import tools.idempotency as idem

    monkeypatch.setattr(idem, "_DB_PATH", tmp_path / "idempotency.sqlite3")
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)
    # Pre-create the DB/table on the main thread so both worker threads race
    # purely on the INSERT, not on CREATE TABLE IF NOT EXISTS.
    idem._ensure_db().close()

    key = "usr_a:racecase"
    results: list[object] = []
    barrier = threading.Barrier(2)

    def _attempt():
        barrier.wait()
        results.append(idem.is_duplicate_invocation(key))

    threads = [threading.Thread(target=_attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(None) == 1
    assert results.count(idem._PENDING_MESSAGE) == 1


# ---------------------------------------------------------------------------
# 1a.7: SMTP timeout — a hung mail server must not hang forever.
# ---------------------------------------------------------------------------

def _make_email_tool():
    from tools.email_tools.email_toolkit import EmailTool

    return EmailTool(
        sender_name="Turtle",
        sender_email="turtle@example.invalid",
        sender_passkey="fake-app-password",
    )


def test_smtp_ssl_gets_timeout_send_multiple(monkeypatch):
    """_send_email_internal_multiple (the live send_email path) must pass
    timeout=20 to smtplib.SMTP_SSL, or a hung mail server hangs forever."""
    import smtplib

    captured = {}

    class _FakeSMTP:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def login(self, *a, **kw):
            pass

        def send_message(self, *a, **kw):
            pass

    monkeypatch.setattr(smtplib, "SMTP_SSL", _FakeSMTP)

    tool = _make_email_tool()
    result = tool.send_email(receiver="to@example.invalid", subject="Hi", body="Body")

    assert not result.startswith("error:")
    assert captured["kwargs"].get("timeout") == 20


def test_smtp_ssl_gets_timeout_test_connection(monkeypatch):
    """test_connection must also pass timeout=20 (third construction site)."""
    import smtplib

    captured = {}

    class _FakeSMTP:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def login(self, *a, **kw):
            pass

    monkeypatch.setattr(smtplib, "SMTP_SSL", _FakeSMTP)

    tool = _make_email_tool()
    result = tool.test_connection()

    assert result.success is True
    assert captured["kwargs"].get("timeout") == 20


def test_all_three_smtp_ssl_call_sites_pass_timeout():
    """Static check: every smtplib.SMTP_SSL(...) construction in the toolkit
    carries timeout=20 (keyword-only per the stdlib signature, so it can't be
    positional). Complements the behavioural tests above."""
    import inspect

    from tools.email_tools import email_toolkit

    source = inspect.getsource(email_toolkit)
    call_sites = [
        line for line in source.splitlines() if "smtplib.SMTP_SSL(" in line
    ]
    assert len(call_sites) == 3, call_sites
    for line in call_sites:
        assert "timeout=20" in line, line


def test_send_email_now_call_site_is_off_event_loop():
    """apps/turtle_server.py's send_email_assistant tool must invoke the
    blocking send_email_now via asyncio.to_thread, not directly, so a hung
    SMTP call can't freeze the event loop (and every connected user with
    it). Source-level check because building a full RunContext/SharedState
    to drive the tool end-to-end is out of scope for this WP's file
    ownership.
    """
    import inspect

    import apps.turtle_server as ts

    source = inspect.getsource(ts)
    assert "await asyncio.to_thread(send_email_now, merged)" in source
    assert "send_result = send_email_now(merged)" not in source


def test_remember_tool_contract_and_registration_present():
    import apps.turtle_server as ts

    repo_root = Path(ts.__file__).resolve().parents[1]
    contract = repo_root / "core" / "system_prompts" / "tools" / "remember.md"
    source = Path(ts.__file__).read_text(encoding="utf-8")

    assert contract.exists()
    contract_text = contract.read_text(encoding="utf-8")
    assert "Only claim \"I'll remember\" after this tool returns ok." in contract_text

    assert "class RememberArgs" in source
    assert "async def remember(" in source
    assert "(\"remember\", remember)" in source
