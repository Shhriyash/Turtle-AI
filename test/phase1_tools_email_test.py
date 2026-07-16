from pathlib import Path


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
        recipients=["USER@example.com"],
        subject="Hello",
        body="Same body",
        cc=[],
        bcc=[],
    )

    monkeypatch.setattr(idem.time, "time", lambda: 125.5)
    key2 = idem.build_email_idempotency_key(
        recipients=["USER@example.com"],
        subject="Hello",
        body="Same body",
        cc=[],
        bcc=[],
    )

    assert key1 == key2


def test_idempotency_records_only_successes(monkeypatch, tmp_path):
    import tools.idempotency as idem

    monkeypatch.setattr(idem, "_DB_PATH", tmp_path / "idempotency.sqlite3")
    monkeypatch.setattr(idem, "_DB_INITIALIZED", False)

    idem.record_invocation("k", "Failed to send email: boom")
    assert idem.is_duplicate_invocation("k") is None

    idem.record_invocation("k2", "Email sent successfully! message id 123")
    assert idem.is_duplicate_invocation("k2") == "Email sent successfully! message id 123"


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
