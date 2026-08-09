"""
Phase 9 — verified cross-channel account linking (one human, one memory).

resolve_user() mints a fresh user_id per channel binding, so the same person on
web and on Discord was two Turtle users with two disjoint memories (observed
live: shriyashbeohar1@gmail.com -> usr_1040a4c93ae5 and discord/759... ->
usr_7dd950955e9c, both named "Shriyash").

The obvious fix — link when a channel user claims an email that matches a web
account — is an ACCOUNT TAKEOVER vector: a self-claimed identifier proves
nothing, so anyone knowing your email could inherit your memory from Discord.

The implemented flow requires BOTH halves to be proven:
  * a single-use, TTL-bounded CLAIM CODE issued on the channel (proves control
    of the channel identity, and names no target account), and
  * redemption inside an AUTHENTICATED web session (proves ownership of the
    target account).

These tests pin the security properties, not just the happy path.
"""
from __future__ import annotations

import asyncio

import pytest

import core.paths as core_paths
from core.account_linking import (
    LINK_CODE_TTL_MINUTES,
    LinkCodeStore,
    merge_memory,
)


@pytest.fixture()
def store(tmp_path):
    return LinkCodeStore(tmp_path / "users.sqlite")


@pytest.fixture()
def pm_root(tmp_path, monkeypatch):
    root = tmp_path / "personal"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root, raising=False)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snap", raising=False)
    return root


def _issue(store, ext="759", src="usr_discord"):
    return store.issue(channel="discord", channel_user_id=ext, source_user_id=src)


# ── claim code security ──────────────────────────────────────────────────────

def test_code_round_trips_and_carries_the_channel_identity():
    pass  # covered by the two below; kept out to avoid a redundant sqlite file


def test_code_is_single_use(store):
    """A replayed code must not link a second time."""
    code = _issue(store).code
    first = store.consume(code)
    assert first is not None and first.channel == "discord"
    assert store.consume(code) is None, "code was redeemable twice"


def test_unknown_code_is_rejected(store):
    assert store.consume("NOPENOPE") is None
    assert store.consume("") is None


def test_expired_code_is_rejected(store, monkeypatch):
    import core.account_linking as al
    from datetime import UTC, datetime, timedelta

    code = _issue(store).code
    later = datetime.now(UTC) + timedelta(minutes=LINK_CODE_TTL_MINUTES + 1)
    monkeypatch.setattr(al, "_utc_now", lambda: later)
    assert store.consume(code) is None, "an expired code was accepted"


def test_reissue_invalidates_the_previous_code(store):
    """Asking twice must not leave a stale redeemable code lying around."""
    first = _issue(store).code
    second = _issue(store).code
    assert first != second
    assert store.consume(first) is None
    assert store.consume(second) is not None


def test_code_names_no_target_account(store):
    """A leaked code must not be usable to target someone else's account: it
    carries only the SOURCE channel identity. The target comes from the
    authenticated session at redemption time."""
    claim = store.consume(_issue(store, src="usr_discord").code)
    assert claim is not None
    assert claim.source_user_id == "usr_discord"
    assert not hasattr(claim, "target_user_id")


def test_codes_are_unguessable(store):
    """Distinct, high-entropy codes — this is a bearer credential."""
    codes = {_issue(store, ext=f"u{i}").code for i in range(25)}
    assert len(codes) == 25
    assert all(len(c) >= 8 for c in codes)


# ── the merge ────────────────────────────────────────────────────────────────

def test_merge_folds_source_memory_into_target(pm_root):
    from core.memory_journal import JournalStore, make_event

    def _fact(key, value, eid):
        return make_event(
            event_id=eid, kind="fact", topic="identity", key=key,
            value={"value": value}, confidence=1.0, source="explicit",
            extractor="deterministic", session_id="s", turn_id="t", applied=True,
        )

    JournalStore(user_id="usr_src").append_many([_fact("identity.city", "Indore", "e1")])
    JournalStore(user_id="usr_dst").append_many([_fact("identity.name", "Shriyash", "e2")])

    result = merge_memory("usr_src", "usr_dst")
    assert result["events_copied"] == 1

    keys = {e.key for e in JournalStore(user_id="usr_dst").load_all()}
    assert {"identity.name", "identity.city"} <= keys, "source memory did not arrive"


def test_merge_is_idempotent(pm_root):
    """Re-linking must not duplicate every event."""
    from core.memory_journal import JournalStore, make_event

    JournalStore(user_id="usr_src2").append_many([
        make_event(
            event_id="dup1", kind="fact", topic="identity", key="identity.city",
            value={"value": "Indore"}, confidence=1.0, source="explicit",
            extractor="deterministic", session_id="s", turn_id="t", applied=True,
        )
    ])
    first = merge_memory("usr_src2", "usr_dst2")
    second = merge_memory("usr_src2", "usr_dst2")
    assert first["events_copied"] == 1
    assert second["events_copied"] == 0


def test_merge_refuses_self_merge(pm_root):
    assert merge_memory("usr_same", "usr_same")["events_copied"] == 0


# ── identity re-point ────────────────────────────────────────────────────────

def test_link_channel_repoints_mapping(tmp_path):
    from core.identity import IdentityManager

    mgr = IdentityManager(db_path=tmp_path / "users.sqlite")

    async def run():
        await mgr.init_db()
        original = await mgr.resolve_user("discord", "759")
        web = await mgr.resolve_user("web_email", "me@example.com")
        assert original != web  # the bug: two ids for one human

        previous = await mgr.link_channel(
            user_id=web, channel="discord", channel_user_id="759"
        )
        assert previous == original
        # the Discord identity now resolves to the WEB account
        assert await mgr.resolve_user("discord", "759") == web
        # re-linking is a no-op, not an error
        assert await mgr.link_channel(
            user_id=web, channel="discord", channel_user_id="759"
        ) == web

    asyncio.run(run())


def test_link_channel_rejects_blank_input(tmp_path):
    from core.identity import IdentityManager

    mgr = IdentityManager(db_path=tmp_path / "users.sqlite")

    async def run():
        await mgr.init_db()
        assert await mgr.link_channel(user_id="", channel="discord", channel_user_id="1") is None
        assert await mgr.link_channel(user_id="u", channel="", channel_user_id="1") is None
        assert await mgr.link_channel(user_id="u", channel="discord", channel_user_id=" ") is None

    asyncio.run(run())


# ── endpoint wiring ──────────────────────────────────────────────────────────
# The unit tests above exercise LinkCodeStore/link_channel directly, which let a
# pair of NameErrors (HTTPException and identity_manager were never imported in
# turtle_server) reach a running server: every POST 500'd. These drive the real
# ASGI route so the wiring itself is covered.

def test_link_endpoint_rejects_bad_code_without_500(monkeypatch):
    from fastapi.testclient import TestClient
    import apps.turtle_server as ts

    monkeypatch.setattr(ts.settings, "deploy_mode", "local", raising=False)
    with TestClient(ts.app) as client:
        resp = client.post("/api/account/link", json={"code": "BOGUSCODE"})
    assert resp.status_code == 400, f"expected a clean 400, got {resp.status_code}"
    assert "invalid" in resp.text.lower() or "expired" in resp.text.lower()


def test_link_endpoint_requires_a_code(monkeypatch):
    from fastapi.testclient import TestClient
    import apps.turtle_server as ts

    monkeypatch.setattr(ts.settings, "deploy_mode", "local", raising=False)
    with TestClient(ts.app) as client:
        resp = client.post("/api/account/link", json={})
    assert resp.status_code == 400


def test_link_endpoint_requires_authentication(monkeypatch):
    """In cloud there is no local_dev_user fallback — unauthenticated must 401,
    never link. Authentication IS the proof of target-account ownership."""
    from fastapi.testclient import TestClient
    import apps.turtle_server as ts

    monkeypatch.setattr(ts.settings, "deploy_mode", "cloud", raising=False)
    with TestClient(ts.app) as client:
        resp = client.post("/api/account/link", json={"code": "WHATEVER"})
    assert resp.status_code == 401
