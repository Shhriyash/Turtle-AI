"""
WP1.D2 (ledger 1a.4 part 3) — web-issued, channel-redeemed link codes.

Mirrors the direction of core/account_linking.py's existing ``link_codes``
table: there the CHANNEL issues a code and an authenticated WEB session
redeems it; here the WEB (already authenticated) issues a code bound to the
caller's account, and the CHANNEL identity that SENDS it is what gets proven
and linked.

Security property pinned by these tests: the code is bound to a TARGET
account, not a channel identity, so whoever sends it (from any channel
identity they control) gets THAT channel identity pointed at the target and
their own channel-side memory merged in. That is why redemption is reserved
to the FIRST channel identity that attempts it (closing the same two-claimant
race the original table closes), single-use, and TTL-bounded.
"""
from __future__ import annotations

import asyncio

import pytest

import core.account_linking as al
import core.paths as core_paths
from core.account_linking import LINK_CODE_TTL_MINUTES, LinkCodeStore


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


# ── store-level: issue / reserve / consume ──────────────────────────────────

def test_issue_target_code_binds_target_not_channel(store):
    issued = store.issue_target_code(target_user_id="usr_web")
    assert issued.target_user_id == "usr_web"
    assert issued.channel == "" and issued.channel_user_id == ""
    assert len(issued.code) >= 8


def test_reissue_invalidates_previous_target_code(store):
    first = store.issue_target_code(target_user_id="usr_web").code
    second = store.issue_target_code(target_user_id="usr_web").code
    assert first != second
    assert store.reserve_target_code(first, "discord", "759")[0] == "invalid"
    assert store.reserve_target_code(second, "discord", "759")[0] == "ok"


def test_reserve_then_consume_round_trip(store):
    code = store.issue_target_code(target_user_id="usr_web").code
    status, claim = store.reserve_target_code(code, "discord", "759")
    assert status == "ok"
    assert claim.target_user_id == "usr_web"
    consumed = store.consume_target_code(code)
    assert consumed is not None
    assert consumed.target_user_id == "usr_web"


def test_code_is_single_use(store):
    code = store.issue_target_code(target_user_id="usr_web").code
    assert store.consume_target_code(code) is not None
    assert store.consume_target_code(code) is None, "code was redeemable twice"


def test_expired_code_is_rejected(store, monkeypatch):
    from datetime import UTC, datetime, timedelta

    code = store.issue_target_code(target_user_id="usr_web").code
    later = datetime.now(UTC) + timedelta(minutes=LINK_CODE_TTL_MINUTES + 1)
    monkeypatch.setattr(al, "_utc_now", lambda: later)
    status, claim = store.reserve_target_code(code, "discord", "759")
    assert status == "invalid"
    assert store.consume_target_code(code) is None


def test_unknown_code_is_rejected(store):
    assert store.reserve_target_code("NOPENOPE", "discord", "1")[0] == "invalid"
    assert store.consume_target_code("NOPENOPE") is None


def test_reserve_same_channel_identity_is_idempotent(store):
    """The same channel identity retrying (e.g. a message-delivery retry)
    inside the reservation window must not be locked out by itself."""
    code = store.issue_target_code(target_user_id="usr_web").code
    assert store.reserve_target_code(code, "discord", "759")[0] == "ok"
    assert store.reserve_target_code(code, "discord", "759")[0] == "ok"


def test_reserve_blocks_a_different_channel_identity(store):
    """A code redeemed by a DIFFERENT channel identity than the one that first
    reserved it must be rejected — this is the pinned behaviour for "a code
    redeemed by a different channel identity than intended": first claimer
    wins, a second identity racing the same code is locked out, never
    revealed who holds it."""
    code = store.issue_target_code(target_user_id="usr_web").code
    status_a, claim_a = store.reserve_target_code(code, "discord", "759")
    assert status_a == "ok" and claim_a is not None

    status_b, claim_b = store.reserve_target_code(code, "discord", "OTHER_USER")
    assert status_b == "locked"
    assert claim_b is None

    # The original identity can still complete redemption afterwards.
    status_a2, claim_a2 = store.reserve_target_code(code, "discord", "759")
    assert status_a2 == "ok"
    assert store.consume_target_code(code) is not None


def test_release_lets_the_reserving_identity_retry_immediately(store):
    code = store.issue_target_code(target_user_id="usr_web").code
    store.reserve_target_code(code, "discord", "759")
    store.release_target_reservation(code, "discord", "759")
    # A different identity can now claim it (reservation was actually dropped).
    status, claim = store.reserve_target_code(code, "discord", "another")
    assert status == "ok"


# ── the redemption core (_redeem_target_link_code_core) ─────────────────────

def _make_ts(monkeypatch):
    """Import apps.turtle_server against a throwaway local users.sqlite so
    the module-level identity_manager singleton doesn't touch the real
    process-wide one across tests."""
    import apps.turtle_server as ts
    return ts


def test_full_round_trip_issue_web_then_redeem_from_channel(monkeypatch, tmp_path, pm_root):
    """Acceptance criterion 2: issue on web -> send from channel -> accounts
    linked, memory merged, code burned — as one test."""
    import apps.turtle_server as ts
    from core.identity import IdentityManager
    from core.memory_journal import JournalStore, make_event

    mgr = IdentityManager(db_path=tmp_path / "users.sqlite")
    asyncio.run(mgr.init_db())
    monkeypatch.setattr("core.identity.identity_manager", mgr)

    link_store = LinkCodeStore(tmp_path / "users.sqlite")
    monkeypatch.setattr(
        "core.storage.factory.get_link_code_store", lambda: link_store
    )

    async def run():
        # 1. Web account already exists (the "target").
        web_user = await mgr.resolve_user("web_email", "me@example.com")
        # 2. Web issues a code bound to that account.
        issued = link_store.issue_target_code(target_user_id=web_user)

        # 3. The channel identity (pre-existing, separate memory) sends it.
        channel_user = await mgr.resolve_user("discord", "759")
        assert channel_user != web_user  # the bug this WP fixes

        JournalStore(user_id=channel_user).append_many([
            make_event(
                event_id="e1", kind="fact", topic="identity", key="identity.city",
                value={"value": "Indore"}, confidence=1.0, source="explicit",
                extractor="deterministic", session_id="s", turn_id="t", applied=True,
            )
        ])

        result_text = await ts._redeem_target_link_code_core(
            channel="discord", channel_user_id="759",
            source_user_id=channel_user, code=issued.code,
        )
        assert "Linked" in result_text or "linked" in result_text.lower()

        # 4. discord/759 now resolves to the web account.
        assert await mgr.resolve_user("discord", "759") == web_user
        # 5. Memory merged.
        keys = {e.key for e in JournalStore(user_id=web_user).load_all()}
        assert "identity.city" in keys
        # 6. Code burned — single use.
        assert link_store.consume_target_code(issued.code) is None

    asyncio.run(run())


def test_redeem_is_single_use(monkeypatch, tmp_path, pm_root):
    import apps.turtle_server as ts
    from core.identity import IdentityManager

    mgr = IdentityManager(db_path=tmp_path / "users.sqlite")
    asyncio.run(mgr.init_db())
    monkeypatch.setattr("core.identity.identity_manager", mgr)

    link_store = LinkCodeStore(tmp_path / "users.sqlite")
    monkeypatch.setattr("core.storage.factory.get_link_code_store", lambda: link_store)

    async def run():
        web_user = await mgr.resolve_user("web_email", "me2@example.com")
        issued = link_store.issue_target_code(target_user_id=web_user)
        channel_user = await mgr.resolve_user("discord", "111")

        first = await ts._redeem_target_link_code_core(
            channel="discord", channel_user_id="111",
            source_user_id=channel_user, code=issued.code,
        )
        assert "invalid" not in first.lower()

        # Fresh channel identity trying the now-consumed code must fail.
        second = await ts._redeem_target_link_code_core(
            channel="discord", channel_user_id="222",
            source_user_id="usr_someone_else", code=issued.code,
        )
        assert "invalid" in second.lower() or "expired" in second.lower()

    asyncio.run(run())


def test_redeem_rejects_expired_code(monkeypatch, tmp_path, pm_root):
    import apps.turtle_server as ts
    from datetime import UTC, datetime, timedelta
    from core.identity import IdentityManager

    mgr = IdentityManager(db_path=tmp_path / "users.sqlite")
    asyncio.run(mgr.init_db())
    monkeypatch.setattr("core.identity.identity_manager", mgr)

    link_store = LinkCodeStore(tmp_path / "users.sqlite")
    monkeypatch.setattr("core.storage.factory.get_link_code_store", lambda: link_store)

    async def run():
        web_user = await mgr.resolve_user("web_email", "me3@example.com")
        issued = link_store.issue_target_code(target_user_id=web_user)
        later = datetime.now(UTC) + timedelta(minutes=LINK_CODE_TTL_MINUTES + 1)
        monkeypatch.setattr(al, "_utc_now", lambda: later)

        result = await ts._redeem_target_link_code_core(
            channel="discord", channel_user_id="333",
            source_user_id="usr_x", code=issued.code,
        )
        assert "invalid" in result.lower() or "expired" in result.lower()

    asyncio.run(run())


def test_redeem_by_a_different_channel_identity_than_reserved_is_rejected(monkeypatch, tmp_path, pm_root):
    """Pinned behaviour: once identity A has reserved the code, identity B
    sending the same code within the reservation window is refused, not
    silently redirected or allowed to race the merge."""
    import apps.turtle_server as ts
    from core.identity import IdentityManager

    mgr = IdentityManager(db_path=tmp_path / "users.sqlite")
    asyncio.run(mgr.init_db())
    monkeypatch.setattr("core.identity.identity_manager", mgr)

    link_store = LinkCodeStore(tmp_path / "users.sqlite")
    monkeypatch.setattr("core.storage.factory.get_link_code_store", lambda: link_store)

    async def run():
        web_user = await mgr.resolve_user("web_email", "me4@example.com")
        issued = link_store.issue_target_code(target_user_id=web_user)

        # Reserve (but don't finish) as identity A by reserving directly.
        status, _claim = link_store.reserve_target_code(issued.code, "discord", "AAA")
        assert status == "ok"

        # Identity B tries to redeem the same still-reserved code.
        result = await ts._redeem_target_link_code_core(
            channel="discord", channel_user_id="BBB",
            source_user_id="usr_b", code=issued.code,
        )
        assert "invalid" in result.lower()
        # A can still complete afterwards.
        result_a = await ts._redeem_target_link_code_core(
            channel="discord", channel_user_id="AAA",
            source_user_id="usr_a", code=issued.code,
        )
        assert "invalid" not in result_a.lower()

    asyncio.run(run())


def test_redeeming_own_already_linked_channel_is_a_noop(monkeypatch, tmp_path, pm_root):
    import apps.turtle_server as ts
    from core.identity import IdentityManager

    mgr = IdentityManager(db_path=tmp_path / "users.sqlite")
    asyncio.run(mgr.init_db())
    monkeypatch.setattr("core.identity.identity_manager", mgr)

    link_store = LinkCodeStore(tmp_path / "users.sqlite")
    monkeypatch.setattr("core.storage.factory.get_link_code_store", lambda: link_store)

    async def run():
        web_user = await mgr.resolve_user("web_email", "me5@example.com")
        issued = link_store.issue_target_code(target_user_id=web_user)
        result = await ts._redeem_target_link_code_core(
            channel="discord", channel_user_id="759",
            source_user_id=web_user, code=issued.code,
        )
        assert "already linked" in result.lower()

    asyncio.run(run())


# ── web issuance endpoint ────────────────────────────────────────────────────

def test_issue_endpoint_requires_authentication(monkeypatch):
    from fastapi.testclient import TestClient
    import apps.turtle_server as ts

    monkeypatch.setattr(ts.settings, "deploy_mode", "local", raising=False)
    monkeypatch.setattr(ts.settings, "dev_anon", False, raising=False)
    with TestClient(ts.app) as client:
        resp = client.post("/api/account/link/issue")
    assert resp.status_code == 401


def test_issue_endpoint_returns_a_code_for_an_authenticated_caller(monkeypatch):
    from fastapi.testclient import TestClient
    import apps.turtle_server as ts

    monkeypatch.setattr(ts.settings, "deploy_mode", "local", raising=False)
    monkeypatch.setattr(ts.settings, "dev_anon", True, raising=False)
    with TestClient(ts.app) as client:
        resp = client.post("/api/account/link/issue")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert len(body["code"]) >= 8
    assert body["expires_in_minutes"] == LINK_CODE_TTL_MINUTES


# ── reachability under TURTLE_CHANNEL_SIGNUP=invite ──────────────────────────

def test_redemption_is_unreachable_for_a_brand_new_identity_under_invite(monkeypatch):
    """Pins the crux finding: under invite-only sign-up, a channel identity
    with NO existing mapping never reaches _channel_dispatch_handler (hence
    never reaches the redeem_link_code tool) at all — the 8 adapters'
    resolve_channel_user() gate refuses first. This is the exact function
    every adapter calls before dispatch; see the WP report for the adapter
    hook this WP does NOT add (apps/channels/* is out of scope)."""
    import core.identity as identity_mod
    from types import SimpleNamespace

    class _FakeManager:
        async def lookup_user(self, channel, channel_user_id):
            return None  # unknown identity — a genuine never-seen-before miss

        async def resolve_user(self, channel, channel_user_id):
            raise AssertionError("invite-only must never mint on a miss")

    monkeypatch.setattr(identity_mod, "identity_manager", _FakeManager(), raising=False)
    monkeypatch.setattr(identity_mod.settings, "channel_signup", "invite", raising=False)

    async def run():
        uid = await identity_mod.resolve_channel_user("discord", "brand_new_unknown_id")
        assert uid is None  # -> adapter replies CHANNEL_INVITE_ONLY_MESSAGE, dispatch never runs

    asyncio.run(run())
