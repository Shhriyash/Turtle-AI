"""
WP1.J — two-sided link-code binding (ledger 1b.6, unblocking the withdrawn
1a.4 part 3).

Before this WP, `/api/account/link` had exactly one gate: an authenticated
web session. ANY authenticated account — not just the one the channel user
intended — could redeem a leaked/intercepted claim code and walk away with
the channel identity's memory merged into it (see core/account_linking.py's
module docstring and the "NEVER emit a claim code into a shared channel"
comment on the `link_account` tool for the pre-existing acknowledgement of
this gap).

Two-sided binding closes it: `link_account` now takes `expected_email` — what
the channel-side user says their web account is — and redemption refuses any
authenticated session whose OWN account email doesn't match. The email is
never trusted as authorization by itself (a self-claimed identifier still
proves nothing); it only narrows who the existing two proofs (channel
control + authenticated session) are allowed to belong to.

These tests pin:
  * a bound code redeemed by the matching account links
  * the SAME code redeemed by a different authenticated account is refused,
    and the mapping / memory are untouched
  * an old row with no expected_email (pre-migration / unbound issue) behaves
    as before — redeemable by any authenticated account
  * unauthenticated redemption is still refused
  * concurrent redemption by two different targets still yields exactly one
    winner (binding must not turn `reserve()` into a check-then-act race)
  * the binding-mismatch refusal is byte-for-byte identical to the
    invalid/locked-code refusal (no oracle for probing a code's binding)
"""
from __future__ import annotations

import asyncio

import pytest


def _no_configured_secret(monkeypatch):
    """Force auth_secret() to hit its dev-random-but-stable branch, exactly
    as phase9_codex_verification_test.py does, so create_session_token /
    verify_token round-trip inside this test process."""
    import core.auth_secret as m
    from core.config import settings

    monkeypatch.setattr(settings, "auth_secret_key", None, raising=False)
    monkeypatch.setattr(settings, "deploy_mode", "local", raising=False)
    m._reset_for_tests()


@pytest.fixture()
def linked_env(tmp_path, monkeypatch):
    """A local IdentityManager + LinkCodeStore sharing one users.sqlite, wired
    into both core.identity.identity_manager (what the endpoint's lazy
    imports resolve) and core.storage.factory.get_link_code_store (via the
    same identity_manager.db_path)."""
    import apps.turtle_server as ts
    import core.identity as identity_mod
    from core.account_linking import LinkCodeStore
    from core.identity import IdentityManager

    db_path = tmp_path / "users.sqlite"
    mgr = IdentityManager(db_path=db_path)
    monkeypatch.setattr(identity_mod, "identity_manager", mgr, raising=False)
    monkeypatch.setattr(ts.settings, "deploy_mode", "local", raising=False)
    monkeypatch.setattr(ts.settings, "dev_anon", False, raising=False)
    _no_configured_secret(monkeypatch)

    asyncio.run(mgr.init_db())
    store = LinkCodeStore(db_path)
    return mgr, store


@pytest.fixture()
def pm_root(tmp_path, monkeypatch):
    """merge_memory() reads/writes under PERSONAL_MEMORY_DIR — redirect it so
    these tests never touch real user data (mirrors phase9_account_linking_test.py)."""
    import core.paths as core_paths

    root = tmp_path / "personal"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root, raising=False)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snap", raising=False)
    return root


def _bearer(user_id: str) -> dict:
    from apps.auth import create_session_token

    return {"Authorization": f"Bearer {create_session_token(user_id)}"}


# ── matching redeemer links ──────────────────────────────────────────────────

def test_bound_code_redeemed_by_matching_account_links(linked_env, pm_root):
    from fastapi.testclient import TestClient
    import apps.turtle_server as ts

    mgr, store = linked_env

    async def setup():
        source = await mgr.resolve_user("discord", "759")
        target = await mgr.resolve_user("web_email", "match@example.com")
        return source, target

    source, target = asyncio.run(setup())
    code = store.issue(
        channel="discord", channel_user_id="759", source_user_id=source,
        expected_email="match@example.com",
    ).code

    with TestClient(ts.app) as client:
        resp = client.post(
            "/api/account/link", json={"code": code}, headers=_bearer(target)
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["linked_to"] == target

    assert asyncio.run(mgr.resolve_user("discord", "759")) == target


# ── wrong redeemer is refused, mapping + memory untouched ───────────────────

def test_bound_code_redeemed_by_wrong_account_is_refused(linked_env, pm_root):
    from fastapi.testclient import TestClient
    import apps.turtle_server as ts
    from core.memory_journal import JournalStore, make_event

    mgr, store = linked_env

    async def setup():
        source = await mgr.resolve_user("discord", "759")
        target = await mgr.resolve_user("web_email", "owner@example.com")
        attacker = await mgr.resolve_user("web_email", "attacker@example.com")
        return source, target, attacker

    source, target, attacker = asyncio.run(setup())

    # The source has real memory that must NOT end up in the attacker's account.
    JournalStore(user_id=source).append_many([
        make_event(
            event_id="e1", kind="fact", topic="identity", key="identity.city",
            value={"value": "Indore"}, confidence=1.0, source="explicit",
            extractor="deterministic", session_id="s", turn_id="t", applied=True,
        )
    ])

    code = store.issue(
        channel="discord", channel_user_id="759", source_user_id=source,
        expected_email="owner@example.com",
    ).code

    with TestClient(ts.app) as client:
        resp = client.post(
            "/api/account/link", json={"code": code}, headers=_bearer(attacker)
        )
    assert resp.status_code == 400, resp.text
    assert resp.json() == {"error": "That code is invalid or has expired"}

    # Mapping unchanged — still the original source, not re-pointed at the attacker.
    assert asyncio.run(mgr.resolve_user("discord", "759")) == source
    # Memory did NOT get copied into the attacker's account.
    attacker_keys = {e.key for e in JournalStore(user_id=attacker).load_all()}
    assert "identity.city" not in attacker_keys

    # The legitimate target can still redeem afterwards — the wrong-target
    # reservation was released, not left to squat out the TTL.
    with TestClient(ts.app) as client:
        resp2 = client.post(
            "/api/account/link", json={"code": code}, headers=_bearer(target)
        )
    assert resp2.status_code == 200, resp2.text
    assert asyncio.run(mgr.resolve_user("discord", "759")) == target


# ── pre-binding rows behave as before ────────────────────────────────────────

def test_unbound_code_is_redeemable_by_any_authenticated_account(linked_env, pm_root):
    """A code with no expected_email (issued before this WP, or issued without
    one) is NOT refused on binding grounds — pinning the deliberate decision
    that pre-existing production codes keep working until they expire."""
    from fastapi.testclient import TestClient
    import apps.turtle_server as ts

    mgr, store = linked_env

    async def setup():
        source = await mgr.resolve_user("discord", "759")
        target = await mgr.resolve_user("web_email", "whoever@example.com")
        return source, target

    source, target = asyncio.run(setup())
    issued = store.issue(channel="discord", channel_user_id="759", source_user_id=source)
    assert issued.expected_email is None

    with TestClient(ts.app) as client:
        resp = client.post(
            "/api/account/link", json={"code": issued.code}, headers=_bearer(target)
        )
    assert resp.status_code == 200, resp.text
    assert asyncio.run(mgr.resolve_user("discord", "759")) == target


# ── authentication is still required ─────────────────────────────────────────

def test_unauthenticated_redemption_still_refused(linked_env, pm_root):
    from fastapi.testclient import TestClient
    import apps.turtle_server as ts

    mgr, store = linked_env
    source = asyncio.run(mgr.resolve_user("discord", "759"))
    code = store.issue(
        channel="discord", channel_user_id="759", source_user_id=source,
        expected_email="someone@example.com",
    ).code

    with TestClient(ts.app) as client:
        resp = client.post("/api/account/link", json={"code": code})
    assert resp.status_code == 401


# ── binding mismatch is indistinguishable from an invalid code ──────────────

def test_binding_mismatch_message_matches_invalid_code_message(linked_env, pm_root):
    from fastapi.testclient import TestClient
    import apps.turtle_server as ts

    mgr, store = linked_env

    async def setup():
        source = await mgr.resolve_user("discord", "759")
        wrong = await mgr.resolve_user("web_email", "not-the-owner@example.com")
        return source, wrong

    source, wrong = asyncio.run(setup())
    code = store.issue(
        channel="discord", channel_user_id="759", source_user_id=source,
        expected_email="owner@example.com",
    ).code

    with TestClient(ts.app) as client:
        mismatch_resp = client.post(
            "/api/account/link", json={"code": code}, headers=_bearer(wrong)
        )
        invalid_resp = client.post(
            "/api/account/link", json={"code": "BOGUSCODE"}, headers=_bearer(wrong)
        )
    assert mismatch_resp.status_code == invalid_resp.status_code == 400
    assert mismatch_resp.json() == invalid_resp.json(), (
        "a binding mismatch must not be distinguishable from an unknown code — "
        "otherwise this endpoint becomes an oracle for who a code is bound to"
    )


# ── binding does not turn reserve() into a check-then-act race ──────────────

def test_reserve_stays_atomic_with_binding_present(tmp_path):
    """Two different targets racing the SAME bound code must still get
    exactly one 'ok' — binding is checked by the CALLER using the claim
    reserve() already returns, not by adding a second read before the write."""
    import core.account_linking as al

    store = al.LinkCodeStore(tmp_path / "users.sqlite")
    code = store.issue(
        channel="discord", channel_user_id="759", source_user_id="usr_src",
        expected_email="owner@example.com",
    ).code

    status_a, claim_a = store.reserve(code, "usr_target_a")
    status_b, claim_b = store.reserve(code, "usr_target_b")

    assert status_a == "ok" and claim_a is not None
    assert claim_a.expected_email == "owner@example.com"
    assert status_b == "locked" and claim_b is None
