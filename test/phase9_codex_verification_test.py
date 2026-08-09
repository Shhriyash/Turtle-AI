"""
Phase 9 — regression tests for the SECOND Codex adversarial-review pass.

Codex verified: Twilio stayed unmounted, unsigned-channel gating held, unapplied
names were excluded. Codex REJECTED: three new findings, all real —

  [high] Fallback JWT secrets. verify_session_cookie/verify_token used
    "dev-fallback-secret" (or the 32b variant) when AUTH_SECRET_KEY was unset,
    so a public deployment that forgot the env var accepted forged tokens for
    any user_id on every memory endpoint, WS, and /api/account/link.
  [medium] Two-target race in /api/account/link (see phase9_link_race_test).
  [medium] Detached post-turn writers escape the source lock (see the same).

This file pins the fix for the FIRST finding.
"""
from __future__ import annotations

import jwt
import pytest

from core.auth_secret import auth_secret, is_using_fallback_secret


# ── auth secret can never be a repo-known literal ────────────────────────────

_KNOWN_LEAKED_LITERALS = (
    "dev-fallback-secret",
    "dev-fallback-secret-do-not-use-in-production-32b",
)


def _no_configured_secret(monkeypatch):
    """Force auth_secret() to hit the unset branch, cleanly."""
    import core.auth_secret as m
    from core.config import settings

    monkeypatch.setattr(settings, "auth_secret_key", None, raising=False)
    monkeypatch.setattr(settings, "deploy_mode", "local", raising=False)
    m._reset_for_tests()


def test_unset_secret_yields_a_random_secret_not_a_literal(monkeypatch):
    _no_configured_secret(monkeypatch)
    got = auth_secret()
    assert got not in _KNOWN_LEAKED_LITERALS
    assert len(got) >= 40, "dev secret must be long enough to resist offline attack"
    assert is_using_fallback_secret() is True


def test_dev_secret_is_stable_within_process(monkeypatch):
    _no_configured_secret(monkeypatch)
    a = auth_secret()
    b = auth_secret()
    assert a == b, "auth_secret must be stable so cookies verify across requests"


def test_configured_secret_wins(monkeypatch):
    import core.auth_secret as m
    from core.config import settings
    from pydantic import SecretStr

    monkeypatch.setattr(settings, "auth_secret_key", SecretStr("explicit-key"), raising=False)
    m._reset_for_tests()
    assert auth_secret() == "explicit-key"
    assert is_using_fallback_secret() is False


def test_cloud_without_secret_hard_fails(monkeypatch):
    import core.auth_secret as m
    from core.config import settings

    monkeypatch.setattr(settings, "auth_secret_key", None, raising=False)
    monkeypatch.setattr(settings, "deploy_mode", "cloud", raising=False)
    m._reset_for_tests()

    with pytest.raises(RuntimeError, match="AUTH_SECRET_KEY is required"):
        auth_secret()


# ── negative tests: tokens forged with the OLD literal must fail everywhere ──

def _forge(literal: str, user_id: str = "usr_victim") -> str:
    return jwt.encode({"sub": user_id, "channel": "web"}, literal, algorithm="HS256")


def test_verify_token_rejects_forged_literal(monkeypatch):
    """apps/auth.verify_token USED TO fall back to 'dev-fallback-secret' — even
    in cloud. A cloud deploy without AUTH_SECRET_KEY set therefore accepted a
    token signed with the literal for any user_id."""
    _no_configured_secret(monkeypatch)
    from apps.auth import verify_token

    for literal in _KNOWN_LEAKED_LITERALS:
        with pytest.raises(ValueError):
            verify_token(_forge(literal))


def test_verify_session_cookie_rejects_forged_literal(monkeypatch):
    """Same hole in onboarding_routes — same fix, verified separately."""
    _no_configured_secret(monkeypatch)
    from apps.onboarding_routes import verify_session_cookie

    for literal in _KNOWN_LEAKED_LITERALS:
        assert verify_session_cookie(_forge(literal)) is None


def test_memory_endpoint_rejects_forged_bearer(monkeypatch):
    """End-to-end: a forged Bearer with the old literal must NOT get 200 on the
    memory profile — the auth path is the actual attack surface."""
    from fastapi.testclient import TestClient
    import apps.turtle_server as ts

    _no_configured_secret(monkeypatch)
    monkeypatch.setattr(ts.settings, "dev_anon", False, raising=False)

    forged = _forge("dev-fallback-secret", user_id="usr_someone_else")
    with TestClient(ts.app) as client:
        r = client.get("/api/memory/profile", headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401, (
        f"memory endpoint accepted a forged token! got {r.status_code}: {r.text[:200]}"
    )


def test_own_token_still_works(monkeypatch):
    """Sanity: a token signed by the actual (dev-random) secret still verifies —
    the fix must not break the normal path."""
    _no_configured_secret(monkeypatch)
    from apps.auth import create_session_token, verify_token

    tok = create_session_token("usr_me", channel="web")
    assert verify_token(tok).get("sub") == "usr_me"
