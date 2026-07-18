"""
apps/auth.py JWT session-token tests.

Migrated from test_tier2_verification.py (TestG4Auth). apps.auth is safe to
import (it is not part of the server rewrite). All behavior: module import,
create/verify round-trip, expired-token rejection, and the dev fallback that
lets local mode work without AUTH_SECRET_KEY set.
"""
from __future__ import annotations


class TestG4Auth:
    """JWT creation/verification; dev fallback when no secret configured."""

    def test_auth_module_importable(self):
        from apps.auth import create_session_token, verify_token, authenticate_websocket
        assert callable(create_session_token)
        assert callable(verify_token)
        assert callable(authenticate_websocket)

    def test_create_and_verify_token(self):
        import unittest.mock as mock
        from apps.auth import create_session_token, verify_token
        from pydantic import SecretStr

        with mock.patch("apps.auth.settings") as s:
            s.auth_secret_key = SecretStr("test-secret-abc")
            token = create_session_token("usr_abc123", "web")
            assert isinstance(token, str) and len(token) > 20

            payload = verify_token(token)
            assert payload["sub"] == "usr_abc123"
            assert payload["channel"] == "web"

    def test_expired_token_raises(self):
        import pytest, unittest.mock as mock
        from apps.auth import verify_token
        from pydantic import SecretStr

        with mock.patch("apps.auth.settings") as s:
            s.auth_secret_key = SecretStr("test-secret-abc")
            # Create a token backdated by more than 24h
            import jwt as _jwt
            from datetime import datetime, UTC, timedelta
            payload = {"sub": "usr_old", "channel": "web",
                       "exp": datetime.now(UTC) - timedelta(hours=25)}
            token = _jwt.encode(payload, "test-secret-abc", algorithm="HS256")

            with pytest.raises(ValueError, match="[Ee]xpired|[Ii]nvalid"):
                verify_token(token)

    def test_dev_fallback_no_crash_without_secret(self):
        """Local mode auth should work even without AUTH_SECRET_KEY set."""
        import unittest.mock as mock
        from apps.auth import create_session_token, verify_token

        with mock.patch("apps.auth.settings") as s:
            s.auth_secret_key = None
            token = create_session_token("usr_dev")
            payload = verify_token(token)
            assert payload["sub"] == "usr_dev"
