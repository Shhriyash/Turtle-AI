"""
Phase 4 — magic-link onboarding round-trip.

The HTTP layer is mostly glue; the load-bearing pieces are:
  * session cookies round-trip (mint → verify)
  * /onboarding/claim seeds identity.md and sets the cookie
  * /onboarding/start refuses traffic when EmailTool is unconfigured
  * the IP rate limiter blocks abusive callers
"""
from __future__ import annotations

import shutil
import unittest
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

try:
    import jwt
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from apps import onboarding_routes
    from apps.onboarding_routes import (
        issue_session_cookie_value,
        router as onboarding_router,
        verify_session_cookie,
    )
    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover — missing optional deps in test env
    _IMPORT_ERROR = _e

from core.config import settings


class StubEmailTool:
    """In-memory replacement for the Gmail SMTP tool used in tests."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send_email(self, *, receiver: str, subject: str, body: str, content_type: str = "plain") -> str:
        self.sent.append({
            "receiver": receiver, "subject": subject, "body": body, "content_type": content_type,
        })
        return "Email sent successfully"


@unittest.skipIf(_IMPORT_ERROR is not None, f"onboarding deps missing: {_IMPORT_ERROR}")
class SessionCookieRoundTripTests(unittest.TestCase):
    def test_issue_then_verify_returns_user_id(self) -> None:
        token = issue_session_cookie_value("usr_alice")
        self.assertEqual(verify_session_cookie(token), "usr_alice")

    def test_verify_rejects_garbage(self) -> None:
        self.assertIsNone(verify_session_cookie(""))
        self.assertIsNone(verify_session_cookie("not-a-jwt"))

    def test_verify_rejects_expired_token(self) -> None:
        secret = (
            settings.auth_secret_key.get_secret_value()
            if settings.auth_secret_key is not None
            else "dev-fallback-secret"
        )
        expired = jwt.encode(
            {
                "sub": "usr_alice",
                "exp": datetime.now(UTC) - timedelta(minutes=1),
            },
            secret,
            algorithm="HS256",
        )
        self.assertIsNone(verify_session_cookie(expired))


@unittest.skipIf(_IMPORT_ERROR is not None, f"onboarding deps missing: {_IMPORT_ERROR}")
class OnboardingEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path("test") / "_tmp" / f"onboarding_{uuid.uuid4().hex}"
        self.tmp.mkdir(parents=True, exist_ok=True)
        # Steer identity SQLite + memory writes into the temp dir so we don't
        # mutate the real data/ tree.
        self._orig_db_path = onboarding_routes.identity_manager.db_path
        onboarding_routes.identity_manager.db_path = self.tmp / "users.sqlite"
        # The /onboarding/claim handler instantiates PersonalMemoryStore with
        # the live PERSONAL_MEMORY_DIR — we redirect that too.
        import core.paths as paths_mod
        self._orig_personal_dir = paths_mod.PERSONAL_MEMORY_DIR
        paths_mod.PERSONAL_MEMORY_DIR = self.tmp / "personal"
        paths_mod.PERSONAL_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        # Clear the per-IP rate-limit history between tests.
        onboarding_routes._recent_starts.clear()

        self.email_tool = StubEmailTool()
        self._patcher = patch(
            "tools.email_tools.config.create_email_tool_from_env",
            return_value=self.email_tool,
        )
        self._patcher.start()

        app = FastAPI()
        app.include_router(onboarding_router)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self._patcher.stop()
        onboarding_routes.identity_manager.db_path = self._orig_db_path
        import core.paths as paths_mod
        paths_mod.PERSONAL_MEMORY_DIR = self._orig_personal_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_start_sends_email_and_returns_sent(self) -> None:
        resp = self.client.post(
            "/onboarding/start",
            json={"email": "alice@example.com", "name": "Alice", "timezone": "Europe/Berlin"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json(), {"status": "sent"})
        self.assertEqual(len(self.email_tool.sent), 1)
        sent = self.email_tool.sent[0]
        self.assertEqual(sent["receiver"], "alice@example.com")
        self.assertIn("/onboarding/claim?token=", sent["body"])

    def test_start_503_when_email_unconfigured(self) -> None:
        # Re-patch with None to simulate missing SMTP config.
        self._patcher.stop()
        with patch(
            "tools.email_tools.config.create_email_tool_from_env",
            return_value=None,
        ):
            resp = self.client.post(
                "/onboarding/start",
                json={"email": "alice@example.com", "name": "Alice", "timezone": "UTC"},
            )
        self.assertEqual(resp.status_code, 503)
        # Restart the outer patcher so tearDown stops cleanly.
        self._patcher.start()

    def test_rate_limit_blocks_after_threshold(self) -> None:
        original = settings.onboarding_rate_limit_per_hour
        settings.onboarding_rate_limit_per_hour = 2
        try:
            for _ in range(2):
                resp = self.client.post(
                    "/onboarding/start",
                    json={"email": "alice@example.com", "name": "Alice", "timezone": "UTC"},
                )
                self.assertEqual(resp.status_code, 200)
            resp = self.client.post(
                "/onboarding/start",
                json={"email": "alice@example.com", "name": "Alice", "timezone": "UTC"},
            )
            self.assertEqual(resp.status_code, 429)
        finally:
            settings.onboarding_rate_limit_per_hour = original

    def test_claim_seeds_identity_md_and_sets_cookie(self) -> None:
        # Walk the full flow: POST /start → grab the link → GET /claim.
        resp = self.client.post(
            "/onboarding/start",
            json={"email": "alice@example.com", "name": "Alice", "timezone": "Europe/Berlin"},
        )
        self.assertEqual(resp.status_code, 200)
        token = self.email_tool.sent[0]["body"].split("token=", 1)[1].split('"', 1)[0]

        # /claim intentionally returns a 200 HTML page with JS/meta-refresh
        # navigation instead of a 303 — email-client link rewriters and
        # prefetchers don't reliably follow Location headers.
        resp = self.client.get(f"/onboarding/claim?token={token}", follow_redirects=False)
        self.assertEqual(resp.status_code, 200)
        cookie = resp.cookies.get("turtle_uid")
        self.assertIsNotNone(cookie)

        # The cookie verifies and carries a real user_id.
        user_id = verify_session_cookie(cookie)
        self.assertIsNotNone(user_id)
        self.assertTrue(user_id.startswith("usr_"))

        # identity.md was seeded with the form facts.
        identity_path = (
            onboarding_routes.PersonalMemoryStore.__module__  # silence import-unused warning
            and (self.tmp / "personal" / user_id / "identity.md")
        )
        self.assertTrue(identity_path.exists())
        body = identity_path.read_text(encoding="utf-8")
        self.assertIn("Alice", body)
        self.assertIn("alice@example.com", body)
        self.assertIn("Europe/Berlin", body)

    def test_claim_rejects_wrong_kind(self) -> None:
        # A session cookie JWT is not a valid claim JWT (kind mismatch).
        bad = issue_session_cookie_value("usr_alice")
        resp = self.client.get(f"/onboarding/claim?token={bad}", follow_redirects=False)
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
