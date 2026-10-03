"""
test/phase6_admin_test.py
-------------------------
Phase 6 W2: admin dashboard page + POST /api/config auth gate.

Proves:
  (a) GET /admin serves the (unauthenticated) dashboard shell as HTML.
  (b) POST /api/config is gated by X-Admin-Token when TURTLE_ADMIN_TOKEN is set —
      401 without / with the wrong header, 200 with the correct header.
  (c) With the admin token unset AND TURTLE_DEV_ANON=1 (explicit unsafe-dev
      opt-in), POST /api/config keeps its open dev behavior locally. Without
      DEV_ANON it fails closed — a tunneled local deploy is otherwise reachable
      and hot-swapping the agent chain from an anon POST would let anyone
      downgrade every user mid-conversation. Same explicit-opt-in rule as the
      channel webhook verifiers.
  (e) POST /api/config fails closed off-cloud WITHOUT DEV_ANON.
  (d) GET /api/config stays open in LOCAL mode regardless (the dev panel
      reads it to render; web/js/devmode.js is the only reader).
  (f) GET /api/config is gated behind X-Admin-Token in CLOUD mode, mirroring
      POST — 401 without the header, 200 with the correct one.
  (g) WP2.C (ledger 2.7 / D7): POST /api/config is unconditionally immutable
      in cloud — 405 even with a VALID admin token, and _save_config is
      never called. The pre-WP2.C code let a correct token through to a
      filesystem write that either crashes (read-only) or silently vanishes
      at the next cold start while agents_mgr.rebuild still mutated the live
      process. The 405 body is JSON carrying the message so
      web/js/devmode.js's existing `result.error` else-branch renders it.

Offline: no network, no live keys. Follows smoke_boot_test.py's guarded-import
pattern so a genuinely missing optional dep skips rather than erroring collection.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from pydantic import SecretStr
    from apps import turtle_server

    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover - optional deps missing in env
    _IMPORT_ERROR = _e


@unittest.skipIf(_IMPORT_ERROR is not None, f"app import failed: {_IMPORT_ERROR!r}")
class AdminDashboardAndConfigGate(unittest.TestCase):
    """The admin page serves, and the config POST honors the admin-token gate."""

    def setUp(self) -> None:
        # Instantiate the client directly (not as a context manager) so no
        # lifespan/startup hooks fire — same approach as smoke_boot_test.
        self.client = TestClient(turtle_server.app)
        # Neutralize the heavy / side-effecting tail of the POST handler so these
        # tests assert only the AUTH behavior, not config persistence or the
        # agent rebuild (which would touch disk / model wiring).
        p_save = patch.object(turtle_server, "_save_config", lambda cfg: None)
        p_rebuild = patch.object(turtle_server.agents_mgr, "rebuild", lambda cfg: None)
        p_save.start()
        p_rebuild.start()
        self.addCleanup(p_save.stop)
        self.addCleanup(p_rebuild.stop)

    # (a) -------------------------------------------------------------------
    def test_get_admin_returns_html(self) -> None:
        resp = self.client.get("/admin")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers.get("content-type", ""))
        self.assertIn("Turtle Admin", resp.text)

    # (b) -------------------------------------------------------------------
    def test_post_config_requires_token_when_set(self) -> None:
        with patch.object(turtle_server.settings, "admin_token", SecretStr("s3cret")):
            # Missing header -> 401
            r_missing = self.client.post("/api/config", json={"temperature": 0.3})
            self.assertEqual(r_missing.status_code, 401)

            # Wrong header -> 401
            r_wrong = self.client.post(
                "/api/config",
                json={"temperature": 0.3},
                headers={"X-Admin-Token": "nope"},
            )
            self.assertEqual(r_wrong.status_code, 401)

            # Correct header -> 200
            r_ok = self.client.post(
                "/api/config",
                json={"temperature": 0.3},
                headers={"X-Admin-Token": "s3cret"},
            )
            self.assertEqual(r_ok.status_code, 200)
            self.assertEqual(r_ok.json().get("status"), "ok")

    # (c) -------------------------------------------------------------------
    def test_post_config_open_when_token_unset_with_dev_anon(self) -> None:
        """No admin token + DEV_ANON=1 + local: open (dev flow preserved)."""
        with patch.object(turtle_server.settings, "admin_token", None), \
             patch.object(turtle_server.settings, "dev_anon", True), \
             patch.object(turtle_server.settings, "deploy_mode", "local"):
            r = self.client.post("/api/config", json={"temperature": 0.3})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json().get("status"), "ok")

    # (e) -------------------------------------------------------------------
    def test_post_config_fails_closed_locally_without_dev_anon(self) -> None:
        """No admin token + no DEV_ANON + local: 503. The pre-hardening code
        stayed OPEN in this case — a publicly tunneled local deploy could
        hot-swap the model for every user from an anonymous POST. Codex
        adversarial review flagged this class as a real blocker."""
        with patch.object(turtle_server.settings, "admin_token", None), \
             patch.object(turtle_server.settings, "dev_anon", False), \
             patch.object(turtle_server.settings, "deploy_mode", "local"):
            r = self.client.post("/api/config", json={"temperature": 0.3})
            self.assertEqual(r.status_code, 503)
            self.assertIn("admin", r.json().get("error", "").lower())

    def test_post_config_fails_closed_in_cloud_without_token(self) -> None:
        """Cloud always fails closed, regardless of DEV_ANON.

        WP2.C (ledger 2.7): cloud's config-immutability 405 is unconditional
        and checked before the admin-token gate, so this now short-circuits
        to 405 rather than the pre-WP2.C 503 ("no admin token configured") --
        immutability is a stronger and more accurate statement than "the
        write mechanism is merely disabled". See (g) below for the
        valid-token case this supersedes.
        """
        with patch.object(turtle_server.settings, "admin_token", None), \
             patch.object(turtle_server.settings, "dev_anon", True), \
             patch.object(turtle_server.settings, "deploy_mode", "cloud"):
            r = self.client.post("/api/config", json={"temperature": 0.3})
            self.assertEqual(r.status_code, 405)

    # (d) -------------------------------------------------------------------
    def test_get_config_open_in_local_regardless(self) -> None:
        with patch.object(turtle_server.settings, "admin_token", SecretStr("s3cret")), \
             patch.object(turtle_server.settings, "deploy_mode", "local"):
            r = self.client.get("/api/config")
            self.assertEqual(r.status_code, 200)
            self.assertIsInstance(r.json(), dict)

    # (f) -------------------------------------------------------------------
    def test_get_config_gated_in_cloud(self) -> None:
        with patch.object(turtle_server.settings, "admin_token", SecretStr("s3cret")), \
             patch.object(turtle_server.settings, "deploy_mode", "cloud"):
            r_missing = self.client.get("/api/config")
            self.assertEqual(r_missing.status_code, 401)

            r_wrong = self.client.get("/api/config", headers={"X-Admin-Token": "nope"})
            self.assertEqual(r_wrong.status_code, 401)

            r_ok = self.client.get("/api/config", headers={"X-Admin-Token": "s3cret"})
            self.assertEqual(r_ok.status_code, 200)
            self.assertIsInstance(r_ok.json(), dict)

    def test_get_config_fails_closed_in_cloud_without_token_configured(self) -> None:
        with patch.object(turtle_server.settings, "admin_token", None), \
             patch.object(turtle_server.settings, "deploy_mode", "cloud"):
            r = self.client.get("/api/config")
            self.assertEqual(r.status_code, 401)

    # (g) -------------------------------------------------------------------
    def test_post_config_immutable_in_cloud_even_with_valid_token(self) -> None:
        """A CORRECT admin token must NOT unlock a config write in cloud --
        the token proves identity, not that a filesystem write will persist.
        setUp already patches _save_config to a no-op; this test additionally
        asserts it was never even called."""
        with patch.object(turtle_server, "_save_config") as mock_save, \
             patch.object(turtle_server.settings, "admin_token", SecretStr("s3cret")), \
             patch.object(turtle_server.settings, "deploy_mode", "cloud"):
            r = self.client.post(
                "/api/config",
                json={"temperature": 0.3},
                headers={"X-Admin-Token": "s3cret"},
            )
            self.assertEqual(r.status_code, 405)
            mock_save.assert_not_called()

    def test_post_config_immutable_in_cloud_body_is_json_with_error(self) -> None:
        """web/js/devmode.js's applyDevConfig does `await res.json()`
        unconditionally and reads `result.error` on any non-'ok' status --
        a bare 405 with no body would throw inside res.json() there and show
        a generic toast instead of this message."""
        with patch.object(turtle_server.settings, "admin_token", SecretStr("s3cret")), \
             patch.object(turtle_server.settings, "deploy_mode", "cloud"):
            r = self.client.post(
                "/api/config",
                json={"temperature": 0.3},
                headers={"X-Admin-Token": "s3cret"},
            )
            self.assertEqual(r.status_code, 405)
            self.assertEqual(r.headers.get("content-type", "").split(";")[0], "application/json")
            body = r.json()
            self.assertIn("error", body)
            self.assertIn("immutable", body["error"].lower())

    def test_post_config_still_works_locally(self) -> None:
        """Local mode is unaffected by the cloud immutability gate."""
        with patch.object(turtle_server.settings, "admin_token", SecretStr("s3cret")), \
             patch.object(turtle_server.settings, "deploy_mode", "local"):
            r = self.client.post(
                "/api/config",
                json={"temperature": 0.3},
                headers={"X-Admin-Token": "s3cret"},
            )
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json().get("status"), "ok")


if __name__ == "__main__":
    unittest.main()
