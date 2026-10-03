"""
test/smoke_boot_test.py
-----------------------
Phase 5 W3: deploy boot smoke. Proves the ASGI app imports and serves its most
basic contracts without any live API, so a broken boot fails the normal test
suite (and by extension CI) instead of only surfacing at deploy time. This is
the in-suite equivalent of the Dockerfile HEALTHCHECK — no separate CI job.

Fully offline: no network, no auth, no writes. The app module boots under CI's
dummy keys (see .github/workflows/tests.yml) and locally imports clean too.

MODE-AWARE (not mode-agnostic). This file used to assume local mode outright
and so could only ever be run in local mode: /api/config is gated behind
X-Admin-Token in cloud (ledger 1a.2) and POST is 405 there (ledger 2.7), so a
cloud-mode run failed on an unauthenticated 200 assertion. The one test meant
to catch a broken boot therefore could not run in the configuration that
actually ships to production — a control wired at one end only.

The fix asserts the per-mode contract rather than loosening to "401 or 200":
each branch below still pins an exact status code, so a genuinely broken
/api/config (500, 404, a crashed handler) fails in BOTH modes. The cloud
branch additionally drives the authenticated path, because a bare 401 proves
the gate fires but not that the handler can still load and serve config.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

# Import the app exactly as test/phase2_gate_ui_test.py does: guard the import
# so a genuinely missing optional dep degrades to a skip rather than a collection
# error, but under CI/local (deps present) the smoke actually runs.
try:
    from fastapi.testclient import TestClient
    from pydantic import SecretStr
    from apps import turtle_server

    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover - optional deps missing in env
    _IMPORT_ERROR = _e

# Read the ambient deploy mode once, at import, so the skip decorators below
# resolve against the mode this run was actually launched in (TURTLE_DEPLOY).
_IS_CLOUD = bool(_IMPORT_ERROR is None and turtle_server.settings.is_cloud)


@unittest.skipIf(_IMPORT_ERROR is not None, f"app import failed: {_IMPORT_ERROR!r}")
class BootSmoke(unittest.TestCase):
    """The app boots and answers its liveness + config + index contracts."""

    def setUp(self) -> None:
        # Instantiate the client directly (not as a context manager) so no
        # lifespan/startup hooks fire — same approach as phase2_gate_ui_test.
        self.client = TestClient(turtle_server.app)

    def test_healthz_returns_ok(self) -> None:
        # WP0.A: /healthz also reports the build SHA (read from TURTLE_BUILD_SHA)
        # so a deploy can prove which commit it is serving. Assert the contract
        # CI relies on directly — status is "ok" and the "sha" key is always
        # present (it may be null when TURTLE_BUILD_SHA is unset) — rather than
        # an exact-dict match that would break the moment a new key is added.
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body.get("status"), "ok")
        self.assertIn("sha", body)

    @unittest.skipIf(_IS_CLOUD, "local-mode contract; cloud gates GET /api/config")
    def test_api_config_returns_dict_local(self) -> None:
        """Local: GET /api/config is open — the dev panel reads it to render."""
        resp = self.client.get("/api/config")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), dict)

    @unittest.skipIf(not _IS_CLOUD, "cloud-mode contract; local GET is open")
    def test_api_config_gated_but_serving_cloud(self) -> None:
        """Cloud: the gate fires AND the handler behind it still works.

        Two assertions, because either alone is a half-check. Unauthenticated
        must be exactly 401 — that is the ledger 1a.2 gate, and asserting it
        here keeps a boot smoke from silently passing on, say, a 500 from a
        handler that blew up before reaching the token check. Then the
        authenticated request must be 200 with a dict body, which is the same
        boot signal the local branch asserts: config loads and serializes.

        The admin token is patched rather than read from the environment so
        this runs under a cloud-mode CI job that sets no TURTLE_ADMIN_TOKEN
        (an unset token is itself a 401, which would make the authed leg
        indistinguishable from the unauthed one). Patching settings.admin_token
        follows test/phase6_admin_test.py, which owns the full gate matrix;
        this file only re-checks that the route boots and serves in cloud.
        """
        r_unauthed = self.client.get("/api/config")
        self.assertEqual(r_unauthed.status_code, 401)

        with patch.object(turtle_server.settings, "admin_token", SecretStr("smoke-token")):
            r_authed = self.client.get(
                "/api/config", headers={"X-Admin-Token": "smoke-token"}
            )
        self.assertEqual(r_authed.status_code, 200)
        self.assertIsInstance(r_authed.json(), dict)

    def test_index_serves_200(self) -> None:
        # Either the chat UI (authed) or the onboarding form (unauthed) — both
        # are 200. We only assert the boot path renders *something*.
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
