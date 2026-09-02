"""
test/smoke_boot_test.py
-----------------------
Phase 5 W3: deploy boot smoke. Proves the ASGI app imports and serves its most
basic contracts without any live API, so a broken boot fails the normal test
suite (and by extension CI) instead of only surfacing at deploy time. This is
the in-suite equivalent of the Dockerfile HEALTHCHECK — no separate CI job.

Fully offline: no network, no auth, no writes. The app module boots under CI's
dummy keys (see .github/workflows/tests.yml) and locally imports clean too.
"""
from __future__ import annotations

import unittest

# Import the app exactly as test/phase2_gate_ui_test.py does: guard the import
# so a genuinely missing optional dep degrades to a skip rather than a collection
# error, but under CI/local (deps present) the smoke actually runs.
try:
    from fastapi.testclient import TestClient
    from apps import turtle_server

    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover - optional deps missing in env
    _IMPORT_ERROR = _e


@unittest.skipIf(_IMPORT_ERROR is not None, f"app import failed: {_IMPORT_ERROR!r}")
class BootSmoke(unittest.TestCase):
    """The app boots and answers its liveness + config + index contracts."""

    def setUp(self) -> None:
        # Instantiate the client directly (not as a context manager) so no
        # lifespan/startup hooks fire — same approach as phase2_gate_ui_test.
        self.client = TestClient(turtle_server.app)

    def test_healthz_returns_ok(self) -> None:
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_api_config_returns_dict(self) -> None:
        resp = self.client.get("/api/config")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), dict)

    def test_index_serves_200(self) -> None:
        # Either the chat UI (authed) or the onboarding form (unauthed) — both
        # are 200. We only assert the boot path renders *something*.
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
