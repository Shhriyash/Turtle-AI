"""
Phase 2 / W4 — memory confirmation UI.

Two halves:

1. Content-level pins on the shipped web client. These catch a regression
   (someone deletes the handler, the fetch, or the panel mount) without a
   browser. They read the source files directly.

2. A server-side exercise of the exact request/response contract the UI was
   coded against: GET /api/memory/pending and POST /api/memory/confirm, driven
   through FastAPI's TestClient against a real ConfirmationGate. Guarded with
   skipIf so a missing optional dep degrades to the content-level pins only.

Contract the UI consumes (apps/turtle_server.py):
  GET  /api/memory/pending -> {"pending": [{event_id, question, topic, key}, ...]}
  POST /api/memory/confirm  body {"event_id": <str>, "accepted": <bool>}
                            -> 200 {"status": "ok", "applied": <bool>}
                               400 (bad body) / 401 (no user) / 404 (no session
                               or unknown event_id), each {"error": <str>}.
"""
from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

# Repo root = turtle/ (this file lives in turtle/test/).
ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def _read(*parts: str) -> str:
    return (WEB.joinpath(*parts)).read_text(encoding="utf-8")


class WebClientContentPins(unittest.TestCase):
    """Regression pins on the shipped vanilla-JS client + HTML mount points."""

    def test_websocket_handles_confirmation_prompt_frame(self) -> None:
        src = _read("js", "websocket.js")
        # The sidecar frame the server pushes must be dispatched by the switch.
        self.assertIn("confirmation_prompt", src)
        self.assertIn("case 'confirmation_prompt'", src)
        # And it must delegate to the memory module that renders the card.
        self.assertIn("renderConfirmationPrompt", src)
        self.assertIn("./memory.js", src)

    def test_client_posts_to_confirm_endpoint(self) -> None:
        src = _read("js", "memory.js")
        # A fetch to the confirm endpoint with the exact body contract.
        self.assertIn("/api/memory/confirm", src)
        self.assertIn("fetch(", src)
        self.assertIn("event_id", src)
        self.assertIn("accepted", src)
        self.assertIn("POST", src)

    def test_pending_panel_reads_pending_endpoint(self) -> None:
        src = _read("js", "memory.js")
        self.assertIn("/api/memory/pending", src)

    def test_index_html_has_panel_mount_and_toggle(self) -> None:
        html = _read("index.html")
        # Panel list mount + header toggle + stylesheet link.
        self.assertIn('id="memory-pending-list"', html)
        self.assertIn('id="memory-panel"', html)
        self.assertIn('id="btn-memory-toggle"', html)
        self.assertIn("/static/css/memory.css", html)

    def test_app_boots_memory_ui(self) -> None:
        src = _read("js", "app.js")
        self.assertIn("initMemoryUI", src)


# --------------------------------------------------------------------------
# Server-side contract exercise
# --------------------------------------------------------------------------
try:
    from fastapi.testclient import TestClient
    from apps import turtle_server
    from core.confirmation_gate import ConfirmationGate
    from core.memory_journal import JournalStore, make_event
    from core.personal_memory_store import PersonalMemoryStore

    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover — optional deps missing in env
    _IMPORT_ERROR = _e


@unittest.skipIf(_IMPORT_ERROR is not None, f"server deps missing: {_IMPORT_ERROR}")
class MemoryEndpointContractTests(unittest.TestCase):
    """Exercises the endpoints as they actually run in production: each
    request builds its own ConfirmationGate straight from storage
    (turtle_server._build_confirmation_gate_for_user), not from a
    process-cached SharedState — see that function's docstring for why
    (this is the fix for the documented multi-worker/cross-instance
    /api/memory/confirm 404). Isolation is via a redirected
    core.paths.PERSONAL_MEMORY_DIR, matching every other test in this suite
    that exercises real storage (e.g. test/production_onboarding_test.py),
    rather than injecting a fake SharedState.
    """

    USER = "local_dev_user"

    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"phase2_gate_ui_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)

        import core.paths as paths_mod
        self._orig_personal_dir = paths_mod.PERSONAL_MEMORY_DIR
        paths_mod.PERSONAL_MEMORY_DIR = self.base / "personal"
        paths_mod.PERSONAL_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        # The gate the test queues candidates through directly, resolving to
        # the SAME storage the endpoint will build its own gate from — both
        # go through the redirected PERSONAL_MEMORY_DIR.
        self.journal = JournalStore(user_id=self.USER)
        self.store = PersonalMemoryStore(user_id=self.USER)
        self.gate = ConfirmationGate(
            journal=self.journal,
            store=self.store,
            state_path=paths_mod.personal_memory_dir(self.USER) / "confirmation_state.json",
        )

        # The endpoints resolve the user via _get_user_id_from_request, which
        # returns "local_dev_user" only when not running in cloud mode. Pin
        # deploy_mode so the test is robust regardless of the ambient env.
        self._orig_deploy_mode = turtle_server.settings.deploy_mode
        turtle_server.settings.deploy_mode = "local"
        # _get_user_id_from_request now requires DEV_ANON=1 to resolve an
        # anonymous local caller to local_dev_user (was implicit off-cloud).
        self._orig_dev_anon = turtle_server.settings.dev_anon
        turtle_server.settings.dev_anon = True

        # No lifespan/startup: instantiate the client directly (not as a
        # context manager) so no server startup hooks fire.
        self.client = TestClient(turtle_server.app)

    def tearDown(self) -> None:
        turtle_server.settings.deploy_mode = self._orig_deploy_mode
        turtle_server.settings.dev_anon = self._orig_dev_anon
        import core.paths as paths_mod
        paths_mod.PERSONAL_MEMORY_DIR = self._orig_personal_dir
        shutil.rmtree(self.base, ignore_errors=True)

    def _queue_candidate(self, *, key: str, value: dict) -> str:
        event = make_event(
            kind="fact",
            topic=key.split(".", 1)[0],
            key=key,
            value=value,
            confidence=0.7,
            source="inferred",
            extractor="llm_turn",
            session_id="s1",
            turn_id="t1",
            applied=False,
        )
        self.assertTrue(self.gate.queue_candidate(event))
        return event.event_id

    def test_pending_lists_queued_candidate(self) -> None:
        eid = self._queue_candidate(key="identity.name", value={"name": "Shriyash"})

        resp = self.client.get("/api/memory/pending")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn("pending", body)
        self.assertEqual(len(body["pending"]), 1)
        item = body["pending"][0]
        # The exact shape the panel renders against.
        self.assertEqual(item["event_id"], eid)
        self.assertEqual(item["topic"], "identity")
        self.assertEqual(item["key"], "identity.name")
        self.assertIn("Shriyash", item["question"])

    def test_confirm_accept_applies_and_drains_queue(self) -> None:
        eid = self._queue_candidate(key="identity.name", value={"name": "Shriyash"})

        resp = self.client.post(
            "/api/memory/confirm", json={"event_id": eid, "accepted": True}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json(), {"status": "ok", "applied": True})

        # Queue is now empty — the panel would show its empty state.
        again = self.client.get("/api/memory/pending")
        self.assertEqual(again.json(), {"pending": []})

    def test_confirm_reject_returns_applied_false(self) -> None:
        eid = self._queue_candidate(key="identity.name", value={"name": "Wrongname"})

        resp = self.client.post(
            "/api/memory/confirm", json={"event_id": eid, "accepted": False}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json(), {"status": "ok", "applied": False})

    def test_confirm_rejects_missing_event_id(self) -> None:
        resp = self.client.post("/api/memory/confirm", json={"accepted": True})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    def test_confirm_rejects_non_bool_accepted(self) -> None:
        eid = self._queue_candidate(key="identity.name", value={"name": "Shriyash"})
        resp = self.client.post(
            "/api/memory/confirm", json={"event_id": eid, "accepted": "yes"}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    def test_confirm_unknown_event_id_is_404(self) -> None:
        resp = self.client.post(
            "/api/memory/confirm", json={"event_id": "evt_missing", "accepted": True}
        )
        self.assertEqual(resp.status_code, 404)
        self.assertIn("error", resp.json())

    def test_pending_empty_when_nothing_queued(self) -> None:
        # A user with no queued candidates (and, in particular, no live
        # SharedState anywhere) must still get a clean empty list, never a
        # 500 — the endpoint builds its own gate straight from storage now,
        # so there is no "active session" precondition left to fail.
        resp = self.client.get("/api/memory/pending")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json(), {"pending": []})


if __name__ == "__main__":
    unittest.main()
