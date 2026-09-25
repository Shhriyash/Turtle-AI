"""
test/cron_tick_routes_test.py
--------------------------------
Endpoint-level coverage for apps/cron_tick_routes.py (Vercel migration
Phase 2) using a standalone FastAPI app with just this router mounted
(mirrors test/production_onboarding_test.py's pattern), rather than the
whole turtle_server.app.
"""
from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.cron_tick_routes import router


class CronTickAuthTest(unittest.TestCase):
    """WP 1.B / S-7.3: /internal/cron-tick now authenticates with
    CRON_TICK_SECRET only (the retired CRON_SHARED_SECRET is never read)."""

    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_no_secret_configured_returns_503(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_tick_secret = None
            resp = self.client.post("/internal/cron-tick")
        self.assertEqual(resp.status_code, 503)

    def test_missing_bearer_token_returns_401(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_tick_secret.get_secret_value.return_value = "secret123"
            resp = self.client.post("/internal/cron-tick")
        self.assertEqual(resp.status_code, 401)

    def test_wrong_bearer_token_returns_401(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_tick_secret.get_secret_value.return_value = "secret123"
            resp = self.client.post(
                "/internal/cron-tick", headers={"Authorization": "Bearer wrong"}
            )
        self.assertEqual(resp.status_code, 401)

    def test_correct_token_but_not_cloud_mode_returns_503(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_tick_secret.get_secret_value.return_value = "secret123"
            fake_settings.is_cloud = False
            resp = self.client.post(
                "/internal/cron-tick", headers={"Authorization": "Bearer secret123"}
            )
        self.assertEqual(resp.status_code, 503)

    def test_correct_token_and_cloud_mode_runs_the_tick(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_tick_secret.get_secret_value.return_value = "secret123"
            fake_settings.is_cloud = True
            with patch(
                "apps.cron_tick_routes._run_tick",
                return_value={
                    "correlation_id": "corr-1",
                    "users_checked": 0,
                    "routines_fired": 0,
                    "error_count": 0,
                    "ticked_at": "x",
                },
            ) as fake_run:
                resp = self.client.post(
                    "/internal/cron-tick", headers={"Authorization": "Bearer secret123"}
                )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["users_checked"], 0)
        self.assertEqual(body["correlation_id"], "corr-1")
        fake_run.assert_called_once()

    def test_internal_job_secret_is_rejected_on_cron_tick(self) -> None:
        # The two secrets are NOT interchangeable: a caller holding only
        # INTERNAL_JOB_SECRET must not be able to authenticate to cron-tick.
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_tick_secret.get_secret_value.return_value = "cron-secret"
            resp = self.client.post(
                "/internal/cron-tick",
                headers={"Authorization": "Bearer job-secret-value"},
            )
        self.assertEqual(resp.status_code, 401)


class RunTickOrchestrationTest(unittest.TestCase):
    """_run_tick's own orchestration logic (list users -> get routines ->
    compute due -> claim -> fire), with every dependency mocked."""

    def _event(self, value, *, applied=True, rejected=False):
        m = Mock()
        m.value = value
        m.applied = applied
        m.rejected = rejected
        return m

    def test_fires_due_routine_for_each_user(self) -> None:
        from apps.cron_tick_routes import _run_tick

        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        routines = {
            "workflow.morning_routine": self._event(
                {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
            )
        }
        with patch(
            "core.storage.cloud.journal_store.list_user_ids_pg", return_value=["usr_a"]
        ), patch(
            "core.routine_scheduler.get_active_routines_for_user", return_value=routines
        ), patch(
            "core.storage.cloud.routine_last_fired_store.try_claim_fire", return_value=True
        ) as fake_claim, patch(
            "core.routine_scheduler._fire_routine"
        ) as fake_fire:
            result = _run_tick(now)

        self.assertEqual(result["users_checked"], 1)
        self.assertEqual(result["routines_fired"], 1)
        self.assertEqual(result["error_count"], 0)
        self.assertIn("correlation_id", result)
        fake_claim.assert_called_once_with("usr_a", "workflow.morning_routine", "2026-09-12T09:00")
        fake_fire.assert_called_once()

    def test_skips_already_claimed_routine(self) -> None:
        from apps.cron_tick_routes import _run_tick

        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        routines = {
            "workflow.morning_routine": self._event(
                {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
            )
        }
        with patch(
            "core.storage.cloud.journal_store.list_user_ids_pg", return_value=["usr_a"]
        ), patch(
            "core.routine_scheduler.get_active_routines_for_user", return_value=routines
        ), patch(
            "core.storage.cloud.routine_last_fired_store.try_claim_fire", return_value=False
        ), patch(
            "core.routine_scheduler._fire_routine"
        ) as fake_fire:
            result = _run_tick(now)

        self.assertEqual(result["routines_fired"], 0)
        self.assertEqual(result["error_count"], 0)
        fake_fire.assert_not_called()

    def test_skips_rejected_and_unapplied_routines(self) -> None:
        from apps.cron_tick_routes import _run_tick

        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        routines = {
            "workflow.rejected_one": self._event(
                {"cadence": "daily", "time": "09:00", "timezone": "UTC"}, rejected=True
            ),
            "workflow.unapplied_one": self._event(
                {"cadence": "daily", "time": "09:00", "timezone": "UTC"}, applied=False
            ),
        }
        with patch(
            "core.storage.cloud.journal_store.list_user_ids_pg", return_value=["usr_a"]
        ), patch(
            "core.routine_scheduler.get_active_routines_for_user", return_value=routines
        ), patch(
            "core.storage.cloud.routine_last_fired_store.try_claim_fire"
        ) as fake_claim, patch(
            "core.routine_scheduler._fire_routine"
        ) as fake_fire:
            result = _run_tick(now)

        self.assertEqual(result["routines_fired"], 0)
        fake_claim.assert_not_called()
        fake_fire.assert_not_called()

    def test_journal_read_error_recorded_but_does_not_abort_tick(self) -> None:
        # WP 1.B / S-7.3 output hygiene: _run_tick's RETURNED dict (what
        # becomes the HTTP response, and what the GH Actions workflow echoes
        # to its log) must carry only a count, never the user id — the
        # original assertion here (`self.assertIn("usr_broken",
        # result["errors"][0])`) asserted the exact thing this WP requires
        # removed, so it's replaced with an assertion that the user id is
        # NOT present anywhere in the response. Per-user detail still goes
        # to the server's own logger (not the HTTP response) — see
        # test_journal_read_error_is_logged_with_correlation_id below.
        from apps.cron_tick_routes import _run_tick

        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        with patch(
            "core.storage.cloud.journal_store.list_user_ids_pg",
            return_value=["usr_broken", "usr_ok"],
        ), patch(
            "core.routine_scheduler.get_active_routines_for_user",
            side_effect=[RuntimeError("boom"), {}],
        ):
            result = _run_tick(now)

        self.assertEqual(result["users_checked"], 2)
        self.assertEqual(result["error_count"], 1)
        self.assertNotIn("errors", result)
        self.assertNotIn("usr_broken", json.dumps(result))

    def test_journal_read_error_is_logged_with_correlation_id(self) -> None:
        from apps.cron_tick_routes import _run_tick

        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        with patch(
            "core.storage.cloud.journal_store.list_user_ids_pg",
            return_value=["usr_broken"],
        ), patch(
            "core.routine_scheduler.get_active_routines_for_user",
            side_effect=RuntimeError("boom"),
        ), self.assertLogs("apps.cron_tick_routes", level="WARNING") as logs:
            result = _run_tick(now)

        self.assertTrue(any(result["correlation_id"] in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
