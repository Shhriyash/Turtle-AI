"""
test/cron_tick_routes_test.py
--------------------------------
Endpoint-level coverage for apps/cron_tick_routes.py (Vercel migration
Phase 2) using a standalone FastAPI app with just this router mounted
(mirrors test/production_onboarding_test.py's pattern), rather than the
whole turtle_server.app.
"""
from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.cron_tick_routes import router


class CronTickAuthTest(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_no_secret_configured_returns_503(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_shared_secret = None
            resp = self.client.post("/internal/cron-tick")
        self.assertEqual(resp.status_code, 503)

    def test_missing_bearer_token_returns_401(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "secret123"
            resp = self.client.post("/internal/cron-tick")
        self.assertEqual(resp.status_code, 401)

    def test_wrong_bearer_token_returns_401(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "secret123"
            resp = self.client.post(
                "/internal/cron-tick", headers={"Authorization": "Bearer wrong"}
            )
        self.assertEqual(resp.status_code, 401)

    def test_correct_token_but_not_cloud_mode_returns_503(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "secret123"
            fake_settings.is_cloud = False
            resp = self.client.post(
                "/internal/cron-tick", headers={"Authorization": "Bearer secret123"}
            )
        self.assertEqual(resp.status_code, 503)

    def test_correct_token_and_cloud_mode_runs_the_tick(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "secret123"
            fake_settings.is_cloud = True
            with patch(
                "apps.cron_tick_routes._run_tick",
                return_value={"users_checked": 0, "routines_fired": 0, "errors": [], "ticked_at": "x"},
            ) as fake_run:
                resp = self.client.post(
                    "/internal/cron-tick", headers={"Authorization": "Bearer secret123"}
                )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["users_checked"], 0)
        fake_run.assert_called_once()

    def test_get_is_accepted_for_vercel_cron(self) -> None:
        # Vercel Cron issues GET, not POST — see vercel.json's `crons`.
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "secret123"
            fake_settings.is_cloud = True
            with patch(
                "apps.cron_tick_routes._run_tick",
                return_value={"users_checked": 0, "routines_fired": 0, "errors": [], "ticked_at": "x"},
            ):
                resp = self.client.get(
                    "/internal/cron-tick", headers={"Authorization": "Bearer secret123"}
                )
        self.assertEqual(resp.status_code, 200)

    def test_get_without_token_is_still_rejected(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "secret123"
            resp = self.client.get("/internal/cron-tick")
        self.assertEqual(resp.status_code, 401)


class RunTickOrchestrationTest(unittest.TestCase):
    """_run_tick's own orchestration logic (read cursor -> list users -> get
    routines -> compute due -> claim -> fire -> advance cursor), with every
    dependency mocked."""

    def setUp(self) -> None:
        # No cursor by default, so the window is the nominal single interval
        # and these cases read as "what came due in the last tick".
        cursor = patch.multiple(
            "core.storage.cloud.cron_tick_cursor_store",
            read_last_tick=Mock(return_value=None),
            write_last_tick=Mock(),
        )
        self.addCleanup(cursor.stop)
        cursor.start()

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
        self.assertEqual(result["errors"], [])
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

    def test_late_tick_fires_what_came_due_while_it_was_away(self) -> None:
        """The whole point of the cursor: the trigger ran 3 hours late and
        the 09:00 routine must still fire, not be silently skipped."""
        from apps.cron_tick_routes import _run_tick

        now = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
        routines = {
            "workflow.morning_routine": self._event(
                {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
            )
        }
        with patch(
            "core.storage.cloud.cron_tick_cursor_store.read_last_tick",
            return_value=datetime(2026, 9, 12, 8, 0, tzinfo=UTC),
        ), patch(
            "core.storage.cloud.cron_tick_cursor_store.write_last_tick"
        ) as fake_write, patch(
            "core.storage.cloud.journal_store.list_user_ids_pg", return_value=["usr_a"]
        ), patch(
            "core.routine_scheduler.get_active_routines_for_user", return_value=routines
        ), patch(
            "core.storage.cloud.routine_last_fired_store.try_claim_fire", return_value=True
        ) as fake_claim, patch(
            "core.routine_scheduler._fire_routine"
        ):
            result = _run_tick(now)

        self.assertEqual(result["routines_fired"], 1)
        fake_claim.assert_called_once_with(
            "usr_a", "workflow.morning_routine", "2026-09-12T09:00"
        )
        fake_write.assert_called_once_with(now)

    def test_catchup_is_bounded_so_a_stale_cursor_does_not_backfill(self) -> None:
        """A 3-day-old cursor must not replay 3 days of routines — only what
        falls inside MAX_CATCHUP_MINUTES."""
        from apps.cron_tick_routes import _run_tick

        now = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
        routines = {
            "workflow.morning_routine": self._event(
                {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
            )
        }
        with patch(
            "core.storage.cloud.cron_tick_cursor_store.read_last_tick",
            return_value=datetime(2026, 9, 9, 12, 0, tzinfo=UTC),
        ), patch(
            "core.storage.cloud.cron_tick_cursor_store.write_last_tick"
        ), patch(
            "core.storage.cloud.journal_store.list_user_ids_pg", return_value=["usr_a"]
        ), patch(
            "core.routine_scheduler.get_active_routines_for_user", return_value=routines
        ), patch(
            "core.storage.cloud.routine_last_fired_store.try_claim_fire", return_value=True
        ) as fake_claim, patch(
            "core.routine_scheduler._fire_routine"
        ):
            result = _run_tick(now)

        self.assertEqual(result["routines_fired"], 1)
        self.assertEqual(
            [c.args[2] for c in fake_claim.call_args_list], ["2026-09-12T09:00"]
        )

    def test_unreadable_cursor_degrades_to_one_interval_and_records_it(self) -> None:
        from apps.cron_tick_routes import _run_tick

        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        routines = {
            "workflow.morning_routine": self._event(
                {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
            )
        }
        with patch(
            "core.storage.cloud.cron_tick_cursor_store.read_last_tick",
            side_effect=RuntimeError("pg down"),
        ), patch(
            "core.storage.cloud.cron_tick_cursor_store.write_last_tick"
        ), patch(
            "core.storage.cloud.journal_store.list_user_ids_pg", return_value=["usr_a"]
        ), patch(
            "core.routine_scheduler.get_active_routines_for_user", return_value=routines
        ), patch(
            "core.storage.cloud.routine_last_fired_store.try_claim_fire", return_value=True
        ), patch(
            "core.routine_scheduler._fire_routine"
        ):
            result = _run_tick(now)

        self.assertEqual(result["routines_fired"], 1)
        self.assertIn("cursor read failed", result["errors"][0])

    def test_journal_read_error_recorded_but_does_not_abort_tick(self) -> None:
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
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("usr_broken", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
