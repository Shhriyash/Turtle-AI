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
from datetime import UTC, datetime, timedelta
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


class _FakeTickState:
    def __init__(self, last_tick_at):
        self.last_tick_at = last_tick_at
        self.advanced_to = None

    def advance(self, tick_at):
        self.advanced_to = tick_at
        self.last_tick_at = tick_at


class _FakeClaims:
    """In-memory stand-in for the routine_last_fired_store functions the tick
    uses. Mirrors their contract (first claim wins; mark_fired only flips
    'claimed'; list_stuck_claims reads an age the test sets). The SQL itself is
    covered by test/cloud_integration/routine_last_fired_store_test.py."""

    def __init__(self):
        self.rows: dict[tuple[str, str, str], dict] = {}
        self.pruned_cutoff = None

    def try_claim_fire(self, user_id, routine_key, fire_bucket, status="claimed"):
        key = (user_id, routine_key, fire_bucket)
        if key in self.rows:
            return False
        self.rows[key] = {"status": status, "age_s": 0}
        return True

    def mark_fired(self, user_id, routine_key, fire_bucket):
        row = self.rows.get((user_id, routine_key, fire_bucket))
        if row and row["status"] == "claimed":
            row["status"] = "fired"
            return True
        return False

    def mark_failed(self, user_id, routine_key, fire_bucket):
        row = self.rows.get((user_id, routine_key, fire_bucket))
        if row and row["status"] == "claimed":
            row["status"] = "failed"
            return True
        return False

    def list_stuck_claims(self, min_age_s, max_age_s):
        return [
            k for k, r in self.rows.items()
            if r["status"] == "claimed" and min_age_s <= r["age_s"] < max_age_s
        ]

    def fail_stale_claims(self, max_age_s):
        n = 0
        for r in self.rows.values():
            if r["status"] == "claimed" and r["age_s"] >= max_age_s:
                r["status"] = "failed"
                n += 1
        return n

    def prune_older_than(self, cutoff_iso):
        self.pruned_cutoff = cutoff_iso
        return 0

    def statuses(self):
        return {k: r["status"] for k, r in self.rows.items()}


_MORNING = {
    "routine": "morning briefing", "cadence": "daily", "time": "09:00", "timezone": "UTC",
}


class RunTickOrchestrationTest(unittest.TestCase):
    """_run_tick's own orchestration (lock -> window -> enumerate -> claim ->
    fire/confirm, missed, re-fire, prune) with the Postgres-touching
    dependencies replaced by in-memory fakes."""

    def setUp(self) -> None:
        self.claims = _FakeClaims()
        self.fire = Mock()
        self.hook = Mock(return_value=True)
        self.state = None
        self.locked = True

    def _event(self, value, *, applied=True, rejected=False):
        m = Mock()
        m.value = value
        m.applied = applied
        m.rejected = rejected
        return m

    def _run(self, now, *, last_tick, routines=None, users=("usr_a",), env=None,
             routines_side_effect=None):
        import contextlib
        import os

        from apps.cron_tick_routes import _run_tick

        self.state = _FakeTickState(last_tick)

        @contextlib.contextmanager
        def fake_lock():
            yield self.state if self.locked else None

        env_clean = {k: v for k, v in os.environ.items() if k != "TURTLE_ROUTINE_LATE_FIRE_LIMIT_S"}
        env_clean.update(env or {})
        routines_patch = (
            patch("core.routine_scheduler.get_active_routines_for_user",
                  side_effect=routines_side_effect)
            if routines_side_effect is not None
            else patch("core.routine_scheduler.get_active_routines_for_user",
                       return_value=routines or {})
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, env_clean, clear=True))
            stack.enter_context(patch(
                "core.storage.cloud.cron_state_store.locked_tick_state", fake_lock))
            stack.enter_context(patch(
                "core.storage.cloud.journal_store.list_user_ids_pg", return_value=list(users)))
            stack.enter_context(routines_patch)
            for name in ("try_claim_fire", "mark_fired", "mark_failed", "list_stuck_claims",
                         "fail_stale_claims", "prune_older_than"):
                stack.enter_context(patch(
                    f"core.storage.cloud.routine_last_fired_store.{name}",
                    getattr(self.claims, name)))
            stack.enter_context(patch("core.routine_scheduler._fire_routine_strict", self.fire))
            stack.enter_context(patch("core.routine_scheduler._delivery_hook", self.hook))
            return _run_tick(now)

    # -- basic fire path ----------------------------------------------------

    def test_fires_due_routine_and_confirms_it(self) -> None:
        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        result = self._run(
            now, last_tick=now - timedelta(minutes=5),
            routines={"workflow.morning_routine": self._event(_MORNING)},
        )
        self.assertEqual(result["users_checked"], 1)
        self.assertEqual(result["routines_fired"], 1)
        self.assertEqual(result["error_count"], 0)
        self.assertIn("correlation_id", result)
        self.assertEqual(
            self.claims.statuses(),
            {("usr_a", "workflow.morning_routine", "2026-09-12T09:00"): "fired"},
        )
        self.fire.assert_called_once()
        kwargs = self.fire.call_args.kwargs
        # Deterministic identity comes from the scheduled occurrence, not the clock.
        self.assertEqual(kwargs["fire_bucket"], "2026-09-12T09:00")
        self.assertEqual(kwargs["fired_at"], "2026-09-12T09:00:00+00:00")
        self.assertEqual(self.state.advanced_to, now)

    def test_skips_already_claimed_routine(self) -> None:
        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        self.claims.rows[("usr_a", "workflow.morning_routine", "2026-09-12T09:00")] = {
            "status": "fired", "age_s": 0}
        result = self._run(
            now, last_tick=now - timedelta(minutes=5),
            routines={"workflow.morning_routine": self._event(_MORNING)},
        )
        self.assertEqual(result["routines_fired"], 0)
        self.assertEqual(result["error_count"], 0)
        self.fire.assert_not_called()

    def test_skips_rejected_and_unapplied_routines(self) -> None:
        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        result = self._run(
            now, last_tick=now - timedelta(minutes=5),
            routines={
                "workflow.rejected_one": self._event(_MORNING, rejected=True),
                "workflow.unapplied_one": self._event(_MORNING, applied=False),
            },
        )
        self.assertEqual(result["routines_fired"], 0)
        self.assertEqual(self.claims.rows, {})
        self.fire.assert_not_called()

    # -- 5.7: late ticks ------------------------------------------------------

    def test_tick_20_minutes_late_fires_the_missed_occurrence_once(self) -> None:
        routines = {"workflow.morning_routine": self._event(_MORNING)}
        late = datetime(2026, 9, 12, 9, 20, tzinfo=UTC)
        first = self._run(late, last_tick=datetime(2026, 9, 12, 8, 55, tzinfo=UTC),
                          routines=routines)
        self.assertEqual(first["routines_fired"], 1)
        self.assertEqual(first["routines_missed"], 0)
        # The next tick resumes from where the first stopped: no second fire.
        second = self._run(late + timedelta(minutes=5), last_tick=self.state.advanced_to,
                           routines=routines)
        self.assertEqual(second["routines_fired"], 0)
        self.assertEqual(self.fire.call_count, 1)

    def test_even_a_replayed_window_fires_only_once(self) -> None:
        # Same window enumerated twice (e.g. last_tick_at was not advanced):
        # the claim, not the window, is what makes it exactly-once.
        routines = {"workflow.morning_routine": self._event(_MORNING)}
        late = datetime(2026, 9, 12, 9, 20, tzinfo=UTC)
        for _ in range(2):
            self._run(late, last_tick=datetime(2026, 9, 12, 8, 55, tzinfo=UTC), routines=routines)
        self.assertEqual(self.fire.call_count, 1)

    def test_tick_2_hours_late_records_missed_and_notifies_once(self) -> None:
        routines = {"workflow.morning_routine": self._event(_MORNING)}
        late = datetime(2026, 9, 12, 11, 0, tzinfo=UTC)
        last = datetime(2026, 9, 12, 8, 55, tzinfo=UTC)
        result = self._run(late, last_tick=last, routines=routines)
        self.assertEqual(result["routines_fired"], 0)
        self.assertEqual(result["routines_missed"], 1)
        self.fire.assert_not_called()
        self.assertEqual(
            self.claims.statuses(),
            {("usr_a", "workflow.morning_routine", "2026-09-12T09:00"): "missed"},
        )
        self.hook.assert_called_once()
        user_id, frame = self.hook.call_args.args
        self.assertEqual(user_id, "usr_a")
        self.assertEqual(frame["code"], "routine_missed")
        self.assertEqual(frame["type"], "routine")
        self.assertIn("09:00", frame["message"])
        self.assertIn("missed", frame["message"])
        # fired_at comes from the occurrence, not the clock -> stable identity.
        self.assertEqual(frame["fired_at"], "2026-09-12T09:00:00+00:00")
        # Replaying the same window must NOT notify again.
        self._run(late + timedelta(minutes=5), last_tick=last, routines=routines)
        self.hook.assert_called_once()

    def test_several_missed_occurrences_produce_one_notice_frame(self) -> None:
        routines = {
            "workflow.a": self._event({**_MORNING, "routine": "alpha", "time": "07:00"}),
            "workflow.b": self._event({**_MORNING, "routine": "beta", "time": "08:00"}),
            "workflow.c": self._event({**_MORNING, "routine": "gamma", "time": "09:00"}),
        }
        result = self._run(
            datetime(2026, 9, 12, 14, 0, tzinfo=UTC),
            last_tick=datetime(2026, 9, 12, 6, 0, tzinfo=UTC), routines=routines,
        )
        self.assertEqual(result["routines_missed"], 3)
        self.hook.assert_called_once()
        self.assertEqual(self.hook.call_args.args[1]["missed_count"], 3)

    def test_missed_notice_is_delivered_before_this_ticks_fire_frames(self) -> None:
        # The outbox keeps the MOST RECENT 5 frames; sending the missed notice
        # first means real fire frames are the newest and evict it, not vice versa.
        order = []
        self.hook.side_effect = lambda u, f: order.append(("hook", f["code"])) or True
        self.fire.side_effect = lambda *a, **k: order.append(("fire", None))
        routines = {
            "workflow.old": self._event({**_MORNING, "time": "07:00"}),
            "workflow.new": self._event({**_MORNING, "time": "10:45"}),
        }
        self._run(
            datetime(2026, 9, 12, 11, 0, tzinfo=UTC),
            last_tick=datetime(2026, 9, 12, 6, 0, tzinfo=UTC), routines=routines,
        )
        self.assertEqual(order, [("hook", "routine_missed"), ("fire", None)])

    def test_bootstrap_first_tick_only_looks_back_one_limit(self) -> None:
        routines = {
            "workflow.old": self._event({**_MORNING, "time": "07:00"}),
            "workflow.recent": self._event({**_MORNING, "time": "09:40"}),
        }
        result = self._run(
            datetime(2026, 9, 12, 10, 0, tzinfo=UTC), last_tick=None, routines=routines
        )
        self.assertEqual(result["routines_fired"], 1)   # 09:40 is within the last hour
        self.assertEqual(result["routines_missed"], 0)  # 07:00 is history, not "missed"
        self.assertEqual(
            list(self.claims.statuses()),
            [("usr_a", "workflow.recent", "2026-09-12T09:40")],
        )
        self.assertIsNotNone(self.state.advanced_to)

    def test_window_is_clamped_to_24_hours(self) -> None:
        routines = {"workflow.morning_routine": self._event(_MORNING)}
        result = self._run(
            datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
            last_tick=datetime(2026, 9, 10, 0, 0, tzinfo=UTC), routines=routines,
        )
        self.assertTrue(result["window_truncated"])
        # 24 h back from 09-20 12:00 covers only the 09-20 09:00 occurrence.
        self.assertEqual(list(self.claims.statuses()),
                         [("usr_a", "workflow.morning_routine", "2026-09-20T09:00")])

    def test_late_fire_limit_is_configurable_by_env(self) -> None:
        routines = {"workflow.morning_routine": self._event(_MORNING)}
        result = self._run(
            datetime(2026, 9, 12, 11, 0, tzinfo=UTC),  # 2 h late
            last_tick=datetime(2026, 9, 12, 8, 55, tzinfo=UTC), routines=routines,
            env={"TURTLE_ROUTINE_LATE_FIRE_LIMIT_S": "10800"},
        )
        self.assertEqual(result["routines_fired"], 1)
        self.assertEqual(result["routines_missed"], 0)

    def test_invalid_late_fire_limit_falls_back_to_one_hour(self) -> None:
        from apps.cron_tick_routes import _late_fire_limit_s

        for bad in ("abc", "0", "-5", ""):
            with patch.dict("os.environ", {"TURTLE_ROUTINE_LATE_FIRE_LIMIT_S": bad}):
                self.assertEqual(_late_fire_limit_s(), 3600, bad)
        with patch.dict("os.environ", {"TURTLE_ROUTINE_LATE_FIRE_LIMIT_S": "7200"}):
            self.assertEqual(_late_fire_limit_s(), 7200)

    def test_overlapping_tick_that_cannot_take_the_lock_does_nothing(self) -> None:
        self.locked = False
        result = self._run(
            datetime(2026, 9, 12, 9, 0, tzinfo=UTC),
            last_tick=None, routines={"workflow.morning_routine": self._event(_MORNING)},
        )
        self.assertTrue(result["skipped"])
        self.assertEqual(result["routines_fired"], 0)
        self.assertEqual(self.claims.rows, {})
        self.fire.assert_not_called()

    # -- 5.9: at-least-once ---------------------------------------------------

    def test_failed_fire_is_not_counted_and_leaves_the_claim_stuck(self) -> None:
        self.fire.side_effect = RuntimeError("journal down")
        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        result = self._run(
            now, last_tick=now - timedelta(minutes=5),
            routines={"workflow.morning_routine": self._event(_MORNING)},
        )
        self.assertEqual(result["routines_fired"], 0)
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(
            self.claims.statuses(),
            {("usr_a", "workflow.morning_routine", "2026-09-12T09:00"): "claimed"},
        )

    def test_one_users_failed_fire_does_not_abort_the_tick(self) -> None:
        self.fire.side_effect = [RuntimeError("boom"), None]
        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        result = self._run(
            now, last_tick=now - timedelta(minutes=5), users=("usr_a", "usr_b"),
            routines={"workflow.morning_routine": self._event(_MORNING)},
        )
        self.assertEqual(result["users_checked"], 2)
        self.assertEqual(result["routines_fired"], 1)
        self.assertEqual(result["error_count"], 1)

    def test_stuck_claim_is_refired_once_then_confirmed(self) -> None:
        key = ("usr_a", "workflow.morning_routine", "2026-09-12T09:00")
        self.claims.rows[key] = {"status": "claimed", "age_s": 900}
        now = datetime(2026, 9, 12, 9, 15, tzinfo=UTC)
        routines = {"workflow.morning_routine": self._event(_MORNING)}
        result = self._run(now, last_tick=now - timedelta(minutes=5), routines=routines)
        self.assertEqual(result["routines_refired"], 1)
        self.assertEqual(self.claims.statuses()[key], "fired")
        self.fire.assert_called_once()
        self.assertEqual(self.fire.call_args.kwargs["fire_bucket"], "2026-09-12T09:00")
        self.assertEqual(self.fire.call_args.kwargs["fired_at"], "2026-09-12T09:00:00+00:00")
        # A later tick finds nothing stuck.
        self._run(now + timedelta(minutes=5), last_tick=now, routines=routines)
        self.fire.assert_called_once()

    def test_recent_claim_is_not_refired(self) -> None:
        key = ("usr_a", "workflow.morning_routine", "2026-09-12T09:00")
        self.claims.rows[key] = {"status": "claimed", "age_s": 120}
        now = datetime(2026, 9, 12, 9, 2, tzinfo=UTC)
        self._run(now, last_tick=now, routines={"workflow.morning_routine": self._event(_MORNING)})
        self.fire.assert_not_called()

    def test_stuck_claim_for_a_retracted_routine_is_failed_not_fired(self) -> None:
        key = ("usr_a", "workflow.morning_routine", "2026-09-12T09:00")
        self.claims.rows[key] = {"status": "claimed", "age_s": 900}
        now = datetime(2026, 9, 12, 9, 15, tzinfo=UTC)
        result = self._run(
            now, last_tick=now,
            routines={"workflow.morning_routine": self._event(_MORNING, rejected=True)},
        )
        self.fire.assert_not_called()
        self.assertEqual(self.claims.statuses()[key], "failed")
        self.assertEqual(result["claims_failed"], 1)

    def test_claim_too_old_to_retry_is_failed(self) -> None:
        key = ("usr_a", "workflow.morning_routine", "2026-09-12T05:00")
        self.claims.rows[key] = {"status": "claimed", "age_s": 5 * 3600}
        now = datetime(2026, 9, 12, 10, 0, tzinfo=UTC)
        result = self._run(now, last_tick=now, routines={})
        self.fire.assert_not_called()
        self.assertEqual(self.claims.statuses()[key], "failed")
        self.assertEqual(result["claims_failed"], 1)

    def test_failure_between_claim_and_fire_refires_once_with_one_journal_event(self) -> None:
        """The ledger acceptance end to end: real _fire_routine_strict and a
        real (local) journal, only delivery fails the first time."""
        import tempfile
        from pathlib import Path

        import core.paths as core_paths
        import core.routine_scheduler as rs
        from core.memory_journal import JournalStore

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            core_paths, "PERSONAL_MEMORY_DIR", Path(tmp)
        ), patch.object(
            core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", Path(tmp) / "snapshots"
        ), patch.object(rs, "PERSONAL_MEMORY_DIR", Path(tmp)):
            self.fire = rs._fire_routine_strict  # the REAL one
            delivered = []
            attempts = {"n": 0}

            def flaky_delivery(user_id, frame):
                attempts["n"] += 1
                if attempts["n"] == 1:
                    raise RuntimeError("delivery hook blew up")
                delivered.append(frame)
                return True

            self.hook = flaky_delivery
            routines = {"workflow.morning_routine": self._event(_MORNING)}
            now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)

            # patch the module-level name _run() would otherwise replace by a Mock
            with patch("core.routine_scheduler._fire_routine_strict", rs._fire_routine_strict):
                first = self._run_real(now, now - timedelta(minutes=5), routines)
                key = ("usr_a", "workflow.morning_routine", "2026-09-12T09:00")
                self.assertEqual(first["routines_fired"], 0)
                self.assertEqual(first["error_count"], 1)
                self.assertEqual(self.claims.statuses()[key], "claimed")

                self.claims.rows[key]["age_s"] = 900  # the claim has now been stuck 15 min
                second = self._run_real(
                    now + timedelta(minutes=15), now + timedelta(minutes=10), routines
                )

            self.assertEqual(second["routines_refired"], 1)
            self.assertEqual(self.claims.statuses()[key], "fired")
            fires = [
                e for e in JournalStore(user_id="usr_a").load_all()
                if e.key == "workflow.scheduled_fire.workflow.morning_routine"
            ]
            self.assertEqual(len(fires), 1)  # the re-fire was a journal no-op
            self.assertEqual(len(delivered), 1)
            self.assertEqual(fires[0].event_id, rs.fire_event_id(*key))

    def _run_real(self, now, last_tick, routines):
        """_run variant that keeps the real _fire_routine_strict and uses
        self.hook as the delivery hook."""
        import contextlib

        from apps.cron_tick_routes import _run_tick

        self.state = _FakeTickState(last_tick)

        @contextlib.contextmanager
        def fake_lock():
            yield self.state

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch(
                "core.storage.cloud.cron_state_store.locked_tick_state", fake_lock))
            stack.enter_context(patch(
                "core.storage.cloud.journal_store.list_user_ids_pg", return_value=["usr_a"]))
            stack.enter_context(patch(
                "core.routine_scheduler.get_active_routines_for_user", return_value=routines))
            for name in ("try_claim_fire", "mark_fired", "mark_failed", "list_stuck_claims",
                         "fail_stale_claims", "prune_older_than"):
                stack.enter_context(patch(
                    f"core.storage.cloud.routine_last_fired_store.{name}",
                    getattr(self.claims, name)))
            stack.enter_context(patch("core.routine_scheduler._delivery_hook", self.hook))
            return _run_tick(now)

    # -- 5.10: pruning ---------------------------------------------------------

    def test_every_tick_prunes_claims_older_than_seven_days(self) -> None:
        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        result = self._run(now, last_tick=now - timedelta(minutes=5), routines={})
        self.assertEqual(self.claims.pruned_cutoff, (now - timedelta(days=7)).isoformat())
        self.assertIn("claims_pruned", result)

    # -- error handling --------------------------------------------------------

    def test_journal_read_error_recorded_but_does_not_abort_tick(self) -> None:
        # WP 1.B / S-7.3 output hygiene: _run_tick's RETURNED dict (what
        # becomes the HTTP response, and what the GH Actions workflow echoes
        # to its log) must carry only a count, never the user id. Per-user
        # detail goes to the server's own logger (see the next test).
        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        result = self._run(
            now, last_tick=now - timedelta(minutes=5), users=("usr_broken", "usr_ok"),
            routines_side_effect=[RuntimeError("boom"), {}],
        )
        self.assertEqual(result["users_checked"], 2)
        self.assertEqual(result["error_count"], 1)
        self.assertNotIn("errors", result)
        self.assertNotIn("usr_broken", json.dumps(result))

    def test_unreadable_user_holds_the_window_back(self) -> None:
        # If a user's routines could not be read their occurrences were not
        # enumerated, so last_tick_at must NOT advance past them.
        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        result = self._run(
            now, last_tick=now - timedelta(minutes=5), users=("usr_broken",),
            routines_side_effect=RuntimeError("boom"),
        )
        self.assertFalse(result["window_advanced"])
        self.assertIsNone(self.state.advanced_to)

    def test_journal_read_error_is_logged_with_correlation_id(self) -> None:
        now = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
        with self.assertLogs("apps.cron_tick_routes", level="WARNING") as logs:
            result = self._run(
                now, last_tick=now - timedelta(minutes=5), users=("usr_broken",),
                routines_side_effect=RuntimeError("boom"),
            )
        self.assertTrue(any(result["correlation_id"] in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
