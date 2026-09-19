"""
test/routine_cron_tick_test.py
---------------------------------
Unit coverage for core/routine_cron_tick.py (Vercel migration Phase 2's
due-routine computation) and core/storage/cloud/routine_last_fired_store.py.
Pure logic, no mocking needed for the due-computation half.
"""
from __future__ import annotations

import unittest
import unittest.mock
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from core.routine_cron_tick import (
    compute_due_occurrences,
    compute_due_routines,
    is_routine_due,
)


def _dt_utc(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


class IsRoutineDueDailyTest(unittest.TestCase):
    def test_due_at_exact_target_time(self) -> None:
        value = {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
        due, bucket = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 9, 0), tick_interval_minutes=5
        )
        self.assertTrue(due)
        self.assertEqual(bucket, "2026-09-12T09:00")

    def test_due_within_tick_window_after_target(self) -> None:
        value = {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
        due, bucket = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 9, 4), tick_interval_minutes=5
        )
        self.assertTrue(due)
        # Bucket is the SCHEDULED time, not the tick's observed time.
        self.assertEqual(bucket, "2026-09-12T09:00")

    def test_not_due_outside_window(self) -> None:
        value = {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
        due, bucket = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 9, 5), tick_interval_minutes=5
        )
        self.assertFalse(due)
        self.assertIsNone(bucket)

    def test_not_due_before_target(self) -> None:
        value = {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
        due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 8, 59), tick_interval_minutes=5
        )
        self.assertFalse(due)

    def test_default_cadence_is_daily_when_absent(self) -> None:
        value = {"time": "09:00", "timezone": "UTC"}
        due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 9, 0), tick_interval_minutes=5
        )
        self.assertTrue(due)

    def test_missing_time_is_unschedulable(self) -> None:
        value = {"cadence": "daily", "timezone": "UTC"}
        due, bucket = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 9, 0), tick_interval_minutes=5
        )
        self.assertFalse(due)
        self.assertIsNone(bucket)

    def test_unknown_cadence_is_unschedulable(self) -> None:
        value = {"cadence": "quarterly", "time": "09:00", "timezone": "UTC"}
        due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 9, 0), tick_interval_minutes=5
        )
        self.assertFalse(due)

    def test_bad_timezone_is_unschedulable(self) -> None:
        value = {"cadence": "daily", "time": "09:00", "timezone": "Not/AZone"}
        due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 9, 0), tick_interval_minutes=5
        )
        self.assertFalse(due)

    def test_malformed_time_is_unschedulable(self) -> None:
        value = {"cadence": "daily", "time": "not-a-time", "timezone": "UTC"}
        due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 9, 0), tick_interval_minutes=5
        )
        self.assertFalse(due)


class IsRoutineDueTimezoneTest(unittest.TestCase):
    def test_non_utc_timezone_localizes_correctly(self) -> None:
        # 09:00 America/New_York in September (EDT, UTC-4) = 13:00 UTC.
        value = {"cadence": "daily", "time": "09:00", "timezone": "America/New_York"}
        due, bucket = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 13, 0), tick_interval_minutes=5
        )
        self.assertTrue(due)
        self.assertEqual(bucket, "2026-09-12T09:00")

    def test_not_due_when_utc_time_does_not_match_local_target(self) -> None:
        value = {"cadence": "daily", "time": "09:00", "timezone": "America/New_York"}
        due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 9, 0), tick_interval_minutes=5
        )
        self.assertFalse(due)  # 9:00 UTC = 5am EDT, not the 9am target


class IsRoutineDueDayFilterTest(unittest.TestCase):
    def test_weekday_cadence_skips_saturday(self) -> None:
        # 2026-09-12 is a Saturday.
        value = {"cadence": "weekday", "time": "09:00", "timezone": "UTC"}
        due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 9, 0), tick_interval_minutes=5
        )
        self.assertFalse(due)

    def test_weekday_cadence_fires_on_monday(self) -> None:
        # 2026-09-14 is a Monday.
        value = {"cadence": "weekday", "time": "09:00", "timezone": "UTC"}
        due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 14, 9, 0), tick_interval_minutes=5
        )
        self.assertTrue(due)

    def test_weekend_cadence_fires_on_saturday(self) -> None:
        value = {"cadence": "weekend", "time": "09:00", "timezone": "UTC"}
        due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 9, 0), tick_interval_minutes=5
        )
        self.assertTrue(due)

    def test_weekly_cadence_only_fires_on_monday(self) -> None:
        value = {"cadence": "weekly", "time": "09:00", "timezone": "UTC"}
        monday_due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 14, 9, 0), tick_interval_minutes=5
        )
        tuesday_due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 15, 9, 0), tick_interval_minutes=5
        )
        self.assertTrue(monday_due)
        self.assertFalse(tuesday_due)

    def test_monthly_cadence_defaults_to_first_of_month(self) -> None:
        value = {"cadence": "monthly", "time": "09:00", "timezone": "UTC"}
        first_due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 1, 9, 0), tick_interval_minutes=5
        )
        second_due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 2, 9, 0), tick_interval_minutes=5
        )
        self.assertTrue(first_due)
        self.assertFalse(second_due)

    def test_monthly_cadence_honors_explicit_day(self) -> None:
        value = {"cadence": "monthly", "time": "09:00", "timezone": "UTC", "day": 15}
        due, bucket = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 15, 9, 0), tick_interval_minutes=5
        )
        self.assertTrue(due)


class IsRoutineDueHourlyTest(unittest.TestCase):
    def test_hourly_fires_every_hour_at_target_minute(self) -> None:
        value = {"cadence": "hourly", "time": "00:15", "timezone": "UTC"}
        due, bucket = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 14, 15), tick_interval_minutes=5
        )
        self.assertTrue(due)
        self.assertEqual(bucket, "2026-09-12T14:15")

    def test_hourly_defaults_to_minute_zero_with_no_time(self) -> None:
        value = {"cadence": "hourly", "timezone": "UTC"}
        due, bucket = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 14, 0), tick_interval_minutes=5
        )
        self.assertTrue(due)
        self.assertEqual(bucket, "2026-09-12T14:00")

    def test_hourly_not_due_outside_minute_window(self) -> None:
        value = {"cadence": "hourly", "time": "00:15", "timezone": "UTC"}
        due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 14, 30), tick_interval_minutes=5
        )
        self.assertFalse(due)


class IsRoutineDueMidnightWrapTest(unittest.TestCase):
    def test_target_near_midnight_wraps_correctly(self) -> None:
        value = {"cadence": "daily", "time": "23:58", "timezone": "UTC"}
        due, bucket = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 12, 23, 59), tick_interval_minutes=5
        )
        self.assertTrue(due)
        self.assertEqual(bucket, "2026-09-12T23:58")

    def test_far_from_target_across_midnight_not_due(self) -> None:
        value = {"cadence": "daily", "time": "23:58", "timezone": "UTC"}
        due, _ = is_routine_due(
            value, now_utc=_dt_utc(2026, 9, 13, 0, 30), tick_interval_minutes=5
        )
        self.assertFalse(due)


class ComputeDueOccurrencesTest(unittest.TestCase):
    """The catch-up half: a tick that arrives late must still fire what came
    due while it was away — the failure mode that made ~98% of routines
    silently never fire under GitHub Actions' real (hours-apart) cadence."""

    def test_late_tick_still_catches_a_daily_routine(self) -> None:
        value = {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
        # Tick due at 09:00 but the trigger only landed at 12:00.
        buckets = compute_due_occurrences(
            value,
            window_start_utc=_dt_utc(2026, 9, 12, 6, 0),
            window_end_utc=_dt_utc(2026, 9, 12, 12, 0),
        )
        self.assertEqual(buckets, ["2026-09-12T09:00"])

    def test_hourly_routine_yields_one_bucket_per_missed_hour(self) -> None:
        value = {"cadence": "hourly", "time": "00:15", "timezone": "UTC"}
        buckets = compute_due_occurrences(
            value,
            window_start_utc=_dt_utc(2026, 9, 12, 10, 0),
            window_end_utc=_dt_utc(2026, 9, 12, 13, 0),
        )
        self.assertEqual(
            buckets,
            ["2026-09-12T10:15", "2026-09-12T11:15", "2026-09-12T12:15"],
        )

    def test_multi_day_window_yields_one_bucket_per_day(self) -> None:
        value = {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
        buckets = compute_due_occurrences(
            value,
            window_start_utc=_dt_utc(2026, 9, 12, 0, 0),
            window_end_utc=_dt_utc(2026, 9, 14, 12, 0),
        )
        self.assertEqual(
            buckets,
            ["2026-09-12T09:00", "2026-09-13T09:00", "2026-09-14T09:00"],
        )

    def test_day_filter_still_applies_across_a_wide_window(self) -> None:
        # 2026-09-12 is a Saturday, so a weekday routine skips the 12th/13th.
        value = {"cadence": "weekday", "time": "09:00", "timezone": "UTC"}
        buckets = compute_due_occurrences(
            value,
            window_start_utc=_dt_utc(2026, 9, 12, 0, 0),
            window_end_utc=_dt_utc(2026, 9, 15, 12, 0),
        )
        self.assertEqual(buckets, ["2026-09-14T09:00", "2026-09-15T09:00"])

    def test_occurrence_exactly_at_window_start_is_excluded(self) -> None:
        # Half-open at the start: the previous tick already claimed 09:00.
        value = {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
        buckets = compute_due_occurrences(
            value,
            window_start_utc=_dt_utc(2026, 9, 12, 9, 0),
            window_end_utc=_dt_utc(2026, 9, 12, 12, 0),
        )
        self.assertEqual(buckets, [])

    def test_occurrence_exactly_at_window_end_is_included(self) -> None:
        value = {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
        buckets = compute_due_occurrences(
            value,
            window_start_utc=_dt_utc(2026, 9, 12, 6, 0),
            window_end_utc=_dt_utc(2026, 9, 12, 9, 0),
        )
        self.assertEqual(buckets, ["2026-09-12T09:00"])

    def test_empty_window_yields_nothing(self) -> None:
        value = {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
        buckets = compute_due_occurrences(
            value,
            window_start_utc=_dt_utc(2026, 9, 12, 12, 0),
            window_end_utc=_dt_utc(2026, 9, 12, 12, 0),
        )
        self.assertEqual(buckets, [])

    def test_timezone_is_honored_over_a_wide_window(self) -> None:
        # 09:00 America/New_York in September (EDT) = 13:00 UTC, so a window
        # ending at 12:00 UTC has not reached it yet.
        value = {"cadence": "daily", "time": "09:00", "timezone": "America/New_York"}
        before = compute_due_occurrences(
            value,
            window_start_utc=_dt_utc(2026, 9, 12, 6, 0),
            window_end_utc=_dt_utc(2026, 9, 12, 12, 0),
        )
        after = compute_due_occurrences(
            value,
            window_start_utc=_dt_utc(2026, 9, 12, 6, 0),
            window_end_utc=_dt_utc(2026, 9, 12, 14, 0),
        )
        self.assertEqual(before, [])
        self.assertEqual(after, ["2026-09-12T09:00"])

    def test_unschedulable_routine_yields_nothing(self) -> None:
        for value in (
            {"cadence": "quarterly", "time": "09:00", "timezone": "UTC"},
            {"cadence": "daily", "timezone": "UTC"},
            {"cadence": "daily", "time": "09:00", "timezone": "Not/AZone"},
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    compute_due_occurrences(
                        value,
                        window_start_utc=_dt_utc(2026, 9, 12, 0, 0),
                        window_end_utc=_dt_utc(2026, 9, 13, 0, 0),
                    ),
                    [],
                )


class ComputeDueRoutinesTest(unittest.TestCase):
    def _event(self, value):
        m = Mock()
        m.value = value
        return m

    def test_returns_only_due_routines(self) -> None:
        routines = {
            "workflow.morning_routine": self._event(
                {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
            ),
            "workflow.evening_routine": self._event(
                {"cadence": "daily", "time": "21:00", "timezone": "UTC"}
            ),
        }
        due = compute_due_routines(
            routines,
            window_start_utc=_dt_utc(2026, 9, 12, 8, 55),
            window_end_utc=_dt_utc(2026, 9, 12, 9, 0),
        )
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0][0], "workflow.morning_routine")
        self.assertEqual(due[0][2], "2026-09-12T09:00")

    def test_one_entry_per_occurrence_under_a_late_tick(self) -> None:
        routines = {
            "workflow.hourly_check": self._event(
                {"cadence": "hourly", "time": "00:30", "timezone": "UTC"}
            )
        }
        due = compute_due_routines(
            routines,
            window_start_utc=_dt_utc(2026, 9, 12, 9, 0),
            window_end_utc=_dt_utc(2026, 9, 12, 12, 0),
        )
        self.assertEqual(
            [bucket for _, _, bucket in due],
            ["2026-09-12T09:30", "2026-09-12T10:30", "2026-09-12T11:30"],
        )


# --- routine_last_fired_store -----------------------------------------------

class _FakeCursor:
    def __init__(self, rowcount):
        self.rowcount = rowcount


class _FakeConn:
    def __init__(self, claimed: set):
        self._claimed = claimed

    def execute(self, sql: str, params=None):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("INSERT INTO routine_last_fired"):
            key = tuple(params)
            if key in self._claimed:
                return _FakeCursor(0)
            self._claimed.add(key)
            return _FakeCursor(1)
        if sql_norm.startswith("DELETE FROM routine_last_fired"):
            return _FakeCursor(0)
        return _FakeCursor(0)  # CREATE TABLE / CREATE INDEX


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


class _FakePool:
    def __init__(self):
        self.claimed: set = set()

    def connection(self):
        return _FakeConnCtx(_FakeConn(self.claimed))


class TryClaimFireTest(unittest.TestCase):
    def setUp(self) -> None:
        import core.storage.cloud.routine_last_fired_store as rlf

        self.pool = _FakePool()
        rlf._initialized = False
        patcher = patch(
            "core.storage.cloud.routine_last_fired_store.get_pg_sync_pool",
            return_value=self.pool,
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_first_claim_succeeds(self) -> None:
        from core.storage.cloud.routine_last_fired_store import try_claim_fire

        self.assertTrue(try_claim_fire("usr_a", "workflow.morning_routine", "2026-09-12T09:00"))

    def test_second_claim_of_same_bucket_fails(self) -> None:
        from core.storage.cloud.routine_last_fired_store import try_claim_fire

        try_claim_fire("usr_a", "workflow.morning_routine", "2026-09-12T09:00")
        self.assertFalse(try_claim_fire("usr_a", "workflow.morning_routine", "2026-09-12T09:00"))

    def test_different_bucket_is_a_new_claim(self) -> None:
        from core.storage.cloud.routine_last_fired_store import try_claim_fire

        try_claim_fire("usr_a", "workflow.morning_routine", "2026-09-12T09:00")
        self.assertTrue(try_claim_fire("usr_a", "workflow.morning_routine", "2026-09-13T09:00"))

    def test_different_users_are_independent(self) -> None:
        from core.storage.cloud.routine_last_fired_store import try_claim_fire

        try_claim_fire("usr_a", "workflow.morning_routine", "2026-09-12T09:00")
        self.assertTrue(try_claim_fire("usr_b", "workflow.morning_routine", "2026-09-12T09:00"))


if __name__ == "__main__":
    unittest.main()
