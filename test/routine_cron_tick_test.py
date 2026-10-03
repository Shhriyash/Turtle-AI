"""
test/routine_cron_tick_test.py
---------------------------------
Unit coverage for core/routine_cron_tick.py (occurrence enumeration, ledger
5.7 / 5.11) and core/storage/cloud/routine_last_fired_store.py.
Pure logic, no mocking needed for the enumeration half.
"""
from __future__ import annotations

import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from core.routine_cron_tick import (
    compute_due_routines,
    fire_instant_for_bucket,
    occurrences_between,
)


def _dt_utc(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def is_routine_due(value, *, now_utc, tick_interval_minutes):
    """The old single-window question, expressed through the enumeration API:
    is there an occurrence in the tick_interval_minutes ending at now_utc?"""
    occs = occurrences_between(
        value,
        after_utc=now_utc - timedelta(minutes=tick_interval_minutes),
        until_utc=now_utc,
    )
    if not occs:
        return False, None
    return True, occs[-1].fire_bucket


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
            after_utc=_dt_utc(2026, 9, 12, 8, 55),
            until_utc=_dt_utc(2026, 9, 12, 9, 0),
        )
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0].routine_key, "workflow.morning_routine")
        self.assertEqual(due[0].fire_bucket, "2026-09-12T09:00")

    def test_wide_window_returns_every_occurrence_oldest_first(self) -> None:
        routines = {
            "workflow.a": self._event({"cadence": "daily", "time": "21:00", "timezone": "UTC"}),
            "workflow.b": self._event({"cadence": "daily", "time": "09:00", "timezone": "UTC"}),
        }
        due = compute_due_routines(
            routines,
            after_utc=_dt_utc(2026, 9, 12, 8, 0),
            until_utc=_dt_utc(2026, 9, 13, 10, 0),
        )
        self.assertEqual(
            [(d.routine_key, d.fire_bucket) for d in due],
            [
                ("workflow.b", "2026-09-12T09:00"),
                ("workflow.a", "2026-09-12T21:00"),
                ("workflow.b", "2026-09-13T09:00"),
            ],
        )


class OccurrenceEnumerationTest(unittest.TestCase):
    """Ledger 5.7: a tick that arrives late still sees what it stepped over."""

    DAILY = {"cadence": "daily", "time": "08:00", "timezone": "UTC"}

    def test_tick_20_minutes_late_still_finds_the_occurrence_once(self) -> None:
        # Previous tick 07:55, this one 08:20: the old 5-minute window would
        # have skipped 08:00 forever.
        occs = occurrences_between(
            self.DAILY, after_utc=_dt_utc(2026, 9, 12, 7, 55), until_utc=_dt_utc(2026, 9, 12, 8, 20)
        )
        self.assertEqual([o.fire_bucket for o in occs], ["2026-09-12T08:00"])
        # ...and the next tick's window, starting at 08:20, does not see it again.
        again = occurrences_between(
            self.DAILY, after_utc=_dt_utc(2026, 9, 12, 8, 20), until_utc=_dt_utc(2026, 9, 12, 8, 25)
        )
        self.assertEqual(again, [])

    def test_window_is_open_at_after_and_closed_at_until(self) -> None:
        at = _dt_utc(2026, 9, 12, 8, 0)
        self.assertEqual(
            occurrences_between(self.DAILY, after_utc=at, until_utc=at + timedelta(minutes=5)), []
        )
        self.assertEqual(
            len(occurrences_between(self.DAILY, after_utc=at - timedelta(minutes=5), until_utc=at)),
            1,
        )

    def test_multi_day_gap_enumerates_each_day(self) -> None:
        occs = occurrences_between(
            self.DAILY, after_utc=_dt_utc(2026, 9, 10, 9, 0), until_utc=_dt_utc(2026, 9, 13, 9, 0)
        )
        self.assertEqual(
            [o.fire_bucket for o in occs],
            ["2026-09-11T08:00", "2026-09-12T08:00", "2026-09-13T08:00"],
        )

    def test_hourly_enumerates_each_hour_in_gap(self) -> None:
        occs = occurrences_between(
            {"cadence": "hourly", "time": "00:15", "timezone": "UTC"},
            after_utc=_dt_utc(2026, 9, 12, 10, 0),
            until_utc=_dt_utc(2026, 9, 12, 13, 0),
        )
        self.assertEqual(
            [o.fire_bucket for o in occs],
            ["2026-09-12T10:15", "2026-09-12T11:15", "2026-09-12T12:15"],
        )

    def test_unschedulable_routine_yields_nothing_even_over_a_wide_window(self) -> None:
        for value in (
            {"cadence": "daily", "timezone": "UTC"},
            {"cadence": "quarterly", "time": "09:00", "timezone": "UTC"},
            {"cadence": "daily", "time": "09:00", "timezone": "Not/AZone"},
        ):
            self.assertEqual(
                occurrences_between(
                    value, after_utc=_dt_utc(2026, 9, 1, 0, 0), until_utc=_dt_utc(2026, 9, 30, 0, 0)
                ),
                [],
            )


class DstTest(unittest.TestCase):
    """Ledger 5.11, pinned on America/New_York 2026-03-08 and 2026-11-01."""

    NY = "America/New_York"

    def _tick_every_5_min(self, value, start_utc, hours=30):
        """Simulate a tick every 5 minutes (each with the window since the
        previous one) and return every (tick time, bucket, fire instant)."""
        seen = []
        prev = start_utc
        for i in range(1, hours * 12 + 1):
            now = start_utc + timedelta(minutes=5 * i)
            for o in occurrences_between(value, after_utc=prev, until_utc=now):
                seen.append((now, o.fire_bucket, o.fire_utc))
            prev = now
        return seen

    def test_spring_forward_0230_fires_at_first_valid_minute_with_stable_bucket(self) -> None:
        # 2026-03-08: 02:00 EST jumps to 03:00 EDT (07:00 UTC). 02:30 never exists.
        value = {"cadence": "daily", "time": "02:30", "timezone": self.NY}
        seen = self._tick_every_5_min(value, _dt_utc(2026, 3, 8, 3, 0))
        on_day = [x for x in seen if x[1] == "2026-03-08T02:30"]
        self.assertEqual(len(on_day), 1, seen)
        tick, bucket, fire_utc = on_day[0]
        self.assertEqual(fire_utc, _dt_utc(2026, 3, 8, 7, 0))  # 03:00 EDT
        self.assertEqual(tick, _dt_utc(2026, 3, 8, 7, 0))
        # The neighbouring day is unaffected (02:30 EDT = 06:30Z on the 9th).
        self.assertEqual([x[1] for x in seen], ["2026-03-08T02:30", "2026-03-09T02:30"])

    def test_spring_forward_0200_fires_at_first_valid_minute(self) -> None:
        value = {"cadence": "daily", "time": "02:00", "timezone": self.NY}
        seen = self._tick_every_5_min(value, _dt_utc(2026, 3, 8, 3, 0), hours=12)
        self.assertEqual(
            [(x[1], x[2]) for x in seen], [("2026-03-08T02:00", _dt_utc(2026, 3, 8, 7, 0))]
        )

    def test_spring_forward_0300_fires_once_at_the_same_instant(self) -> None:
        # A 03:00 routine and a 02:30 routine land on the same instant but
        # keep distinct buckets; each fires once.
        value = {"cadence": "daily", "time": "03:00", "timezone": self.NY}
        seen = self._tick_every_5_min(value, _dt_utc(2026, 3, 8, 3, 0), hours=12)
        self.assertEqual(
            [(x[1], x[2]) for x in seen], [("2026-03-08T03:00", _dt_utc(2026, 3, 8, 7, 0))]
        )

    def test_spring_forward_gap_does_not_raise_for_any_window_shape(self) -> None:
        value = {"cadence": "daily", "time": "02:30", "timezone": self.NY}
        occs = occurrences_between(
            value, after_utc=_dt_utc(2026, 3, 7, 0, 0), until_utc=_dt_utc(2026, 3, 10, 0, 0)
        )
        self.assertEqual(
            [o.fire_bucket for o in occs],
            ["2026-03-07T02:30", "2026-03-08T02:30", "2026-03-09T02:30"],
        )

    def test_fall_back_0130_fires_exactly_once(self) -> None:
        # 2026-11-01: 02:00 EDT falls back to 01:00 EST, so 01:30 happens twice
        # (05:30Z and 06:30Z). Only the first is enumerated.
        value = {"cadence": "daily", "time": "01:30", "timezone": self.NY}
        seen = self._tick_every_5_min(value, _dt_utc(2026, 11, 1, 3, 0), hours=12)
        self.assertEqual(
            [(x[1], x[2]) for x in seen], [("2026-11-01T01:30", _dt_utc(2026, 11, 1, 5, 30))]
        )

    def test_fall_back_wide_window_spanning_both_passes_fires_once(self) -> None:
        value = {"cadence": "daily", "time": "01:30", "timezone": self.NY}
        occs = occurrences_between(
            value, after_utc=_dt_utc(2026, 11, 1, 4, 0), until_utc=_dt_utc(2026, 11, 1, 8, 0)
        )
        self.assertEqual(len(occs), 1)

    def test_hourly_skips_the_nonexistent_hour(self) -> None:
        value = {"cadence": "hourly", "time": "00:15", "timezone": self.NY}
        occs = occurrences_between(
            value, after_utc=_dt_utc(2026, 3, 8, 5, 0), until_utc=_dt_utc(2026, 3, 8, 9, 0)
        )
        # 05:15Z=00:15 EST, 06:15Z=01:15 EST, (02:15 does not exist),
        # 07:15Z=03:15 EDT, 08:15Z=04:15 EDT
        self.assertEqual(
            [o.fire_bucket for o in occs],
            ["2026-03-08T00:15", "2026-03-08T01:15", "2026-03-08T03:15", "2026-03-08T04:15"],
        )


class FireInstantForBucketTest(unittest.TestCase):
    def test_round_trips_a_normal_bucket(self) -> None:
        value = {"cadence": "daily", "time": "09:00", "timezone": "America/New_York"}
        self.assertEqual(
            fire_instant_for_bucket(value, "2026-09-12T09:00"), _dt_utc(2026, 9, 12, 13, 0)
        )

    def test_gap_bucket_resolves_to_first_valid_minute(self) -> None:
        value = {"cadence": "daily", "time": "02:30", "timezone": "America/New_York"}
        self.assertEqual(
            fire_instant_for_bucket(value, "2026-03-08T02:30"), _dt_utc(2026, 3, 8, 7, 0)
        )

    def test_garbage_is_none(self) -> None:
        good = {"cadence": "daily", "time": "09:00", "timezone": "UTC"}
        self.assertIsNone(fire_instant_for_bucket(good, "nope"))
        self.assertIsNone(
            fire_instant_for_bucket({"timezone": "Not/AZone"}, "2026-09-12T09:00")
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
            key = tuple(params[:3])  # PK is (user, routine, bucket); status is not part of it
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
