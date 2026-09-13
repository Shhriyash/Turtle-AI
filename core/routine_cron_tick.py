"""
core/routine_cron_tick.py
--------------------------
Cloud (TURTLE_DEPLOY=cloud) replacement for core/routine_scheduler.py's
in-process APScheduler: pure "is this routine due right now?" logic for a
periodic cron-tick endpoint (apps/cron_tick_routes.py) rather than a
persistent per-minute-evaluating scheduler process.

APScheduler evaluates every job every minute against a live CronTrigger; a
tick-based caller instead runs every N minutes (N = tick_interval_minutes,
driven by a GitHub Actions `on: schedule` workflow — see
.github/workflows/cron-tick.yml) and must ask "has the target time already
passed since my last tick?" for a WINDOW of width N, not a single instant.
This module owns exactly that translation, reusing the same cadence
vocabulary and "refuse rather than guess" posture as
core.routine_scheduler._routine_to_cron_trigger (same _CADENCE_TO_CRON
mapping, imported from there so the two never drift on what a cadence means)
without depending on APScheduler's CronTrigger class at all — a tick only
needs a due/not-due verdict, not an object encoding "recur forever".

Idempotency: is_routine_due also returns a `fire_bucket` string uniquely
identifying the SCHEDULED occurrence (not the tick that observed it) — e.g.
"2026-09-12T09:05" for a 9:05am daily routine. The cron-tick endpoint claims
this bucket in a Postgres table (core/storage/cloud/routine_last_fired_store.py)
before firing, so a routine is fired exactly once per scheduled occurrence
even if two ticks both land inside the same due window (misfire/retry) or a
GitHub Actions run overlaps the previous one.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from core.routine_scheduler import _CADENCE_TO_CRON


def _resolve_cadence(value: dict[str, Any]) -> Optional[str]:
    cadence_raw = value.get("cadence")
    if cadence_raw is None or (isinstance(cadence_raw, str) and not cadence_raw.strip()):
        cadence = "daily"
    else:
        cadence = str(cadence_raw).strip().lower()
    return cadence if cadence in _CADENCE_TO_CRON else None


def _parse_time(value: dict[str, Any]) -> Optional[tuple[int, int]]:
    time_str = value.get("time")
    if not isinstance(time_str, str) or ":" not in time_str:
        return None
    try:
        hh_s, mm_s = time_str.split(":", 1)
        hh, mm = int(hh_s), int(mm_s)
        if not (0 <= hh < 24 and 0 <= mm < 60):
            return None
        return hh, mm
    except Exception:
        return None


def _day_matches(cadence: str, value: dict[str, Any], local_now: datetime) -> bool:
    """Mirrors _CADENCE_TO_CRON's day_of_week/day mapping exactly (weekly
    hardcodes Monday, matching the local scheduler's own semantics)."""
    weekday = local_now.weekday()  # Monday = 0
    if cadence == "daily" or cadence == "hourly":
        return True
    if cadence in ("weekday", "weekdays"):
        return weekday < 5
    if cadence in ("weekend", "weekends"):
        return weekday >= 5
    if cadence == "weekly":
        return weekday == 0
    if cadence == "monthly":
        day = value.get("day") or value.get("day_of_month")
        try:
            target_day = int(day) if day is not None else 1
        except Exception:
            target_day = 1
        return local_now.day == target_day
    return False


def _in_window(target_minutes: int, current_minutes: int, window_minutes: int) -> bool:
    """True when current_minutes is within [target_minutes, target_minutes +
    window_minutes) on a 24h wheel — wraps correctly for a target near
    midnight. Minutes-since-midnight, both sides already mod 1440."""
    delta = (current_minutes - target_minutes) % 1440
    return 0 <= delta < window_minutes


def is_routine_due(
    value: dict[str, Any], *, now_utc: datetime, tick_interval_minutes: int
) -> tuple[bool, Optional[str]]:
    """Returns (due, fire_bucket). fire_bucket is None when not due.

    An unschedulable cadence/time/timezone is a refuse (False, None), same
    posture as core.routine_scheduler._routine_to_cron_trigger — a
    misconfigured routine should silently not fire, not fire wrong.
    """
    cadence = _resolve_cadence(value)
    if cadence is None:
        return False, None

    tz_name = value.get("timezone") or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return False, None
    local_now = now_utc.astimezone(tz)

    if cadence == "hourly":
        parsed = _parse_time(value)
        target_minute = parsed[1] if parsed else 0
        if not _in_window(target_minute, local_now.minute, tick_interval_minutes):
            return False, None
        fire_bucket = f"{local_now:%Y-%m-%dT%H}:{target_minute:02d}"
        return True, fire_bucket

    parsed = _parse_time(value)
    if parsed is None:
        return False, None
    target_hh, target_mm = parsed
    if not _day_matches(cadence, value, local_now):
        return False, None
    target_minutes = target_hh * 60 + target_mm
    current_minutes = local_now.hour * 60 + local_now.minute
    if not _in_window(target_minutes, current_minutes, tick_interval_minutes):
        return False, None
    fire_bucket = f"{local_now:%Y-%m-%d}T{target_hh:02d}:{target_mm:02d}"
    return True, fire_bucket


def compute_due_routines(
    routines: dict[str, Any], *, now_utc: datetime, tick_interval_minutes: int
) -> list[tuple[str, Any, str]]:
    """routines: {routine_key: MemoryEvent}, already filtered to
    applied-and-not-rejected by the caller (see
    core.routine_scheduler.get_active_routines_for_user + apps/cron_tick_routes.py).
    Returns [(routine_key, event, fire_bucket), ...] for every due routine.
    """
    due: list[tuple[str, Any, str]] = []
    for key, event in routines.items():
        is_due, fire_bucket = is_routine_due(
            event.value, now_utc=now_utc, tick_interval_minutes=tick_interval_minutes
        )
        if is_due and fire_bucket is not None:
            due.append((key, event, fire_bucket))
    return due
