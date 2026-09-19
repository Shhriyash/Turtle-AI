"""
core/routine_cron_tick.py
--------------------------
Cloud (TURTLE_DEPLOY=cloud) replacement for core/routine_scheduler.py's
in-process APScheduler: pure "is this routine due right now?" logic for a
periodic cron-tick endpoint (apps/cron_tick_routes.py) rather than a
persistent per-minute-evaluating scheduler process.

APScheduler evaluates every job every minute against a live CronTrigger; a
tick-based caller instead runs periodically and must ask "which scheduled
occurrences have passed since my last tick?". This module owns exactly that
translation, reusing the same cadence vocabulary and "refuse rather than
guess" posture as core.routine_scheduler._routine_to_cron_trigger (same
_CADENCE_TO_CRON mapping, imported from there so the two never drift on what
a cadence means) without depending on APScheduler's CronTrigger class at all
— a tick only needs the list of occurrences, not an object encoding
"recur forever".

The unit of work is a WINDOW, not a fixed-width one. An earlier version
assumed ticks arrive every 5 minutes and asked only "is a target within the
last 5 minutes?", which silently dropped a routine whenever the trigger ran
late. It always ran late: measured over the first 5 days of
.github/workflows/cron-tick.yml, GitHub delivered 34 of ~1484 scheduled
`*/5` runs (2.2%), with a median gap of 220 minutes and not one gap under
115. A 5-minute verdict against a 220-minute cadence misses ~98% of
routines. So compute_due_occurrences takes an explicit (start, end] range
and enumerates EVERY occurrence inside it: a tick that arrives three hours
late still fires what came due while it was away. The caller supplies the
range from its persisted tick cursor (see apps/cron_tick_routes.py) and
bounds how far back it is willing to look.

Idempotency: is_routine_due also returns a `fire_bucket` string uniquely
identifying the SCHEDULED occurrence (not the tick that observed it) — e.g.
"2026-09-12T09:05" for a 9:05am daily routine. The cron-tick endpoint claims
this bucket in a Postgres table (core/storage/cloud/routine_last_fired_store.py)
before firing, so a routine is fired exactly once per scheduled occurrence
even if two ticks both land inside the same due window (misfire/retry) or a
GitHub Actions run overlaps the previous one.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
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


def compute_due_occurrences(
    value: dict[str, Any], *, window_start_utc: datetime, window_end_utc: datetime
) -> list[str]:
    """Every scheduled occurrence of this routine in (start, end], oldest
    first, as fire_bucket strings.

    The range is half-open at the start so consecutive ticks sharing a
    boundary can't both claim the same occurrence — the bucket dedupe in
    core/storage/cloud/routine_last_fired_store.py is the backstop, not the
    first line of defence.

    An unschedulable cadence/time/timezone yields [], the same "refuse
    rather than guess" posture as
    core.routine_scheduler._routine_to_cron_trigger — a misconfigured
    routine should silently not fire, not fire wrong.
    """
    cadence = _resolve_cadence(value)
    if cadence is None:
        return []
    tz_name = value.get("timezone") or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return []
    if window_end_utc <= window_start_utc:
        return []

    local_start = window_start_utc.astimezone(tz)
    local_end = window_end_utc.astimezone(tz)
    buckets: list[str] = []

    if cadence == "hourly":
        parsed = _parse_time(value)
        target_minute = parsed[1] if parsed else 0
        # Walk hour boundaries, not the raw window: the occurrence inside an
        # hour sits at target_minute, which may fall on either side of the
        # window edge, so each candidate is range-checked in UTC below.
        hour = local_start.replace(minute=0, second=0, microsecond=0)
        while hour <= local_end:
            occurrence = hour.replace(minute=target_minute)
            if window_start_utc < occurrence.astimezone(timezone.utc) <= window_end_utc:
                buckets.append(f"{occurrence:%Y-%m-%dT%H}:{target_minute:02d}")
            hour += timedelta(hours=1)
        return buckets

    parsed = _parse_time(value)
    if parsed is None:
        return []
    target_hh, target_mm = parsed
    day = local_start.date()
    last_day = local_end.date()
    while day <= last_day:
        occurrence = datetime(day.year, day.month, day.day, target_hh, target_mm, tzinfo=tz)
        if _day_matches(cadence, value, occurrence) and (
            window_start_utc < occurrence.astimezone(timezone.utc) <= window_end_utc
        ):
            buckets.append(f"{day:%Y-%m-%d}T{target_hh:02d}:{target_mm:02d}")
        day += timedelta(days=1)
    return buckets


def is_routine_due(
    value: dict[str, Any], *, now_utc: datetime, tick_interval_minutes: int
) -> tuple[bool, Optional[str]]:
    """Single-window convenience form: (due, fire_bucket) for the fixed
    (now - tick_interval, now] range. Kept because "did this routine come due
    in the last N minutes?" is the natural question at a call site that has
    no tick cursor to work from; compute_due_occurrences is the general form
    and the only place the cadence rules live.
    """
    buckets = compute_due_occurrences(
        value,
        window_start_utc=now_utc - timedelta(minutes=tick_interval_minutes),
        window_end_utc=now_utc,
    )
    if not buckets:
        return False, None
    return True, buckets[-1]


def compute_due_routines(
    routines: dict[str, Any], *, window_start_utc: datetime, window_end_utc: datetime
) -> list[tuple[str, Any, str]]:
    """routines: {routine_key: MemoryEvent}, already filtered to
    applied-and-not-rejected by the caller (see
    core.routine_scheduler.get_active_routines_for_user + apps/cron_tick_routes.py).
    Returns [(routine_key, event, fire_bucket), ...] — one entry per due
    OCCURRENCE, so a routine that came due more than once inside the window
    (an hourly one under a late tick) appears once per occurrence.
    """
    due: list[tuple[str, Any, str]] = []
    for key, event in routines.items():
        for fire_bucket in compute_due_occurrences(
            event.value, window_start_utc=window_start_utc, window_end_utc=window_end_utc
        ):
            due.append((key, event, fire_bucket))
    return due
