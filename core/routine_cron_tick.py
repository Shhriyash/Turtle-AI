"""
core/routine_cron_tick.py
--------------------------
Cloud (TURTLE_DEPLOY=cloud) replacement for core/routine_scheduler.py's
in-process APScheduler: pure "which scheduled occurrences fall in this time
window?" logic for a periodic cron-tick endpoint (apps/cron_tick_routes.py)
rather than a persistent per-minute-evaluating scheduler process.

Why enumeration, not a due window. The tick is driven by a GitHub Actions
`on: schedule` workflow (.github/workflows/cron-tick.yml), and GitHub does
not deliver it reliably: measured over 19 days only ~2% of the expected
every-5-minutes runs arrived, with gaps of 2-8 hours between them. A check
of the form "is the target time inside the last 5 minutes?" silently skips
every occurrence a late tick steps over. So instead the endpoint remembers
when it last ticked (core/storage/cloud/cron_state_store.py) and asks this
module for EVERY occurrence in (last_tick_at, now]; the caller then decides,
per occurrence, whether it is recent enough to fire or old enough to be
recorded as missed. This module only does the calendar arithmetic.

It reuses the same cadence vocabulary and "refuse rather than guess" posture
as core.routine_scheduler._routine_to_cron_trigger (same _CADENCE_TO_CRON
mapping, imported from there so the two never drift on what a cadence means)
without depending on APScheduler's CronTrigger class.

Idempotency: every Occurrence carries a `fire_bucket` string uniquely
identifying the SCHEDULED occurrence (not the tick that observed it) -- e.g.
"2026-09-12T09:05" for a 9:05am daily routine, always the LOCAL wall time the
user asked for. The endpoint claims this bucket in Postgres
(core/storage/cloud/routine_last_fired_store.py) before firing.

DST (ledger 5.11). The bucket is always the scheduled wall time, but the
instant it fires is resolved against the real timezone rules:
  - Spring forward: a wall time inside the skipped hour does not exist (New
    York 2026-03-08 02:30). Its bucket stays "2026-03-08T02:30" so dedup is
    stable, but it fires at the first valid minute after the change (03:00
    local). Matches classic cron behaviour for skipped fixed times. Hourly
    routines are the exception: an hour that does not exist is simply not
    enumerated (a wildcard-hour cron job skips it too).
  - Fall back: a wall time that happens twice (New York 2026-11-01 01:30) is
    enumerated once, at its FIRST occurrence, and the bucket would dedupe the
    second anyway.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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


@dataclass(frozen=True)
class Occurrence:
    """One scheduled occurrence of a routine.

    fire_bucket: the scheduled LOCAL wall time, "YYYY-MM-DDTHH:MM" (stable
        dedup key, even when that wall time does not exist -- see module DST
        notes).
    fire_utc: the instant it is due. Equals the wall time converted to UTC
        except for a spring-forward gap, where it is the first valid minute
        after the clock change.
    """

    fire_bucket: str
    fire_utc: datetime


@dataclass(frozen=True)
class DueOccurrence:
    """An Occurrence tied to the routine that produced it."""

    routine_key: str
    event: Any
    fire_bucket: str
    fire_utc: datetime


def _resolve_wall_time(wall: datetime, tz: ZoneInfo) -> tuple[datetime, bool]:
    """Map a naive local wall time to (instant in UTC, in_dst_gap).

    Ambiguous (fall-back) times resolve to their FIRST occurrence (fold=0).
    A time inside a spring-forward gap resolves to the first valid minute
    after the clock change, and in_dst_gap is True. Never raises for a
    nonexistent wall time.
    """
    first = wall.replace(tzinfo=tz, fold=0)
    utc_first = first.astimezone(UTC)
    if utc_first.astimezone(tz).replace(tzinfo=None) == wall:
        return utc_first, False
    # Nonexistent wall time: the zone's utcoffset() for fold=0 is the offset
    # BEFORE the change, fold=1 the offset AFTER it. The change happened
    # between the two candidate instants; walk minute by minute to the first
    # instant already on the post-change offset.
    off_after = wall.replace(tzinfo=tz, fold=1).utcoffset()
    utc_after = wall.replace(tzinfo=tz, fold=1).astimezone(UTC)
    lo, hi = sorted((utc_first, utc_after))
    probe = lo
    while probe <= hi:
        if probe.astimezone(tz).utcoffset() == off_after:
            return probe, True
        probe += timedelta(minutes=1)
    return hi, True  # unreachable for real zone data; stay total anyway


def _routine_timezone(value: dict[str, Any]) -> Optional[ZoneInfo]:
    try:
        return ZoneInfo(value.get("timezone") or "UTC")
    except Exception:
        return None


def occurrences_between(
    value: dict[str, Any], *, after_utc: datetime, until_utc: datetime
) -> list[Occurrence]:
    """Every scheduled occurrence with fire_utc in (after_utc, until_utc],
    oldest first.

    An unschedulable cadence/time/timezone yields [] (same refuse-rather-
    than-guess posture as core.routine_scheduler._routine_to_cron_trigger).
    """
    cadence = _resolve_cadence(value)
    if cadence is None:
        return []
    tz = _routine_timezone(value)
    if tz is None:
        return []
    parsed = _parse_time(value)
    if cadence == "hourly":
        hours_minutes = [(h, parsed[1] if parsed else 0) for h in range(24)]
    elif parsed is None:
        return []
    else:
        hours_minutes = [parsed]
    if until_utc <= after_utc:
        return []

    # +/- 1 day of slack on the local date range so a timezone whose local
    # date differs from the UTC date at either edge is still covered.
    first_date = after_utc.astimezone(tz).date() - timedelta(days=1)
    last_date = until_utc.astimezone(tz).date() + timedelta(days=1)
    out: list[Occurrence] = []
    day = first_date
    while day <= last_date:
        for hh, mm in hours_minutes:
            wall = datetime(day.year, day.month, day.day, hh, mm)
            if not _day_matches(cadence, value, wall):
                continue
            fire_utc, in_gap = _resolve_wall_time(wall, tz)
            if in_gap and cadence == "hourly":
                continue
            if after_utc < fire_utc <= until_utc:
                out.append(Occurrence(f"{wall:%Y-%m-%dT%H:%M}", fire_utc))
        day += timedelta(days=1)
    out.sort(key=lambda o: o.fire_utc)
    return out


def fire_instant_for_bucket(value: dict[str, Any], fire_bucket: str) -> Optional[datetime]:
    """The UTC instant a previously-issued fire_bucket was due, or None when
    the routine's timezone or the bucket string is unusable. Used to give a
    re-fired claim the same deterministic fired_at as its first attempt."""
    tz = _routine_timezone(value)
    if tz is None:
        return None
    try:
        wall = datetime.strptime(fire_bucket, "%Y-%m-%dT%H:%M")
    except ValueError:
        return None
    return _resolve_wall_time(wall, tz)[0]


def compute_due_routines(
    routines: dict[str, Any], *, after_utc: datetime, until_utc: datetime
) -> list[DueOccurrence]:
    """routines: {routine_key: MemoryEvent}, already filtered to
    applied-and-not-rejected by the caller (see
    core.routine_scheduler.get_active_routines_for_user + apps/cron_tick_routes.py).
    Returns a DueOccurrence for every occurrence of every routine in
    (after_utc, until_utc], oldest first.
    """
    due: list[DueOccurrence] = []
    for key, event in routines.items():
        for occ in occurrences_between(
            event.value, after_utc=after_utc, until_utc=until_utc
        ):
            due.append(DueOccurrence(key, event, occ.fire_bucket, occ.fire_utc))
    due.sort(key=lambda d: d.fire_utc)
    return due
