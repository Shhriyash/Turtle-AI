"""
Phase 3 — routine scheduler hardening (W4).

Covers the defects fixed in core/routine_scheduler.py:
  (a) _routine_to_cron_trigger cadence mapping — daily/weekdays/weekend/weekly
      plus the newly-supported hourly/monthly, malformed/absent time handling,
      unknown-cadence refusal (no silent daily fallback), and timezone honoring.
  (b) register_for_user against a tmp journal — an applied routine registers a
      job; a later rejected/unapplied event removes it; re-registration is
      idempotent (stable job id, replace_existing).
  (c) _fire_routine appends a scheduled_fire event carrying the corrected
      extractor provenance ("scheduler", not "dream_pass").

Offline: (a) and (c) need no running scheduler. (b) starts a real
AsyncIOScheduler backed by an in-memory jobstore (monkeypatched in place of the
SQLAlchemy/sqlite store) so nothing touches data/scheduler.sqlite, and runs
inside asyncio.run so replace_existing semantics behave like production.
"""
from __future__ import annotations

import asyncio

import pytest
from apscheduler.jobstores.memory import MemoryJobStore

import core.paths as core_paths
import core.routine_scheduler as rs
from core.memory_journal import JournalStore, make_event
from core.routine_scheduler import (
    RoutineScheduler,
    _fire_routine,
    _job_id_for,
    _routine_to_cron_trigger,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _fields(trigger) -> dict[str, str]:
    """Flatten a CronTrigger into {field_name: expression_str}."""
    return {f.name: str(f) for f in trigger.fields}


@pytest.fixture()
def pm_root(tmp_path, monkeypatch):
    """Redirect all per-user memory/journal writes into a tmp dir."""
    root = tmp_path / "pm"
    root.mkdir()
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", root)
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_SNAPSHOTS_DIR", root / "snapshots")
    # routine_scheduler binds PERSONAL_MEMORY_DIR at import time, so its
    # _scan_and_register_all_users reads this module-local copy — patch it too.
    monkeypatch.setattr(rs, "PERSONAL_MEMORY_DIR", root)
    return root


def _routine_event(user_id: str, *, key: str, value: dict, applied: bool, observed_at: str):
    return make_event(
        kind="behavior",
        topic="workflow",
        key=key,
        value=value,
        confidence=1.0,
        source="explicit",
        extractor="llm_turn",
        session_id="s_sched",
        turn_id=f"{key}_{observed_at}",
        applied=applied,
        observed_at=observed_at,
    )


# ── (a) _routine_to_cron_trigger cadence mapping ─────────────────────────

def test_daily_maps_to_every_day_at_time():
    t = _routine_to_cron_trigger({"cadence": "daily", "time": "07:30"})
    f = _fields(t)
    assert f["day_of_week"] == "*"
    assert f["hour"] == "7"
    assert f["minute"] == "30"


def test_weekdays_and_weekend_and_weekly_map_correctly():
    assert _fields(_routine_to_cron_trigger({"cadence": "weekdays", "time": "09:00"}))["day_of_week"] == "mon-fri"
    # singular alias too
    assert _fields(_routine_to_cron_trigger({"cadence": "weekday", "time": "09:00"}))["day_of_week"] == "mon-fri"
    assert _fields(_routine_to_cron_trigger({"cadence": "weekend", "time": "10:00"}))["day_of_week"] == "sat,sun"
    assert _fields(_routine_to_cron_trigger({"cadence": "weekends", "time": "10:00"}))["day_of_week"] == "sat,sun"
    assert _fields(_routine_to_cron_trigger({"cadence": "weekly", "time": "08:00"}))["day_of_week"] == "mon"


def test_hourly_fires_every_hour_at_minute_from_time():
    t = _routine_to_cron_trigger({"cadence": "hourly", "time": "06:45"})
    f = _fields(t)
    assert f["hour"] == "*"
    assert f["minute"] == "45"
    assert f["day_of_week"] == "*"


def test_hourly_without_time_defaults_to_minute_zero():
    # hourly is the one cadence that is schedulable without a clock time.
    t = _routine_to_cron_trigger({"cadence": "hourly"})
    assert t is not None
    f = _fields(t)
    assert f["hour"] == "*"
    assert f["minute"] == "0"


def test_monthly_defaults_to_first_of_month():
    t = _routine_to_cron_trigger({"cadence": "monthly", "time": "09:00"})
    f = _fields(t)
    assert f["day"] == "1"
    assert f["hour"] == "9"
    assert f["minute"] == "0"


def test_monthly_honors_explicit_day_field():
    t = _routine_to_cron_trigger({"cadence": "monthly", "time": "09:00", "day": 15})
    assert _fields(t)["day"] == "15"


def test_unknown_cadence_returns_none_not_daily():
    # The core defect: an unschedulable cadence must NOT silently become daily.
    assert _routine_to_cron_trigger({"cadence": "quarterly", "time": "09:00"}) is None
    assert _routine_to_cron_trigger({"cadence": "fortnightly", "time": "09:00"}) is None


def test_missing_cadence_defaults_to_daily():
    # Absent cadence keeps the schema renderer's documented daily default.
    t = _routine_to_cron_trigger({"time": "09:00"})
    assert t is not None
    assert _fields(t)["day_of_week"] == "*"


def test_no_time_returns_none_for_clock_bound_cadences():
    assert _routine_to_cron_trigger({"cadence": "daily"}) is None
    assert _routine_to_cron_trigger({"cadence": "weekly"}) is None
    assert _routine_to_cron_trigger({"cadence": "monthly"}) is None


def test_malformed_time_returns_none():
    assert _routine_to_cron_trigger({"cadence": "daily", "time": "not-a-time:xx"}) is None


def test_timezone_is_honored():
    t = _routine_to_cron_trigger({"cadence": "daily", "time": "07:00", "timezone": "Asia/Kolkata"})
    assert str(t.timezone) == "Asia/Kolkata"


# ── (b) register_for_user lifecycle ──────────────────────────────────────

def test_register_lifecycle_and_idempotency(pm_root, monkeypatch, tmp_path):
    # Use an in-memory jobstore so nothing touches data/scheduler.sqlite and the
    # started-scheduler replace_existing semantics behave like production.
    monkeypatch.setattr(rs, "SQLAlchemyJobStore", lambda url: MemoryJobStore())

    user_id = "usr_sched"
    key = "workflow.morning_routine"
    value = {"routine": "morning briefing", "cadence": "daily", "time": "07:30", "timezone": "Asia/Kolkata"}

    # Seed one APPLIED routine before the scheduler scans.
    JournalStore(user_id=user_id).append(
        _routine_event(user_id, key=key, value=value, applied=True, observed_at="2026-07-01T00:00:00Z")
    )

    async def _run():
        sched = RoutineScheduler(db_path=tmp_path / "sched.sqlite")
        sched.start()  # scans pm_root, registers the applied routine
        try:
            job_id = _job_id_for(user_id, key)
            jobs = sched.list_jobs()
            assert [j["id"] for j in jobs] == [job_id]
            assert jobs[0]["next_run_time"] is not None

            # Re-register: idempotent — same stable id, still exactly one job.
            sched.register_for_user(user_id)
            assert [j["id"] for j in sched.list_jobs()] == [job_id]

            # A later unapplied (superseding) event must remove the job.
            JournalStore(user_id=user_id).append(
                _routine_event(user_id, key=key, value=value, applied=False, observed_at="2026-07-02T00:00:00Z")
            )
            sched.register_for_user(user_id)
            assert sched.list_jobs() == []
        finally:
            sched.shutdown()

    asyncio.run(_run())


def test_unschedulable_routine_registers_no_job(pm_root, monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "SQLAlchemyJobStore", lambda url: MemoryJobStore())

    user_id = "usr_bad_cadence"
    key = "workflow.morning_routine"
    # Applied, but the cadence cannot be translated to a cron trigger.
    value = {"routine": "quarterly review", "cadence": "quarterly", "time": "09:00"}
    JournalStore(user_id=user_id).append(
        _routine_event(user_id, key=key, value=value, applied=True, observed_at="2026-07-01T00:00:00Z")
    )

    async def _run():
        sched = RoutineScheduler(db_path=tmp_path / "sched.sqlite")
        sched._scheduler.start()
        sched._started = True
        try:
            registered = sched.register_for_user(user_id)
            assert registered == 0
            assert sched.list_jobs() == []
        finally:
            sched.shutdown()

    asyncio.run(_run())


# ── (c) _fire_routine provenance ─────────────────────────────────────────

def test_fire_routine_writes_scheduled_fire_with_scheduler_extractor(pm_root):
    user_id = "usr_fire"
    routine_key = "workflow.morning_routine"
    value = {"routine": "morning briefing", "items": ["Indore news"], "cadence": "daily", "time": "07:30"}

    _fire_routine(user_id, routine_key, value)

    events = JournalStore(user_id=user_id).load_all()
    fires = [e for e in events if e.key == f"workflow.scheduled_fire.{routine_key}"]
    assert len(fires) == 1
    fire = fires[0]
    # The provenance fix: honest "scheduler" label, not the borrowed "dream_pass".
    assert fire.extractor == "scheduler"
    assert fire.source == "synthesized"
    # A scheduled fire is surfaced as context, never auto-applied.
    assert fire.applied is False
    assert fire.value["source_routine"] == routine_key
    assert fire.value["items"] == ["Indore news"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
