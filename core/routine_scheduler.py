"""
core/routine_scheduler.py
-------------------------
Phase 4 / E1+E2: APScheduler-based in-process routine firing.

Reads applied workflow.{morning_routine,daily_briefing,recurring_request.*}
events from each user's journal, parses cadence + time + timezone, and
registers an APScheduler CronTrigger per routine. On fire, appends a
`workflow.scheduled_fire` event to the user's journal so the agent picks
up the trigger as context on the user's next turn.

Delivery channels (email push, voice push, etc.) are out of scope for
Phase 4 — the journal-event handoff is the minimum viable delivery path
the plan calls for.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.memory_journal import JournalStore, MemoryEvent, make_event
from core.paths import DATA_DIR, PERSONAL_MEMORY_DIR
from core.personal_memory_extract import is_routine_key


_log = logging.getLogger(__name__)

_SCHEDULER_DB_PATH = DATA_DIR / "scheduler.sqlite"

_CADENCE_TO_CRON: dict[str, dict[str, str]] = {
    "daily":    {"day_of_week": "*"},
    "weekday":  {"day_of_week": "mon-fri"},
    "weekdays": {"day_of_week": "mon-fri"},
    "weekend":  {"day_of_week": "sat,sun"},
    "weekends": {"day_of_week": "sat,sun"},
    "weekly":   {"day_of_week": "mon"},
    # Hourly + monthly are accepted by the schema but require additional
    # fields; left out until we have a real user case.
}


class RoutineScheduler:
    """Owns a single AsyncIOScheduler shared across all users."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _SCHEDULER_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        jobstore = SQLAlchemyJobStore(url=f"sqlite:///{self._db_path}")
        self._scheduler = AsyncIOScheduler(jobstores={"default": jobstore})
        self._started = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._started:
            return
        self._scheduler.start()
        self._started = True
        registered = self._scan_and_register_all_users()
        print(f"LOG: RoutineScheduler started — {registered} job(s) registered")

    def shutdown(self) -> None:
        if not self._started:
            return
        try:
            self._scheduler.shutdown(wait=False)
        except Exception as e:
            print(f"LOG: RoutineScheduler shutdown error: {e}")
        self._started = False

    # ── Registration ─────────────────────────────────────────────────────

    def _scan_and_register_all_users(self) -> int:
        """Walk every user dir under PERSONAL_MEMORY_DIR and register routines."""
        if not PERSONAL_MEMORY_DIR.exists():
            return 0
        total = 0
        for user_dir in PERSONAL_MEMORY_DIR.iterdir():
            if not user_dir.is_dir():
                continue
            if user_dir.name in ("snapshots",):
                continue
            total += self.register_for_user(user_dir.name)
        return total

    def register_for_user(self, user_id: str) -> int:
        """Scan a single user's journal and register all applied routine events.

        Idempotent: APScheduler uses a stable job_id derived from (user_id, key),
        so re-registering replaces the previous job in the SQLite store.
        """
        try:
            journal = JournalStore(user_id=user_id)
        except Exception as e:
            print(f"LOG: RoutineScheduler skip user {user_id} (journal init failed: {e})")
            return 0

        # Resolve latest event per (topic, key) so a rejected/superseded
        # routine doesn't keep firing.
        latest_by_key: dict[str, MemoryEvent] = {}
        for event in journal.iter_events():
            if event.topic != "workflow" or not is_routine_key(event.key):
                continue
            prev = latest_by_key.get(event.key)
            if prev is None or event.observed_at > prev.observed_at:
                latest_by_key[event.key] = event

        registered = 0
        for key, event in latest_by_key.items():
            if event.rejected or not event.applied:
                self._remove_job(user_id, key)
                continue
            if self._register_event(user_id, event):
                registered += 1
        return registered

    def on_event_applied(self, user_id: str, event: MemoryEvent) -> bool:
        """Live registration hook for a single newly-applied routine event."""
        if event.topic != "workflow" or not is_routine_key(event.key):
            return False
        if event.rejected or not event.applied:
            self._remove_job(user_id, event.key)
            return False
        return self._register_event(user_id, event)

    def _register_event(self, user_id: str, event: MemoryEvent) -> bool:
        trigger = _routine_to_cron_trigger(event.value)
        if trigger is None:
            print(
                f"LOG: RoutineScheduler skip {user_id}/{event.key} — "
                f"unschedulable value {event.value!r}"
            )
            return False
        job_id = _job_id_for(user_id, event.key)
        self._scheduler.add_job(
            _fire_routine,
            trigger=trigger,
            id=job_id,
            args=[user_id, event.key, dict(event.value)],
            replace_existing=True,
            misfire_grace_time=300,
        )
        print(
            f"LOG: RoutineScheduler registered {job_id} "
            f"(cadence={event.value.get('cadence')}, time={event.value.get('time')}, "
            f"tz={event.value.get('timezone')})"
        )
        return True

    def _remove_job(self, user_id: str, key: str) -> None:
        job_id = _job_id_for(user_id, key)
        try:
            self._scheduler.remove_job(job_id)
            print(f"LOG: RoutineScheduler removed {job_id}")
        except Exception:
            pass

    # ── Introspection (test/debug) ───────────────────────────────────────

    def list_jobs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for job in self._scheduler.get_jobs():
            out.append({
                "id": job.id,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
            })
        return out


# ────────────────────────────────────────────────────────────────────────
# Module-level helpers — top-level so APScheduler can pickle job refs
# across restarts via the SQLAlchemy jobstore.
# ────────────────────────────────────────────────────────────────────────

def _job_id_for(user_id: str, key: str) -> str:
    safe_key = key.replace("/", "_").replace(":", "_")
    return f"routine::{user_id}::{safe_key}"


def _routine_to_cron_trigger(value: dict[str, Any]) -> CronTrigger | None:
    cadence = str(value.get("cadence") or "daily").strip().lower()
    time_str = value.get("time")
    tz = value.get("timezone") or None

    cron_kwargs = dict(_CADENCE_TO_CRON.get(cadence, _CADENCE_TO_CRON["daily"]))

    if isinstance(time_str, str) and ":" in time_str:
        try:
            hh, mm = time_str.split(":", 1)
            cron_kwargs["hour"] = str(int(hh))
            cron_kwargs["minute"] = str(int(mm))
        except Exception:
            return None
    else:
        # No clock time — can't schedule a cron without one.
        return None

    if tz:
        cron_kwargs["timezone"] = tz

    try:
        return CronTrigger(**cron_kwargs)
    except Exception as e:
        print(f"LOG: invalid cron spec {cron_kwargs!r}: {e}")
        return None


def _fire_routine(user_id: str, routine_key: str, value: dict[str, Any]) -> None:
    """APScheduler job body — runs at the cron tick.

    Writes a `workflow.scheduled_fire` event into the user's journal so the
    agent surfaces it on the user's next turn (via the workflow.md snapshot
    + memory context). Real-time channel delivery (email/push) is layered on
    top of this in a follow-up.
    """
    fired_at = datetime.now(UTC).isoformat()
    fire_key = f"workflow.scheduled_fire.{routine_key}"
    fire_value = {
        "source_routine": routine_key,
        "fired_at": fired_at,
        "routine": value.get("routine"),
        "items": value.get("items") or [],
        "cadence": value.get("cadence"),
        "time": value.get("time"),
        "timezone": value.get("timezone"),
    }
    try:
        journal = JournalStore(user_id=user_id)
        event = make_event(
            kind="behavior",
            topic="workflow",
            key=fire_key,
            value=fire_value,
            confidence=1.0,
            source="synthesized",
            extractor="dream_pass",  # closest valid extractor label
            session_id=f"scheduler_{fired_at[:10]}",
            turn_id=f"scheduler_{fired_at}",
            applied=False,
        )
        journal.append(event)
        print(f"LOG: routine fired user={user_id} key={routine_key} event_id={event.event_id}")
    except Exception as e:
        print(f"LOG: routine fire failed user={user_id} key={routine_key}: {e}")
