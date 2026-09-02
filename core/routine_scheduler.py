"""
core/routine_scheduler.py
-------------------------
Phase 4 / E1+E2 + Phase 5 (W2): APScheduler-based in-process routine firing.

Reads applied workflow.{morning_routine,daily_briefing,recurring_request.*}
events from each user's journal, parses cadence + time + timezone, and
registers an APScheduler CronTrigger per routine.

On fire (_fire_routine, which runs on a ThreadPoolExecutor worker thread), two
things happen, in order:

  1. Journal append (source of truth). A `workflow.scheduled_fire.<key>` event
     is written with applied=False. NOTE: this event is an AUDIT record only —
     it is NOT surfaced by any memory read path. The replayer renders
     applied-only, the sqlite search defaults to applied_only, and nothing
     reads scheduled_fire. The old docstring's claim that the agent "surfaces it
     on the user's next turn" was fiction; there is no such consumer.

  2. Live delivery (Phase 5 / W2, best-effort). A routine WS frame is pushed to
     the user's open socket(s) via the server's deliver_routine_notice (imported
     lazily through _delivery_hook so this module never hard-depends on
     apps.turtle_server — tests drive _fire_routine standalone). If the user has
     no live socket, the frame is queued server-side and drained on their next
     connect. This live push is the ONLY real delivery of a routine fire.

The journal write is first and unconditional; a delivery failure (import error,
degraded server module, send error) only logs and never affects step 1.
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

# Single registry of cadences the scheduler knows how to translate into a cron
# trigger. A cadence absent from this map is *unschedulable* — the trigger
# builder returns None rather than silently firing it every day (see
# _routine_to_cron_trigger). hourly/monthly carry distinctive base fields that
# the builder augments with the routine's clock time / day-of-month.
_CADENCE_TO_CRON: dict[str, dict[str, str]] = {
    "daily":    {"day_of_week": "*"},
    "weekday":  {"day_of_week": "mon-fri"},
    "weekdays": {"day_of_week": "mon-fri"},
    "weekend":  {"day_of_week": "sat,sun"},
    "weekends": {"day_of_week": "sat,sun"},
    "weekly":   {"day_of_week": "mon"},
    "hourly":   {"hour": "*"},   # every hour; minute filled from the clock time (else :00)
    "monthly":  {"day": "1"},    # 1st of month by default; overridable via value["day"]
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
    cadence_raw = value.get("cadence")
    time_str = value.get("time")
    tz = value.get("timezone") or None

    # Missing cadence entirely keeps the schema renderer's documented default
    # (daily). A *present but unknown* cadence is a different case, handled below.
    if cadence_raw is None or (isinstance(cadence_raw, str) and not cadence_raw.strip()):
        cadence = "daily"
    else:
        cadence = str(cadence_raw).strip().lower()

    mapping = _CADENCE_TO_CRON.get(cadence)
    if mapping is None:
        # Unknown cadence: do NOT silently fall back to daily. A routine the user
        # asked for "every quarter" firing every morning is worse than not firing
        # — skip it loudly so the misconfiguration is visible in the logs.
        print(f"LOG: routine has unschedulable cadence {cadence!r}; skipping")
        return None
    cron_kwargs = dict(mapping)

    # Parse the clock time once (HH:MM). A malformed value is a refuse, not a
    # guess — better no fire than a fire at the wrong time.
    hh: int | None = None
    mm: int | None = None
    if isinstance(time_str, str) and ":" in time_str:
        try:
            hh_s, mm_s = time_str.split(":", 1)
            hh, mm = int(hh_s), int(mm_s)
        except Exception:
            return None

    if cadence == "hourly":
        # Fires every hour at a fixed minute; the clock *hour* is irrelevant, so
        # a routine with no time is still schedulable (defaults to :00).
        cron_kwargs["minute"] = str(mm if mm is not None else 0)
    else:
        # Every other cadence pins to a wall-clock time; without one we cannot
        # build a cron trigger at all.
        if hh is None or mm is None:
            return None
        cron_kwargs["hour"] = str(hh)
        cron_kwargs["minute"] = str(mm)
        if cadence == "monthly":
            # Honor an explicit day-of-month when the value carries one; the
            # mapping already defaults to the 1st.
            day = value.get("day") or value.get("day_of_month")
            if day is not None:
                try:
                    cron_kwargs["day"] = str(int(day))
                except Exception:
                    pass  # unparseable day → keep the default 1st-of-month

    if tz:
        cron_kwargs["timezone"] = tz

    try:
        return CronTrigger(**cron_kwargs)
    except Exception as e:
        print(f"LOG: invalid cron spec {cron_kwargs!r}: {e}")
        return None


def _humanize_routine_key(routine_key: str) -> str:
    """Fallback display name from a key: 'workflow.morning_routine' → 'morning routine'."""
    tail = routine_key.rsplit(".", 1)[-1]
    return tail.replace("_", " ").strip() or routine_key


def _build_routine_frame(routine_key: str, value: dict[str, Any], fired_at: str) -> dict[str, Any]:
    """Build the WS frame the browser renders as a (non-error) routine toast.

    Shape mirrors the storage-cap notice frame: type/code/message, plus routine
    metadata. Handled by websocket.js case 'routine' → showToast(msg.message).
    """
    name = value.get("routine") or _humanize_routine_key(routine_key)
    items = value.get("items") or []
    message = f"⏰ Routine: {name}"
    if items:
        shown = ", ".join(str(i) for i in items[:3])
        if len(items) > 3:
            shown += f" (+{len(items) - 3} more)"
        message += f" — {shown}"
    return {
        "type": "routine",
        "code": "routine_fire",
        "message": message,
        "routine_key": routine_key,
        "fired_at": fired_at,
    }


def _default_delivery_hook(user_id: str, frame: dict[str, Any]) -> bool:
    """Bridge a routine frame to the server's live-socket delivery.

    Imported LAZILY so this module carries no import-time dependency on
    apps.turtle_server (tests exercise _fire_routine standalone; the server is
    also a much heavier import). deliver_routine_notice itself never raises.
    """
    from apps.turtle_server import deliver_routine_notice
    return deliver_routine_notice(user_id, frame)


# Indirection seam for testability: production points at the lazy server bridge;
# tests inject a fake to assert _fire_routine calls delivery with the right frame
# without importing the server module.
_delivery_hook = _default_delivery_hook


def _fire_routine(user_id: str, routine_key: str, value: dict[str, Any]) -> None:
    """APScheduler job body — runs at the cron tick (on a worker thread).

    Two ordered steps (see the module docstring for the full contract):

      1. Journal append — the SOURCE OF TRUTH. Writes a
         `workflow.scheduled_fire.<key>` event with applied=False. This is an
         AUDIT record: it is NOT surfaced by any memory read path (replayer /
         sqlite search are applied-only; nothing consumes scheduled_fire).

      2. Live delivery — best-effort. Builds a routine WS frame and hands it to
         the delivery hook (the server's live-socket push, with a pending-queue
         fallback drained on the user's next connect). This is the ONLY real
         delivery of a fire.

    The journal write is first and unconditional. A delivery failure (import
    error, degraded server module, send error) only logs and never affects the
    journal write.
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
            extractor="scheduler",  # honest provenance (was mislabeled "dream_pass")
            session_id=f"scheduler_{fired_at[:10]}",
            turn_id=f"scheduler_{fired_at}",
            applied=False,
        )
        journal.append(event)
        print(f"LOG: routine fired user={user_id} key={routine_key} event_id={event.event_id}")
    except Exception as e:
        # Journal write is the source of truth; if it failed there is nothing to
        # deliver. Do not attempt a push on a failed fire.
        print(f"LOG: routine fire failed user={user_id} key={routine_key}: {e}")
        return

    # Step 2: best-effort live delivery. Strictly additive — wrapped so a failed
    # import or a degraded server module only logs and never re-raises into the
    # scheduler (which would surface as an unhandled job error).
    try:
        frame = _build_routine_frame(routine_key, value, fired_at)
        _delivery_hook(user_id, frame)
    except Exception as e:
        print(f"LOG: routine delivery skipped user={user_id} key={routine_key}: {e}")
