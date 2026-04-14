"""Dream pass (Stage C) — batch LLM review of pending memory candidates.

Called at session end, after Stage B. Trigger conditions (either must be true):
  - pending confirmation-gate candidates >= min_candidates (default 3), OR
  - last pass was >= min_hours ago AND there is at least one pending candidate.

Process:
  1. Take a snapshot of topics/ to snapshots/<iso-ts>/.
  2. Collect pending (applied=False) candidates from the journal.
  3. Ask the LLM (Groq) to decide ``promote|drop`` for each candidate.
  4. Promote: write an applied event that supersedes the candidate.
     Drop: write a contradiction event that supersedes the candidate.
  5. Remove processed candidates from the confirmation-gate pending queue.
  6. Replay to update topic markdown files.
  7. Run sanity checks (shrinkage, index count, duplicate lines).
  8. On failure: write rollback contradiction events for all promoted events,
     then restore topic files from the snapshot.
  9. On success: persist last-pass timestamp.

If the LLM is unavailable, the pass is skipped silently without side effects.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from core.io_atomic import atomic_write_json
from core.memory_journal import JournalStore, MemoryEvent, make_event
from core.memory_replayer import ALL_TOPICS, replay
from core.personal_memory_store import PersonalMemoryStore


DREAM_PASS_MIN_CANDIDATES = 3
DREAM_PASS_MIN_HOURS = 24


@dataclass
class DreamPassResult:
    promoted_count: int = 0
    dropped_count: int = 0
    rolled_back: bool = False
    skipped_reason: str = ""
    sanity_failures: list[str] = field(default_factory=list)


class DreamPass:
    """Offline batch LLM review of pending memory candidates (Stage C).

    This class is intentionally free of voice-loop or LLM-client imports at
    module level so it remains unit-testable in isolation. The model instance
    is injected at ``run()`` time.
    """

    def __init__(
        self,
        *,
        journal: JournalStore,
        store: PersonalMemoryStore,
        confirmation_gate: Any,  # ConfirmationGate — typed Any to avoid circular import
        state_path: Path,
        snapshots_dir: Path,
        min_candidates: int = DREAM_PASS_MIN_CANDIDATES,
        min_hours: int = DREAM_PASS_MIN_HOURS,
    ) -> None:
        self.journal = journal
        self.store = store
        self.confirmation_gate = confirmation_gate
        self.state_path = state_path
        self.snapshots_dir = snapshots_dir
        self.min_candidates = min_candidates
        self.min_hours = min_hours
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_run(self) -> bool:
        """Return True if at least one trigger condition is satisfied.

        Trigger 1: enough pending candidates.
        Trigger 2: enough time has passed since the last pass AND at least
                   one candidate is waiting.
        """
        pending = self.confirmation_gate.pending_count()
        if pending == 0:
            return False
        if pending >= self.min_candidates:
            return True
        last_run_at = self._state.get("last_run_at")
        if last_run_at:
            try:
                last_dt = datetime.fromisoformat(last_run_at.replace("Z", "+00:00"))
                if datetime.now(UTC) - last_dt >= timedelta(hours=self.min_hours):
                    return True
            except Exception:
                # Unparseable timestamp — treat as never-ran, so time trigger fires.
                return True
        return False

    async def run(
        self,
        *,
        session_id: str,
        model: Any = None,
        _decisions_override: list[dict] | None = None,
    ) -> DreamPassResult:
        """Execute a dream pass.

        Parameters
        ----------
        session_id:
            The session that triggered this pass (used for event metadata).
        model:
            A pydantic_ai model instance (e.g. GroqModel). Required unless
            ``_decisions_override`` is supplied.
        _decisions_override:
            Testing hook. If provided, skip the LLM call entirely and use
            this list as the decision set. Each item must be a dict with
            ``event_id`` and ``decision`` (``"promote"`` or ``"drop"``),
            plus an optional ``supersedes_existing`` key.
        """
        result = DreamPassResult()

        if _decisions_override is None and model is None:
            result.skipped_reason = "no_model"
            return result

        # ---- Collect pending candidates --------------------------------
        pending_ids = self.confirmation_gate.get_pending_ids()
        if not pending_ids:
            result.skipped_reason = "no_pending_candidates"
            return result

        all_events = self.journal.load_all()
        event_by_id = {e.event_id: e for e in all_events}
        candidates = [event_by_id[eid] for eid in pending_ids if eid in event_by_id]
        if not candidates:
            result.skipped_reason = "candidates_missing_from_journal"
            return result

        # ---- Snapshot --------------------------------------------------
        snap_dir = _take_snapshot(self.store, self.snapshots_dir)
        before_line_counts = _measure_topic_line_counts(self.store)
        before_index_count = len(self.store.load_index())

        # ---- Get decisions ---------------------------------------------
        if _decisions_override is not None:
            decisions = list(_decisions_override)
        else:
            profile = self.store.load_profile_snapshot()
            decisions = await _call_model_for_decisions(
                model=model,
                candidates=candidates,
                profile=profile,
                session_id=session_id,
            )
            if decisions is None:
                result.skipped_reason = "model_error"
                return result

        if not decisions:
            result.skipped_reason = "no_parseable_decisions"
            return result

        # ---- Build new events from decisions ---------------------------
        existing_applied_ids = {
            e.event_id for e in all_events if e.applied and not e.rejected
        }
        new_events: list[MemoryEvent] = []

        for dec in decisions:
            event_id = str(dec.get("event_id", "")).strip()
            decision = str(dec.get("decision", "")).strip().lower()
            supersedes_existing = dec.get("supersedes_existing")

            candidate = event_by_id.get(event_id)
            if candidate is None:
                continue

            if decision == "promote":
                sup: str | None = None
                if (
                    supersedes_existing
                    and isinstance(supersedes_existing, str)
                    and supersedes_existing in existing_applied_ids
                ):
                    sup = supersedes_existing
                digest = hashlib.sha1(
                    f"dream_promote_{session_id}_{event_id}".encode()
                ).hexdigest()
                new_events.append(
                    make_event(
                        event_id=f"dreamc_{digest[:20]}",
                        kind=candidate.kind,
                        topic=candidate.topic,
                        key=candidate.key,
                        value=candidate.value,
                        confidence=candidate.confidence,
                        source="synthesized",
                        extractor="dream_pass",
                        session_id=session_id,
                        turn_id=f"{session_id}_dreamc",
                        evidence={
                            "dream_pass_decision": "promote",
                            "from_candidate": event_id,
                        },
                        supersedes=sup or candidate.event_id,
                        applied=True,
                    )
                )
                result.promoted_count += 1

            elif decision == "drop":
                digest = hashlib.sha1(
                    f"dream_drop_{session_id}_{event_id}".encode()
                ).hexdigest()
                new_events.append(
                    make_event(
                        event_id=f"dreamd_{digest[:20]}",
                        kind="contradiction",
                        topic=candidate.topic,
                        key=candidate.key,
                        value={
                            "rejected": True,
                            "dream_drop": True,
                            "original_key": candidate.key,
                        },
                        confidence=1.0,
                        source="synthesized",
                        extractor="dream_pass",
                        session_id=session_id,
                        turn_id=f"{session_id}_dreamc",
                        evidence={
                            "dream_pass_decision": "drop",
                            "from_candidate": event_id,
                        },
                        supersedes=candidate.event_id,
                        applied=False,
                    )
                )
                result.dropped_count += 1

        if result.promoted_count == 0 and result.dropped_count == 0:
            result.skipped_reason = "no_valid_decisions"
            return result

        # ---- Persist to journal ----------------------------------------
        self.journal.append_many(new_events)

        # Remove processed candidates from gate pending queue
        processed_ids = {
            str(dec.get("event_id", ""))
            for dec in decisions
            if dec.get("event_id")
        }
        for eid in processed_ids:
            self.confirmation_gate.force_remove_pending(eid)

        # ---- Replay and sanity-check -----------------------------------
        replay(self.journal.load_all(), store=self.store)

        failures = _run_sanity_checks(
            self.store,
            before_line_counts=before_line_counts,
            before_index_count=before_index_count,
        )

        if failures:
            result.rolled_back = True
            result.sanity_failures = list(failures)
            # Undo all promoted events via compensating contradictions.
            _write_rollback_events(
                journal=self.journal,
                promoted_events=[e for e in new_events if e.applied],
                session_id=session_id,
                sanity_failures=failures,
            )
            # Restore topic markdown files from pre-pass snapshot.
            _restore_snapshot(self.store, snap_dir)
            print(f"LOG: Dream pass rolled back for {session_id}: {failures}")
        else:
            self._record_pass(session_id)
            try:
                self.store.append_daily_log(
                    f"Dream pass: promoted={result.promoted_count}, "
                    f"dropped={result.dropped_count}",
                    session_id=session_id,
                )
            except Exception:
                pass
            print(
                f"LOG: Dream pass complete for {session_id}: "
                f"promoted={result.promoted_count}, dropped={result.dropped_count}"
            )

        return result

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _record_pass(self, session_id: str) -> None:
        self._state["last_run_at"] = (
            datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        )
        self._state["pass_count"] = int(self._state.get("pass_count", 0)) + 1
        self._state["last_session_id"] = session_id
        atomic_write_json(self.state_path, self._state)

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

async def _call_model_for_decisions(
    *,
    model: Any,
    candidates: list[MemoryEvent],
    profile: dict[str, Any],
    session_id: str,
) -> list[dict] | None:
    from pydantic_ai import Agent  # local import to keep module importable without pydantic_ai

    candidates_payload = [_candidate_to_payload(c) for c in candidates]
    prompt = (
        "You are reviewing candidate memory events for a personal assistant.\n"
        "Each candidate was inferred from a recent session but not yet confirmed.\n"
        "Decide what to do with each:\n\n"
        '- "promote": accept and apply as persistent memory.\n'
        '- "drop": reject — noise, wrong, or already covered by existing memory.\n\n'
        "If promoting and the candidate clearly supersedes an already-applied memory "
        "entry, set \"supersedes_existing\" to that entry's event_id. Otherwise null.\n\n"
        "Return ONLY a JSON array. No prose. Schema:\n"
        '[{"event_id": "...", "decision": "promote|drop", '
        '"supersedes_existing": "...|null"}]\n\n'
        f"Current applied profile:\n"
        f"{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
        f"Pending candidates:\n"
        f"{json.dumps(candidates_payload, ensure_ascii=False, indent=2)}"
    )

    extractor_agent = Agent(
        model,
        output_type=str,
        output_retries=1,
        instructions="Return only valid JSON array.",
    )
    try:
        agent_result = await extractor_agent.run(prompt)
    except Exception as e:
        print(f"LOG: Dream pass model call failed for {session_id}: {e}")
        return None

    return _parse_decisions(agent_result.output)


def _candidate_to_payload(event: MemoryEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "kind": event.kind,
        "topic": event.topic,
        "key": event.key,
        "value": event.value,
        "confidence": event.confidence,
        "source": event.source,
        "observed_at": event.observed_at,
        "evidence": event.evidence,
    }


def _parse_decisions(raw: str) -> list[dict]:
    """Extract a decision list from raw LLM output. Best-effort JSON extraction."""
    if not raw:
        return []
    text = raw.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    result: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        event_id = str(item.get("event_id", "")).strip()
        decision = str(item.get("decision", "")).strip().lower()
        if not event_id or decision not in {"promote", "drop"}:
            continue
        result.append(
            {
                "event_id": event_id,
                "decision": decision,
                "supersedes_existing": item.get("supersedes_existing"),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _take_snapshot(store: PersonalMemoryStore, snapshots_dir: Path) -> Path:
    """Copy all topic files and MEMORY.md into ``snapshots/<iso-ts>/``."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    snap_dir = snapshots_dir / ts
    snap_dir.mkdir(parents=True, exist_ok=True)
    for path in [store.index_path, *store.topic_paths.values()]:
        if path.exists():
            shutil.copy2(path, snap_dir / path.name)
    return snap_dir


def _restore_snapshot(store: PersonalMemoryStore, snap_dir: Path) -> None:
    """Overwrite topic files with their snapshot copies, deleting any that
    were not present before the pass."""
    for path in [store.index_path, *store.topic_paths.values()]:
        backed = snap_dir / path.name
        if backed.exists():
            shutil.copy2(backed, path)
        elif path.exists():
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _count_bullet_lines(path: Path) -> int:
    """Count lines that start with ``- `` (content lines, skipping frontmatter)."""
    if not path.exists():
        return 0
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- ")
    )


def _measure_topic_line_counts(store: PersonalMemoryStore) -> dict[str, int]:
    """Return bullet-line counts per topic (frontmatter excluded)."""
    counts: dict[str, int] = {}
    for topic in ALL_TOPICS:
        counts[topic] = _count_bullet_lines(store.get_topic_path(topic))
    return counts


def _run_sanity_checks(
    store: PersonalMemoryStore,
    *,
    before_line_counts: dict[str, int],
    before_index_count: int,
) -> list[str]:
    """Return a list of failure messages; empty if all checks pass."""
    failures: list[str] = []

    # 1. No topic file may lose more than 50% of its bullet lines.
    for topic in ALL_TOPICS:
        before = before_line_counts.get(topic, 0)
        if before <= 2:
            continue  # too sparse to make a meaningful comparison
        after = _count_bullet_lines(store.get_topic_path(topic))
        if after < before * 0.5:
            failures.append(
                f"topic '{topic}' shrank too much: {before} -> {after} lines"
            )

    # 2. MEMORY.md index must not drop below 3 entries if it had >= 3 before.
    if before_index_count >= 3:
        after_count = len(store.load_index())
        if after_count < 3:
            failures.append(
                f"MEMORY.md index shrank below 3 entries: now {after_count}"
            )

    # 3. No duplicate bullet lines within any topic file.
    for topic in ALL_TOPICS:
        path = store.get_topic_path(topic)
        if not path.exists():
            continue
        bullets = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("- ")
        ]
        if len(bullets) != len(set(bullets)):
            failures.append(f"topic '{topic}' has duplicate bullet lines")

    return failures


# ---------------------------------------------------------------------------
# Rollback helpers
# ---------------------------------------------------------------------------

def _write_rollback_events(
    *,
    journal: JournalStore,
    promoted_events: list[MemoryEvent],
    session_id: str,
    sanity_failures: list[str],
) -> None:
    """Append journal contradiction events that supersede each promoted event.

    This keeps the journal append-only while ensuring that a subsequent
    ``replay()`` call excludes the rolled-back promotions.
    """
    rollback_events: list[MemoryEvent] = []
    for evt in promoted_events:
        digest = hashlib.sha1(f"rollback_{evt.event_id}".encode()).hexdigest()
        rollback_events.append(
            make_event(
                event_id=f"rollbk_{digest[:20]}",
                kind="contradiction",
                topic=evt.topic,
                key=evt.key,
                value={"rolled_back": True, "from_dream_pass": evt.event_id},
                confidence=1.0,
                source="synthesized",
                extractor="dream_pass",
                session_id=session_id,
                turn_id=f"{session_id}_rollback",
                evidence={
                    "rollback": True,
                    "sanity_failures": sanity_failures,
                },
                supersedes=evt.event_id,
                applied=False,
            )
        )
    if rollback_events:
        journal.append_many(rollback_events)
