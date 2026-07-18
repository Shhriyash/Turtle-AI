"""Periodic reflector — runs Stage B mid-session.

Phase 2 of the memory architecture. Memory writes used to wait until session
archive (taskkill orphans them). The reflector fires every N turns OR after an
idle gap, re-using the existing Stage B pipeline.

Episodic summarization and rolling-window summary are wired in Phases 4 and 6;
this module exposes the hook so they slot in without changing call sites.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from core.config import settings
from core.personal_memory_extract import run_stage_b_session_extractor
from core.episodic_summarizer import summarize_rolling_window


@dataclass
class _SessionState:
    turn_counter: int = 0
    last_reflected_turn: int = 0
    last_activity_at: float = field(default_factory=time.time)
    last_reflected_at: float = 0.0
    consecutive_failures: int = 0
    in_flight: bool = False


class PeriodicReflector:
    """Owns one watermark per active session; fires Stage B + dream pass.

    Watermark is in-memory only — the on-disk archive sweep at startup
    re-runs Stage B for any pending_finalization sessions, so a crash
    between reflections is recovered there.
    """

    def __init__(
        self,
        *,
        every_turns: int | None = None,
        idle_seconds: int | None = None,
        max_consecutive_failures: int | None = None,
    ) -> None:
        self.every_turns = every_turns or settings.reflect_every_turns
        self.idle_seconds = idle_seconds or settings.reflect_idle_seconds
        self.max_consecutive_failures = (
            max_consecutive_failures or settings.reflect_max_consecutive_failures
        )
        self._state: dict[str, _SessionState] = {}

    def _get(self, session_id: str) -> _SessionState:
        st = self._state.get(session_id)
        if st is None:
            st = _SessionState()
            self._state[session_id] = st
        return st

    def reset(self, session_id: str) -> None:
        self._state.pop(session_id, None)

    async def on_turn(
        self,
        state: Any,
        *,
        session_id: str,
        message_history: list[Any],
    ) -> None:
        if not settings.reflect_enabled or not session_id:
            return

        sess = self._get(session_id)
        sess.turn_counter += 1
        now = time.time()
        idle_gap = now - sess.last_activity_at
        sess.last_activity_at = now

        turns_since = sess.turn_counter - sess.last_reflected_turn
        idle_trigger = (
            sess.last_reflected_turn > 0
            and idle_gap >= self.idle_seconds
            and turns_since > 0
        )
        turn_trigger = turns_since >= self.every_turns

        if not (turn_trigger or idle_trigger):
            return
        if sess.in_flight:
            return

        # Fire-and-forget: never block the user turn.
        asyncio.create_task(
            self._reflect(
                state,
                session_id=session_id,
                message_history=list(message_history),
            ),
            name=f"reflect_{session_id}",
        )

    async def _reflect(
        self,
        state: Any,
        *,
        session_id: str,
        message_history: list[Any],
    ) -> None:
        sess = self._get(session_id)
        sess.in_flight = True
        success = False
        try:
            turn_records: list[dict[str, Any]] = []
            if getattr(state, "rag_system", None) is not None:
                try:
                    turn_records = state.rag_system._extract_turn_records_from_messages(  # noqa: SLF001
                        message_history
                    )
                except Exception:
                    turn_records = []
            written = await run_stage_b_session_extractor(
                state,
                session_id=session_id,
                message_history=message_history,
            )

            if turn_records and getattr(state, "rag_system", None) is not None:
                try:
                    await state.rag_system.add_episodic_summary(
                        session_id=session_id,
                        turn_records=turn_records,
                    )
                except Exception as e:
                    print(f"LOG: Reflector episodic summary failed for {session_id}: {e}")

            if turn_records and getattr(state, "session_store", None) is not None:
                window_records = turn_records[-60:]
                try:
                    bullets = await summarize_rolling_window(
                        window_records,
                        model=settings.episodic_summary_model,
                    )
                    if bullets:
                        await state.session_store.append_summary(
                            bullets=bullets,
                            turn_id_range=(
                                max(0, len(turn_records) - len(window_records)),
                                max(0, len(turn_records) - 1),
                            ),
                        )
                except Exception as e:
                    print(f"LOG: Reflector rolling summary failed for {session_id}: {e}")
            success = True
            print(
                f"LOG: Reflector tick session={session_id} "
                f"turn={sess.turn_counter} stage_b_events={written}"
            )
        except Exception as e:
            print(f"LOG: Reflector Stage B failed for {session_id}: {e}")
        finally:
            sess.in_flight = False
            if success:
                sess.last_reflected_turn = sess.turn_counter
                sess.last_reflected_at = time.time()
                sess.consecutive_failures = 0
            else:
                sess.consecutive_failures += 1
                if sess.consecutive_failures >= self.max_consecutive_failures:
                    print(
                        f"LOG: Reflector skip-cap hit for {session_id}; "
                        f"advancing watermark past turn {sess.turn_counter}"
                    )
                    sess.last_reflected_turn = sess.turn_counter
                    sess.consecutive_failures = 0
