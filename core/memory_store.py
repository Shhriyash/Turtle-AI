from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.graph_store import GraphContextQuery, GraphStore


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class MemoryCheckpointResult:
    triggered: bool
    reason: str = ""
    episode_id: str | None = None


class MemoryStore:
    """Canonical persistent memory built on JSON/JSONL files."""

    EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    def __init__(
        self,
        *,
        profile_path: Path,
        events_path: Path,
        episodes_path: Path,
        state_path: Path,
        graph_store: GraphStore,
        flush_turns: int,
        flush_tokens: int,
        profile_max_lines: int,
    ) -> None:
        self.profile_path = profile_path
        self.events_path = events_path
        self.episodes_path = episodes_path
        self.state_path = state_path
        self.graph_store = graph_store
        self.flush_turns = max(1, flush_turns)
        self.flush_tokens = max(1000, flush_tokens)
        self.profile_max_lines = max(1, profile_max_lines)
        self._ensure_files()

    def record_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_text: str,
        assistant_text: str,
        task_type: str,
    ) -> MemoryCheckpointResult:
        events = self._extract_events(
            session_id=session_id,
            turn_id=turn_id,
            user_text=user_text,
            task_type=task_type,
        )
        self.append_events(events)

        state = self.load_state()
        state["turns_since_flush"] = int(state.get("turns_since_flush", 0)) + 1
        state["tokens_since_flush"] = int(state.get("tokens_since_flush", 0)) + estimate_tokens(user_text) + estimate_tokens(assistant_text)
        self.save_state(state)

        should_flush = (
            state["turns_since_flush"] >= self.flush_turns
            or state["tokens_since_flush"] >= self.flush_tokens
        )
        if not should_flush:
            return MemoryCheckpointResult(triggered=False)
        return self.checkpoint(
            session_id=session_id,
            reason="threshold",
            latest_user=user_text,
            latest_assistant=assistant_text,
        )

    def record_task_outcome(
        self,
        *,
        session_id: str,
        turn_id: str,
        task_type: str,
        summary: str,
        success: bool,
    ) -> MemoryCheckpointResult:
        event = self._make_event(
            session_id=session_id,
            turn_id=turn_id,
            kind="task_outcome",
            key=f"task.{task_type}",
            value={"success": success, "summary": summary[:280]},
            confidence=1.0 if success else 0.6,
            source="explicit",
        )
        self.append_events([event])
        return self.checkpoint(
            session_id=session_id,
            reason=f"task_{task_type}_{'success' if success else 'failure'}",
            latest_user="",
            latest_assistant=summary,
        )

    def record_common_recipients(self, *, session_id: str, turn_id: str, recipients: list[str]) -> None:
        events: list[dict[str, Any]] = []
        for recipient in recipients:
            normalized = str(recipient).strip().lower()
            if not normalized:
                continue
            events.append(
                self._make_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    kind="behavior",
                    key="workflow.common_recipient",
                    value={"recipient": normalized},
                    confidence=0.7,
                    source="inferred",
                )
            )
        self.append_events(events)

    def checkpoint(
        self,
        *,
        session_id: str,
        reason: str,
        latest_user: str,
        latest_assistant: str,
    ) -> MemoryCheckpointResult:
        events = self.load_events()
        profile = self._reduce_profile(events)
        self.save_profile(profile)

        graph = self.graph_store.rebuild_from_profile(profile)
        self.graph_store.save_graph(graph)

        episode_id = self._append_episode(
            session_id=session_id,
            reason=reason,
            latest_user=latest_user,
            latest_assistant=latest_assistant,
        )

        state = self.load_state()
        state["turns_since_flush"] = 0
        state["tokens_since_flush"] = 0
        state["last_checkpoint_at"] = _utc_now()
        self.save_state(state)
        return MemoryCheckpointResult(triggered=True, reason=reason, episode_id=episode_id)

    def force_checkpoint(self, *, session_id: str, reason: str) -> MemoryCheckpointResult:
        return self.checkpoint(session_id=session_id, reason=reason, latest_user="", latest_assistant="")

    def get_context_lines(self, *, task_type: str, query: str) -> list[str]:
        profile = self.load_profile()
        lines: list[str] = []
        identity = profile.get("identity", {})
        preferences = profile.get("preferences", {})
        workflow = profile.get("workflow", {})
        tools = profile.get("tool_preferences", {})

        if identity.get("name"):
            lines.append(f"User name: {identity['name']}")
        if task_type == "email" and identity.get("emails"):
            lines.append(f"Known user email: {identity['emails'][0]}")
        if preferences.get("response_style"):
            lines.append(f"Preferred response style: {preferences['response_style']}")
        if preferences.get("humor_level"):
            lines.append(f"Preferred humor level: {preferences['humor_level']}")
        if task_type == "email" and preferences.get("email_tone"):
            lines.append(f"Preferred email tone: {preferences['email_tone']}")
        if task_type == "email" and workflow.get("prefers_draft_before_send") is not None:
            lines.append(f"Draft before send preference: {workflow['prefers_draft_before_send']}")
        if task_type == "email" and workflow.get("common_recipients"):
            lines.append(f"Common recipient: {workflow['common_recipients'][0]}")
        if tools.get("primary_llm"):
            lines.append(f"Preferred model: {tools['primary_llm']}")

        graph_lines = self.graph_store.query_context(
            GraphContextQuery(query=query, task_type=task_type, max_lines=2)
        )
        lines.extend(graph_lines)

        deduped: list[str] = []
        seen: set[str] = set()
        for line in lines:
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(line)
            if len(deduped) >= self.profile_max_lines:
                break
        return deduped

    def load_profile(self) -> dict[str, Any]:
        if not self.profile_path.exists():
            return self._default_profile()
        try:
            payload = json.loads(self.profile_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return self._default_profile()

    def save_profile(self, profile: dict[str, Any]) -> None:
        profile["meta"] = profile.get("meta", {})
        profile["meta"]["updated_at"] = _utc_now()
        profile["meta"]["version"] = 1
        self.profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    def load_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not self.events_path.exists():
            return events
        with self.events_path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    if isinstance(payload, dict):
                        events.append(payload)
                except Exception:
                    continue
        return events

    def append_events(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as file:
            for event in events:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                merged = self._default_state()
                merged.update(payload)
                return merged
        except Exception:
            pass
        return self._default_state()

    def save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _append_episode(
        self,
        *,
        session_id: str,
        reason: str,
        latest_user: str,
        latest_assistant: str,
    ) -> str:
        episode_id = f"ep_{uuid4().hex}"
        summary = latest_assistant.strip()[:420] or "Checkpoint summary"
        if latest_user.strip():
            summary = f"User: {latest_user.strip()[:180]} | Assistant: {summary}"
        payload = {
            "episode_id": episode_id,
            "session_id": session_id,
            "kind": "session_summary",
            "title": f"Checkpoint {reason}",
            "summary": summary,
            "turn_range": [0, 0],
            "tags": [reason],
            "created_at": _utc_now(),
        }
        with self.episodes_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return episode_id

    def _extract_events(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_text: str,
        task_type: str,
    ) -> list[dict[str, Any]]:
        text = user_text.strip()
        lower = text.lower()
        events: list[dict[str, Any]] = []

        if task_type == "email":
            events.append(
                self._make_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    kind="behavior",
                    key="workflow.email_interaction",
                    value={"count": 1},
                    confidence=0.5,
                    source="inferred",
                )
            )

        emails = self.EMAIL_REGEX.findall(text)
        if emails and ("my email" in lower or "my mail" in lower):
            events.append(
                self._make_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    kind="fact",
                    key="identity.email",
                    value={"emails": emails},
                    confidence=1.0,
                    source="explicit",
                )
            )

        name_match = re.search(r"\bmy name is\s+([a-zA-Z][a-zA-Z\s'-]{1,40})", text, flags=re.IGNORECASE)
        if name_match:
            events.append(
                self._make_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    kind="fact",
                    key="identity.name",
                    value={"name": name_match.group(1).strip()},
                    confidence=1.0,
                    source="explicit",
                )
            )

        if "concise" in lower or "brief" in lower or "short response" in lower:
            events.append(
                self._make_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    kind="preference",
                    key="preferences.response_style",
                    value={"response_style": "concise"},
                    confidence=0.9,
                    source="explicit",
                )
            )
        elif "detailed" in lower or "in detail" in lower:
            events.append(
                self._make_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    kind="preference",
                    key="preferences.response_style",
                    value={"response_style": "detailed"},
                    confidence=0.9,
                    source="explicit",
                )
            )

        if "no humor" in lower or "less humor" in lower:
            events.append(
                self._make_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    kind="preference",
                    key="preferences.humor_level",
                    value={"humor_level": "low"},
                    confidence=0.9,
                    source="explicit",
                )
            )
        elif "more humor" in lower:
            events.append(
                self._make_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    kind="preference",
                    key="preferences.humor_level",
                    value={"humor_level": "medium"},
                    confidence=0.8,
                    source="explicit",
                )
            )

        if "draft before" in lower or "ask before send" in lower:
            events.append(
                self._make_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    kind="preference",
                    key="workflow.prefers_draft_before_send",
                    value={"prefers_draft_before_send": True},
                    confidence=0.95,
                    source="explicit",
                )
            )

        if "use groq" in lower or "groq model" in lower:
            events.append(
                self._make_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    kind="preference",
                    key="tool_preferences.primary_llm",
                    value={"primary_llm": "groq/openai-gpt-oss-120b"},
                    confidence=0.95,
                    source="explicit",
                )
            )

        return events

    def _reduce_profile(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        profile = self._default_profile()
        recipient_counter: dict[str, int] = {}
        email_interactions = 0

        for event in events:
            key = str(event.get("key", ""))
            value = event.get("value", {})
            if not isinstance(value, dict):
                continue
            if key == "identity.email":
                emails = value.get("emails", [])
                if isinstance(emails, list):
                    profile["identity"]["emails"] = self._dedupe_list([str(item).strip() for item in emails if str(item).strip()])
            elif key == "identity.name" and value.get("name"):
                profile["identity"]["name"] = str(value["name"]).strip()
            elif key == "preferences.response_style" and value.get("response_style"):
                profile["preferences"]["response_style"] = str(value["response_style"]).strip()
            elif key == "preferences.humor_level" and value.get("humor_level"):
                profile["preferences"]["humor_level"] = str(value["humor_level"]).strip()
            elif key == "preferences.email_tone" and value.get("email_tone"):
                profile["preferences"]["email_tone"] = str(value["email_tone"]).strip()
            elif key == "workflow.prefers_draft_before_send":
                profile["workflow"]["prefers_draft_before_send"] = bool(value.get("prefers_draft_before_send"))
            elif key == "tool_preferences.primary_llm" and value.get("primary_llm"):
                profile["tool_preferences"]["primary_llm"] = str(value["primary_llm"]).strip()
            elif key == "workflow.common_recipient" and value.get("recipient"):
                recipient = str(value["recipient"]).strip().lower()
                if recipient:
                    recipient_counter[recipient] = recipient_counter.get(recipient, 0) + 1
            elif key == "workflow.email_interaction":
                email_interactions += int(value.get("count", 0) or 0)

        if recipient_counter:
            ordered = sorted(recipient_counter.items(), key=lambda item: item[1], reverse=True)
            profile["workflow"]["common_recipients"] = [item[0] for item in ordered[:5]]
        profile["workflow"]["email_interactions"] = email_interactions
        profile["meta"]["updated_at"] = _utc_now()
        return profile

    @staticmethod
    def _make_event(
        *,
        session_id: str,
        turn_id: str,
        kind: str,
        key: str,
        value: dict[str, Any],
        confidence: float,
        source: str,
    ) -> dict[str, Any]:
        return {
            "event_id": f"evt_{uuid4().hex}",
            "session_id": session_id,
            "turn_id": turn_id,
            "kind": kind,
            "key": key,
            "value": value,
            "confidence": confidence,
            "source": source,
            "created_at": _utc_now(),
        }

    def _ensure_files(self) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.profile_path.exists():
            self.save_profile(self._default_profile())
        for path in [self.events_path, self.episodes_path]:
            if not path.exists():
                path.touch()
        if not self.state_path.exists():
            self.save_state(self._default_state())
        if not self.graph_store.graph_path.exists():
            self.graph_store.save_graph(self.graph_store.load_graph())

    @staticmethod
    def _default_profile() -> dict[str, Any]:
        return {
            "identity": {"name": None, "emails": [], "timezone": None},
            "preferences": {"response_style": None, "humor_level": None, "email_tone": None},
            "workflow": {"prefers_draft_before_send": None, "common_recipients": [], "email_interactions": 0},
            "tool_preferences": {"primary_llm": None},
            "meta": {"updated_at": _utc_now(), "version": 1},
        }

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {
            "turns_since_flush": 0,
            "tokens_since_flush": 0,
            "last_checkpoint_at": None,
        }

    @staticmethod
    def _dedupe_list(values: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            output.append(value)
        return output
