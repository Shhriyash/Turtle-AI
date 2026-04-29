from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from core.memory_extractor import extract_memory_event_specs


def _detect_task_type(user_text: str) -> str:
    lowered = user_text.lower()
    if "email" in lowered or "mail" in lowered:
        return "email"
    if "http://" in lowered or "https://" in lowered:
        return "url"
    if any(token in lowered for token in ["search", "latest", "news", "top ", "price"]):
        return "web"
    return "general"


def _confidence_from_event(event: dict[str, Any]) -> str:
    confidence_class = str(event.get("confidence_class", "")).strip().lower()
    if confidence_class == "high":
        return "confirmed"
    if confidence_class == "medium":
        return "inferred"
    return "weak_signal"


@dataclass(frozen=True)
class PersonalMemoryCandidate:
    topic: str
    key: str
    value: str
    line: str
    overwrite_policy: str
    confidence: str
    sensitivity: str
    source_session_id: str | None
    evidence: str
    source: str
    extraction_source: str


def _unwrap_user_request(user_text: str) -> str:
    text = str(user_text or "").strip()
    if not text:
        return ""

    # Some runtimes wrap user input with memory preamble before it reaches the model.
    # Memory extraction should operate on the original user request only.
    if "Relevant user memory:" in text and "User request:" in text:
        match = re.search(r"User request:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            unwrapped = match.group(1).strip()
            if unwrapped:
                return unwrapped
    return text


def extract_memory_candidates_from_messages(
    *,
    message_history: list[ModelMessage],
    session_id: str | None = None,
    profile: dict[str, Any] | None = None,
    mode: str = "deterministic",
) -> list[PersonalMemoryCandidate]:
    candidates: list[PersonalMemoryCandidate] = []

    for message in message_history:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, UserPromptPart):
                continue
            user_text = _unwrap_user_request(str(part.content))
            if not user_text:
                continue
            task_type = _detect_task_type(user_text)
            events = extract_memory_event_specs(
                user_text=user_text,
                task_type=task_type,
                profile=profile,
                mode=mode,
            )
            for event in events:
                candidates.extend(
                    _event_to_candidates(
                        event=event,
                        evidence=user_text,
                        session_id=session_id,
                    )
                )

    return _dedupe_candidates(candidates)


def _event_to_candidates(
    *,
    event: dict[str, Any],
    evidence: str,
    session_id: str | None,
) -> list[PersonalMemoryCandidate]:
    key = str(event.get("key", "")).strip()
    value = event.get("value", {})
    if not key or not isinstance(value, dict):
        return []

    confidence = _confidence_from_event(event)
    source = str(event.get("source", "unknown")).strip() or "unknown"
    extraction_source = str(event.get("extraction_source", "unknown")).strip() or "unknown"
    sensitivity = "normal"

    candidates: list[PersonalMemoryCandidate] = []

    def add(topic: str, candidate_key: str, candidate_value: str, line: str, overwrite_policy: str) -> None:
        normalized_value = str(candidate_value).strip()
        if not normalized_value:
            return
        candidates.append(
            PersonalMemoryCandidate(
                topic=topic,
                key=candidate_key,
                value=normalized_value,
                line=line,
                overwrite_policy=overwrite_policy,
                confidence=confidence,
                sensitivity=sensitivity,
                source_session_id=session_id,
                evidence=evidence,
                source=source,
                extraction_source=extraction_source,
            )
        )

    if key == "identity.name" and value.get("name"):
        name = str(value["name"]).strip()
        add("identity", "name", name, f"- Name: {name}", "replace")
    elif key == "identity.home_city" and value.get("home_city"):
        home_city = str(value["home_city"]).strip()
        add("identity", "home_city", home_city, f"- Home city: {home_city}", "replace")
    elif key == "identity.current_city" and value.get("current_city"):
        current_city = str(value["current_city"]).strip()
        add("identity", "current_city", current_city, f"- Current city: {current_city}", "replace")
    elif key == "identity.country" and value.get("country"):
        country = str(value["country"]).strip()
        add("identity", "country", country, f"- Country: {country}", "replace")
    elif key == "identity.email":
        primary_email = str(value.get("primary_email", "")).strip().lower()
        if primary_email:
            add("identity", "primary_email", primary_email, f"- Primary email: {primary_email}", "replace")
        for email in value.get("emails", []):
            normalized = str(email).strip().lower()
            if normalized and normalized != primary_email:
                add("identity", f"known_email:{normalized}", normalized, f"- Known email: {normalized}", "append_unique")
    elif key == "identity.timezone" and value.get("timezone"):
        timezone = str(value["timezone"]).strip()
        add("identity", "timezone", timezone, f"- Timezone: {timezone}", "replace")
    elif key == "identity.preferred_language" and value.get("preferred_language"):
        language = str(value["preferred_language"]).strip()
        add("identity", "preferred_language", language, f"- Preferred language: {language}", "replace")
    elif key == "identity.occupation" and value.get("occupation"):
        occupation = str(value["occupation"]).strip()
        add("identity", "occupation", occupation, f"- Occupation: {occupation}", "replace")
    elif key == "identity.company" and value.get("company"):
        company = str(value["company"]).strip()
        add("identity", "company", company, f"- Company: {company}", "replace")
    elif key == "preferences.response_style" and value.get("response_style"):
        response_style = str(value["response_style"]).strip()
        add("preferences", "response_style", response_style, f"- Response style: {response_style}", "replace")
    elif key == "preferences.humor_level" and value.get("humor_level"):
        humor_level = str(value["humor_level"]).strip()
        add("preferences", "humor_level", humor_level, f"- Humor level: {humor_level}", "replace")
    elif key == "preferences.email_tone" and value.get("email_tone"):
        email_tone = str(value["email_tone"]).strip()
        add("preferences", "email_tone", email_tone, f"- Email tone: {email_tone}", "replace")
    elif key == "workflow.prefers_draft_before_send":
        prefers_draft = str(bool(value.get("prefers_draft_before_send"))).lower()
        add("workflow", "prefers_draft_before_send", prefers_draft, f"- Prefers draft before send: {prefers_draft}", "replace")
    elif key == "tool_preferences.primary_llm" and value.get("primary_llm"):
        primary_llm = str(value["primary_llm"]).strip()
        add("workflow", "primary_llm", primary_llm, f"- Preferred primary model: {primary_llm}", "replace")
    elif key == "workflow.common_recipient" and value.get("recipient"):
        recipient = str(value["recipient"]).strip().lower()
        add("contacts", f"frequent_recipient:{recipient}", recipient, f"- Frequent recipient: {recipient}", "append_unique")

    return candidates


def _dedupe_candidates(candidates: list[PersonalMemoryCandidate]) -> list[PersonalMemoryCandidate]:
    ordered: list[PersonalMemoryCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        dedupe_key = (candidate.topic, candidate.key, candidate.value.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        ordered.append(candidate)
    return ordered
