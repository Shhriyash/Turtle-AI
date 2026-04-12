from __future__ import annotations

import re
from typing import Any


EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

_NAME_STOPWORDS = {
    "fine",
    "good",
    "great",
    "okay",
    "ok",
    "doing",
    "ready",
    "trying",
    "here",
    "there",
    "busy",
    "available",
    "tired",
}


def _normalize_email(value: str) -> str:
    return value.strip().lower()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(value)
    return output


def _extract_name(user_text: str) -> str | None:
    patterns = [
        r"\bmy name is\s+([a-zA-Z][a-zA-Z\s'-]{1,40})",
        r"\bi am\s+([a-zA-Z][a-zA-Z\s'-]{1,40})",
        r"\bi['’]?m\s+([a-zA-Z][a-zA-Z\s'-]{1,40})",
        r"\bcall me\s+([a-zA-Z][a-zA-Z\s'-]{1,40})",
        r"\bname\s*[:\-]\s*([a-zA-Z][a-zA-Z\s'-]{1,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip(" .,!?:;")
        candidate = re.split(
            r"\b(?:and|but|because|email|mail|reach me|contact me|from)\b",
            candidate,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .,!?:;")
        candidate = " ".join(candidate.split())
        words = candidate.split()
        if not words or len(words) > 4:
            continue
        if any(word.lower() in _NAME_STOPWORDS for word in words):
            continue
        if all(re.fullmatch(r"[A-Za-z][A-Za-z'-]*", word) for word in words):
            return candidate
    return None


def _has_identity_email_intent(lowered: str) -> bool:
    identity_markers = [
        "my email",
        "my mail",
        "email is",
        "mail is",
        "email address",
        "reach me at",
        "contact me at",
        "use this email",
        "save this info about me",
        "about me",
    ]
    return any(marker in lowered for marker in identity_markers)


def _has_email_correction_intent(lowered: str) -> bool:
    correction_markers = [
        "correct it to",
        "correct email",
        "actually",
        "instead",
        "the right email",
        "it is ",
        "it's ",
        "primary email",
        "use this email",
    ]
    return any(marker in lowered for marker in correction_markers)


def _extract_preference_events(lowered: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    concise_markers = [
        "keep responses concise",
        "be concise",
        "concise response",
        "short response",
        "brief response",
        "keep it short",
    ]
    detailed_markers = [
        "be detailed",
        "in detail",
        "detailed response",
        "more detailed",
    ]
    if any(marker in lowered for marker in concise_markers):
        events.append(
            {
                "kind": "preference",
                "key": "preferences.response_style",
                "value": {"response_style": "concise"},
                "confidence": 0.95,
                "source": "explicit",
                "extraction_source": "deterministic_phrase",
                "confidence_class": "high",
            }
        )
    elif any(marker in lowered for marker in detailed_markers):
        events.append(
            {
                "kind": "preference",
                "key": "preferences.response_style",
                "value": {"response_style": "detailed"},
                "confidence": 0.95,
                "source": "explicit",
                "extraction_source": "deterministic_phrase",
                "confidence_class": "high",
            }
        )

    low_humor_markers = ["no humor", "less humor", "avoid humor"]
    med_humor_markers = ["more humor", "add humor"]
    if any(marker in lowered for marker in low_humor_markers):
        events.append(
            {
                "kind": "preference",
                "key": "preferences.humor_level",
                "value": {"humor_level": "low"},
                "confidence": 0.95,
                "source": "explicit",
                "extraction_source": "deterministic_phrase",
                "confidence_class": "high",
            }
        )
    elif any(marker in lowered for marker in med_humor_markers):
        events.append(
            {
                "kind": "preference",
                "key": "preferences.humor_level",
                "value": {"humor_level": "medium"},
                "confidence": 0.85,
                "source": "explicit",
                "extraction_source": "deterministic_phrase",
                "confidence_class": "medium",
            }
        )

    if "formal email" in lowered or "professional email tone" in lowered:
        events.append(
            {
                "kind": "preference",
                "key": "preferences.email_tone",
                "value": {"email_tone": "formal"},
                "confidence": 0.9,
                "source": "explicit",
                "extraction_source": "deterministic_phrase",
                "confidence_class": "high",
            }
        )
    elif "casual email" in lowered or "friendly email tone" in lowered:
        events.append(
            {
                "kind": "preference",
                "key": "preferences.email_tone",
                "value": {"email_tone": "casual"},
                "confidence": 0.9,
                "source": "explicit",
                "extraction_source": "deterministic_phrase",
                "confidence_class": "high",
            }
        )

    if "draft before" in lowered or "ask before send" in lowered:
        events.append(
            {
                "kind": "preference",
                "key": "workflow.prefers_draft_before_send",
                "value": {"prefers_draft_before_send": True},
                "confidence": 0.95,
                "source": "explicit",
                "extraction_source": "deterministic_phrase",
                "confidence_class": "high",
            }
        )

    if "use groq" in lowered or "groq model" in lowered:
        events.append(
            {
                "kind": "preference",
                "key": "tool_preferences.primary_llm",
                "value": {"primary_llm": "groq/openai-gpt-oss-120b"},
                "confidence": 0.95,
                "source": "explicit",
                "extraction_source": "deterministic_phrase",
                "confidence_class": "high",
            }
        )
    elif "use openrouter" in lowered:
        events.append(
            {
                "kind": "preference",
                "key": "tool_preferences.primary_llm",
                "value": {"primary_llm": "openrouter"},
                "confidence": 0.95,
                "source": "explicit",
                "extraction_source": "deterministic_phrase",
                "confidence_class": "high",
            }
        )

    return events


def extract_memory_event_specs(
    *,
    user_text: str,
    task_type: str,
    profile: dict[str, Any] | None,
    mode: str = "deterministic",
) -> list[dict[str, Any]]:
    _ = mode  # hybrid mode reserved for future use
    text = user_text.strip()
    lower = text.lower()
    profile_identity = (profile or {}).get("identity", {})
    profile_emails = [
        _normalize_email(value)
        for value in profile_identity.get("emails", [])
        if isinstance(value, str) and value.strip()
    ]

    events: list[dict[str, Any]] = []
    if task_type == "email":
        events.append(
            {
                "kind": "behavior",
                "key": "workflow.email_interaction",
                "value": {"count": 1},
                "confidence": 0.5,
                "source": "inferred",
                "extraction_source": "task_type_inference",
                "confidence_class": "low",
            }
        )

    name = _extract_name(text)
    if name:
        events.append(
            {
                "kind": "fact",
                "key": "identity.name",
                "value": {"name": name, "correction": False},
                "confidence": 1.0,
                "source": "explicit",
                "extraction_source": "deterministic_name_pattern",
                "confidence_class": "high",
            }
        )

    extracted_emails = [_normalize_email(value) for value in EMAIL_REGEX.findall(text)]
    deduped_emails = _dedupe_preserve_order(extracted_emails)
    has_identity_email_intent = _has_identity_email_intent(lower)
    has_correction_intent = _has_email_correction_intent(lower)

    if deduped_emails and (has_identity_email_intent or has_correction_intent):
        primary_email = deduped_emails[0]
        if "first one" in lower:
            primary_email = deduped_emails[0]
        events.append(
            {
                "kind": "fact",
                "key": "identity.email",
                "value": {
                    "emails": deduped_emails,
                    "primary_email": primary_email,
                    "correction": has_correction_intent,
                },
                "confidence": 1.0 if has_correction_intent else 0.95,
                "source": "explicit",
                "extraction_source": "deterministic_email_pattern",
                "confidence_class": "high",
            }
        )
    elif "first one" in lower and profile_emails:
        events.append(
            {
                "kind": "fact",
                "key": "identity.email",
                "value": {
                    "emails": profile_emails,
                    "primary_email": profile_emails[0],
                    "correction": True,
                },
                "confidence": 0.9,
                "source": "explicit",
                "extraction_source": "deterministic_reference_resolution",
                "confidence_class": "high",
            }
        )

    events.extend(_extract_preference_events(lower))
    return events
