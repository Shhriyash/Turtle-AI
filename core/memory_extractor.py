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

_ROLE_KEYWORDS = {
    "engineer",
    "developer",
    "manager",
    "analyst",
    "designer",
    "researcher",
    "architect",
    "consultant",
    "student",
    "founder",
    "teacher",
    "marketer",
    "writer",
    "player",
    "athlete",
    "freelancer",
    "entrepreneur",
    "intern",
    "scientist",
}

_CLAUSE_SPLIT_REGEX = re.compile(
    r"\b(?:and|but|because|so|since|while|though|however|email|mail|contact|reach me|my name is|my timezone|timezone|i need|please)\b",
    flags=re.IGNORECASE,
)

_COUNTRY_HINTS = {
    "india",
    "united states",
    "usa",
    "us",
    "uk",
    "united kingdom",
    "canada",
    "australia",
    "germany",
    "france",
    "singapore",
    "uae",
    "netherlands",
    "japan",
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


def _normalize_phrase(value: str) -> str:
    compact = " ".join(value.replace("\n", " ").split()).strip(" .,!?:;\"'()[]{}")
    compact = re.split(r"[.!?]\s+", compact, maxsplit=1)[0].strip(" .,!?:;\"'()[]{}")
    compact = _CLAUSE_SPLIT_REGEX.split(compact, maxsplit=1)[0].strip(" .,!?:;\"'()[]{}")
    return " ".join(compact.split())


def _to_title_case_phrase(value: str) -> str:
    words = []
    for token in value.split():
        if token.isupper() and len(token) <= 5:
            words.append(token)
            continue
        words.append(token.capitalize())
    return " ".join(words)


def _normalize_timezone(value: str) -> str:
    normalized = _normalize_phrase(value)
    if not normalized:
        return ""
    if re.fullmatch(r"[a-zA-Z]{2,5}", normalized):
        return normalized.upper()
    if "/" in normalized:
        return "/".join(_to_title_case_phrase(part) for part in normalized.split("/"))
    return normalized


def _normalize_language(value: str) -> str:
    normalized = _normalize_phrase(value)
    if not normalized:
        return ""
    if len(normalized.split()) > 3:
        return ""
    return _to_title_case_phrase(normalized)


def _normalize_location(value: str) -> str:
    normalized = _normalize_phrase(value)
    normalized = re.sub(r"^(?:the\s+city\s+of|city\s+of|the\s+state\s+of)\s+", "", normalized, flags=re.IGNORECASE)
    if not normalized:
        return ""
    if len(normalized.split()) > 6:
        return ""
    return _to_title_case_phrase(normalized)


def _looks_like_country(value: str) -> bool:
    normalized = _normalize_phrase(value).lower()
    if not normalized:
        return False
    if normalized in _COUNTRY_HINTS:
        return True
    return bool(re.fullmatch(r"[a-zA-Z][a-zA-Z\s'-]{1,30}", normalized) and len(normalized.split()) <= 3)


def _extract_location_events(user_text: str, lowered: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    home_city = ""
    current_city = ""
    country = ""

    home_patterns = [
        r"\b(?:i am|i['’]?m|im)\s+from\s+([a-zA-Z][a-zA-Z\s,.'-]{1,60})",
        r"\bmy\s+home\s+(?:city|town|place)\s+is\s+([a-zA-Z][a-zA-Z\s,.'-]{1,60})",
    ]
    current_patterns = [
        r"\b(?:i live|i['’]?m\s+living|i am living|i stay|i['’]?m\s+based|i am based)\s+(?:in|at)\s+([a-zA-Z][a-zA-Z\s,.'-]{1,60})",
        r"\b(?:currently\s+in|located\s+in|based\s+in)\s+([a-zA-Z][a-zA-Z\s,.'-]{1,60})",
    ]

    for pattern in home_patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if not match:
            continue
        location = _normalize_location(match.group(1))
        if location:
            home_city = location
            break

    for pattern in current_patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if not match:
            continue
        location = _normalize_location(match.group(1))
        if location:
            current_city = location
            break

    explicit_country_match = re.search(
        r"\b(?:my\s+country\s+is|country\s*[:=])\s*([a-zA-Z][a-zA-Z\s'-]{1,30})",
        user_text,
        flags=re.IGNORECASE,
    )
    if explicit_country_match:
        explicit_country = _normalize_location(explicit_country_match.group(1))
        if _looks_like_country(explicit_country):
            country = explicit_country

    if "," in home_city:
        parts = [segment.strip() for segment in home_city.split(",") if segment.strip()]
        if parts:
            home_city = _normalize_location(parts[0])
        if len(parts) > 1 and not country and _looks_like_country(parts[-1]):
            country = _normalize_location(parts[-1])

    if "," in current_city:
        parts = [segment.strip() for segment in current_city.split(",") if segment.strip()]
        if parts:
            current_city = _normalize_location(parts[0])
        if len(parts) > 1 and not country and _looks_like_country(parts[-1]):
            country = _normalize_location(parts[-1])

    if not country:
        from_match = re.search(
            r"\b(?:i am|i['’]?m|im)\s+from\s+([a-zA-Z][a-zA-Z\s'-]{1,30})\s*$",
            lowered,
            flags=re.IGNORECASE,
        )
        if from_match:
            maybe_country = _normalize_location(from_match.group(1))
            if _looks_like_country(maybe_country):
                country = maybe_country

    if home_city:
        events.append(
            {
                "kind": "fact",
                "key": "identity.home_city",
                "value": {"home_city": home_city},
                "confidence": 0.95,
                "source": "explicit",
                "extraction_source": "deterministic_location_pattern",
                "confidence_class": "high",
            }
        )

    if current_city:
        events.append(
            {
                "kind": "fact",
                "key": "identity.current_city",
                "value": {"current_city": current_city},
                "confidence": 0.95,
                "source": "explicit",
                "extraction_source": "deterministic_location_pattern",
                "confidence_class": "high",
            }
        )

    if country:
        events.append(
            {
                "kind": "fact",
                "key": "identity.country",
                "value": {"country": country},
                "confidence": 0.9,
                "source": "explicit",
                "extraction_source": "deterministic_location_pattern",
                "confidence_class": "high",
            }
        )

    return events


def _extract_identity_detail_events(user_text: str, lowered: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    events.extend(_extract_location_events(user_text, lowered))

    timezone_patterns = [
        r"\b(?:my\s+)?time\s*zone\s*(?:is|=|:)\s*([a-zA-Z0-9_+\-/\s]{2,40})",
        r"\b(?:i am|i['’]?m)\s+in\s+([a-zA-Z]{2,6}(?:/[a-zA-Z_]+)?)\s+time\s*zone\b",
    ]
    for pattern in timezone_patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if not match:
            continue
        timezone = _normalize_timezone(match.group(1))
        if not timezone:
            continue
        events.append(
            {
                "kind": "fact",
                "key": "identity.timezone",
                "value": {"timezone": timezone},
                "confidence": 0.95,
                "source": "explicit",
                "extraction_source": "deterministic_timezone_pattern",
                "confidence_class": "high",
            }
        )
        break

    language_patterns = [
        r"\b(?:my\s+)?preferred\s+language\s*(?:is|=|:)\s*([a-zA-Z][a-zA-Z\s-]{1,30})",
        r"\b(?:reply|respond|speak)\s+(?:to\s+me\s+)?in\s+([a-zA-Z][a-zA-Z\s-]{1,30})",
    ]
    for pattern in language_patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if not match:
            continue
        language = _normalize_language(match.group(1))
        if not language:
            continue
        events.append(
            {
                "kind": "fact",
                "key": "identity.preferred_language",
                "value": {"preferred_language": language},
                "confidence": 0.9,
                "source": "explicit",
                "extraction_source": "deterministic_language_pattern",
                "confidence_class": "high",
            }
        )
        break

    company_match = re.search(
        r"\b(?:i work at|i work for|i['’]?m\s+at|i am at)\s+([a-zA-Z0-9][a-zA-Z0-9&.,'\-\s]{1,60})",
        user_text,
        flags=re.IGNORECASE,
    )
    if company_match:
        company = _normalize_phrase(company_match.group(1))
        if company and len(company.split()) <= 6:
            events.append(
                {
                    "kind": "fact",
                    "key": "identity.company",
                    "value": {"company": company},
                    "confidence": 0.85,
                    "source": "explicit",
                    "extraction_source": "deterministic_company_pattern",
                    "confidence_class": "medium",
                }
            )

    # Explicit role performatives only. The bare copula ("I'm an AI engineer")
    # is left to the LLM extractor — same reasoning as _extract_name: regex
    # can't tell occupation from name/location/mood, so it must not claim a
    # strong signal that short-circuits the LLM.
    role_patterns = [
        r"\b(?:i work as|working as)\s+(?:an?\s+)?([a-zA-Z][a-zA-Z\s/&\-]{2,50})",
    ]
    for pattern in role_patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if not match:
            continue
        role = _normalize_phrase(match.group(1))
        if not role or len(role.split()) > 5:
            continue
        role_lower = role.lower()
        if not any(keyword in role_lower for keyword in _ROLE_KEYWORDS):
            continue
        events.append(
            {
                "kind": "fact",
                "key": "identity.occupation",
                "value": {"occupation": _to_title_case_phrase(role)},
                "confidence": 0.85,
                "source": "explicit",
                "extraction_source": "deterministic_occupation_pattern",
                "confidence_class": "medium",
            }
        )
        break

    return events


def _extract_name(user_text: str) -> str | None:
    # Only explicit name performatives. The bare copula ("I'm X" / "I am X") is
    # semantically ambiguous — name vs. occupation vs. location vs. mood — and
    # cannot be disambiguated by pattern alone, so it is intentionally NOT here.
    # Those statements fall through to the LLM extractor, which classifies the
    # role correctly instead of the regex confidently mislabelling it as a name.
    patterns = [
        r"\bmy name is\s+([a-zA-Z][a-zA-Z\s'-]{1,40})",
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
        # Defense-in-depth for the explicit patterns: "my name is a doctor" /
        # "call me the boss" — a leading article or a role word means it isn't
        # a name. (The ambiguous bare-copula patterns were removed above.)
        if words[0].lower() in {"a", "an", "the"}:
            continue
        if any(word.lower() in _ROLE_KEYWORDS for word in words):
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

    events.extend(_extract_identity_detail_events(text, lower))
    events.extend(_extract_preference_events(lower))
    return events
