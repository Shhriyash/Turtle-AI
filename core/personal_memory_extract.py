from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
import re
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart

from core.memory_extractor import extract_memory_event_specs



_ROUTINE_KEY_PREFIXES = ("workflow.morning_routine", "workflow.daily_briefing", "workflow.recurring_request")
_IANA_TZ_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_+\-]*(?:/[A-Za-z][A-Za-z0-9_+\-]*)+$|^UTC$")
_CLOCK_RE = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")
_VALID_CADENCES = {"daily", "weekly", "weekday", "weekdays", "weekend", "weekends", "monthly", "hourly"}


def is_routine_key(key: str) -> bool:
    return any(key == p or key.startswith(p + ".") or key.startswith(p + ":") for p in _ROUTINE_KEY_PREFIXES)


def _validate_routine_value(value: dict[str, Any]) -> dict[str, Any] | None:
    """D1: validate the extended routine value shape.

    Required: routine (or name) AND cadence.
    Optional but validated when present: time (HH:MM), timezone (IANA-ish).
    items normalized to a list of strings.
    Returns the normalized dict, or None if the shape is unusable.
    """
    if not isinstance(value, dict):
        return None
    routine = value.get("routine") or value.get("name")
    cadence_raw = value.get("cadence") or value.get("frequency")
    if not (isinstance(routine, str) and routine.strip()):
        return None
    if not (isinstance(cadence_raw, str) and cadence_raw.strip()):
        return None
    cadence = cadence_raw.strip().lower()
    if cadence not in _VALID_CADENCES:
        # Unknown cadence: allow through but normalize whitespace; the scheduler
        # in Phase 4 will reject anything it can't translate to a cron trigger.
        cadence = cadence_raw.strip().lower()

    normalized: dict[str, Any] = {"routine": routine.strip(), "cadence": cadence}

    items = value.get("items") or value.get("steps") or []
    if isinstance(items, str):
        items = [items]
    if isinstance(items, list):
        norm_items = [str(i).strip() for i in items if str(i).strip()]
        if norm_items:
            normalized["items"] = norm_items

    time_raw = value.get("time")
    if isinstance(time_raw, str) and time_raw.strip():
        t = time_raw.strip()
        if _CLOCK_RE.match(t):
            # Zero-pad single-digit hours so "8:00" stays "08:00".
            hh, mm = t.split(":")
            normalized["time"] = f"{int(hh):02d}:{mm}"
        # Silently drop invalid clock strings rather than half-store.

    tz_raw = value.get("timezone") or value.get("tz")
    if isinstance(tz_raw, str) and tz_raw.strip():
        tz = tz_raw.strip()
        if _IANA_TZ_RE.match(tz):
            normalized["timezone"] = tz

    return normalized


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
    # Structured payload for keys whose journal event needs a dict (e.g.
    # workflow.morning_routine carries cadence/time/timezone). When set, the
    # candidate→event converter uses this verbatim instead of wrapping the
    # flat descriptor `value` in a single-field object.
    value_struct: dict[str, Any] | None = None


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


async def _extract_with_llm(
    user_text: str,
    *,
    session_id: str | None = None,
    profile: dict[str, Any] | None = None,
) -> list[PersonalMemoryCandidate]:
    """D1: LLM-based memory extraction using groq:llama-3.1-8b-instant.

    Only fires when the deterministic regex path returns nothing or only
    weak_signal candidates.  Uses the memory_extractor.txt prompt for structured
    JSON output (same schema the regex path already produces).
    """
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2")
    if not api_key:
        return []  # Graceful degrade when key is absent

    # Load the memory extractor system prompt
    try:
        from pathlib import Path
        prompt_path = (
            Path(__file__).resolve().parent
            / "system_prompts" / "memory_extractor.txt"
        )
        system_prompt = prompt_path.read_text(encoding="utf-8")
    except Exception:
        system_prompt = (
            "Extract personal facts the user revealed. "
            "Return a JSON array of objects with keys: "
            "topic, key, value, confidence, source, evidence."
        )

    profile_ctx = ""
    if profile:
        try:
            profile_ctx = f"\n\nExisting profile snapshot:\n{json.dumps(profile, ensure_ascii=False)[:800]}"
        except Exception:
            pass

    prompt = f"{system_prompt}{profile_ctx}\n\nConversation message:\n{user_text[:1000]}"

    try:
        from core.config import settings as _cfg
        from groq import AsyncGroq
        client = AsyncGroq(api_key=api_key)
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=_cfg.personal_memory_turn_extractor_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
                response_format={"type": "json_object"},
            ),
            timeout=10.0,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        print(f"LOG: LLM memory extractor failed: {exc}")
        return []

    # primary contract: {"facts": [...]}; bare-array and items kept for tolerance
    try:
        data = json.loads(raw)
        items: list[dict] = data if isinstance(data, list) else data.get("facts", data.get("items", []))
    except Exception:
        return []

    candidates: list[PersonalMemoryCandidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic", "")).strip().lower()
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        confidence_raw = str(item.get("confidence", "inferred")).lower()
        confidence = "confirmed" if confidence_raw in {"confirmed", "high"} else "inferred"
        source_raw = str(item.get("source", "inferred")).lower()
        source = "explicit" if source_raw in {"explicit", "confirmed"} else "inferred"
        evidence = str(item.get("evidence", user_text[:200])).strip()
        if not topic or not key or not value:
            continue
        candidates.append(
            PersonalMemoryCandidate(
                topic=topic,
                key=key,
                value=value,
                line=f"- {key}: {value}",
                overwrite_policy="replace",
                confidence=confidence,
                sensitivity="normal",
                source_session_id=session_id,
                evidence=evidence,
                source=source,
                extraction_source="llm_turn",
            )
        )
    return candidates




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
    elif key in {"workflow.morning_routine", "workflow.daily_briefing"} or key.startswith("workflow.recurring_request"):
        normalized = _validate_routine_value(value)
        if normalized is not None:
            routine = normalized["routine"]
            cadence = normalized["cadence"]
            items_str = ", ".join(normalized.get("items") or [])
            descriptor = f"{routine} ({cadence})"
            if items_str:
                descriptor = f"{descriptor}: {items_str}"
            cand_key = key.split(".", 1)[-1]
            candidates.append(
                PersonalMemoryCandidate(
                    topic="workflow",
                    key=cand_key,
                    value=descriptor,
                    line=f"- Routine: {descriptor}",
                    overwrite_policy="replace",
                    confidence=confidence,
                    sensitivity=sensitivity,
                    source_session_id=session_id,
                    evidence=evidence,
                    source=source,
                    extraction_source=extraction_source,
                    value_struct=normalized,
                )
            )
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


# ---------------------------------------------------------------------------
# Deterministic routine extractor (D1 fix)
# ---------------------------------------------------------------------------
_ROUTINE_INTENT_RE = re.compile(
    r"\b("
    r"every\s+(?P<freq>morning|day|night|evening|afternoon|weekday|weekdays|weekend|weekends|week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|daily|weekly|each\s+(?P<freq2>morning|day|night|evening|afternoon|weekday|weekend|week)"
    r")\b",
    re.IGNORECASE,
)
_ROUTINE_TIME_RE = re.compile(
    r"\bat\s+(?P<hh>[01]?\d|2[0-3])(?::(?P<mm>[0-5]\d))?\s*(?P<ampm>am|pm|a\.m\.|p\.m\.)?\b",
    re.IGNORECASE,
)

_CADENCE_NORMALIZE: dict[str, str] = {
    "morning": "daily", "day": "daily", "night": "daily",
    "evening": "daily", "afternoon": "daily",
    "weekday": "weekday", "weekdays": "weekday",
    "weekend": "weekend", "weekends": "weekend",
    "week": "weekly",
    "monday": "weekly", "tuesday": "weekly", "wednesday": "weekly",
    "thursday": "weekly", "friday": "weekly",
    "saturday": "weekly", "sunday": "weekly",
}

_DEFAULT_HOUR_FOR_PART: dict[str, int] = {
    "morning": 8, "day": 9, "afternoon": 14, "evening": 19, "night": 21,
}


def _slug_action(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return text[:48] or "routine"


def _extract_routine_candidate(
    user_text: str,
    *,
    session_id: str | None,
    profile: dict[str, Any] | None,
    evidence: str,
) -> PersonalMemoryCandidate | None:
    """Detect 'every (morning|day|...) at HH[:MM][am|pm]...' style routines."""
    if not user_text:
        return None
    m_freq = _ROUTINE_INTENT_RE.search(user_text)
    if not m_freq:
        return None
    freq = (m_freq.group("freq") or m_freq.group("freq2") or "").lower()
    if not freq:
        token = m_freq.group(1).lower()
        freq = "day" if token == "daily" else ("week" if token == "weekly" else "")
    cadence = _CADENCE_NORMALIZE.get(freq, "daily")

    hh: int | None = None
    mm: int = 0
    m_time = _ROUTINE_TIME_RE.search(user_text)
    if m_time:
        hh = int(m_time.group("hh"))
        mm = int(m_time.group("mm") or 0)
        ampm = (m_time.group("ampm") or "").lower().replace(".", "")
        if ampm.startswith("p") and hh < 12:
            hh += 12
        elif ampm.startswith("a") and hh == 12:
            hh = 0
    elif freq in _DEFAULT_HOUR_FOR_PART:
        hh = _DEFAULT_HOUR_FOR_PART[freq]

    if hh is None:
        return None  # need a clock time for the scheduler to register

    # Build a name + items from the surrounding action ("send me a daily news brief").
    text_lower = user_text.lower()
    action = re.sub(_ROUTINE_INTENT_RE, " ", text_lower)
    action = re.sub(_ROUTINE_TIME_RE, " ", action)
    action = re.sub(r"\b(please|kindly|hey turtle|turtle|me|i\b)", " ", action)
    action = re.sub(r"\s+", " ", action).strip(" ,.!?")

    if "news" in text_lower or "brief" in text_lower or "digest" in text_lower:
        key_suffix = "daily_briefing"
        routine_name = "daily briefing"
        items = ["news brief"]
    elif freq in {"morning"} and hh < 12:
        key_suffix = "morning_routine"
        routine_name = "morning routine"
        items = [action] if action else []
    else:
        key_suffix = f"recurring_request.{_slug_action(action)}"
        routine_name = action or "recurring task"
        items = [action] if action else []

    tz = "UTC"
    if profile:
        identity = profile.get("identity") if isinstance(profile, dict) else None
        if isinstance(identity, dict):
            tz_val = identity.get("timezone")
            if isinstance(tz_val, str) and tz_val.strip():
                tz = tz_val.strip()

    value_struct = {
        "routine": routine_name,
        "cadence": cadence,
        "time": f"{hh:02d}:{mm:02d}",
        "timezone": tz,
    }
    if items:
        value_struct["items"] = items

    descriptor = f"{routine_name} ({cadence}) {value_struct['time']} {tz}"
    if items:
        descriptor += " : " + ", ".join(items)

    return PersonalMemoryCandidate(
        topic="workflow",
        key=key_suffix,
        value=descriptor,
        line=f"- Routine: {descriptor}",
        overwrite_policy="replace",
        # Routines need explicit user confirmation before they fire on a
        # schedule. Keep confidence at "inferred" so _should_auto_apply_event
        # leaves them pending in the confirmation gate.
        confidence="inferred",
        sensitivity="normal",
        source_session_id=session_id,
        evidence=evidence,
        source="inferred",
        extraction_source="deterministic",
        value_struct=value_struct,
    )


_RELATION_RE = re.compile(
    r"\bmy\s+(?P<rel>best\s+friend|wife|husband|partner|girlfriend|boyfriend|mother|father|mom|dad|brother|sister|son|daughter|boss|manager|cofounder|co-founder|friend|colleague)\s+(?:is\s+|named\s+|called\s+)?(?P<name>[A-Z][\w''.-]+(?:\s+[A-Z][\w''.-]+)?)",
    re.IGNORECASE,
)


def _extract_relation_candidate(
    user_text: str,
    *,
    session_id: str | None,
    evidence: str,
) -> PersonalMemoryCandidate | None:
    m = _RELATION_RE.search(user_text)
    if not m:
        return None
    rel = re.sub(r"\s+", "_", m.group("rel").strip().lower())
    rel = rel.replace("co-founder", "cofounder")
    name = m.group("name").strip().rstrip(".,!?;:")
    if not name or name.lower() in {"me", "you", "him", "her", "them"}:
        return None
    return PersonalMemoryCandidate(
        topic="relations",
        key=rel,
        value=name,
        line=f"- {rel.replace('_', ' ').title()}: {name}",
        overwrite_policy="replace",
        confidence="confirmed",
        sensitivity="normal",
        source_session_id=session_id,
        evidence=evidence,
        source="explicit",
        extraction_source="deterministic",
    )


def extract_memory_candidates_from_messages(
    *,
    message_history: list[ModelMessage],
    session_id: str | None = None,
    profile: dict[str, Any] | None = None,
    mode: str = "deterministic",
) -> list[PersonalMemoryCandidate]:
    """Synchronous deterministic (regex) extraction path.

    D1: When this returns nothing or only weak_signal candidates, callers
    should use extract_memory_candidates_from_messages_async() which falls
    back to the LLM extractor.
    """
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
            # D1 fix: deterministic routine pass — emits a structured workflow
            # candidate when the user describes a cadence + clock time
            # ("every morning at 8am ...").
            routine_candidate = _extract_routine_candidate(
                user_text,
                session_id=session_id,
                profile=profile,
                evidence=user_text,
            )
            if routine_candidate is not None:
                candidates.append(routine_candidate)
            # D1 fix: deterministic relations pass.
            relation_candidate = _extract_relation_candidate(
                user_text, session_id=session_id, evidence=user_text,
            )
            if relation_candidate is not None:
                candidates.append(relation_candidate)

    return _dedupe_candidates(candidates)


async def extract_memory_candidates_from_messages_async(
    *,
    message_history: list[ModelMessage],
    session_id: str | None = None,
    profile: dict[str, Any] | None = None,
) -> list[PersonalMemoryCandidate]:
    """D1: Regex-first extraction with LLM fallback.

    Runs the cheap deterministic path first.  If it returns nothing or only
    weak_signal candidates, fires the LLM extractor (groq:llama-3.1-8b-instant)
    to catch subtler disclosures.

    This is the preferred async entry point for per-turn extraction.
    """
    # Step 1: deterministic path (fast, free)
    candidates = extract_memory_candidates_from_messages(
        message_history=message_history,
        session_id=session_id,
        profile=profile,
    )

    # Step 2: only fire LLM when regex found nothing useful
    strong = [c for c in candidates if c.confidence in {"confirmed", "inferred"}]
    if strong:
        return candidates  # regex found strong signals, skip LLM

    # Collect user text across all messages for LLM pass
    user_texts: list[str] = []
    for message in message_history:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, UserPromptPart):
                continue
            text = _unwrap_user_request(str(part.content))
            if text:
                user_texts.append(text)

    if not user_texts:
        return candidates

    combined_text = " | ".join(user_texts[:3])  # At most 3 turns for LLM
    llm_candidates = await _extract_with_llm(
        combined_text,
        session_id=session_id,
        profile=profile,
    )

    merged = _dedupe_candidates(candidates + llm_candidates)
    return merged


# ---------------------------------------------------------------------------
# Stage B — LLM-based session-level memory extractor (lifted from turtle_voice)
# ---------------------------------------------------------------------------

def _extract_stage_b_json_array(raw_text: str) -> list[dict[str, object]]:
    text = str(raw_text or "").strip()
    if not text:
        return []

    candidates: list[str] = [text]

    fenced = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    bracketed = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if bracketed:
        candidates.append(bracketed.group(0).strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        return [item for item in payload if isinstance(item, dict)]

    return []


def _evidence_supports_value(value: dict, evidence: dict) -> bool:
    """Explicit facts must be grounded in what the user actually said — every leaf
    of the value must be traceable to the evidence, else it's downgraded to
    inferred (→ pending). Normalization-robust: a lightly-canonicalized value
    still counts as grounded (the model rendering the user's "...@gmail" as
    "...@gmail.com", or spacing/casing differences), while a hallucinated value
    (a distinctive token the user never said) does not.

    A verbatim case-insensitive substring passes immediately. Otherwise every
    SIGNIFICANT (>=4 char) token of the value must appear in the evidence,
    ignoring case/punctuation/spacing — short tokens (a trailing "com", articles)
    are not required, so canonicalization doesn't strand a genuine self-disclosure.
    """
    evidence_text = json.dumps(evidence, ensure_ascii=False).lower()
    ev_squashed = re.sub(r"[^a-z0-9]", "", evidence_text)
    leaves = [str(v).strip().lower() for v in value.values() if isinstance(v, (str, int, float)) and str(v).strip()]
    if not leaves:
        return False
    for leaf in leaves:
        if leaf in evidence_text:
            continue
        sig = [t for t in re.split(r"[^a-z0-9]+", leaf) if len(t) >= 4]
        if sig and all(t in ev_squashed for t in sig):
            continue
        return False
    return True


def _stage_b_turns_from_messages(
    message_history: list[ModelMessage], max_turns: int
) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for message in message_history:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if not isinstance(part, UserPromptPart):
                    continue
                # Drop injected memory context before session extraction.
                text = _unwrap_user_request(str(part.content)).strip()
                if not text:
                    continue
                turns.append({"role": "user", "text": text})
        elif isinstance(message, ModelResponse):
            text = str(message.text or "").strip()
            if text:
                turns.append({"role": "assistant", "text": text})

    if max_turns <= 0:
        return turns
    return turns[-max_turns:]


# Request markers: keys the Stage-B model invents when it mis-models a one-off
# request ("give me a quick high-protein breakfast idea") as a durable
# preference — e.g. value {"requested": true}. A value dict whose keys are ALL
# request markers and that carries no substantive text leaf is a transient ask,
# not a standing fact, so we drop it. Deliberately narrow: keyed on this marker
# set, NOT on "value is boolean", so legitimate boolean prefs like
# {"prefers_draft_before_send": true} pass through untouched.
_REQUEST_MARKER_KEYS = frozenset({
    "requested", "request", "asked", "asking", "one_off", "one_time",
    "wants", "want", "query", "ask", "requesting",
})


def _is_request_shaped(value: Any) -> bool:
    """True when a Stage-B value dict encodes a one-off request, not a fact.

    Rejects only dicts whose keys are all request markers AND that carry no
    substantive string content (a real preference would describe the preference
    in text). This precisely kills {"requested": true} while leaving legitimate
    boolean-valued preferences intact.
    """
    if not isinstance(value, dict) or not value:
        return False
    keys = {str(k).strip().lower() for k in value.keys()}
    if not keys <= _REQUEST_MARKER_KEYS:
        return False
    for leaf in value.values():
        if isinstance(leaf, str) and len(leaf.strip()) > 2:
            return False  # descriptive text present — may be a real fact, keep it
    return True


async def run_stage_b_session_extractor(
    state: Any,
    *,
    session_id: str,
    message_history: list[ModelMessage],
    model_settings: dict[str, Any] | None = None,
) -> int:
    """Stage B session-level LLM extractor (shared by voice + web paths).

    Reads config flags from core.config.settings; writes events through
    state.journal_store and queues inferred candidates via state.confirmation_gate.
    Returns number of events written.
    """
    from core.config import settings as _settings
    from core.llm_client import get_google_models, get_groq_model, get_openrouter_models
    from core.memory_journal import ALLOWED_TOPICS, make_event
    from core.memory_schema import decide_write_policy, statement_for

    if not _settings.personal_memory_enabled or not _settings.personal_memory_stage_b_enabled:
        return 0
    if not session_id or not message_history:
        return 0

    stage_b_model = get_groq_model(
        model_name=_settings.personal_memory_stage_b_model,
        settings=model_settings,
    )
    stage_b_models = [m for m in [stage_b_model, *get_google_models(), *get_openrouter_models()] if m is not None]
    if not stage_b_models:
        print(f"LOG: Stage B skipped for {session_id} (Groq unavailable)")
        return 0

    max_turns = _settings.personal_memory_stage_b_max_turns
    max_candidates = _settings.personal_memory_stage_b_max_candidates

    turns = _stage_b_turns_from_messages(message_history, max_turns)
    if not turns:
        return 0

    profile = state.personal_memory_store.load_profile_snapshot()
    prompt = (
        "Extract candidate personal-memory events from this session.\n"
        "Return ONLY JSON array. No prose.\n"
        "Each item must include: kind, topic, key, value, confidence, source, evidence.\n"
        "Rules:\n"
        "- source: explicit when the user stated the fact about THEMSELVES in their "
        "own words (a first-person self-disclosure like \"my email is X\", \"that's my "
        "gmail\", \"call me Y\", \"I live in Z\") — even if said while asking for "
        "something else — and evidence quotes them verbatim; otherwise inferred or synthesized.\n"
        "- confidence in [0,1].\n"
        "- topic in: identity, preferences, workflow, contacts, relations, projects, corrections, working_style, communication_style, tool_preferences, decision_style.\n"
        "- key should be stable dotted path.\n"
        "- value must be an object.\n"
        "- Recurring patterns: when the user describes a habit, routine, or scheduled "
        "request (\"every morning\", \"every day\", \"when I start my day\", \"I like to ... "
        "first\", \"always\", \"usually\", \"daily/weekly\"), emit a workflow event with key "
        "workflow.morning_routine, workflow.daily_briefing, or workflow.recurring_request:<slug>. "
        "Value MUST include what to do, cadence, AND a clock time + IANA timezone when "
        "the user specified one (or one can be inferred from their stated location). Example: "
        "{\"routine\": \"morning briefing\", \"items\": [\"Indore news\"], \"cadence\": \"daily\", "
        "\"time\": \"08:00\", \"timezone\": \"Asia/Kolkata\"}. "
        "time is 24-hour HH:MM. timezone is an IANA name. Omit time/timezone only when the "
        "user genuinely did not state one. Treat these as preferences, not one-off tasks.\n"
        "- Do NOT emit a candidate for a one-off request, question, or task "
        "instruction (\"give me...\", \"show me...\", \"what's...\", \"suggest...\", "
        "\"recommend...\"). Only durable standing facts or preferences the user will "
        "still hold next week. A single request for a breakfast idea is NOT a food "
        "preference.\n"
        f"- Max candidates: {max_candidates}.\n\n"
        f"Current profile snapshot:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
        f"Session turns:\n{json.dumps(turns, ensure_ascii=False, indent=2)}"
    )

    result = None
    last_error: Exception | None = None
    for model in stage_b_models:
        extractor_agent = Agent(
            model,
            deps_type=type(state),
            output_type=str,
            output_retries=1,
            instructions="Return only valid JSON array.",
        )

        try:
            result = await extractor_agent.run(prompt, deps=state)
            break
        except Exception as e:
            last_error = e
            print(f"LOG: Stage B model hop ({e.__class__.__name__}), trying next")

    if result is None:
        print(f"LOG: Stage B skipped for {session_id} (model unavailable: {last_error})")
        return 0

    raw_items = _extract_stage_b_json_array(result.output)
    if not raw_items:
        return 0

    events = []
    for index, item in enumerate(raw_items[:max_candidates]):
        kind = str(item.get("kind", "")).strip().lower()
        topic = str(item.get("topic", "")).strip().lower()
        key = str(item.get("key", "")).strip()
        value = item.get("value", {})
        source = str(item.get("source", "inferred")).strip().lower()
        evidence = item.get("evidence", {})
        try:
            confidence = float(item.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0

        if kind not in {"fact", "preference", "behavior", "correction", "contradiction"}:
            continue
        if topic not in ALLOWED_TOPICS:
            continue
        if not key or not isinstance(value, dict):
            continue
        # Bug A: a one-off request ("give me a quick high-protein breakfast idea")
        # mis-modeled as a preference — value like {"requested": true} with no
        # standing content. Drop it rather than journaling a durable pref the user
        # never actually expressed.
        if _is_request_shaped(value):
            continue
        # D1: validate the routine value shape; drop half-formed entries
        # rather than half-storing them. Non-routine workflow events pass through.
        if topic == "workflow" and is_routine_key(key):
            validated = _validate_routine_value(value)
            if validated is None:
                continue
            value = validated
        if source not in {"inferred", "synthesized", "explicit"}:
            source = "inferred"
        if source == "explicit" and not _evidence_supports_value(value, evidence):
            source = "inferred"
        confidence = max(0.0, min(1.0, confidence))
        if not isinstance(evidence, dict):
            evidence = {"note": str(evidence)}

        payload_for_id = {
            "session": session_id,
            "kind": kind,
            "topic": topic,
            "key": key,
            "value": value,
            "idx": index,
        }
        digest = hashlib.sha1(
            json.dumps(payload_for_id, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        # Auto-apply is now the single registry decision (core.memory_schema).
        # The former stage-B special case auto-applied explicit identity at
        # confidence >= 0.95; decide_write_policy uses >= 0.9 for evidence-
        # supported explicit facts of any topic — an intended widening (the
        # phase1 0.99 case still applies).
        applied = decide_write_policy(
            source=source,
            topic=topic,
            confidence=confidence,
            evidence_supported=_evidence_supports_value(value, evidence),
        ) == "applied"

        events.append(
            make_event(
                event_id=f"stageb_{digest[:20]}",
                kind=kind,
                topic=topic,
                key=key,
                value=value,
                confidence=confidence,
                source=source,
                extractor="llm_turn",
                session_id=session_id,
                turn_id=f"{session_id}_stageb_{index}",
                evidence=evidence,
                applied=applied,
                # Snapshot the projection now so the replayer renders verbatim.
                statement=statement_for(topic, key, value),
            )
        )

    if not events:
        return 0

    state.journal_store.append_many(events)
    queued = 0
    for event in events:
        if event.applied:
            continue
        if state.confirmation_gate.queue_candidate(event):
            queued += 1

    state.personal_memory_store.append_daily_log(
        f"Stage B candidates written: {len(events)} (queued: {queued})",
        session_id=session_id,
    )
    print(f"LOG: Stage B wrote {len(events)} candidates for {session_id} (queued: {queued})")
    return len(events)
