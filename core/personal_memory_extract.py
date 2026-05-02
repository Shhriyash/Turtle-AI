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
        from groq import AsyncGroq
        client = AsyncGroq(api_key=api_key)
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
                response_format={"type": "json_object"},
            ),
            timeout=6.0,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        print(f"LOG: LLM memory extractor failed: {exc}")
        return []

    # Parse the returned JSON array
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
        routine = str(value.get("routine") or value.get("name") or key.split(".", 1)[-1]).strip()
        items = value.get("items") or value.get("steps") or []
        if isinstance(items, str):
            items = [items]
        cadence = str(value.get("cadence") or value.get("frequency") or "daily").strip()
        items_str = ", ".join(str(i).strip() for i in items if str(i).strip())
        if routine:
            descriptor = f"{routine} ({cadence})" if cadence else routine
            if items_str:
                descriptor = f"{descriptor}: {items_str}"
            add("workflow", key.split(".", 1)[-1], descriptor, f"- Routine: {descriptor}", "replace")
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


def _stage_b_turns_from_messages(
    message_history: list[ModelMessage], max_turns: int
) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for message in message_history:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if not isinstance(part, UserPromptPart):
                    continue
                text = str(part.content).strip()
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
    from core.llm_client import get_groq_model
    from core.memory_journal import make_event

    if not _settings.personal_memory_enabled or not _settings.personal_memory_stage_b_enabled:
        return 0
    if not session_id or not message_history:
        return 0

    stage_b_model = get_groq_model(
        model_name=_settings.personal_memory_stage_b_model,
        settings=model_settings,
    )
    if stage_b_model is None:
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
        "- source must be inferred or synthesized (never explicit).\n"
        "- confidence in [0,1].\n"
        "- topic in: identity, preferences, workflow, contacts, projects, corrections.\n"
        "- key should be stable dotted path.\n"
        "- value must be an object.\n"
        "- Recurring patterns: when the user describes a habit, routine, or scheduled "
        "request (\"every morning\", \"every day\", \"when I start my day\", \"I like to ... "
        "first\", \"always\", \"usually\", \"daily/weekly\"), emit a workflow event with key "
        "workflow.morning_routine, workflow.daily_briefing, or workflow.recurring_request:<slug>. "
        "Value should include what to do and cadence, e.g. "
        "{\"routine\": \"morning briefing\", \"items\": [\"city news\", \"AI engineer jobs\"], \"cadence\": \"daily\"}. "
        "Treat these as preferences, not one-off tasks.\n"
        f"- Max candidates: {max_candidates}.\n\n"
        f"Current profile snapshot:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
        f"Session turns:\n{json.dumps(turns, ensure_ascii=False, indent=2)}"
    )

    extractor_agent = Agent(
        stage_b_model,
        deps_type=type(state),
        output_type=str,
        output_retries=1,
        instructions="Return only valid JSON array.",
    )

    try:
        result = await extractor_agent.run(prompt, deps=state)
    except Exception as e:
        print(f"LOG: Stage B skipped for {session_id} (model unavailable: {e})")
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
        if topic not in {"identity", "preferences", "workflow", "contacts", "projects", "corrections"}:
            continue
        if not key or not isinstance(value, dict):
            continue
        if source not in {"inferred", "synthesized"}:
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

        # Phase 1: confidence-tiered auto-apply for non-identity topics
        auto_apply_topics = {"preferences", "workflow", "projects"}
        applied = topic in auto_apply_topics and confidence >= 0.85

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
