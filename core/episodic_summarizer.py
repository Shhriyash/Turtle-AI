from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class EpisodicChunk:
    summary: str
    topics: list[str]
    turn_id_range: tuple[int, int]
    timestamp: str


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _extract_keywords(text: str, limit: int = 6) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "have",
        "has",
        "was",
        "were",
        "you",
        "your",
        "about",
        "into",
        "they",
        "them",
        "their",
        "what",
        "when",
        "where",
        "which",
        "why",
        "how",
        "there",
        "here",
        "been",
        "will",
        "would",
        "could",
        "should",
        "then",
        "than",
        "also",
        "just",
        "user",
        "assistant",
        "tool",
        "call",
        "return",
        "message",
    }
    counts: dict[str, int] = {}
    for word in words:
        if word in stop:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [item[0] for item in ranked[:limit]]


def _build_fallback_summary(turn_records: list[dict[str, Any]]) -> tuple[str, list[str]]:
    lines: list[str] = []
    for record in turn_records:
        kind = str(record.get("kind", "")).strip().lower()
        content = str(record.get("content", "")).strip()
        if not content:
            continue
        if kind in {"tool_call", "tool_return", "retry"}:
            tool_name = str(record.get("tool_name", "")).strip()
            label = f"{kind}:{tool_name}" if tool_name else kind
            content = _truncate(content, 160)
            lines.append(f"{label} {content}")
            continue
        if kind in {"user", "assistant"}:
            lines.append(f"{kind}: {_truncate(content, 200)}")
        if len(lines) >= 12:
            break

    summary_text = " ".join(lines).strip()
    if not summary_text:
        summary_text = "Session activity summary unavailable."
    summary_text = _truncate(summary_text, 300)
    topics = _extract_keywords(summary_text)
    return summary_text, topics


def _normalize_turn_records(turn_records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    total_chars = 0
    for record in turn_records:
        kind = str(record.get("kind", "")).strip().lower()
        content = str(record.get("content", "")).strip()
        if not content:
            continue
        tool_name = str(record.get("tool_name", "")).strip()
        if kind in {"tool_call", "tool_return", "retry"}:
            content = _truncate(content, 240)
        else:
            content = _truncate(content, 400)
        prefix = kind or "event"
        if tool_name:
            prefix = f"{prefix}:{tool_name}"
        line = f"{prefix} {content}"
        lines.append(line)
        total_chars += len(line)
        if len(lines) >= 40 or total_chars > 5000:
            break
    return lines


def _build_fallback_bullets(turn_records: list[dict[str, Any]]) -> list[str]:
    lines = _normalize_turn_records(turn_records)
    bullets: list[str] = []
    for line in lines[:8]:
        bullet = _truncate(line, 160)
        if bullet:
            bullets.append(bullet)
    return bullets


async def summarize_rolling_window(
    turn_records: list[dict[str, Any]],
    *,
    model: str | None = None,
) -> list[str]:
    """Summarize a window into 5-8 short bullet points."""
    if not turn_records:
        return []

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2")
    if not api_key:
        return _build_fallback_bullets(turn_records)

    lines = _normalize_turn_records(turn_records)
    if not lines:
        return _build_fallback_bullets(turn_records)

    prompt = (
        "Summarize this window into 5 to 8 short bullet points. "
        "Return JSON with key bullets (array of strings).\n\n"
        "Conversation:\n"
        + "\n".join(lines)
    )

    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=api_key)
        response = await client.chat.completions.create(
            model=model or "llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        payload = json.loads(raw)
        bullets_raw = payload.get("bullets", [])
        if not isinstance(bullets_raw, list):
            bullets_raw = []
        bullets = [str(item).strip() for item in bullets_raw if str(item).strip()]
        bullets = bullets[:8]
        if len(bullets) < 5:
            fallback = _build_fallback_bullets(turn_records)
            if fallback:
                return fallback[:8]
        return bullets
    except Exception:
        return _build_fallback_bullets(turn_records)


async def summarize_window(
    turn_records: list[dict[str, Any]],
    *,
    model: str | None = None,
) -> EpisodicChunk:
    """Summarize a window of turn records into an episodic chunk."""
    turn_count = len(turn_records)
    turn_range = (0, max(0, turn_count - 1))
    timestamp = _utc_now()

    if not turn_records:
        return EpisodicChunk(
            summary="Session activity summary unavailable.",
            topics=[],
            turn_id_range=turn_range,
            timestamp=timestamp,
        )

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2")
    if not api_key:
        summary_text, topics = _build_fallback_summary(turn_records)
        return EpisodicChunk(
            summary=summary_text,
            topics=topics,
            turn_id_range=turn_range,
            timestamp=timestamp,
        )

    lines = _normalize_turn_records(turn_records)
    if not lines:
        summary_text, topics = _build_fallback_summary(turn_records)
        return EpisodicChunk(
            summary=summary_text,
            topics=topics,
            turn_id_range=turn_range,
            timestamp=timestamp,
        )

    prompt = (
        "Summarize this conversation window for retrieval. "
        "Return JSON with keys: summary (<=300 chars) and topics (3-6 short phrases).\n\n"
        "Conversation:\n"
        + "\n".join(lines)
    )

    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=api_key)
        response = await client.chat.completions.create(
            model=model or "llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        payload = json.loads(raw)
        summary_raw = str(payload.get("summary", "")).strip()
        topics_raw = payload.get("topics", [])
        if not isinstance(topics_raw, list):
            topics_raw = []
        topics = [str(item).strip() for item in topics_raw if str(item).strip()][:6]
        if not summary_raw:
            summary_raw, topics = _build_fallback_summary(turn_records)
        summary_text = _truncate(summary_raw, 300)
        if not topics:
            topics = _extract_keywords(summary_text)
        return EpisodicChunk(
            summary=summary_text,
            topics=topics,
            turn_id_range=turn_range,
            timestamp=timestamp,
        )
    except Exception:
        summary_text, topics = _build_fallback_summary(turn_records)
        return EpisodicChunk(
            summary=summary_text,
            topics=topics,
            turn_id_range=turn_range,
            timestamp=timestamp,
        )
