"""Retrieval broker — assembles prompt-time memory plus recall-tool lookups.

Prompt-time injection is intentionally small: MEMORY.md index + identity block,
trimmed under a hard token budget. Episodic and task history are now accessed
only through the recall tool.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

from core.personal_memory_prompt import PersonalMemoryPromptBuilder
from core.personal_memory_store import PersonalMemoryStore
from core.task_history import TaskHistoryStore
from core.memory_journal import JournalStore
from core.session_store import SessionStore


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalBudget:
    index_tokens: int = 240
    summary_tokens: int = 150
    total_tokens: int = 390


DEFAULT_BUDGET = RetrievalBudget()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars (consistent with complete_rag.py)."""
    return max(0, len(text) // 4)


def _trim_to_tokens(text: str, max_tokens: int) -> str:
    """Hard-trim *text* to at most *max_tokens* estimated tokens.

    Appends ``...`` when content is cut so the model knows the context is
    incomplete.
    """
    if max_tokens <= 0:
        return ""
    if _estimate_tokens(text) <= max_tokens:
        return text
    max_chars = max_tokens * 4
    # Walk backwards to avoid splitting a multi-byte UTF-8 sequence mid-char.
    truncated = text[:max_chars].rstrip()
    if len(truncated) < len(text):
        truncated += "..."
    return truncated


# History/time trigger words that activate Tier 3 (episodic) and Tier 4 (task).
# The set is intentionally broad to avoid false negatives.
_HISTORY_TRIGGERS: frozenset[str] = frozenset({
    "yesterday",
    "last week",
    "last month",
    "last year",
    "remember",
    "remembered",
    "previously",
    "used to",
    "before",
    "history",
    "past",
    "earlier",
    "when did",
    "what did",
    "did we",
    "did you",
    "told you",
    "told me",
    "we discussed",
    "we talked",
    "you said",
    "i said",
    "last time",
    "few days ago",
    "week ago",
    "month ago",
    "a while ago",
    "some time ago",
    "recently told",
})


def _has_history_trigger(query: str) -> bool:
    """Return True if *query* contains any history/time trigger phrase."""
    lowered = query.lower()
    return any(trigger in lowered for trigger in _HISTORY_TRIGGERS)


# Filler the model prepends ("do you remember", "tell me", "what is"...) — these
# never appear in stored memory lines, so they kill the substring match.
_RECALL_QUERY_FILLER_RE = re.compile(
    r"^(?:do\s+you\s+remember|do\s+you\s+know|can\s+you\s+tell\s+me|tell\s+me|"
    r"what\s+is|what's|whats|who\s+is|who's|whos|recall|remember)\s+",
    re.IGNORECASE,
)

# Possessives the model adds because the system prompt taught it to phrase
# queries from the assistant's perspective ("user's <noun>", "my <noun>",
# "your <noun>"). Stored memory lines never contain these tokens.
_RECALL_QUERY_POSSESSIVE_RE = re.compile(
    r"\b(?:the\s+user'?s?|user'?s?|my|your|their|his|her)\s+",
    re.IGNORECASE,
)


def _normalize_recall_query(query: str) -> str:
    """Lowercase + strip natural-language filler + strip possessives.

    The personal-memory lexical search compares the query as a substring of
    each topic line. If the model emits "user's best friend" but the line is
    "- Best Friend: Aarav", the substring fails. Normalization brings the
    query into the same shape the store uses ("best friend") so the match has
    a chance of hitting.
    """
    text = " ".join(str(query or "").split()).strip().lower()
    if not text:
        return ""
    # Strip leading filler — may chain ("do you know if my..." → "if my..." → "if...").
    for _ in range(3):
        replaced = _RECALL_QUERY_FILLER_RE.sub("", text)
        if replaced == text:
            break
        text = replaced
    text = _RECALL_QUERY_POSSESSIVE_RE.sub("", text)
    text = " ".join(text.split())
    text = text.rstrip("?.!,; ")
    return text


# Maps query keywords → topic file(s) we should dump in full when lexical
# search misses. Order within each tuple is preserved so the highest-signal
# keywords match first.
_TOPIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("relations", (
        "best friend", "boyfriend", "girlfriend",
        "wife", "husband", "spouse", "partner",
        "brother", "sister", "sibling", "cousin",
        "mom", "mother", "dad", "father", "parent",
        "family", "relative", "friend",
    )),
    ("identity", (
        "name", "email", "timezone", "time zone",
        "language", "city", "country",
        "occupation", "job", "role", "profession",
        "company", "employer", "work",
    )),
    ("workflow", (
        "morning routine", "daily briefing",
        "routine", "schedule", "habit",
        "every morning", "every day", "every week",
        "daily", "weekly", "briefing",
    )),
    ("contacts", ("recipient", "mail to", "send to", "contact")),
    ("projects", ("project", "repo", "repository", "codebase", "working on")),
    ("preferences", ("prefer", "tone", "style", "humor", "concise", "detailed")),
)


def _topics_for_keywords(query: str) -> list[str]:
    """Return candidate topic names whose keyword family appears in *query*."""
    if not query:
        return []
    topics: list[str] = []
    for topic_name, keywords in _TOPIC_KEYWORDS:
        if topic_name in topics:
            continue
        if any(kw in query for kw in keywords):
            topics.append(topic_name)
    return topics


# ---------------------------------------------------------------------------
# RetrievalBroker
# ---------------------------------------------------------------------------

class RetrievalBroker:
    """Assembles per-turn memory context from four tiers under a token budget.

    Replaces ``PersonalMemoryPromptBuilder.build_memory_block`` as the main
    entry point for prompt memory injection.  The existing prompt builder is
    kept as a fallback and is reused here for topic selection logic.
    """

    def __init__(
        self,
        *,
        store: PersonalMemoryStore,
        task_store: TaskHistoryStore,
        journal_store: JournalStore | None = None,
        session_store: SessionStore | None = None,
        rag_system: Any | None = None,
        vector_store: Any | None = None,
        budget: RetrievalBudget = DEFAULT_BUDGET,
    ) -> None:
        self.store = store
        self.task_store = task_store
        self.journal_store = journal_store
        self.session_store = session_store
        self.rag_system = rag_system
        self.vector_store = vector_store
        self.budget = budget
        # Reuse existing topic-selection logic from PersonalMemoryPromptBuilder.
        self._prompt_builder = PersonalMemoryPromptBuilder(store)

    async def build_context(self, *, task_type: str, query: str) -> str:
        """Assemble and return the memory context block for one turn.

        The result is a plain-text string ready for injection into the prompt.
        Returns ``""`` when no relevant memory exists.
        """
        sections: list[str] = []
        tokens_used = 0

        # --- Tier 1: MEMORY.md index (always) ---------------------------
        index_text = self._build_index_tier()
        if index_text:
            budget_1 = min(self.budget.index_tokens, self.budget.total_tokens - tokens_used)
            trimmed = _trim_to_tokens(index_text, budget_1)
            if trimmed:
                sections.append(trimmed)
                tokens_used += _estimate_tokens(trimmed)

        # --- Tier 2: rolling summary tail -------------------------------
        remaining = self.budget.total_tokens - tokens_used
        budget_2 = min(self.budget.summary_tokens, remaining)
        if budget_2 > 0:
            summary_text = self._build_summary_tier()
            if summary_text:
                trimmed = _trim_to_tokens(summary_text, budget_2)
                if trimmed:
                    sections.append(trimmed)
                    tokens_used += _estimate_tokens(trimmed)

        return "\n\n".join(sections).strip()

    async def recall(
        self,
        *,
        query: str,
        scope: str,
        message_history: list[Any] | None = None,
        trim_fn: Callable[[list[Any]], list[Any]] | None = None,
    ) -> str:
        """Dispatch recall queries by scope."""
        scope_key = str(scope or "").strip().lower()
        if scope_key == "personal":
            return await self._build_personal_tier(query)
        if scope_key == "episodic":
            return await self._build_episodic_tier(query)
        if scope_key == "tasks":
            return self._build_task_tier(query)
        if scope_key == "working":
            return self._build_working_tier(query, message_history, trim_fn)
        return ""

    # ------------------------------------------------------------------
    # Tier builders
    # ------------------------------------------------------------------

    def _build_index_tier(self) -> str:
        """Tier 1: MEMORY.md index + full bodies of identity.md and preferences.md.

        The index lines alone are just descriptions, not actual values. We
        always expand identity to ensure key facts are present in the prompt.
        """
        sections: list[str] = []
        entries = self.store.load_index()
        if entries:
            lines = [f"- {entry.title}: {entry.summary}" for entry in entries]
            sections.append("[Memory Index]\n" + "\n".join(lines))

        for topic_label, topic_key in (("Identity", "identity"), ("Preferences", "preferences")):
            try:
                doc = self.store.load_topic(topic_key)
            except Exception:
                continue
            body_lines = [str(line).strip() for line in (doc.lines or []) if str(line).strip()]
            if not body_lines:
                continue
            sections.append(f"[{topic_label}]\n" + "\n".join(body_lines))

        return "\n\n".join(sections).strip()

    def _build_summary_tier(self) -> str:
        if self.session_store is None:
            return ""
        entries = self.session_store.get_summary_tail(max_entries=6)
        if not entries:
            return ""
        lines = ["[Recent Summary]"]
        for entry in entries:
            timestamp = str(entry.get("timestamp", "")).strip()
            turn_range = entry.get("turn_id_range")
            bullets = entry.get("bullets") or []
            header = timestamp
            if isinstance(turn_range, (list, tuple)) and len(turn_range) == 2:
                header = f"{header} (turns {turn_range[0]}-{turn_range[1]})" if header else f"turns {turn_range[0]}-{turn_range[1]}"
            if header:
                lines.append(header)
            for bullet in bullets:
                bullet_text = str(bullet).strip()
                if bullet_text:
                    lines.append(f"- {bullet_text}")
        return "\n".join(lines).strip()

    async def _build_personal_tier(self, query: str) -> str:
        """Search personal memory topics + journal for relevant matches.

        The model frequently phrases queries as "user's X" or "my X" — neither
        token appears in stored memory. We normalize the query first so the
        substring match has a chance of hitting. If lexical match still finds
        nothing, we fall back to dumping the full body of any topic file the
        query keyword-routes to, letting the model do the matching itself.
        """
        if not str(query or "").strip():
            return ""
        lowered = _normalize_recall_query(query)
        if not lowered:
            return ""

        sections: list[str] = []
        topic_hits: list[str] = []
        for topic_name in sorted(self.store.topic_paths.keys()):
            try:
                doc = self.store.load_topic(topic_name)
            except Exception:
                continue
            matches = []
            for line in doc.lines or []:
                line_text = str(line).strip()
                if not line_text:
                    continue
                if lowered in line_text.lower():
                    matches.append(line_text)
                if len(matches) >= 4:
                    break
            if matches:
                title = doc.metadata.get("title") or topic_name.replace("_", " ").title()
                topic_hits.append("\n".join([f"[{title}]", *matches]))
            if len(topic_hits) >= 3:
                break

        if topic_hits:
            sections.append("\n\n".join(topic_hits))
        else:
            # Lexical miss — dump full topic bodies the query keyword-routes to.
            fallback_topics = _topics_for_keywords(lowered)
            fallback_hits: list[str] = []
            for topic_name in fallback_topics:
                try:
                    doc = self.store.load_topic(topic_name)
                except Exception:
                    continue
                body_lines = [
                    str(line).strip()
                    for line in (doc.lines or [])
                    if str(line).strip()
                ]
                if not body_lines:
                    continue
                title = doc.metadata.get("title") or topic_name.replace("_", " ").title()
                fallback_hits.append("\n".join([f"[{title}]", *body_lines]))
                if len(fallback_hits) >= 2:
                    break
            if fallback_hits:
                sections.append("\n\n".join(fallback_hits))

        if self.journal_store is not None:
            journal_hits: list[str] = []
            try:
                events = self.journal_store.load_all()
            except Exception:
                events = []
            for event in reversed(events[-200:]):
                text = json.dumps(event.to_payload(), ensure_ascii=False).lower()
                if lowered not in text:
                    continue
                value_text = json.dumps(event.value, ensure_ascii=False)
                journal_hits.append(
                    f"- {event.topic} {event.key}: {value_text}"
                )
                if len(journal_hits) >= 5:
                    break
            if journal_hits:
                sections.append("[Journal]\n" + "\n".join(journal_hits))

        return "\n\n".join(sections).strip()

    async def _build_episodic_tier(self, query: str) -> str:
        """Tier 3: top-k episodic RAG hits formatted for prompt injection."""
        try:
            raw = await self.rag_system.query_history(query)
            if not raw or raw == "cannot find in history":
                return ""
            try:
                chunks = json.loads(raw)
            except Exception:
                return ""
            if not isinstance(chunks, list) or not chunks:
                return ""
            window = _infer_temporal_window(query)
            lines = ["[Past Conversations]"]
            added = 0
            for chunk in chunks:
                content = str(chunk.get("content", "")).strip()
                ts = str(chunk.get("timestamp", "")).strip()
                if window and not _timestamp_in_window(ts, window):
                    continue
                if not content:
                    continue
                prefix = f"[{ts}] " if ts else ""
                # Keep individual chunk excerpts short
                excerpt = content[:200]
                lines.append(f"- {prefix}{excerpt}")
                added += 1
                if added >= 3:
                    break
            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception:
            return ""

    def _build_task_tier(self, query: str) -> str:
        """Tier 4: single best task history hit."""
        try:
            return self.task_store.format_search_results(query, max_results=1)
        except Exception:
            return ""

    def _build_working_tier(
        self,
        query: str,
        message_history: list[Any] | None,
        trim_fn: Callable[[list[Any]], list[Any]] | None,
    ) -> str:
        query_text = str(query or "").strip()
        if not query_text or not message_history:
            return ""
        lowered = query_text.lower()

        visible = trim_fn(message_history) if trim_fn else []
        hidden = message_history
        if visible:
            hidden = message_history[:-len(visible)]
        if not hidden:
            return ""

        lines: list[str] = []
        try:
            from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart, ToolCallPart, ToolReturnPart
        except Exception:
            ModelRequest = ModelResponse = UserPromptPart = TextPart = ToolCallPart = ToolReturnPart = None

        for message in hidden:
            if ModelRequest and isinstance(message, ModelRequest):
                for part in message.parts:
                    if UserPromptPart and isinstance(part, UserPromptPart):
                        content = str(part.content).strip()
                        if content and lowered in content.lower():
                            lines.append(f"user: {content}")
            elif ModelResponse and isinstance(message, ModelResponse):
                for part in message.parts:
                    if TextPart and isinstance(part, TextPart):
                        content = str(part.content).strip()
                        if content and lowered in content.lower():
                            lines.append(f"assistant: {content}")
                    elif ToolCallPart and isinstance(part, ToolCallPart):
                        args = str(part.args).strip()
                        tool_name = str(part.tool_name).strip()
                        content = f"tool_call {tool_name}: {args}".strip()
                        if lowered in content.lower():
                            lines.append(content)
                    elif ToolReturnPart and isinstance(part, ToolReturnPart):
                        content = str(part.content).strip()
                        tool_name = str(part.tool_name).strip()
                        line = f"tool_return {tool_name}: {content}".strip()
                        if content and lowered in line.lower():
                            lines.append(line)
            else:
                raw = str(message).strip()
                if raw and lowered in raw.lower():
                    lines.append(raw)

            if len(lines) >= 8:
                break

        if not lines:
            return ""
        return "[Earlier In This Conversation]\n" + "\n".join(f"- {line}" for line in lines)


def _infer_temporal_window(query: str) -> tuple[datetime, datetime] | None:
    lowered = query.lower()
    now = datetime.now(UTC)
    if "today" in lowered:
        start = datetime(now.year, now.month, now.day, tzinfo=UTC)
        end = start + timedelta(days=1)
        return (start, end)
    if "yesterday" in lowered:
        end = datetime(now.year, now.month, now.day, tzinfo=UTC)
        start = end - timedelta(days=1)
        return (start, end)
    if "last week" in lowered or "week ago" in lowered:
        start = now - timedelta(days=7)
        return (start, now)
    if "last month" in lowered or "month ago" in lowered:
        start = now - timedelta(days=30)
        return (start, now)
    if "last year" in lowered or "year ago" in lowered:
        start = now - timedelta(days=365)
        return (start, now)
    return None


def _timestamp_in_window(timestamp: str, window: tuple[datetime, datetime]) -> bool:
    if not timestamp:
        return False
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except Exception:
        return False
    start, end = window
    return start <= parsed <= end
