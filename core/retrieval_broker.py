"""Retrieval broker (Step 7) — assembles per-turn memory context under a hard token budget.

Four tiers, assembled in priority order, each capped at a sub-budget:

  Tier 1  MEMORY.md index       60 tokens   always present
  Tier 2  topic slice          200 tokens   always present, relevance-scored
  Tier 3  episodic RAG hits    100 tokens   only on history/time trigger words
  Tier 4  task history hit      40 tokens   only on history/time trigger words

  Hard cap: 400 tokens total.

"Tokens" are estimated at 1 token per 4 characters, consistent with the rest
of the codebase.  The budget is intentionally conservative — these numbers are
embedded in system-prompt context that already competes with the conversation.

Tier 3 and 4 are gated behind ``_has_history_trigger`` to avoid a FAISS vector
search and SQLite FTS query on every turn.  Explicit history queries (the
``history_tool`` agent tool) run unconditionally via a separate code path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core.personal_memory_prompt import PersonalMemoryPromptBuilder
from core.personal_memory_store import PersonalMemoryStore
from core.task_history import TaskHistoryStore


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalBudget:
    index_tokens: int = 60
    topic_tokens: int = 200
    episodic_tokens: int = 100
    task_tokens: int = 40
    total_tokens: int = 400


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
        rag_system: Any | None = None,
        budget: RetrievalBudget = DEFAULT_BUDGET,
    ) -> None:
        self.store = store
        self.task_store = task_store
        self.rag_system = rag_system
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

        # --- Tier 2: topic slice (always) --------------------------------
        remaining = self.budget.total_tokens - tokens_used
        budget_2 = min(self.budget.topic_tokens, remaining)
        if budget_2 > 0:
            topic_text = self._build_topic_tier(task_type=task_type, query=query)
            if topic_text:
                trimmed = _trim_to_tokens(topic_text, budget_2)
                if trimmed:
                    sections.append(trimmed)
                    tokens_used += _estimate_tokens(trimmed)

        # --- Tier 3 + 4: episodic + task (history triggers only) ---------
        if _has_history_trigger(query):
            # Tier 3: episodic RAG
            remaining = self.budget.total_tokens - tokens_used
            budget_3 = min(self.budget.episodic_tokens, remaining)
            if budget_3 > 0 and self.rag_system is not None:
                episodic_text = await self._build_episodic_tier(query)
                if episodic_text:
                    trimmed = _trim_to_tokens(episodic_text, budget_3)
                    if trimmed:
                        sections.append(trimmed)
                        tokens_used += _estimate_tokens(trimmed)

            # Tier 4: task history
            remaining = self.budget.total_tokens - tokens_used
            budget_4 = min(self.budget.task_tokens, remaining)
            if budget_4 > 0:
                task_text = self._build_task_tier(query)
                if task_text:
                    trimmed = _trim_to_tokens(task_text, budget_4)
                    if trimmed:
                        sections.append(trimmed)
                        # tokens_used += ... (not needed, this is the last tier)

        return "\n\n".join(sections).strip()

    # ------------------------------------------------------------------
    # Tier builders
    # ------------------------------------------------------------------

    def _build_index_tier(self) -> str:
        """Tier 1: compact MEMORY.md index summary."""
        entries = self.store.load_index()
        if not entries:
            return ""
        lines = [f"- {entry.title}: {entry.summary}" for entry in entries]
        return "[Memory Index]\n" + "\n".join(lines)

    def _build_topic_tier(self, *, task_type: str, query: str) -> str:
        """Tier 2: relevance-scored topic file content.

        Topic selection reuses the existing ``PersonalMemoryPromptBuilder``
        logic (task-type priority + keyword signals in the query).
        """
        entries = self.store.load_index()
        # _select_topics is a private method we own; accessing it here is fine.
        selected = self._prompt_builder._select_topics(  # noqa: SLF001
            task_type=task_type,
            query=query,
            entries=entries,
        )
        sections: list[str] = []
        for topic_name in selected[:3]:  # at most 3 before token trimming
            document = self.store.load_topic(topic_name)
            if document.lines:
                title = (
                    document.metadata.get("title")
                    or topic_name.replace("_", " ").title()
                )
                section = "\n".join([f"[{title}]", *document.lines])
                sections.append(section)
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
            lines = ["[Past Conversations]"]
            for chunk in chunks[:3]:
                content = str(chunk.get("content", "")).strip()
                ts = str(chunk.get("timestamp", "")).strip()
                if not content:
                    continue
                prefix = f"[{ts}] " if ts else ""
                # Keep individual chunk excerpts short
                excerpt = content[:200]
                lines.append(f"- {prefix}{excerpt}")
            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception:
            return ""

    def _build_task_tier(self, query: str) -> str:
        """Tier 4: single best task history hit."""
        try:
            return self.task_store.format_search_results(query, max_results=1)
        except Exception:
            return ""
