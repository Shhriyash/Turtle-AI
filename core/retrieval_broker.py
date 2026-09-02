"""Retrieval broker — assembles prompt-time memory plus recall-tool lookups.

Prompt-time injection is intentionally small: MEMORY.md index + identity block,
trimmed under a hard token budget. Episodic and task history are now accessed
only through the recall tool.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

from core.config import settings
from core.memory_sqlite import MemoryEventRow, MemorySQLiteIndex
from core.personal_memory_prompt import PersonalMemoryPromptBuilder
from core.personal_memory_store import PersonalMemoryStore
from core.task_history import TaskHistoryStore
from core.memory_journal import JournalStore
from core.session_store import SessionStore

try:
    import logfire as _logfire  # type: ignore
except Exception:  # pragma: no cover - logfire optional
    _logfire = None  # type: ignore


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalBudget:
    index_tokens: int = 240
    # Phase 2 W2: query-aware [Relevant Memory] tier — the always-on injection
    # that surfaces stored facts outside identity/preferences without a
    # separate recall-tool call.
    relevant_tokens: int = 160
    summary_tokens: int = 150
    total_tokens: int = 600


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

def _normalize_recall_query(query: str) -> str:
    """Lowercase + collapse whitespace + strip leading filler.

    FTS5 tokenization (porter stemmer, unicode61) handles inflection,
    possessives, and word order, so we only strip the leading conversational
    filler ("do you remember", "what is") to keep the MATCH expression tight.
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
    text = " ".join(text.split())
    text = text.rstrip("?.!,; ")
    return text


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _token_overlap(query: str, line: str) -> float:
    """Fraction of query tokens that also appear in *line*.

    Corpus-independent and interpretable — the primary "is lexical strong
    enough?" signal. Returns 0.0 when the query has no tokens.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    line_tokens = _tokenize(line)
    return len(q_tokens & line_tokens) / len(q_tokens)


def _row_searchable_text(row: MemoryEventRow) -> str:
    """The text FTS5 actually matched on — topic, key, value, evidence.

    Used for token-overlap scoring so a query like "best friend" (stored in the
    *key* ``relations.best_friend`` with value "Aarav") counts as a strong hit.
    """
    return " ".join(
        part for part in (row.topic, row.key, row.value_text, row.evidence_text) if part
    )


def _format_personal_hits(hits: list[MemoryEventRow]) -> str:
    """Render BM25-ranked FTS rows as a compact prompt block."""
    lines: list[str] = []
    seen: set[str] = set()
    for row in hits:
        text = (row.value_text or "").strip()
        if not text:
            continue
        dedupe_key = f"{row.topic}:{text.lower()}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        label = row.key or row.topic
        lines.append(f"- {label}: {text}")
    if not lines:
        return ""
    return "[Personal Memory]\n" + "\n".join(lines)


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
        sqlite_index: MemorySQLiteIndex | None = None,
        session_store: SessionStore | None = None,
        rag_system: Any | None = None,
        vector_store: Any | None = None,
        user_id: str = "",
        budget: RetrievalBudget = DEFAULT_BUDGET,
    ) -> None:
        self.store = store
        self.task_store = task_store
        self.journal_store = journal_store
        self.sqlite_index = sqlite_index
        self.session_store = session_store
        self.rag_system = rag_system
        self.vector_store = vector_store
        self.user_id = user_id
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
                # Clamp to the tier budget: _trim_to_tokens appends "..." on cut,
                # so re-estimating can exceed the budget and overdraw the total.
                tokens_used += min(_estimate_tokens(trimmed), budget_1)

        # --- Tier 1.5: query-aware relevant memory (always-on) ----------
        # The autopsy's single most damaging read-path finding: build_context
        # ignored *query* and injected only the static identity/preferences
        # card, so any stored fact outside those two topics was invisible unless
        # the model separately called the recall tool. This tier fixes that by
        # running the Phase 1 search layer against the actual query.
        remaining = self.budget.total_tokens - tokens_used
        budget_r = min(self.budget.relevant_tokens, remaining)
        if budget_r > 0:
            relevant_text = self._build_relevant_tier(query, exclude_text=index_text)
            if relevant_text:
                trimmed = _trim_to_tokens(relevant_text, budget_r)
                if trimmed:
                    sections.append(trimmed)
                    tokens_used += min(_estimate_tokens(trimmed), budget_r)

        # --- Tier 2: rolling summary tail -------------------------------
        remaining = self.budget.total_tokens - tokens_used
        budget_2 = min(self.budget.summary_tokens, remaining)
        if budget_2 > 0:
            summary_text = await self._build_summary_tier()
            if summary_text:
                trimmed = _trim_to_tokens(summary_text, budget_2)
                if trimmed:
                    sections.append(trimmed)
                    tokens_used += min(_estimate_tokens(trimmed), budget_2)

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

    def _build_relevant_tier(self, query: str, *, exclude_text: str = "") -> str:
        """Tier 1.5: query-aware always-on injection of relevant stored facts.

        Runs the Phase 1 search layer (stopword AND-first/OR-fallback,
        latest-per-key, superseded exclusion) then applies an injection-grade
        relevance floor: only rows whose token overlap with the normalized
        query clears 0.3 survive. An always-on tier must prefer empty over
        wrong — the autopsy's "what editor do I use → taekwondo" failure — so
        weak hits are dropped rather than injected. Hits whose value already
        appears in the Tier-1 identity/preferences body are deduped away.
        """
        if not str(query or "").strip() or self.sqlite_index is None:
            return ""
        normalized = _normalize_recall_query(query)
        if not normalized:
            return ""
        try:
            hits = self.sqlite_index.search(normalized, limit=5)
            # Values already served by a Tier-1 line ("- Name: Shriyash" →
            # "shriyash"). Exact match on the parsed line value, NOT substring
            # containment over the whole block: substring matching suppressed
            # distinct short facts ("Python", "yes") whenever the same word
            # happened to appear inside an unrelated Tier-1 sentence.
            exclude_values = {
                line.split(":", 1)[1].strip().lower()
                for line in str(exclude_text or "").splitlines()
                if ":" in line
            }
            exclude_values.discard("")
            lines: list[str] = []
            seen: set[str] = set()
            for row in hits:
                # Injection-grade floor — confidently-wrong beats honestly-empty.
                # Scored against what actually gets injected (topic/key label +
                # value), NOT the full searchable text: evidence_text is never
                # shown, so a row whose only query overlap lives in hidden
                # evidence must not clear the floor on its strength.
                injectable = " ".join(
                    part for part in (row.topic, row.key, row.value_text) if part
                )
                if _token_overlap(normalized, injectable) < 0.3:
                    continue
                text = (row.value_text or "").strip()
                if not text:
                    continue
                # Dedupe against the Tier-1 identity/preferences body already present.
                if text.lower() in exclude_values:
                    continue
                dedupe_key = f"{row.topic}:{text.lower()}"
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                label = row.key or row.topic
                lines.append(f"- {label}: {text}")
            if not lines:
                return ""
            return "[Relevant Memory]\n" + "\n".join(lines)
        except Exception as exc:
            print(f"LOG: RetrievalBroker relevant tier failed: {exc}")
            return ""

    async def _build_summary_tier(self) -> str:
        if self.session_store is None:
            return ""
        try:
            # Prefer cross-session carryover when the store exposes it: a fresh
            # session's own rolling_summary is empty, so this is what seeds a
            # brand-new session with the previous session's [Recent Summary] —
            # the continuity the tier had never actually provided in production.
            if hasattr(self.session_store, "get_summary_tail_with_carryover"):
                entries = await self.session_store.get_summary_tail_with_carryover(max_entries=6)
            else:
                entries = self.session_store.get_summary_tail(max_entries=6)
        except Exception as exc:
            print(f"LOG: RetrievalBroker summary tier failed: {exc}")
            return ""
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
        """Hybrid personal recall: FTS5-first, vector fallback.

        FTS5 handles the common case (exact / inflected / possessive / word
        order) at sub-millisecond cost. When lexical retrieval is weak or empty
        we pay for a query-embedding round-trip and consult the vector store,
        which closes the synonym/semantic gap ("closest pal" → "Best Friend").
        """
        started = time.perf_counter()
        if not str(query or "").strip() or self.sqlite_index is None:
            return ""
        normalized = _normalize_recall_query(query)
        if not normalized:
            return ""

        try:
            fts_hits = self.sqlite_index.search(normalized, limit=8)
        except Exception as exc:
            print(f"LOG: RetrievalBroker FTS5 search failed: {exc}")
            fts_hits = []

        top_overlap = (
            _token_overlap(normalized, _row_searchable_text(fts_hits[0])) if fts_hits else 0.0
        )
        top_rank = fts_hits[0].rank if fts_hits else None

        # Common case: strong lexical match — done, no network call.
        if self._fts_is_strong(fts_hits, normalized):
            result = _format_personal_hits(fts_hits)
            self._log_recall(
                query=query, normalized=normalized, fts_hits=fts_hits,
                top_overlap=top_overlap, top_rank=top_rank, path="fts",
                vector_triggered=False, vec_hits=[], chosen_source="fts",
                started=started,
            )
            return result

        # Weak or empty lexical result → pay for semantic.
        vec_hits: list[Any] = []
        if self.vector_store is not None and self.user_id:
            try:
                vec_hits = await self.vector_store.search(self.user_id, query, 5)
            except Exception as exc:
                print(f"LOG: RetrievalBroker vector search failed: {exc}")
                vec_hits = []

        if not vec_hits and not fts_hits:
            self._log_recall(
                query=query, normalized=normalized, fts_hits=fts_hits,
                top_overlap=top_overlap, top_rank=top_rank, path="miss",
                vector_triggered=True, vec_hits=vec_hits, chosen_source="none",
                started=started,
            )
            return ""

        merged = self._combine(fts_hits, vec_hits, query=normalized)
        chosen = "both" if (fts_hits and vec_hits) else ("fts" if fts_hits else "vector")
        self._log_recall(
            query=query, normalized=normalized, fts_hits=fts_hits,
            top_overlap=top_overlap, top_rank=top_rank, path="vector_fallback",
            vector_triggered=True, vec_hits=vec_hits, chosen_source=chosen,
            started=started,
        )
        return merged

    def _fts_is_strong(self, hits: list[MemoryEventRow], query: str) -> bool:
        """Whether FTS5 was strong enough to skip the vector fallback.

        Token overlap is the primary signal (interpretable, corpus-independent);
        BM25 rank is secondary (corpus-dependent, kept loose).
        """
        if not hits:
            return False
        top = hits[0]
        overlap = _token_overlap(query, _row_searchable_text(top))
        if overlap < settings.personal_recall_overlap_threshold:
            return False
        # BM25 rank: more-negative = better. A rank above the ceiling is weak.
        if top.rank > settings.personal_recall_bm25_ceiling:
            return False
        return True

    def _combine(
        self,
        fts_hits: list[MemoryEventRow],
        vec_hits: list[Any],
        *,
        query: str = "",
    ) -> str:
        """Simple union: FTS hits first (precise), then novel vector hits.

        No normalized score merge — BM25 and cosine are opposite-direction,
        different-scale distributions. Dedupe by topic+text. Capped at 8 lines.

        Only called on the weak-lexical path, so both arms apply a relevance
        floor: confidently-wrong beats honestly-empty was the production
        failure ("what editor do I use" returned taekwondo); below the floor we
        return nothing so the tool reports no relevant information.
        """
        lines: list[str] = []
        seen: set[str] = set()

        for row in fts_hits:
            if query and _token_overlap(query, _row_searchable_text(row)) < 0.3:
                continue
            text = (row.value_text or "").strip()
            if not text:
                continue
            dedupe_key = f"{row.topic}:{text.lower()}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            label = row.key or row.topic
            lines.append(f"- {label}: {text}")
            if len(lines) >= 8:
                break

        for hit in vec_hits:
            if len(lines) >= 8:
                break
            score = getattr(hit, "score", None)
            if isinstance(score, (int, float)) and score < 0.35:
                continue
            text = str(getattr(hit, "text", "") or "").strip()
            if not text:
                continue
            topic = ""
            meta = getattr(hit, "metadata", None)
            if isinstance(meta, dict):
                topic = str(meta.get("topic", "") or "")
            dedupe_key = f"{topic}:{text.lower()}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            lines.append(f"- {text}")

        if not lines:
            return ""
        return "[Personal Memory]\n" + "\n".join(lines)

    def _log_recall(
        self,
        *,
        query: str,
        normalized: str,
        fts_hits: list[MemoryEventRow],
        top_overlap: float,
        top_rank: float | None,
        path: str,
        vector_triggered: bool,
        vec_hits: list[Any],
        chosen_source: str,
        started: float,
    ) -> None:
        """One structured log/span per recall — the data behind §12.3 tuning."""
        vec_top_cosine = None
        if vec_hits:
            try:
                vec_top_cosine = float(getattr(vec_hits[0], "score", None))
            except Exception:
                vec_top_cosine = None
        attrs = {
            "normalized_query": normalized,
            "fts_hit_count": len(fts_hits),
            "fts_top_rank": top_rank,
            "fts_top_overlap": round(top_overlap, 3),
            "path": path,
            "vector_triggered": vector_triggered,
            "vector_hit_count": len(vec_hits),
            "vector_top_cosine": vec_top_cosine,
            "chosen_source": chosen_source,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        print(f"LOG: RetrievalBroker personal recall {attrs}")
        if _logfire is not None:
            try:
                _logfire.info("turtle.personal_recall", **attrs)
            except Exception:
                pass

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
        """Tier 4: single best task history hit — scoped to THIS user.

        task history lives in one global sqlite shared by every user, so the
        owner is passed explicitly here as well as being set on the store: this
        tier feeds straight into the prompt, and an unscoped hit would splice
        another user's task text into this user's context.
        """
        try:
            return self.task_store.format_search_results(
                query, max_results=1, user_id=self.user_id
            )
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
