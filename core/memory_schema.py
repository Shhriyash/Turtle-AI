"""Single home for the memory topic vocabulary, statement rendering, and the
write-apply policy.

Before Phase 2 these three concerns were scattered: the topic set was declared
independently in the journal validator, the replayer renderer whitelist, the
extractor prompts, and the store — five vocabularies that silently disagreed and
dropped user facts. Auto-apply logic was duplicated across four call sites with
inverted trust. This module is the one place each of those now lives; every other
module re-exports or delegates here so the vocabularies can never drift again.

Kept deliberately dependency-light (only ``dataclasses`` + ``types``) so the
journal, replayer, extractor, and server can all import it without a cycle.
``render_statement``/``statement_for`` accept any event-like object by duck
typing rather than importing ``MemoryEvent``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace


@dataclass(frozen=True)
class TopicSpec:
    title: str
    summary: str


# ---------------------------------------------------------------------------
# Decay policy — the SINGLE definition of "this fact has aged out".
#
# Previously the decay rule lived only in the markdown replayer, so the
# markdown projection would forget an aged inferred fact while the SQLite FTS
# read model (which retrieval actually queries) kept returning it — two
# contradictory truths injected into the same prompt (brutal review H2). Both
# projections now import this one predicate so they agree.
# ---------------------------------------------------------------------------
DECAY_DAYS = 30
# Identity facts (name, email, timezone) are load-bearing and never expire.
DECAY_EXEMPT_TOPICS = frozenset({"identity"})
# Explicit user statements persist until superseded/corrected; migration events
# are authoritative. Only *inferred/synthesized behavioral* signals decay.
DECAY_EXEMPT_SOURCES = frozenset({"explicit", "migration"})


def is_decayed(
    topic: str,
    source: str,
    observed_at: str,
    *,
    reference_time: datetime | None = None,
    decay_days: int = DECAY_DAYS,
) -> bool:
    """True if a fact has aged past ``decay_days`` without reinforcement.

    Reinforcement is implicit at every call site: callers pass the *latest*
    event for a (topic, key), so a newer restatement resets the clock. Pure and
    side-effect-free so the replayer (markdown) and MemorySQLiteIndex (FTS +
    read model) can share it verbatim. Unparseable/missing timestamps and
    exempt topics/sources never decay.
    """
    if topic in DECAY_EXEMPT_TOPICS:
        return False
    if source in DECAY_EXEMPT_SOURCES:
        return False
    if not observed_at:
        return False
    try:
        observed = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00"))
    except Exception:
        return False
    ref = reference_time or datetime.now(UTC)
    return (ref - observed) > timedelta(days=decay_days)


# The 11 canonical topics. Titles/summaries are lifted verbatim from the
# replayer's former TOPIC_TITLES/TOPIC_SUMMARIES so rendered topic files and the
# MEMORY.md index do not churn. Insertion order is load-bearing: the replayer
# derives ALL_TOPICS from this dict and iterates it.
TOPICS: dict[str, TopicSpec] = {
    "identity": TopicSpec(
        title="Identity",
        summary="Name, email, location, timezone, language, and role",
    ),
    "preferences": TopicSpec(
        title="Preferences",
        summary="Tone, response style, and delivery defaults",
    ),
    "workflow": TopicSpec(
        title="Workflow",
        summary="Recurring habits and operational defaults",
    ),
    "contacts": TopicSpec(
        title="Contacts",
        summary="Frequent recipients and confirmed aliases",
    ),
    "projects": TopicSpec(
        title="Projects",
        summary="Project context and recurring work references",
    ),
    "corrections": TopicSpec(
        title="Corrections",
        summary="User corrections and how to apply them",
    ),
    "relations": TopicSpec(
        title="Relations",
        summary="People in the user's life and their roles",
    ),
    "working_style": TopicSpec(
        title="Working Style",
        summary="How the user approaches tasks",
    ),
    "communication_style": TopicSpec(
        title="Communication Style",
        summary="How the user likes to receive information",
    ),
    "tool_preferences": TopicSpec(
        title="Tool Preferences",
        summary="Tools, languages, and environments the user favours",
    ),
    "decision_style": TopicSpec(
        title="Decision Style",
        summary="How the user makes decisions",
    ),
}

# Re-exported by core.memory_journal for backwards compatibility with existing
# importers (`from core.memory_journal import ALLOWED_TOPICS`).
ALLOWED_TOPICS = frozenset(TOPICS)

# Inferred/synthesized facts auto-apply only for these low-risk topics; every
# other topic waits in the confirmation gate. See decide_write_policy.
_AUTO_APPLY_TOPICS = frozenset({"preferences", "workflow", "projects"})

# Rendered statements are injected into prompts verbatim, so they must stay
# one line and bounded — an oversized or newline-carrying statement can smuggle
# fake sections into the memory block or crowd real memories out of budget.
_MAX_STATEMENT_CHARS = 300
_MAX_LEAF_CHARS = 120


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def render_statement(event) -> str:
    """Return a human-readable one-line statement for ANY event (no leading "- ").

    Layers, first match wins:
      1. If the event carries a non-empty ``statement`` attribute, return it
         after one-line normalization and a length cap. This is statement-based
         rendering: extractors snapshot the statement at write time so
         projection is decoupled from the key templates below (and old journal
         events without one still render via the templates). Normalization
         collapses newlines so a journal payload can't inject fake markdown
         sections into the rendered memory block.
      2. Per-key templates for every whitelisted key shape.
      3. Generic flatten fallback (label from the key tail, depth-first leaf
         join) so an applied event is never invisible just because its key is
         missing from the whitelist — that silent drop is how the confirmed
         corrections.name_role and preferences.sport facts once vanished.

    Returns "" only for genuinely empty values; never for an event with a
    non-empty renderable value.
    """
    statement = _clean_text(getattr(event, "statement", "") or "")
    if statement:
        return statement[:_MAX_STATEMENT_CHARS].rstrip()

    value = getattr(event, "value", None) or {}
    key = getattr(event, "key", "") or ""
    topic = getattr(event, "topic", "") or ""

    if key == "identity.name":
        name = _clean_text(value.get("name"))
        return f"Name: {name}" if name else ""

    if key == "identity.home_city":
        home_city = _clean_text(value.get("home_city"))
        return f"Home city: {home_city}" if home_city else ""

    if key == "identity.current_city":
        current_city = _clean_text(value.get("current_city"))
        return f"Current city: {current_city}" if current_city else ""

    if key == "identity.country":
        country = _clean_text(value.get("country"))
        return f"Country: {country}" if country else ""

    if key == "identity.primary_email":
        email = _clean_text(value.get("primary_email") or value.get("email"))
        return f"Primary email: {email.lower()}" if email else ""

    if key.startswith("identity.known_email."):
        email = _clean_text(value.get("email"))
        return f"Known email: {email.lower()}" if email else ""

    if key == "identity.timezone":
        tz = _clean_text(value.get("timezone"))
        return f"Timezone: {tz}" if tz else ""

    if key == "identity.preferred_language":
        language = _clean_text(value.get("preferred_language"))
        return f"Preferred language: {language}" if language else ""

    if key == "identity.occupation":
        occupation = _clean_text(value.get("occupation"))
        return f"Occupation: {occupation}" if occupation else ""

    if key == "identity.company":
        company = _clean_text(value.get("company"))
        return f"Company: {company}" if company else ""

    if key == "preferences.response_style":
        style = _clean_text(value.get("response_style"))
        return f"Response style: {style}" if style else ""

    if key == "preferences.humor_level":
        humor = _clean_text(value.get("humor_level"))
        return f"Humor level: {humor}" if humor else ""

    if key == "preferences.email_tone":
        tone = _clean_text(value.get("email_tone"))
        return f"Email tone: {tone}" if tone else ""

    if key == "workflow.prefers_draft_before_send":
        flag = value.get("prefers_draft_before_send")
        if flag is None:
            return ""
        # Old journal payloads sometimes carry the flag as a string; bool("false")
        # is True, so parse the common textual forms instead of coercing blindly.
        if isinstance(flag, str):
            lowered = flag.strip().lower()
            if lowered in {"true", "yes", "1"}:
                flag = True
            elif lowered in {"false", "no", "0"}:
                flag = False
            else:
                return ""
        return f"Prefers draft before send: {'true' if bool(flag) else 'false'}"

    if key == "workflow.primary_llm":
        model = _clean_text(value.get("primary_llm"))
        return f"Preferred primary model: {model}" if model else ""

    if key == "workflow.email_interactions_recorded":
        count = value.get("count")
        if count is None:
            return ""
        try:
            return f"Email interactions recorded: {int(count)}"
        except Exception:
            return ""

    # D4: routines — a parseable single-line summary so the profile snapshot can
    # read them back without re-touching the journal.
    if key in {"workflow.morning_routine", "workflow.daily_briefing"} or key.startswith("workflow.recurring_request"):
        routine = _clean_text(value.get("routine") or value.get("name"))
        if not routine:
            return ""
        cadence = _clean_text(value.get("cadence") or value.get("frequency") or "daily")
        clock = _clean_text(value.get("time"))
        tz = _clean_text(value.get("timezone"))
        items_field = value.get("items") or value.get("steps") or []
        if isinstance(items_field, str):
            items_field = [items_field]
        items = [_clean_text(i) for i in items_field if _clean_text(i)] if isinstance(items_field, list) else []

        schedule_parts = [cadence] if cadence else []
        if clock:
            schedule_parts.append(clock)
        if tz:
            schedule_parts.append(tz)
        schedule = " ".join(schedule_parts) or "daily"

        line = f"Routine: {routine} | {schedule}"
        if items:
            line += f" | items: {', '.join(items)}"
        return line

    if key.startswith("contacts.frequent_recipient."):
        email = _clean_text(value.get("email"))
        if not email:
            return ""
        count = value.get("count")
        if count is None:
            return f"Frequent recipient: {email.lower()}"
        try:
            return f"Frequent recipient: {email.lower()} (count: {int(count)})"
        except Exception:
            return f"Frequent recipient: {email.lower()}"

    if key.startswith("projects.project."):
        name = _clean_text(value.get("name"))
        summary = _clean_text(value.get("summary"))
        if not name:
            return ""
        if summary:
            return f"Project: {name} — {summary}"
        return f"Project: {name}"

    if topic == "relations" and key.startswith("relations."):
        role = _clean_text(value.get("role") or key.split(".", 1)[-1])
        name = _clean_text(value.get("name"))
        if not name:
            return ""
        label = role.replace("_", " ").title() if role else "Person"
        return f"{label}: {name}"

    if topic == "corrections":
        summary = _clean_text(value.get("summary") or value.get("text"))
        if summary:
            return f"Correction: {summary}"
        # fall through to the generic renderer for corrections whose value
        # carries structured fields instead of a summary/text string

    # Generic fallback: label from the key tail, depth-first leaf join.
    label_source = key.rsplit(".", 1)[-1] if "." in key else (key or topic)
    label = _clean_text(label_source.replace("_", " ")).title() or (topic.title() if topic else "")

    parts: list[str] = []

    def _collect(node: object) -> None:
        if isinstance(node, dict):
            for item in node.values():
                _collect(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _collect(item)
        elif isinstance(node, bool):
            parts.append("true" if node else "false")
        elif isinstance(node, (str, int, float)):
            text = _clean_text(node)
            if text:
                parts.append(text[:_MAX_LEAF_CHARS])

    _collect(value)
    flattened = ", ".join(parts)
    if not flattened:
        return ""
    return f"{label}: {flattened}"[:_MAX_STATEMENT_CHARS].rstrip()


def statement_for(topic: str, key: str, value: dict) -> str:
    """Compute render_statement over an assembled (topic, key, value) triple.

    Extractors call this to snapshot the statement onto an event at write time,
    before the full MemoryEvent exists. Wraps render_statement over an ad-hoc
    event-like object whose empty ``statement`` forces the key-template layer.
    """
    probe = SimpleNamespace(statement="", topic=topic, key=key, value=value)
    return render_statement(probe)


def decide_write_policy(
    *,
    source: str,
    topic: str,
    confidence: float,
    evidence_supported: bool,
) -> str:
    """Single write-apply decision for every candidate: "applied" | "pending" | "rejected".

    Consolidates auto-apply logic that used to be copy-pasted across four call
    sites with inverted trust. In production that duplication auto-applied
    deterministic regex garbage at confidence 1.0 straight into the profile while
    the correct Stage-B name fix sat gated behind the confirmation gate. The rule
    now: explicit facts must be evidence-grounded before they auto-apply, and
    inferred/synthesized facts auto-apply only for the low-risk topics at high
    confidence; everything else waits in the gate.

    Semantics:
      - explicit + evidence_supported + confidence >= 0.9  -> "applied" (any topic)
      - explicit + not evidence_supported                  -> "pending"
      - inferred/synthesized + topic in {preferences, workflow, projects}
        + confidence >= 0.85                               -> "applied"
      - other inferred/synthesized                         -> "pending"
      - unknown source / topic not in TOPICS / confidence
        outside [0, 1]                                     -> "rejected"
    """
    # Structural rejection first: anything the journal could not legally hold.
    if topic not in TOPICS:
        return "rejected"
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return "rejected"
    if not 0.0 <= conf <= 1.0:
        return "rejected"

    if source == "explicit":
        if evidence_supported and conf >= 0.9:
            return "applied"
        return "pending"

    if source in {"inferred", "synthesized"}:
        if topic in _AUTO_APPLY_TOPICS and conf >= 0.85:
            return "applied"
        return "pending"

    # Unrecognized source (e.g. migration is applied directly, never routed here).
    return "rejected"
