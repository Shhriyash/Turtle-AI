#!/usr/bin/env python
"""
scripts/model_compare.py
========================
§7 controlled model-comparison harness for TURTLE's personal-memory pipeline.

  ┌───────────────────────────────────────────────────────────────────────┐
  │  THIS SCRIPT MAKES **LIVE** LLM API CALLS AND REQUIRES PROVIDER KEYS.   │
  │  It is a human-run benchmarking tool — it is NOT part of the CI suite   │
  │  and must never be imported by a test. Set GROQ_API_KEY / GEMINI_API_  │
  │  KEY / OPEN_ROUTER_API_KEY_* in your .env before running.               │
  └───────────────────────────────────────────────────────────────────────┘

It seeds a temp fixture user with ~25 facts across all 11 memory topics, then
scores each candidate model across three phases:

  1. extraction accuracy  — run the per-turn extractor with the model over ~15
     canonical utterances; score (topic, key, value) against gold.
  2. recall-tool selection — run the real main-agent toolset with the model over
     ~8 memory questions; score whether the model chose recall(scope=personal).
  3. end-to-end answer     — after a simulated restart (index rebuilt from the
     journal alone), ask the model seeded questions and score whether the
     seeded fact appears in the final answer.

Output: a per-model per-phase accuracy table (markdown) to stdout, plus a JSON
file (``--output``).

Usage:
    venv\\Scripts\\python.exe scripts/model_compare.py --help
    venv\\Scripts\\python.exe scripts/model_compare.py \\
        --models groq:openai/gpt-oss-120b gemini:gemini-2.5-flash \\
        --output output/model_compare.json

A model whose provider key is missing (or that errors mid-run) is skipped with a
clear message rather than crashing the whole comparison.
"""
from __future__ import annotations

# NOTE: keep top-level imports light and key-free so ``--help`` works anywhere.
# All heavy repo imports happen lazily inside functions.
import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_MODELS = [
    "gemini:gemini-2.5-flash",
    "groq:openai/gpt-oss-120b",
    "groq:llama-3.3-70b-versatile",
]


# ---------------------------------------------------------------------------
# Fixture data — ~25 seeded facts across all 11 topics, ~15 extraction gold
# utterances, ~8 recall questions, and the E2E answer-contains set.
# ---------------------------------------------------------------------------

# (topic, key, value_dict) — journaled applied so the seeded user "remembers".
SEED_FACTS: list[tuple[str, str, dict]] = [
    ("identity", "identity.name", {"name": "Priya"}),
    ("identity", "identity.current_city", {"current_city": "Mumbai"}),
    ("identity", "identity.country", {"country": "India"}),
    ("identity", "identity.timezone", {"timezone": "Asia/Kolkata"}),
    ("identity", "identity.occupation", {"occupation": "data scientist"}),
    ("identity", "identity.company", {"company": "Acme Corp"}),
    ("identity", "identity.primary_email", {"primary_email": "priya@acme.com"}),
    ("preferences", "preferences.response_style", {"response_style": "concise"}),
    ("preferences", "preferences.humor_level", {"humor_level": "light"}),
    ("preferences", "preferences.email_tone", {"email_tone": "friendly"}),
    ("preferences", "preferences.favourite_editor", {"value": "VS Code"}),
    ("workflow", "workflow.prefers_draft_before_send", {"prefers_draft_before_send": True}),
    ("workflow", "workflow.primary_llm", {"primary_llm": "Claude"}),
    ("contacts", "contacts.frequent_recipient.keshav@acme.com", {"email": "keshav@acme.com"}),
    ("projects", "projects.project.zephyr", {"name": "Zephyr"}),
    ("relations", "relations.best_friend", {"role": "best_friend", "name": "Elvin"}),
    ("relations", "relations.manager", {"role": "manager", "name": "Keshav"}),
    ("relations", "relations.sister", {"role": "sister", "name": "Anaya"}),
    ("corrections", "corrections.name_pronunciation", {"summary": "Priya is pree-yah"}),
    ("working_style", "working_style.focus", {"note": "deep work in the mornings"}),
    ("communication_style", "communication_style.format", {"note": "prefers bullet points"}),
    ("tool_preferences", "tool_preferences.language", {"tool": "Python"}),
    ("tool_preferences", "tool_preferences.shell", {"tool": "PowerShell"}),
    ("decision_style", "decision_style.approach", {"note": "data-driven, weighs trade-offs"}),
    ("preferences", "preferences.units", {"value": "metric"}),
]

# (utterance, gold_topic, gold_value) — the extractor should surface the value.
EXTRACTION_GOLD: list[tuple[str, str, str]] = [
    ("My name is Priya.", "identity", "Priya"),
    ("I currently live in Mumbai.", "identity", "Mumbai"),
    ("My timezone is Asia/Kolkata.", "identity", "Asia/Kolkata"),
    ("I work as a data scientist.", "identity", "data scientist"),
    ("My company is Acme Corp.", "identity", "Acme Corp"),
    ("You can email me at priya@acme.com.", "identity", "priya@acme.com"),
    ("I prefer concise responses.", "preferences", "concise"),
    ("Keep the humor light please.", "preferences", "light"),
    ("My favourite editor is VS Code.", "preferences", "VS Code"),
    ("Always draft my emails before sending them.", "workflow", "draft"),
    ("My best friend is Elvin.", "relations", "Elvin"),
    ("My manager is Keshav.", "relations", "Keshav"),
    ("I'm working on Project Zephyr.", "projects", "Zephyr"),
    ("I mostly code in Python.", "tool_preferences", "Python"),
    ("I like to receive information as bullet points.", "communication_style", "bullet"),
]

# Memory questions — the model should choose recall(scope=personal).
RECALL_QUESTIONS: list[str] = [
    "Who is my best friend?",
    "What's my favourite editor?",
    "Which city do I live in?",
    "What's my email address?",
    "Who is my manager?",
    "What company do I work for?",
    "What timezone am I in?",
    "What's the name of my project?",
]

# (question, expected_substring) — the seeded fact must appear in the answer.
E2E_QUESTIONS: list[tuple[str, str]] = [
    ("Who is my best friend?", "Elvin"),
    ("What's my favourite editor?", "VS Code"),
    ("Which city do I live in?", "Mumbai"),
    ("Who is my manager?", "Keshav"),
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PhaseScore:
    hits: int = 0
    total: int = 0
    error: str | None = None

    @property
    def accuracy(self) -> float | None:
        if self.error is not None or self.total == 0:
            return None
        return self.hits / self.total

    def cell(self) -> str:
        if self.error is not None:
            return f"ERR ({self.error[:24]})"
        if self.total == 0:
            return "-"
        return f"{self.accuracy:.0%} ({self.hits}/{self.total})"


@dataclass
class ModelResult:
    model: str
    skipped: str | None = None
    extraction: PhaseScore = field(default_factory=PhaseScore)
    recall_selection: PhaseScore = field(default_factory=PhaseScore)
    end_to_end: PhaseScore = field(default_factory=PhaseScore)


# ---------------------------------------------------------------------------
# Temp fixture wiring (isolated; never touches the repo data/ tree)
# ---------------------------------------------------------------------------

def _seed_fixture(temp_root: Path, user_id: str):
    """Journal all SEED_FACTS as applied events, backfill the FTS index, and
    replay into the markdown store. Returns (journal_dir, store, index)."""
    import core.paths as core_paths
    core_paths.PERSONAL_MEMORY_DIR = temp_root  # runtime redirect (script-only)
    core_paths.PERSONAL_MEMORY_SNAPSHOTS_DIR = temp_root / "snapshots"

    from core.memory_journal import JournalStore, make_event
    from core.memory_schema import statement_for
    from core.memory_sqlite import MemorySQLiteIndex
    from core.memory_replayer import replay
    from core.personal_memory_store import PersonalMemoryStore

    journal_dir = temp_root / "journal"
    index = MemorySQLiteIndex(db_path=temp_root / "memory.sqlite")
    journal = JournalStore(user_id="default", journal_dir=journal_dir, on_append=index.index_event)
    store = PersonalMemoryStore()

    for ordinal, (topic, key, value) in enumerate(SEED_FACTS):
        kind = "fact" if topic in {"identity", "contacts", "projects", "relations"} else "preference"
        journal.append(
            make_event(
                kind=kind, topic=topic, key=key, value=value, confidence=1.0,
                source="explicit", extractor="deterministic", session_id="seed",
                turn_id=f"seed_{ordinal}", observed_at="2026-05-01T10:00:00Z",
                applied=True, evidence={"text": f"seed fact {key}"},
                statement=statement_for(topic, key, value),
            )
        )
    replay(journal.load_all(), store=store)
    return journal_dir, store, index


def _build_broker(store, index, temp_root: Path, *, journal_store=None):
    from core.retrieval_broker import RetrievalBroker
    from core.task_history import TaskHistoryStore
    return RetrievalBroker(
        store=store,
        task_store=TaskHistoryStore(temp_root / "tasks" / "history.jsonl"),
        journal_store=journal_store,
        sqlite_index=index,
        session_store=None,
        rag_system=None,
        vector_store=None,
        user_id="default",
    )


def _build_deps(http_client, store, journal, index, broker, temp_root: Path, user_id: str):
    """A duck-typed deps object for the main agent's tools (pydantic-ai does not
    runtime-enforce deps_type, so a SimpleNamespace suffices — see the repo's
    phase0 fallback-tools test)."""
    from types import SimpleNamespace
    from core.confirmation_gate import ConfirmationGate
    from core.personal_memory_prompt import PersonalMemoryPromptBuilder
    from core.task_history import TaskHistoryStore

    gate = ConfirmationGate(
        journal=journal, store=store, state_path=temp_root / "confirmation_state.json"
    )
    return SimpleNamespace(
        http_client=http_client,
        session_store=SimpleNamespace(message_history=[], session_id="model_compare"),
        memory_store=None,
        personal_memory_store=store,
        personal_memory_prompt=PersonalMemoryPromptBuilder(store),
        journal_store=journal,
        confirmation_gate=gate,
        task_history_store=TaskHistoryStore(temp_root / "tasks" / "history.jsonl"),
        rag_system=None,
        sqlite_index=index,
        retrieval_broker=broker,
        search_cache={},
        turn_counter=0,
        user_id=user_id,
        intent="",
        memory_context="",
    )


# ---------------------------------------------------------------------------
# Phase 1 — extraction accuracy
# ---------------------------------------------------------------------------

async def _score_extraction(model_obj) -> PhaseScore:
    from pydantic_ai import Agent
    from core.personal_memory_extract import _extract_stage_b_json_array

    prompt_path = ROOT_DIR / "core" / "system_prompts" / "memory_extractor.txt"
    try:
        system_prompt = prompt_path.read_text(encoding="utf-8")
    except Exception:
        system_prompt = (
            "Extract personal facts the user revealed. Return ONLY a JSON array of "
            "objects with keys: topic, key, value, confidence, source, evidence."
        )

    agent = Agent(model_obj, output_type=str, output_retries=1,
                  instructions="Return only a valid JSON array. No prose.")

    score = PhaseScore(total=len(EXTRACTION_GOLD))
    for utterance, gold_topic, gold_value in EXTRACTION_GOLD:
        try:
            result = await agent.run(f"{system_prompt}\n\nConversation message:\n{utterance}")
            items = _extract_stage_b_json_array(result.output)
        except Exception as exc:
            score.error = f"{exc.__class__.__name__}"
            return score
        if _extraction_matches(items, gold_topic, gold_value):
            score.hits += 1
    return score


def _extraction_matches(items: list[dict], gold_topic: str, gold_value: str) -> bool:
    gold_value_low = gold_value.lower()
    for item in items:
        topic = str(item.get("topic", "")).strip().lower()
        blob = json.dumps(
            {"key": item.get("key", ""), "value": item.get("value", "")},
            ensure_ascii=False,
        ).lower()
        if topic == gold_topic and gold_value_low in blob:
            return True
    return False


# ---------------------------------------------------------------------------
# Phase 2 — recall-tool selection
# ---------------------------------------------------------------------------

async def _score_recall_selection(model_obj, main_agent, deps) -> PhaseScore:
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.usage import RunUsage

    score = PhaseScore(total=len(RECALL_QUESTIONS))
    for question in RECALL_QUESTIONS:
        try:
            result = await main_agent.run(
                question, model=model_obj, deps=deps, usage=RunUsage()
            )
        except Exception as exc:
            score.error = f"{exc.__class__.__name__}"
            return score
        chose_recall = any(
            isinstance(part, ToolCallPart) and part.tool_name == "recall"
            for message in result.all_messages()
            if isinstance(message, ModelResponse)
            for part in message.parts
        )
        if chose_recall:
            score.hits += 1
    return score


# ---------------------------------------------------------------------------
# Phase 3 — end-to-end answer-contains across a simulated restart
# ---------------------------------------------------------------------------

async def _score_end_to_end(model_obj, main_agent, deps) -> PhaseScore:
    from pydantic_ai.usage import RunUsage

    score = PhaseScore(total=len(E2E_QUESTIONS))
    for question, expected in E2E_QUESTIONS:
        try:
            result = await main_agent.run(
                question, model=model_obj, deps=deps, usage=RunUsage()
            )
        except Exception as exc:
            score.error = f"{exc.__class__.__name__}"
            return score
        if expected.lower() in str(result.output).lower():
            score.hits += 1
    return score


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------

async def _run_model(model_str: str, temp_root: Path) -> ModelResult:
    import httpx
    import apps.turtle_server as ts

    out = ModelResult(model=model_str)

    model_obj = ts._build_model_from_str(model_str, ts.agents_mgr.model_settings)
    if model_obj is None:
        out.skipped = "no provider/key available for this model string"
        return out

    user_id = "model_compare"
    model_root = temp_root / model_str.replace(":", "_").replace("/", "_")
    model_root.mkdir(parents=True, exist_ok=True)

    # Seed a fresh fixture per model so runs never cross-contaminate.
    journal_dir, store, live_index = _seed_fixture(model_root, user_id)

    # Phase 1: extraction accuracy.
    out.extraction = await _score_extraction(model_obj)

    async with httpx.AsyncClient() as client:
        broker = _build_broker(store, live_index, model_root, journal_store=None)
        deps = _build_deps(client, store, _reopen_journal(journal_dir), live_index, broker, model_root, user_id)

        # Phase 2: recall-tool selection against the live main-agent toolset.
        out.recall_selection = await _score_recall_selection(
            model_obj, ts.agents_mgr.main_assistant, deps
        )
        live_index.close()

        # Phase 3: simulate a restart — rebuild the index from the journal alone.
        restart_journal, restart_index = _restart(journal_dir, model_root)
        restart_store = _fresh_store()
        restart_broker = _build_broker(restart_store, restart_index, model_root, journal_store=restart_journal)
        restart_deps = _build_deps(
            client, restart_store, restart_journal, restart_index, restart_broker, model_root, user_id
        )
        out.end_to_end = await _score_end_to_end(
            model_obj, ts.agents_mgr.main_assistant, restart_deps
        )
        restart_index.close()

    return out


def _reopen_journal(journal_dir: Path):
    from core.memory_journal import JournalStore
    return JournalStore(user_id="default", journal_dir=journal_dir)


def _fresh_store():
    from core.personal_memory_store import PersonalMemoryStore
    return PersonalMemoryStore()


def _restart(journal_dir: Path, model_root: Path):
    from core.memory_journal import JournalStore
    from core.memory_sqlite import MemorySQLiteIndex
    journal = JournalStore(user_id="default", journal_dir=journal_dir)
    index = MemorySQLiteIndex(db_path=model_root / "memory_restart.sqlite")
    index.backfill_from_journal(journal)
    return journal, index


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _render_markdown(results: list[ModelResult]) -> str:
    lines = [
        "| Model | Extraction | Recall selection | End-to-end |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        if r.skipped:
            lines.append(f"| `{r.model}` | SKIPPED — {r.skipped} | | |")
            continue
        lines.append(
            f"| `{r.model}` | {r.extraction.cell()} | "
            f"{r.recall_selection.cell()} | {r.end_to_end.cell()} |"
        )
    return "\n".join(lines)


def _to_json(results: list[ModelResult]) -> list[dict]:
    payload = []
    for r in results:
        if r.skipped:
            payload.append({"model": r.model, "skipped": r.skipped})
            continue
        payload.append({
            "model": r.model,
            "extraction": {"hits": r.extraction.hits, "total": r.extraction.total,
                           "accuracy": r.extraction.accuracy, "error": r.extraction.error},
            "recall_selection": {"hits": r.recall_selection.hits, "total": r.recall_selection.total,
                                 "accuracy": r.recall_selection.accuracy, "error": r.recall_selection.error},
            "end_to_end": {"hits": r.end_to_end.hits, "total": r.end_to_end.total,
                           "accuracy": r.end_to_end.accuracy, "error": r.end_to_end.error},
        })
    return payload


async def _main_async(models: list[str], output: Path | None) -> int:
    temp_root = Path(tempfile.mkdtemp(prefix="turtle_model_compare_"))
    print(f"# TURTLE model comparison (§7)\n\nScratch dir: {temp_root}\n")

    results: list[ModelResult] = []
    for model_str in models:
        print(f"→ {model_str} ...")
        try:
            result = await _run_model(model_str, temp_root)
        except Exception as exc:  # a whole-model failure must not abort the sweep
            result = ModelResult(model=model_str, skipped=f"crashed: {exc.__class__.__name__}: {exc}")
        results.append(result)
        if result.skipped:
            print(f"   skipped: {result.skipped}")
        else:
            print(f"   extraction={result.extraction.cell()} "
                  f"recall={result.recall_selection.cell()} e2e={result.end_to_end.cell()}")

    table = _render_markdown(results)
    print("\n" + table + "\n")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"models": _to_json(results)}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote JSON results to {output}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live model-comparison harness for TURTLE personal memory (§7). "
                    "Makes real API calls; requires provider keys. Not run in CI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="Model strings (provider:name). Default: the config stack "
             f"({', '.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write JSON results (e.g. output/model_compare.json).",
    )
    args = parser.parse_args()

    try:
        from core.env import load_env
        load_env(override=True)
    except Exception as exc:
        print(f"WARNING: could not load .env ({exc}); relying on process environment.")

    exit_code = asyncio.run(_main_async(args.models, args.output))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
