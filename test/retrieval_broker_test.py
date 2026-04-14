"""Tests for Step 7 — RetrievalBroker (token-budgeted memory context assembly)."""
import shutil
import unittest
import uuid
from pathlib import Path

from core.personal_memory_store import PersonalMemoryStore
from core.retrieval_broker import (
    DEFAULT_BUDGET,
    RetrievalBroker,
    RetrievalBudget,
    _estimate_tokens,
    _has_history_trigger,
    _trim_to_tokens,
)
from core.task_history import TaskHistoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(base: Path) -> PersonalMemoryStore:
    return PersonalMemoryStore(
        base_dir=base,
        index_path=base / "MEMORY.md",
        logs_dir=base / "logs",
        topic_paths={
            "identity": base / "identity.md",
            "preferences": base / "preferences.md",
            "workflow": base / "workflow.md",
            "contacts": base / "contacts.md",
            "projects": base / "projects.md",
            "corrections": base / "corrections.md",
        },
    )


class _FakeRagSystem:
    """Synchronous-looking async mock for TurtleRAGSystem.query_history."""

    def __init__(self, response: str = "") -> None:
        self._response = response

    async def query_history(self, query: str) -> str:
        return self._response


# ---------------------------------------------------------------------------
# Unit: token helpers
# ---------------------------------------------------------------------------

class TokenEstimatorTests(unittest.TestCase):
    def test_empty_string_is_zero_tokens(self) -> None:
        self.assertEqual(_estimate_tokens(""), 0)

    def test_four_chars_is_one_token(self) -> None:
        self.assertEqual(_estimate_tokens("abcd"), 1)

    def test_eight_chars_is_two_tokens(self) -> None:
        self.assertEqual(_estimate_tokens("abcdefgh"), 2)

    def test_odd_length_rounds_down(self) -> None:
        self.assertEqual(_estimate_tokens("abc"), 0)
        self.assertEqual(_estimate_tokens("abcde"), 1)


class TrimToTokensTests(unittest.TestCase):
    def test_short_text_unchanged(self) -> None:
        text = "Hello"
        self.assertEqual(_trim_to_tokens(text, 100), text)

    def test_text_at_exact_budget_unchanged(self) -> None:
        text = "a" * 40  # 40 chars = 10 tokens
        self.assertEqual(_trim_to_tokens(text, 10), text)

    def test_long_text_is_truncated_with_ellipsis(self) -> None:
        text = "a" * 100  # 100 chars = 25 tokens
        result = _trim_to_tokens(text, 10)  # max 40 chars
        self.assertLessEqual(len(result), 43)  # 40 + "..."
        self.assertTrue(result.endswith("..."))

    def test_zero_budget_returns_empty(self) -> None:
        self.assertEqual(_trim_to_tokens("hello world", 0), "")

    def test_negative_budget_returns_empty(self) -> None:
        self.assertEqual(_trim_to_tokens("hello world", -5), "")


# ---------------------------------------------------------------------------
# Unit: history trigger detection
# ---------------------------------------------------------------------------

class HistoryTriggerTests(unittest.TestCase):
    def test_plain_query_has_no_trigger(self) -> None:
        self.assertFalse(_has_history_trigger("What is the weather today?"))

    def test_yesterday_triggers(self) -> None:
        self.assertTrue(_has_history_trigger("What did we do yesterday?"))

    def test_last_week_triggers(self) -> None:
        self.assertTrue(_has_history_trigger("Remind me what we discussed last week"))

    def test_remember_triggers(self) -> None:
        self.assertTrue(_has_history_trigger("Do you remember the email I sent?"))

    def test_history_triggers(self) -> None:
        self.assertTrue(_has_history_trigger("Check the history of this project"))

    def test_told_you_triggers(self) -> None:
        self.assertTrue(_has_history_trigger("I told you my timezone before"))

    def test_case_insensitive(self) -> None:
        self.assertTrue(_has_history_trigger("YESTERDAY I mentioned this"))

    def test_email_task_no_trigger(self) -> None:
        self.assertFalse(_has_history_trigger("Send an email to Alice about the report"))


# ---------------------------------------------------------------------------
# Integration: RetrievalBroker.build_context
# ---------------------------------------------------------------------------

class RetrievalBrokerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.base = Path("test") / "_tmp" / f"broker_{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.store = _make_store(self.base)
        self.task_store = TaskHistoryStore(self.base / "tasks" / "history.jsonl")

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def _make_broker(self, rag_system=None, budget=DEFAULT_BUDGET) -> RetrievalBroker:
        return RetrievalBroker(
            store=self.store,
            task_store=self.task_store,
            rag_system=rag_system,
            budget=budget,
        )

    # --- empty store ---

    async def test_empty_store_returns_empty_string(self) -> None:
        broker = self._make_broker()
        result = await broker.build_context(task_type="general", query="hello")
        self.assertEqual(result, "")

    # --- Tier 1: index ---

    async def test_index_tier_present_when_topics_exist(self) -> None:
        self.store.write_topic("preferences", ["- Response style: concise"], {"title": "Preferences"})
        self.store.update_index_entry("preferences", "Tone and delivery defaults")
        broker = self._make_broker()
        result = await broker.build_context(task_type="general", query="hello")
        self.assertIn("[Memory Index]", result)
        self.assertIn("Preferences", result)

    async def test_index_tier_trimmed_to_budget(self) -> None:
        # Create many topics to push index beyond 60-token limit
        for i in range(10):
            self.store.write_topic(f"preferences", [f"- Pref {i}: value_{i}"], {"title": f"Topic {i}"})
            try:
                self.store.update_index_entry("preferences", f"Summary for topic {i} with extra text to push tokens")
            except Exception:
                pass
        budget = RetrievalBudget(index_tokens=5, topic_tokens=200, episodic_tokens=100, task_tokens=40, total_tokens=400)
        broker = self._make_broker(budget=budget)
        result = await broker.build_context(task_type="general", query="hello")
        # Index section should be heavily truncated
        index_section = result.split("\n\n")[0] if result else ""
        self.assertLessEqual(_estimate_tokens(index_section), 8)  # 5 + small rounding

    # --- Tier 2: topics ---

    async def test_topic_tier_contains_content(self) -> None:
        self.store.write_topic(
            "preferences",
            ["- Response style: concise", "- Humor level: low"],
            {"title": "Preferences"},
        )
        self.store.update_index_entry("preferences", "Tone defaults")
        broker = self._make_broker()
        result = await broker.build_context(task_type="general", query="what is my response style?")
        self.assertIn("- Response style: concise", result)

    async def test_email_task_type_loads_identity_and_contacts(self) -> None:
        self.store.write_topic("identity", ["- Name: Shriyash", "- Primary email: s@example.com"], {"title": "Identity"})
        self.store.update_index_entry("identity", "Name, email, timezone")
        broker = self._make_broker()
        result = await broker.build_context(task_type="email", query="send an email to Alice")
        self.assertIn("- Name: Shriyash", result)

    # --- Tier 3 + 4: episodic + task (history triggers) ---

    async def test_no_history_trigger_skips_episodic(self) -> None:
        rag = _FakeRagSystem(response='[{"content": "old conversation", "timestamp": "2026-01-01"}]')
        broker = self._make_broker(rag_system=rag)
        result = await broker.build_context(task_type="general", query="what is my name?")
        self.assertNotIn("[Past Conversations]", result)

    async def test_history_trigger_includes_episodic(self) -> None:
        rag = _FakeRagSystem(
            response='[{"content": "We discussed the Turtle project", "timestamp": "2026-01-01"}]'
        )
        broker = self._make_broker(rag_system=rag)
        result = await broker.build_context(task_type="general", query="Do you remember what we discussed last week?")
        self.assertIn("[Past Conversations]", result)
        self.assertIn("Turtle project", result)

    async def test_history_trigger_no_rag_system_skips_episodic(self) -> None:
        broker = self._make_broker(rag_system=None)
        result = await broker.build_context(task_type="general", query="what did we talk about yesterday?")
        self.assertNotIn("[Past Conversations]", result)

    async def test_history_trigger_rag_returns_not_found(self) -> None:
        rag = _FakeRagSystem(response="cannot find in history")
        broker = self._make_broker(rag_system=rag)
        result = await broker.build_context(task_type="general", query="do you remember last week?")
        self.assertNotIn("[Past Conversations]", result)

    async def test_task_tier_fires_on_history_trigger(self) -> None:
        # Record a task so the task store has something to return
        self.task_store.record(
            session_id="s1",
            turn_id="t1",
            task_type="email",
            status="completed",
            query="sent email to Alice",
            tool_used="send_email",
            outcome="sent",
        )
        broker = self._make_broker()
        result = await broker.build_context(task_type="general", query="remember when I sent email to Alice yesterday?")
        self.assertIn("Task history", result)

    async def test_task_tier_skipped_without_history_trigger(self) -> None:
        self.task_store.record(
            session_id="s1",
            turn_id="t1",
            task_type="email",
            status="completed",
            query="sent email to Alice",
            tool_used="send_email",
            outcome="sent",
        )
        broker = self._make_broker()
        result = await broker.build_context(task_type="general", query="send email to Bob")
        self.assertNotIn("Task history", result)

    # --- Hard budget cap ---

    async def test_total_budget_is_respected(self) -> None:
        # Fill all topics with lots of content
        for topic in ("preferences", "workflow", "contacts"):
            lines = [f"- Key {i}: {'value ' * 20}" for i in range(20)]
            self.store.write_topic(topic, lines, {"title": topic.title()})
            self.store.update_index_entry(topic, f"{topic} summary")
        rag = _FakeRagSystem(
            response='[{"content": "' + "word " * 200 + '", "timestamp": "2026-01-01"}]'
        )
        broker = self._make_broker(rag_system=rag)
        result = await broker.build_context(
            task_type="general",
            query="do you remember everything from last week?",
        )
        tokens = _estimate_tokens(result)
        # Allow some overhead for section separators but enforce the hard cap
        self.assertLessEqual(tokens, DEFAULT_BUDGET.total_tokens + 10)

    async def test_rag_error_is_swallowed(self) -> None:
        class _BrokenRag:
            async def query_history(self, q: str) -> str:
                raise RuntimeError("vector db offline")

        self.store.write_topic("preferences", ["- Response style: concise"], {"title": "Preferences"})
        self.store.update_index_entry("preferences", "Tone defaults")
        broker = self._make_broker(rag_system=_BrokenRag())
        # Should not raise; episodic tier is silently skipped
        result = await broker.build_context(
            task_type="general",
            query="do you remember what we talked about yesterday?",
        )
        self.assertIsInstance(result, str)
        self.assertNotIn("[Past Conversations]", result)


class RetrievalBudgetTests(unittest.TestCase):
    def test_default_budget_values(self) -> None:
        b = DEFAULT_BUDGET
        self.assertEqual(b.index_tokens, 60)
        self.assertEqual(b.topic_tokens, 200)
        self.assertEqual(b.episodic_tokens, 100)
        self.assertEqual(b.task_tokens, 40)
        self.assertEqual(b.total_tokens, 400)
        self.assertEqual(
            b.index_tokens + b.topic_tokens + b.episodic_tokens + b.task_tokens,
            b.total_tokens,
        )

    def test_custom_budget(self) -> None:
        b = RetrievalBudget(index_tokens=30, topic_tokens=100, episodic_tokens=50, task_tokens=20, total_tokens=200)
        self.assertEqual(b.total_tokens, 200)


if __name__ == "__main__":
    unittest.main()
