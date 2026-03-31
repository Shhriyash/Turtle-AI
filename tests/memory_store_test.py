import shutil
import unittest
import uuid
from pathlib import Path

from core.graph_store import GraphStore
from core.memory_store import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def _make_store(self, base: Path) -> MemoryStore:
        return MemoryStore(
            profile_path=base / "profile.json",
            events_path=base / "events.jsonl",
            episodes_path=base / "episodes.jsonl",
            state_path=base / "state.json",
            graph_store=GraphStore(graph_path=base / "graph.json"),
            flush_turns=2,
            flush_tokens=200,
            profile_max_lines=6,
        )

    def test_fact_and_preference_persist_to_profile(self) -> None:
        base = Path("tests") / "_tmp" / f"memory_store_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = self._make_store(base)
            store.record_turn(
                session_id="s1",
                turn_id="t1",
                user_text="My email is user@example.com and keep responses concise.",
                assistant_text="Noted.",
                task_type="general",
            )
            store.force_checkpoint(session_id="s1", reason="test")
            profile = store.load_profile()
            self.assertEqual(profile["identity"]["emails"], ["user@example.com"])
            self.assertEqual(profile["preferences"]["response_style"], "concise")
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_threshold_checkpoint_triggers_episode(self) -> None:
        base = Path("tests") / "_tmp" / f"memory_store_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            store = self._make_store(base)
            first = store.record_turn(
                session_id="s2",
                turn_id="t1",
                user_text="Hello",
                assistant_text="Hi",
                task_type="general",
            )
            second = store.record_turn(
                session_id="s2",
                turn_id="t2",
                user_text="My name is Shriyash",
                assistant_text="Saved",
                task_type="general",
            )
            self.assertFalse(first.triggered)
            self.assertTrue(second.triggered)
            episodes = (base / "episodes.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(episodes), 1)
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()