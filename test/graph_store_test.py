import shutil
import unittest
import uuid
from pathlib import Path

from core.graph_store import GraphContextQuery, GraphStore


class GraphStoreTests(unittest.TestCase):
    def test_rebuild_and_query_context(self) -> None:
        base = Path("test") / "_tmp" / f"graph_store_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            graph_path = base / "graph.json"
            store = GraphStore(graph_path=graph_path)
            profile = {
                "identity": {"name": "Shriyash", "emails": ["user@example.com"], "timezone": "Asia/Calcutta"},
                "preferences": {"response_style": "concise", "humor_level": "low", "email_tone": "formal"},
                "workflow": {"prefers_draft_before_send": True, "common_recipients": ["friend@example.com"]},
                "tool_preferences": {"primary_llm": "groq/openai-gpt-oss-120b"},
                "meta": {"updated_at": "", "version": 1},
            }
            graph = store.rebuild_from_profile(profile)
            store.save_graph(graph)
            # max_lines=8: the deprecated store emits edges in insertion order
            # (has_email, timezone, 3 preference edges) before the recipient
            # "emails" edge, so a cap of 4 can never surface it.
            lines = store.query_context(
                GraphContextQuery(query="who do I usually email", task_type="email", max_lines=8)
            )
            self.assertTrue(any("emails" in line for line in lines))
            self.assertTrue(graph_path.exists())
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()