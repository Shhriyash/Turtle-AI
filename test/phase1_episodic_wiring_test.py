import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

from rag.system import complete_rag
from rag.system.complete_rag import TurtleRAGSystem


def test_per_user_staging_uses_user_storage_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(complete_rag, "RAG_DATA_DIR", tmp_path)
    monkeypatch.setattr(complete_rag, "get_vector_storage", lambda *args, **kwargs: MagicMock())

    system = TurtleRAGSystem(user_id="usr_x")

    assert system.temp_session_file == tmp_path / "usr_x" / "current_session.json"


def test_leftover_staging_is_indexed_before_new_session(tmp_path):
    async def run():
        storage_dir = tmp_path / "s"

        first = TurtleRAGSystem(storage_dir=storage_dir)
        await first.start_session(session_id="old_sess")
        first.add_conversation("first user message", "first assistant response")
        first.add_conversation("second user message", "second assistant response")

        resumed_calls = []

        def resumed_index_stub(*, session_id, conversations, creation_time=None):
            resumed_calls.append((session_id, list(conversations)))

        resumed = TurtleRAGSystem(storage_dir=storage_dir)
        resumed._index_session_conversations = resumed_index_stub

        resumed_session_id = await resumed.start_session(session_id="old_sess")

        assert resumed_session_id == "old_sess"
        assert resumed_calls == []

        indexed_calls = []

        def index_stub(*, session_id, conversations, creation_time=None):
            indexed_calls.append((session_id, list(conversations)))

        fresh = TurtleRAGSystem(storage_dir=storage_dir)
        fresh._index_session_conversations = index_stub

        new_session_id = await fresh.start_session(session_id="new_sess")

        assert new_session_id == "new_sess"
        assert len(indexed_calls) == 1
        assert indexed_calls[0][0] == "old_sess"
        assert len(indexed_calls[0][1]) == 2

        staged = json.loads((storage_dir / "current_session.json").read_text(encoding="utf-8"))
        assert staged["session_id"] == "new_sess"

    asyncio.run(run())


def test_history_tool_routes_personal_facts_to_recall():
    prompt_path = Path(__file__).resolve().parents[1] / "core" / "system_prompts" / "tools" / "history_tool.md"
    text = prompt_path.read_text(encoding="utf-8")

    assert 'recall(scope="personal")' in text or 'scope="personal"' in text
    assert "authoritative source for anything the user has told Turtle before" not in text
