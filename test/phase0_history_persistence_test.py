from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel

import apps.turtle_server as ts


def test_persist_history_appends_new_messages_without_shrinking_prior():
    class FakeResponse:
        def new_messages(self):
            return ["n1", "n2"]

        def all_messages(self):
            return ["x"]

    assert ts._persist_history(["p1"], FakeResponse()) == ["p1", "n1", "n2"]


def test_persist_history_falls_back_to_all_messages_when_new_messages_unavailable():
    class FakeResponse:
        def new_messages(self):
            raise RuntimeError("missing")

        def all_messages(self):
            return ["x"]

    assert ts._persist_history(["p1"], FakeResponse()) == ["x"]


def test_persist_history_preserves_full_conversation_when_history_processor_trims():
    agent = Agent(TestModel(), history_processors=[lambda msgs: msgs[-2:]])
    history = []

    for i in range(5):
        result = agent.run_sync(f"turn {i}", message_history=list(history) or None)
        history = ts._persist_history(history, result)

    assert any("turn 0" in str(m) for m in history)
    assert _count_user_turns(history) == 5


def _count_user_turns(history):
    count = 0
    for msg in history:
        if not isinstance(msg, ModelRequest):
            continue
        if any(isinstance(part, UserPromptPart) for part in msg.parts):
            count += 1
    return count
