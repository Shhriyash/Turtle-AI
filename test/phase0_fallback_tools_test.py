import asyncio
import types

from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

import apps.turtle_server as ts


EXPECTED = {
    "search_web",
    "search_url",
    "send_email_assistant",
    "recall",
    "calendar_create",
    "calendar_list",
    "remember",
}


def test_every_rung_has_tools():
    agents = [ts.agents_mgr.main_assistant, *ts.agents_mgr.main_assistant_fallbacks]
    assert agents

    for i, ag in enumerate(agents):
        tm = TestModel(call_tools=[])
        asyncio.run(
            ag.run(
                "hi",
                model=tm,
                deps=types.SimpleNamespace(intent="", user_id=""),
                usage=RunUsage(),
            )
        )
        names = {t.name for t in tm.last_model_request_parameters.function_tools}
        assert EXPECTED <= names, f"agent {i} missing {EXPECTED - names}"
