import apps.turtle_server as ts


def test_memory_tools_visible_on_every_intent():
    memory_tools = {"recall", "history_tool"}

    for tool_names in ts._TOOL_NAMES_BY_INTENT.values():
        assert tool_names >= memory_tools

    assert "send_email_assistant" in ts._TOOL_NAMES_BY_INTENT["email"]
    assert ts._TOOL_NAMES_BY_INTENT["chitchat"] == {"recall", "history_tool"}
