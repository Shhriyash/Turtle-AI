import datetime
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import ModelRequest, UserPromptPart


def test_turn_instructions_include_current_memory_and_live_year(monkeypatch, tmp_path):
    import core.paths as paths

    monkeypatch.setattr(paths, "PERSONAL_MEMORY_DIR", tmp_path)

    from apps.turtle_server import _build_turn_instructions

    state = SimpleNamespace(user_id="", memory_context="[Identity]\n- Name: Shriyash")

    first = _build_turn_instructions(state)
    second = _build_turn_instructions(state)

    current_year = str(datetime.datetime.now(datetime.UTC).year)

    assert "[Identity]\n- Name: Shriyash" in first
    assert "Current user memory (authoritative; this is the only current copy" in first
    assert "Current date and time:" in first
    assert current_year in first
    assert current_year in second


def test_turn_instructions_omit_memory_section_when_empty(monkeypatch, tmp_path):
    import core.paths as paths

    monkeypatch.setattr(paths, "PERSONAL_MEMORY_DIR", tmp_path)

    from apps.turtle_server import _build_turn_instructions

    state = SimpleNamespace(user_id="", memory_context="")

    result = _build_turn_instructions(state)

    assert "Current user memory" not in result


def test_strip_legacy_memory_wrappers_unwraps_only_wrapped_user_turns():
    from core.session_store import _strip_legacy_memory_wrappers

    wrapped = ModelRequest(
        parts=[
            UserPromptPart(
                content=(
                    "Relevant user memory:\n"
                    "[Identity]\n"
                    "- Name: an AI engineer\n\n"
                    "User request:\n"
                    "its Shriyash"
                )
            )
        ]
    )
    bare = ModelRequest(parts=[UserPromptPart(content="hello there")])

    stripped = _strip_legacy_memory_wrappers([wrapped, bare])

    assert stripped[0].parts[0].content == "its Shriyash"
    assert stripped[1].parts[0].content == "hello there"


def test_static_main_prompt_has_no_frozen_clock_and_rubric_mentions_authoritative_subset():
    import apps.turtle_server as turtle_server

    assert "Current date and time:" not in turtle_server.MAIN_ASSISTANT_PROMPT

    prompt_path = Path(__file__).resolve().parents[1] / "core" / "system_prompts" / "main_assistant.txt"
    assert "authoritative and may be a subset" in prompt_path.read_text(encoding="utf-8")
