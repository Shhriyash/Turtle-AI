import json
from pathlib import Path

from pydantic_ai.messages import ModelRequest, UserPromptPart

from core.config import settings
from core.personal_memory_extract import (
    _evidence_supports_value,
    _extract_stage_b_json_array,
    _stage_b_turns_from_messages,
)


def parse_turn_contract(raw: str) -> list[dict]:
    data = json.loads(raw)
    return data if isinstance(data, list) else data.get("facts", data.get("items", []))


def test_turn_contract_accepts_primary_and_tolerated_shapes():
    item = {"topic": "identity"}

    assert parse_turn_contract(json.dumps({"facts": [item]})) == [item]
    assert parse_turn_contract(json.dumps([item])) == [item]
    assert parse_turn_contract(json.dumps({"items": [item]})) == [item]


def test_stage_b_array_parser_unchanged():
    item = {
        "kind": "fact",
        "topic": "identity",
        "key": "name",
        "value": {"name": "Shriyash"},
        "confidence": 0.99,
        "source": "explicit",
        "evidence": {"quote": "my name is Shriyash"},
    }

    assert _extract_stage_b_json_array(json.dumps([item])) == [item]
    assert _extract_stage_b_json_array("prose\n" + json.dumps([item]) + "\nmore prose") == [item]


def test_memory_extractor_prompt_contract_mentions_facts_and_relations():
    prompt_path = Path(__file__).resolve().parents[1] / "core" / "system_prompts" / "memory_extractor.txt"
    text = prompt_path.read_text(encoding="utf-8")

    assert '"facts"' in text
    assert "relations" in text
    assert "Return a JSON array" not in text


def test_stage_b_turns_unwrap_injected_memory_preamble():
    message = ModelRequest(
        parts=[
            UserPromptPart(
                content=(
                    "Relevant user memory:\n"
                    "[Identity]\n"
                    "- Name: X\n\n"
                    "User request:\n"
                    "my editor is VS Code"
                )
            )
        ]
    )

    turns = _stage_b_turns_from_messages([message], max_turns=10)

    assert turns == [{"role": "user", "text": "my editor is VS Code"}]


def test_evidence_supports_value_requires_grounded_leaf_values():
    assert _evidence_supports_value(
        {"name": "Shriyash"},
        {"note": "user said: my name is Shriyash"},
    )
    assert not _evidence_supports_value(
        {"name": "Shriyash"},
        {"note": "unrelated"},
    )
    assert not _evidence_supports_value({}, {"note": "my name is Shriyash"})


def test_turn_extractor_default_model_is_70b():
    assert settings.personal_memory_turn_extractor_model == "llama-3.3-70b-versatile"
