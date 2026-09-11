"""
core/system_prompts/main_assistant.txt and the per-turn instruction builder.

These tests pin two behaviours the user relies on:
1. Persona rule: the assistant is told, in the static prompt, never to use
   em (—) or en (–) dashes, and the prompt itself contains no such dashes.
2. Channel-aware formatting hint: _build_turn_instructions injects a line
   saying which output ruleset is active this turn (voice vs text/chat).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "core" / "system_prompts" / "main_assistant.txt"
)


def _prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def test_main_prompt_bans_long_dashes_in_output():
    text = _prompt_text()
    # An explicit ban clause the model can read and follow.
    assert "Never use em dashes" in text
    # The clause references both dashes so the model sees which characters.
    assert "—" in text  # only as an example inside the rule itself
    assert "–" in text
    # The clause is repeated near the formatting rules so bulleted / bold
    # replies don't smuggle a dash back in.
    assert text.count("em dash") >= 2 or text.count("em (—)") >= 1


def test_main_prompt_defines_text_formatting_rules():
    text = _prompt_text()
    assert "<text_formatting_rules>" in text
    # Chat channels are the audience for the markdown rules.
    assert "Telegram" in text and "Discord" in text
    assert "**bold**" in text
    assert "- item" in text or "`- ` bullets" in text


def test_main_prompt_keeps_voice_rules_scoped_to_voice_only():
    text = _prompt_text()
    # The voice rules must NOT be an unconditional ban that leaks into chat.
    assert "<voice_first_output_rules>" in text
    assert "ONLY when the per-turn instructions say the output channel is voice" in text


def test_build_turn_instructions_emits_voice_hint_on_voice_channel():
    from apps import turtle_server

    state = SimpleNamespace(user_id="", memory_context="", channel="web_voice")
    got = turtle_server._build_turn_instructions(state)
    assert "Output channel: voice" in got
    assert "voice_first_output_rules" in got


def test_build_turn_instructions_emits_text_hint_on_chat_channel():
    from apps import turtle_server

    for ch in ("telegram", "discord", "web", "whatsapp"):
        state = SimpleNamespace(user_id="", memory_context="", channel=ch)
        got = turtle_server._build_turn_instructions(state)
        assert f"Output channel: {ch} (text)" in got, f"missing text hint for {ch}"
        assert "text_formatting_rules" in got
        # The dash ban is repeated in the per-turn line so a fallback rung
        # that ignored part of the static prompt still sees it.
        assert "em (—)" in got or "em dash" in got.lower()


def test_build_turn_instructions_no_channel_hint_when_channel_unknown():
    """An empty channel (test harness or legacy caller) should skip the
    formatting line entirely rather than fabricate a rule."""
    from apps import turtle_server

    state = SimpleNamespace(user_id="", memory_context="", channel="")
    got = turtle_server._build_turn_instructions(state)
    assert "Output channel:" not in got
