"""
Phase 8 — the assistant must use the in-session conversation, not treat it as a
memory-store lookup.

Live symptom (Discord): the model answered "I don't have a record" / "I don't
have access to that" for a fact the user stated one turn earlier, even though the
message_history was correctly threaded into the model's context. Root cause was
the system prompt framing "something earlier in this conversation" as a `recall`
operation, so the model searched the (empty) memory store instead of reading the
visible turns.

This guards the prompt fix so it can't silently regress.
"""
from __future__ import annotations

from pathlib import Path

import core.system_prompts as sp

_PROMPT = (Path(sp.__file__).resolve().parent / "main_assistant.txt").read_text(encoding="utf-8").lower()


def test_prompt_directs_model_to_use_visible_conversation():
    assert "conversation in front of you" in _PROMPT
    # it must explicitly tell the model NOT to recall for something already visible
    assert "do not call recall" in _PROMPT


def test_prompt_forbids_no_record_for_visible_conversation():
    # the "I don't have a record" phrasing must be explicitly forbidden for
    # something said earlier in the conversation (not left as a bare instruction).
    assert "do not say \"i don't have a record\"" in _PROMPT
    assert "earlier in this conversation" in _PROMPT


def test_prompt_scopes_recall_to_previous_session_or_scrolled_out():
    # recall is for a previous session / content scrolled out of context — not
    # for anything still visible in the current conversation.
    assert "previous session" in _PROMPT
    assert "scrolled out" in _PROMPT
