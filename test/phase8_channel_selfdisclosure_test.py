"""
Phase 8 — self-disclosed facts must be remembered on channels (Discord/Slack),
not stranded in the confirmation gate.

Live symptom: a Discord user said "send it to shriyashbeohar1@gmail (that's my
gmail)". It was extracted but marked source=inferred / applied=False (pending),
and Discord has no "To confirm" surface, so it was lost. Meanwhile decide_write_
policy WOULD auto-apply an explicit+grounded identity fact on any channel — two
things blocked it:
  1. the extractor prompt never defined a self-disclosure as `explicit`;
  2. _evidence_supports_value was a brittle verbatim-substring check, so the
     moment the model canonicalized "...@gmail" -> "...@gmail.com" the value
     stopped being a substring of the evidence and `explicit` was downgraded to
     `inferred` (-> pending).

These tests cover the code lever (2): grounding is now normalization-robust, so a
lightly-canonicalized self-disclosure stays `explicit` and decide_write_policy
returns `applied`. (Lever 1 is a prompt change, asserted by contract below.)
"""
from __future__ import annotations

from core.personal_memory_extract import _evidence_supports_value
from core.memory_schema import decide_write_policy


_EMAIL_EVIDENCE = {"quote": "send it to shriyashbeohar1@gmail address, that's my gmail"}


def test_grounding_tolerates_email_canonicalization():
    """The exact failing case: the user wrote '...@gmail', the model stored
    '...@gmail.com' — still grounded (the appended 'com' is a short token)."""
    assert _evidence_supports_value(
        {"value": "shriyashbeohar1@gmail.com"}, _EMAIL_EVIDENCE
    ) is True


def test_grounding_still_accepts_verbatim():
    assert _evidence_supports_value({"value": "Indore"}, {"quote": "I live in Indore"}) is True
    assert _evidence_supports_value({"value": "nvim"}, {"quote": "my editor is nvim"}) is True


def test_grounding_rejects_hallucination():
    """A value whose distinctive token the user never said is NOT grounded —
    we must not auto-apply a fabricated or third-party email as the user's own."""
    assert _evidence_supports_value(
        {"value": "johndoe@gmail.com"}, _EMAIL_EVIDENCE
    ) is False
    assert _evidence_supports_value(
        {"value": "Mumbai"}, {"quote": "I live in Indore"}
    ) is False


def test_explicit_grounded_identity_applies_on_any_channel():
    """With grounding passing, a self-disclosed identity email auto-applies —
    no web confirmation surface needed. This is the policy the channel path hits."""
    grounded = _evidence_supports_value({"value": "shriyashbeohar1@gmail.com"}, _EMAIL_EVIDENCE)
    assert grounded is True
    assert decide_write_policy(
        source="explicit", topic="identity", confidence=0.95, evidence_supported=grounded
    ) == "applied"


def test_inferred_identity_still_gated():
    """A genuinely inferred identity fact must still be gated (don't reintroduce
    the junk-extraction problem)."""
    assert decide_write_policy(
        source="inferred", topic="identity", confidence=0.99, evidence_supported=True
    ) == "pending"


def test_extractor_prompt_defines_self_disclosure_as_explicit():
    """Contract: the per-turn extractor prompt teaches the LLM that a first-person
    self-disclosure is `explicit` (lever 1, so the classification even reaches the
    grounded-apply path)."""
    from pathlib import Path
    import core.system_prompts as sp
    text = (Path(sp.__file__).resolve().parent / "memory_extractor.txt").read_text(encoding="utf-8").lower()
    assert "explicit" in text and "self" in text and "first person" in text
    assert "verbatim" in text  # evidence must be the user's own words for grounding
