"""
core/channel_gate.py — unit tests.

Covers the narrow-match rules that let a channel (Discord/Telegram/WhatsApp/
Slack/iMessage) user answer a ConfirmationGate prompt in plain chat text
without reopening the pre-Phase-4 "bare yes" hazard: an outstanding prompt
must exist for the exact key, within TTL, and the reply must parse as a
short, unambiguous yes/no token.
"""
from __future__ import annotations

from core.channel_gate import ChannelGateBuffer, parse_gate_answer


# ---------------------------------------------------------------------------
# parse_gate_answer
# ---------------------------------------------------------------------------

def test_parse_accepts_common_yes_tokens():
    for word in ["yes", "Yes", "YES ", "y", "yep", "yeah", "yup", "sure", "ok", "okay", "confirm", "confirmed", "correct"]:
        assert parse_gate_answer(word) is True, f"{word!r} should parse as accept"


def test_parse_accepts_common_no_tokens():
    for word in ["no", "No", "n", "nope", "nah", "cancel", "skip", "negative", "reject"]:
        assert parse_gate_answer(word) is False, f"{word!r} should parse as reject"


def test_parse_trims_punctuation_and_case():
    assert parse_gate_answer("Yes!") is True
    assert parse_gate_answer("  NO.  ") is False
    assert parse_gate_answer("Yep,") is True


def test_parse_allows_short_two_word_phrase():
    assert parse_gate_answer("yes please") is True
    assert parse_gate_answer("no thanks") is False


def test_parse_rejects_longer_sentences_even_if_they_contain_yes():
    """This is the exact hazard Phase 4 closed off — a sentence that merely
    CONTAINS "yes" must never be treated as a gate answer."""
    assert parse_gate_answer("yes but only save my name, not my city") is None
    assert parse_gate_answer("well yes I suppose that's fine") is None


def test_parse_rejects_unrelated_text():
    assert parse_gate_answer("what's the weather today") is None
    assert parse_gate_answer("send an invoice to Acme") is None


def test_parse_rejects_empty():
    assert parse_gate_answer("") is None
    assert parse_gate_answer("   ") is None
    assert parse_gate_answer(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ChannelGateBuffer — the three guards
# ---------------------------------------------------------------------------

def test_no_outstanding_prompt_returns_none():
    buf = ChannelGateBuffer()
    assert buf.try_consume_answer(("u1", "telegram"), "yes") is None


def test_answers_within_ttl_when_outstanding():
    buf = ChannelGateBuffer(ttl_seconds=300)
    key = ("u1", "telegram")
    buf.note_prompt(key, ("ev1", "ev2"), now=1000.0)
    result = buf.try_consume_answer(key, "yes", now=1010.0)
    assert result == (True, ("ev1", "ev2"))


def test_reject_answer_returns_false_verdict():
    buf = ChannelGateBuffer(ttl_seconds=300)
    key = ("u1", "telegram")
    buf.note_prompt(key, ("ev1",), now=1000.0)
    result = buf.try_consume_answer(key, "no", now=1010.0)
    assert result == (False, ("ev1",))


def test_expired_prompt_is_not_answered():
    buf = ChannelGateBuffer(ttl_seconds=60)
    key = ("u1", "telegram")
    buf.note_prompt(key, ("ev1",), now=1000.0)
    # 61s later — past the 60s TTL.
    result = buf.try_consume_answer(key, "yes", now=1061.0)
    assert result is None


def test_non_answer_does_not_consume_the_outstanding_prompt():
    """A message that fails to parse as yes/no must NOT burn the pending
    prompt — the user should still be able to answer correctly on a later
    turn within the TTL window."""
    buf = ChannelGateBuffer(ttl_seconds=300)
    key = ("u1", "telegram")
    buf.note_prompt(key, ("ev1",), now=1000.0)
    assert buf.try_consume_answer(key, "what time is it", now=1005.0) is None
    # Still outstanding — a real answer 5s later succeeds.
    result = buf.try_consume_answer(key, "yes", now=1010.0)
    assert result == (True, ("ev1",))


def test_answer_is_consumed_once():
    """Answering clears the outstanding prompt — a second "yes" with no new
    prompt surfaced must not silently re-trigger anything."""
    buf = ChannelGateBuffer(ttl_seconds=300)
    key = ("u1", "telegram")
    buf.note_prompt(key, ("ev1",), now=1000.0)
    first = buf.try_consume_answer(key, "yes", now=1005.0)
    assert first == (True, ("ev1",))
    second = buf.try_consume_answer(key, "yes", now=1006.0)
    assert second is None


def test_different_channel_keys_are_isolated():
    """The same user answering "yes" on Discord must not resolve a prompt
    that was actually surfaced on Telegram."""
    buf = ChannelGateBuffer(ttl_seconds=300)
    buf.note_prompt(("u1", "telegram"), ("ev1",), now=1000.0)
    assert buf.try_consume_answer(("u1", "discord"), "yes", now=1005.0) is None
    # The telegram prompt is untouched.
    assert buf.try_consume_answer(("u1", "telegram"), "yes", now=1006.0) == (True, ("ev1",))


def test_note_prompt_overwrites_previous_outstanding_for_same_key():
    """A second prompt surfaced before the first was answered replaces it —
    only the most recently asked question should be answerable."""
    buf = ChannelGateBuffer(ttl_seconds=300)
    key = ("u1", "telegram")
    buf.note_prompt(key, ("ev1",), now=1000.0)
    buf.note_prompt(key, ("ev2",), now=1001.0)
    result = buf.try_consume_answer(key, "yes", now=1002.0)
    assert result == (True, ("ev2",))


def test_note_prompt_ignores_empty_event_ids():
    buf = ChannelGateBuffer()
    buf.note_prompt(("u1", "telegram"), (), now=1000.0)
    assert buf.try_consume_answer(("u1", "telegram"), "yes", now=1001.0) is None


def test_has_outstanding():
    buf = ChannelGateBuffer(ttl_seconds=60)
    key = ("u1", "telegram")
    assert buf.has_outstanding(key, now=1000.0) is False
    buf.note_prompt(key, ("ev1",), now=1000.0)
    assert buf.has_outstanding(key, now=1030.0) is True
    assert buf.has_outstanding(key, now=1061.0) is False


def test_clear_removes_outstanding():
    buf = ChannelGateBuffer(ttl_seconds=300)
    key = ("u1", "telegram")
    buf.note_prompt(key, ("ev1",), now=1000.0)
    buf.clear(key)
    assert buf.try_consume_answer(key, "yes", now=1001.0) is None


def test_max_entries_evicts_soonest_to_expire():
    """Defensive cap: under a pathological number of distinct senders, the
    buffer must not grow unbounded — it drops the entry nearest to expiry."""
    buf = ChannelGateBuffer(ttl_seconds=300, max_entries=2)
    buf.note_prompt(("u1", "telegram"), ("ev1",), now=1000.0)   # expires 1300
    buf.note_prompt(("u2", "telegram"), ("ev2",), now=1100.0)   # expires 1400
    # Adding a third distinct key over cap should evict u1 (soonest expiry).
    buf.note_prompt(("u3", "telegram"), ("ev3",), now=1150.0)   # expires 1450
    assert buf.try_consume_answer(("u1", "telegram"), "yes", now=1160.0) is None
    assert buf.try_consume_answer(("u2", "telegram"), "yes", now=1160.0) == (True, ("ev2",))
    assert buf.try_consume_answer(("u3", "telegram"), "yes", now=1160.0) == (True, ("ev3",))
