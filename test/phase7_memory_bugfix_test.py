"""Regression tests for the two memory-coherence bugs the brutal review proved.

H1 — dedup flip-back: JournalStore.append_many must journal a restatement whose
     value differs from the CURRENT latest for a (topic, key), and must still
     suppress a true no-op (same value as the current latest).
H2 — decay divergence: a fact aged past DECAY_DAYS must be dropped from the
     SQLite read model (search / latest_for_key) the same way the replayer drops
     it from markdown, so retrieval and the projection agree. Identity / explicit
     / migration facts and fresh facts are never decayed.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.memory_journal import JournalStore, make_event
from core.memory_replayer import _is_decayed
from core.memory_schema import DECAY_DAYS, is_decayed
from core.memory_sqlite import MemorySQLiteIndex


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _pref(value: str, i: int, *, applied: bool = True) -> object:
    return make_event(
        event_id=f"e{i}",
        kind="preference",
        topic="preferences",
        key="preferences.humor_level",
        value={"humor_level": value},
        confidence=1.0,
        source="explicit",
        extractor="deterministic",
        session_id="s",
        turn_id=f"t{i}",
        evidence={"user_text": f"turn {i}"},
        applied=applied,
        observed_at=_iso(datetime(2026, 7, 21, 10, 0, i, tzinfo=timezone.utc)),
    )


# ---------------------------------------------------------------------------
# H1 — dedup flip-back
# ---------------------------------------------------------------------------

def test_dedup_flip_back_lands(tmp_path: Path) -> None:
    j = JournalStore(user_id="usr_h1a", journal_dir=tmp_path / "j")
    j.append_many([_pref("medium", 1)])
    j.append_many([_pref("low", 2)])
    j.append_many([_pref("high", 3)])  # differs from current latest 'low' → must land
    values = [e.value.get("humor_level") for e in j.load_all()]
    assert values == ["medium", "low", "high"]


def test_dedup_true_noop_suppressed(tmp_path: Path) -> None:
    j = JournalStore(user_id="usr_h1b", journal_dir=tmp_path / "j")
    j.append_many([_pref("high", 1)])
    before = len(j.load_all())
    j.append_many([_pref("high", 2)])  # same value as current latest → no-op
    assert len(j.load_all()) == before


def test_dedup_flip_back_within_one_batch(tmp_path: Path) -> None:
    j = JournalStore(user_id="usr_h1c", journal_dir=tmp_path / "j")
    # medium, low, medium in a single call: the third differs from the running
    # latest ('low') and must land; a trailing exact repeat must not.
    j.append_many([_pref("medium", 1), _pref("low", 2), _pref("medium", 3), _pref("medium", 4)])
    values = [e.value.get("humor_level") for e in j.load_all()]
    assert values == ["medium", "low", "medium"]


def test_dedup_applied_flag_not_suppressed_by_candidate(tmp_path: Path) -> None:
    # An applied=True restatement must never be suppressed by a lingering
    # applied=False candidate of the same value (gate-held fact must journal).
    j = JournalStore(user_id="usr_h1d", journal_dir=tmp_path / "j")
    j.append_many([_pref("high", 1, applied=False)])
    j.append_many([_pref("high", 2, applied=True)])
    applied_flags = [e.applied for e in j.load_all()]
    assert applied_flags == [False, True]


# ---------------------------------------------------------------------------
# H2 — decay divergence
# ---------------------------------------------------------------------------

def _idx(tmp_path: Path) -> MemorySQLiteIndex:
    return MemorySQLiteIndex(user_id="usr_h2", db_path=tmp_path / "memory.sqlite")


def _old() -> str:
    return _iso(datetime.now(timezone.utc) - timedelta(days=DECAY_DAYS + 10))


def _inferred_style(observed_at: str, value: str = "concise") -> object:
    return make_event(
        event_id=f"style_{value}_{observed_at[:10]}",
        kind="preference",
        topic="preferences",
        key="preferences.response_style",
        value={"response_style": value},
        confidence=0.9,
        source="inferred",
        extractor="llm_turn",
        session_id="s",
        turn_id="t",
        evidence={"user_text": "seems concise"},
        applied=True,
        observed_at=observed_at,
    )


def test_decayed_inferred_fact_dropped_from_search(tmp_path: Path) -> None:
    idx = _idx(tmp_path)
    idx.index_event(_inferred_style(_old()))
    assert idx.search("response style concise") == []


def test_latest_for_key_returns_none_when_decayed(tmp_path: Path) -> None:
    idx = _idx(tmp_path)
    idx.index_event(_inferred_style(_old()))
    assert idx.latest_for_key("preferences", "preferences.response_style") is None


def test_fresh_inferred_fact_still_searchable(tmp_path: Path) -> None:
    idx = _idx(tmp_path)
    idx.index_event(_inferred_style(_iso(datetime.now(timezone.utc))))
    assert idx.search("response style concise")
    assert idx.latest_for_key("preferences", "preferences.response_style") is not None


def test_identity_fact_exempt_from_decay(tmp_path: Path) -> None:
    idx = _idx(tmp_path)
    idx.index_event(make_event(
        event_id="id1", kind="fact", topic="identity", key="identity.name",
        value={"name": "Sam"}, confidence=1.0, source="explicit",
        extractor="deterministic", session_id="s", turn_id="t",
        evidence={"user_text": "I'm Sam"}, applied=True, observed_at=_old(),
    ))
    assert idx.search("Sam")  # identity never decays even when old


def test_explicit_fact_exempt_from_decay(tmp_path: Path) -> None:
    idx = _idx(tmp_path)
    idx.index_event(make_event(
        event_id="ex1", kind="preference", topic="preferences",
        key="preferences.email_tone", value={"email_tone": "formal"}, confidence=1.0,
        source="explicit", extractor="deterministic", session_id="s", turn_id="t",
        evidence={"user_text": "keep it formal"}, applied=True, observed_at=_old(),
    ))
    assert idx.search("email tone formal")  # explicit statements persist


def test_search_and_replayer_agree_on_decay(tmp_path: Path) -> None:
    # The whole point of H2: the SQLite read model and the markdown replayer
    # must reach the same verdict for the same event via the shared predicate.
    ref = datetime.now(timezone.utc)
    old_at = _iso(ref - timedelta(days=DECAY_DAYS + 5))
    ev = _inferred_style(old_at)
    # replayer path (adapter) and the shared predicate must both say "decayed".
    assert _is_decayed(ev, ref) is True
    assert is_decayed(ev.topic, ev.source, ev.observed_at, reference_time=ref) is True
    # and the sqlite search path drops it.
    idx = _idx(tmp_path)
    idx.index_event(ev)
    assert idx.search("response style concise") == []
