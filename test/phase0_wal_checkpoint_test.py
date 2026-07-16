import shutil
import sqlite3

from core.memory_journal import MemoryEvent
from core.memory_sqlite import MemorySQLiteIndex


def _event(event_id: str, key: str, value: dict) -> MemoryEvent:
    return MemoryEvent(
        event_id=event_id,
        session_id="probe-session",
        turn_id=event_id,
        observed_at="2026-07-16T10:00:00Z",
        kind="fact",
        topic="identity",
        key=key,
        value=value,
        confidence=1.0,
        source="explicit",
        extractor="deterministic",
        evidence={},
        supersedes=None,
        applied=True,
        rejected=False,
    )


def test_wal_checkpoint_makes_main_sqlite_copy_complete(tmp_path):
    db_path = tmp_path / "m.sqlite"
    idx = MemorySQLiteIndex(user_id="probe_wal", db_path=db_path)

    idx.index_event(_event("evt-name", "identity.name", {"name": "Shriyash"}))
    idx.index_event(_event("evt-city", "identity.current_city", {"city": "Bhopal"}))
    idx.index_event(_event("evt-tz", "identity.timezone", {"timezone": "Asia/Calcutta"}))

    assert idx.count() == 3

    idx.checkpoint()

    copy_path = tmp_path / "copy.sqlite"
    # This is the operational copy scenario that silently lost the production user's corrections.
    shutil.copyfile(db_path, copy_path)

    ro = sqlite3.connect(f"file:{copy_path}?mode=ro", uri=True)
    try:
        assert ro.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 3
    finally:
        ro.close()

    idx.close()
