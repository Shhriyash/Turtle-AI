"""
test/tenant_purge_enumeration_test.py
---------------------------------------
WP 2.A (ledger 2.3): "the enumeration IS the deliverable" — a purge that
reports success while leaving rows in a forgotten table is worse than one
that fails loudly. This test needs no live Postgres (it's a pure source
cross-check), so it runs unconditionally in every CI job, not just the
`cloud` marker's live-services one — the whole point is catching the next
table added under core/storage/cloud/ WITHOUT a matching
core/tenant_purge.py entry, on every run, local or cloud.
"""
from __future__ import annotations

import re
from pathlib import Path

from core.tenant_purge import table_enumeration

_CLOUD_DIR = Path(__file__).resolve().parents[1] / "core" / "storage" / "cloud"
# Require an opening paren shortly after the table name so prose that merely
# MENTIONS "CREATE TABLE IF NOT EXISTS" in a docstring (e.g.
# purge_log_store.py's own module docstring) isn't mistaken for real DDL.
_TABLE_PATTERN = re.compile(r"CREATE TABLE IF NOT EXISTS\s+(\w+)\s*\(")

# purge_log (core/storage/cloud/purge_log_store.py) is deliberately NOT in
# core/tenant_purge.py's enumeration: it IS the erasure audit trail (ledger
# 2.4), content-free by construction (only a SHA-256 of the user_id, never
# the plaintext id) — a purge must never delete its own proof that it ran.
_INTENTIONALLY_UNPURGED_TABLES = {"purge_log"}


def _tables_in_ddl() -> set[str]:
    found: set[str] = set()
    for py_file in _CLOUD_DIR.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        found.update(_TABLE_PATTERN.findall(text))
    return found - _INTENTIONALLY_UNPURGED_TABLES


def test_table_enumeration_matches_ddl() -> None:
    found = _tables_in_ddl()
    enumerated = set(table_enumeration().keys())
    missing = found - enumerated
    stale = enumerated - found
    assert enumerated == found, (
        "core/tenant_purge.py's table enumeration is out of sync with the "
        f"DDL under core/storage/cloud/. Missing from the enumeration: "
        f"{missing}; stale entries no longer backed by any DDL: {stale}. "
        "Fix is one of two things, not a third: (1) if the new table holds "
        "durable per-user data an erasure request must cover, add it to "
        "core/tenant_purge.py's _TABLE_USER_COLUMNS with the column(s) that "
        "identify the user (see telemetry_once there for a worked example, "
        "including how to flag a real consequence of purging it); or (2) if "
        "it must NEVER be purged (it's data ABOUT the purge itself, or "
        "otherwise not user data an erasure request covers — purge_log is "
        "the existing example), add it to this test's own "
        "_INTENTIONALLY_UNPURGED_TABLES with a comment saying which and why. "
        "Do not silence this test any other way — that is what it's for."
    )


def test_link_codes_has_both_user_columns_enumerated() -> None:
    """link_codes is the one two-column case (ledger 1b.6/2.3): a code either
    ORIGINATES from a user (source_user_id) or is RESERVED for one
    (reserved_for). Both must be purged or a purge leaves the other side's
    row behind."""
    columns = table_enumeration()["link_codes"]
    assert set(columns) == {"source_user_id", "reserved_for"}


def test_every_table_has_at_least_one_user_column() -> None:
    for table, columns in table_enumeration().items():
        assert columns, f"{table!r} has no user-scoping column declared"
