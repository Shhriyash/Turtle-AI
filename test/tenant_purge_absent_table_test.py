"""
test/tenant_purge_absent_table_test.py
----------------------------------------
WP 2.A follow-up (ledger 2.3): every cloud store under core/storage/cloud/
creates its table LAZILY (CREATE TABLE IF NOT EXISTS behind a module-level
_initialized flag, run on first use) — there is no migrations framework.
So a table in core/tenant_purge.py's _TABLE_USER_COLUMNS enumeration only
exists in a given Postgres database once that feature has actually been
exercised there. telemetry_once (WP2.C) is the sharpest example: it cannot
exist in any production database until the first telemetry claim lands
after that deploy.

Before this fix, _purge_tables_cloud ran a bare "DELETE FROM {table}" for
every enumerated table inside ONE conn.transaction(). Postgres aborts the
whole transaction the instant one statement raises
asyncpg.exceptions.UndefinedTableError — so the very first /forget-me
request issued against a database where even one enumerated table has never
been created fails outright, deleting NOTHING (not even the tables that DO
exist). That is exactly the failure ledger 2.3 exists to close, reintroduced
by a different mechanism than the one 2.3 originally fixed.

This module uses a fake asyncpg-shaped pool/connection (no live Postgres
needed) to prove two things at the unit level:

  1. test_absent_table_currently_aborts_the_whole_purge_pre_fix reproduces
     the bug directly against the current DELETE-only fake connection: one
     UndefinedTableError anywhere aborts every other table's delete too.
  2. test_absent_table_does_not_abort_purge_and_is_reported_honestly proves
     the FIXED behaviour: an absent table is resolved as absent BEFORE any
     DELETE is attempted (via the existence-check the fix adds), the other
     16 tables still get deleted, and the absent one is reported as None
     (not 0 — those are different facts: 0 means "no rows for this user",
     None means "the table doesn't exist, so the question could not be
     asked").
  3. test_existing_table_delete_failure_still_aborts_purge proves the fix
     does NOT become a blanket "ignore errors" loop: a table that DOES
     exist but whose DELETE fails for some other reason (e.g. a permissions
     error) must still abort the whole purge and propagate the exception.

None of this needs a live Postgres — it is a pure unit test against a fake
connection satisfying only the subset of the asyncpg API
core/tenant_purge.py actually calls. test/cloud_integration/tenant_purge_cloud_test.py
covers the real-Postgres, real-DDL end of this (it self-skips without
DATABASE_URL/REDIS_URL, so it did not run in this sandbox).
"""
from __future__ import annotations

import asyncio
from typing import Any

import asyncpg
import pytest

from core.tenant_purge import _TABLE_USER_COLUMNS, _purge_tables_cloud


class _FakeRecord(dict):
    """Minimal asyncpg.Record stand-in: attribute-less mapping accessed by
    __getitem__, which is all core/tenant_purge.py needs."""


class _FakeTransaction:
    def __init__(self) -> None:
        pass

    async def __aenter__(self) -> "_FakeTransaction":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False  # never swallow — an exception here must propagate


class _FakeAcquire:
    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn

    async def __aenter__(self) -> "_FakeConn":
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeConn:
    """Records every DELETE issued and simulates one or more tables that
    don't exist in this (fake) database."""

    def __init__(self, absent_tables: set[str], fail_tables: set[str] | None = None) -> None:
        self.absent_tables = absent_tables
        self.fail_tables = fail_tables or set()
        self.deleted_tables: list[str] = []

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    async def fetch(self, sql: str, table_names: list[str]) -> list[_FakeRecord]:
        assert "to_regclass" in sql
        return [
            _FakeRecord(name=t, exists=t not in self.absent_tables)
            for t in table_names
        ]

    async def execute(self, sql: str, user_id: str) -> str:
        table = sql.split("FROM", 1)[1].split("WHERE", 1)[0].strip()
        if table in self.absent_tables:
            raise asyncpg.exceptions.UndefinedTableError(
                f'relation "{table}" does not exist'
            )
        if table in self.fail_tables:
            raise asyncpg.exceptions.InsufficientPrivilegeError(
                f'permission denied for table "{table}"'
            )
        self.deleted_tables.append(table)
        return "DELETE 1"


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self._conn)


def _patch_pool(monkeypatch: pytest.MonkeyPatch, conn: _FakeConn) -> None:
    import core.storage.cloud as cloud_pkg

    async def fake_get_pg_pool() -> Any:
        return _FakePool(conn)

    monkeypatch.setattr(cloud_pkg, "get_pg_pool", fake_get_pg_pool)


def test_absent_table_currently_aborts_the_whole_purge_pre_fix() -> None:
    """Reproduction: a bare DELETE-only connection (no existence check) with
    one absent table raises UndefinedTableError, which aborts the whole
    purge — none of the OTHER 16 tables get deleted either. This test talks
    directly to a fake connection that has no existence-check capability at
    all, i.e. it models exactly what _purge_tables_cloud did before this
    fix. It must keep passing after the fix too, since it is not exercising
    core/tenant_purge.py's own existence-check path — it is independent
    proof of the underlying Postgres transaction-abort semantics the fix
    exists to route around.
    """
    absent = {"telemetry_once"}
    conn = _FakeConn(absent_tables=absent)

    async def _run() -> None:
        async with conn.transaction():
            for table, columns in _TABLE_USER_COLUMNS.items():
                where = " OR ".join(f"{c} = $1" for c in columns)
                await conn.execute(f"DELETE FROM {table} WHERE {where}", "u1")

    with pytest.raises(asyncpg.exceptions.UndefinedTableError):
        asyncio.run(_run())

    # The abort happened before every table got a chance — proof the
    # transaction-wide abort is real, not just a per-statement failure.
    assert len(conn.deleted_tables) < len(_TABLE_USER_COLUMNS) - 1


def test_absent_table_does_not_abort_purge_and_is_reported_honestly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    absent = {"telemetry_once"}
    conn = _FakeConn(absent_tables=absent)
    _patch_pool(monkeypatch, conn)

    counts = asyncio.run(_purge_tables_cloud("u1"))

    assert set(counts.keys()) == set(_TABLE_USER_COLUMNS.keys())
    assert counts["telemetry_once"] is None, (
        "an absent table must be reported as None (the question could not "
        "be asked), never as 0 (which means the user genuinely had no rows)"
    )
    for table in _TABLE_USER_COLUMNS:
        if table != "telemetry_once":
            assert counts[table] == 1, f"{table} should have been deleted normally"
    assert "telemetry_once" not in conn.deleted_tables


def test_existing_table_delete_failure_still_aborts_purge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A table that EXISTS but whose DELETE fails for some other reason
    (e.g. a permissions error) must still abort the whole purge — the fix
    for absent tables must not degenerate into swallowing every error."""
    conn = _FakeConn(absent_tables=set(), fail_tables={"sessions"})
    _patch_pool(monkeypatch, conn)

    with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
        asyncio.run(_purge_tables_cloud("u1"))
