"""
test/cloud_integration/migrations_schema_parity_test.py
---------------------------------------------------------
Ledger 3.1's control that matters most: proves migrations/versions/0001_baseline.py
produces a schema identical to what core/storage/cloud/*.py's own DDL constants
declare TODAY — not "identical to what the author of 0001_baseline believed the
stores said when they transcribed it".

Method: build two schemas in the SAME real Postgres (the cloud-tests service
container), independently —

  1. "stores" schema: execute the stores' own `_CREATE_TABLE_SQL` /
     `_CREATE_TABLES_SQL` / `_CREATE_INDEX_SQL` module constants directly,
     imported live from core/storage/cloud/*.py. This is the source of truth:
     whatever these constants say IS what a running app creates.
  2. "alembic" schema: run `alembic upgrade head` as a real subprocess against
     the same database, with the session's search_path pointed at that schema.

Then diff `information_schema`/`pg_indexes` between the two schemas: tables,
columns (type + nullability, defaults normalised for the schema-qualified
sequence name BIGSERIAL creates), primary keys, and index definitions
(including partial index WHERE clauses). A string-diff of the CREATE TABLE
text is deliberately NOT what this compares — Postgres's own catalogs are the
ground truth for what got built, matching the ledger's instruction to query
information_schema/pg_indexes rather than compare strings.

This test does NOT compare against 0001_baseline.py's own source text (that
would just check the migration agrees with itself) — side 1 is read from the
live store modules independently of side 2's migration file, so a future edit
to a store's DDL that isn't mirrored into the migration turns this red.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psycopg
import pytest

pytestmark = pytest.mark.cloud

ROOT_DIR = Path(__file__).resolve().parents[2]

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

_STORES_SCHEMA = "wp3a_parity_stores"
_ALEMBIC_SCHEMA = "wp3a_parity_alembic"

# alembic_version (Alembic's own revision-bookkeeping table, created by
# `alembic upgrade head` itself — NOT part of 0001_baseline.py's own
# CREATE TABLE statements, and never declared by any core/storage/cloud/*.py
# store) is expected to exist in the "alembic" schema and NOT in the "stores"
# schema, and is expected to SURVIVE `alembic downgrade base` (that is how
# Alembic itself knows a database is at "base" rather than unmanaged).
# Excluded by name here, mirroring test/tenant_purge_enumeration_test.py's
# `_INTENTIONALLY_UNPURGED_TABLES` shape — a named, commented exemption, not
# a loosened assertion, so a future REAL table the baseline creates but no
# store declares still turns this test red.
_ALEMBIC_BOOKKEEPING_TABLES = {"alembic_version"}
# alembic_version's own PRIMARY KEY index, created alongside it by the same
# `alembic upgrade head` bookkeeping — the literal name Alembic gives it
# (confirmed via `alembic upgrade head --sql`: "CONSTRAINT alembic_version_pkc
# PRIMARY KEY (version_num)"), not a pattern match, for the same reason.
_ALEMBIC_BOOKKEEPING_INDEXES = {"alembic_version_pkc"}


def _schema_url(schema: str) -> str:
    sep = "&" if "?" in DATABASE_URL else "?"
    # `options=-c search_path=...` sets the session's default search_path for
    # any connection opened with this DSN — both our own psycopg connections
    # below and the subprocess `alembic upgrade head` connection use it the
    # same way. `public` stays on the path so the `vector` type (installed
    # into the default schema by the session-fixture's CREATE EXTENSION, see
    # conftest.py) is visible to `vector_docs`/`vector_chunks`.
    return f"{DATABASE_URL}{sep}options=-c%20search_path%3D{schema}%2Cpublic"


def _reset_schema(schema: str) -> None:
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.execute(f"CREATE SCHEMA {schema}")


# ---------------------------------------------------------------------------
# Side 1: build the "stores" schema straight from the live store constants.
# ---------------------------------------------------------------------------


def _store_ddl_statements() -> list[str]:
    """Every CREATE TABLE / CREATE INDEX statement declared by
    core/storage/cloud/*.py, imported live (not retyped) so this side of the
    comparison tracks the real source of truth automatically.
    """
    from core.storage.cloud import account_linking_store as als
    from core.storage.cloud import calendar_token_store as cts
    from core.storage.cloud import confirmation_state_store as css
    from core.storage.cloud import identity_store as ids
    from core.storage.cloud import journal_store as js
    from core.storage.cloud import personal_memory_store as pms
    from core.storage.cloud import pgvector_store as pvs
    from core.storage.cloud import postgres_store as pss
    from core.storage.cloud import purge_log_store as pls
    from core.storage.cloud import rag_session_staging_store as rsss
    from core.storage.cloud import routine_last_fired_store as rlfs
    from core.storage.cloud import routine_outbox_store as ros
    from core.storage.cloud import telemetry_claim_store as tcs

    statements: list[str] = ["CREATE EXTENSION IF NOT EXISTS vector"]
    statements.append(als._CREATE_TABLE_SQL)
    statements.append(cts._CREATE_TABLE_SQL)
    statements.append(css._CREATE_TABLE_SQL)
    statements.extend(ids._CREATE_TABLES_SQL)
    statements.append(js._CREATE_TABLE_SQL)
    statements.append(js._CREATE_INDEX_SQL)
    statements.extend(pms._CREATE_TABLES_SQL)
    statements.append(pvs._CREATE_VECTOR_DOCS_SQL)
    statements.append(pvs._CREATE_VECTOR_DOCS_INDEX_SQL)
    statements.append(pvs._CREATE_VECTOR_CHUNKS_SQL)
    statements.append(pvs._CREATE_VECTOR_CHUNKS_INDEX_SQL)
    statements.append(pss._CREATE_TABLE_SQL)
    statements.append(pss._CREATE_INDEX_SQL)
    statements.append(pls._CREATE_TABLE_SQL)
    statements.append(pls._CREATE_INDEX_SQL)
    statements.append(rsss._CREATE_TABLE_SQL)
    statements.append(rlfs._CREATE_TABLE_SQL)
    statements.append(rlfs._CREATE_INDEX_SQL)
    statements.append(ros._CREATE_TABLE_SQL)
    statements.append(tcs._CREATE_TABLE_SQL)
    return statements


def _build_stores_schema() -> None:
    _reset_schema(_STORES_SCHEMA)
    with psycopg.connect(_schema_url(_STORES_SCHEMA), autocommit=True) as conn:
        for stmt in _store_ddl_statements():
            conn.execute(stmt)


# ---------------------------------------------------------------------------
# Side 2: build the "alembic" schema via a real `alembic upgrade head`.
# ---------------------------------------------------------------------------


def _build_alembic_schema() -> None:
    _reset_schema(_ALEMBIC_SCHEMA)
    env = dict(os.environ)
    env["DATABASE_URL_DIRECT"] = _schema_url(_ALEMBIC_SCHEMA)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(ROOT_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"alembic upgrade head failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Snapshot + comparison via information_schema / pg_indexes.
# ---------------------------------------------------------------------------


def _normalize_default(default: "str | None", schema: str) -> "str | None":
    if default is None:
        return None
    # BIGSERIAL columns get a DEFAULT nextval('<schema>.<table>_<col>_seq'::regclass)
    # — the schema name is the only expected difference between the two sides.
    return default.replace(f"{schema}.", "<schema>.")


def _columns(conn: psycopg.Connection, schema: str) -> dict:
    rows = conn.execute(
        """
        SELECT table_name, column_name, data_type, udt_name, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = %s
        """,
        (schema,),
    ).fetchall()
    out = {}
    for table_name, column_name, data_type, udt_name, is_nullable, column_default in rows:
        out[(table_name, column_name)] = (
            data_type,
            udt_name,
            is_nullable,
            _normalize_default(column_default, schema),
        )
    return out


def _tables(conn: psycopg.Connection, schema: str) -> set:
    rows = conn.execute(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = %s AND table_type = 'BASE TABLE'
        """,
        (schema,),
    ).fetchall()
    return {r[0] for r in rows}


def _primary_keys(conn: psycopg.Connection, schema: str) -> dict:
    rows = conn.execute(
        """
        SELECT tc.table_name, kcu.column_name, kcu.ordinal_position
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        WHERE tc.table_schema = %s AND tc.constraint_type = 'PRIMARY KEY'
        ORDER BY tc.table_name, kcu.ordinal_position
        """,
        (schema,),
    ).fetchall()
    out: dict = {}
    for table_name, column_name, _pos in rows:
        out.setdefault(table_name, []).append(column_name)
    return out


def _indexes(conn: psycopg.Connection, schema: str) -> dict:
    rows = conn.execute(
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = %s",
        (schema,),
    ).fetchall()
    out = {}
    for indexname, indexdef in rows:
        # indexdef embeds the schema-qualified table name
        # (e.g. "... ON wp3a_parity_stores.vector_docs USING btree ...");
        # normalise it away so only the structural definition is compared.
        out[indexname] = indexdef.replace(f"{schema}.", "<schema>.")
    return out


def test_baseline_matches_store_ddl() -> None:
    _build_stores_schema()
    _build_alembic_schema()

    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        stores_tables = _tables(conn, _STORES_SCHEMA)
        # Named exclusion only (see _ALEMBIC_BOOKKEEPING_TABLES above) — every
        # other table alembic_tables reports still goes into the comparison
        # below unfiltered, so a real drift (a store table the baseline
        # doesn't create, or vice versa) still fails this assertion.
        alembic_tables = _tables(conn, _ALEMBIC_SCHEMA) - _ALEMBIC_BOOKKEEPING_TABLES
        assert stores_tables == alembic_tables, (
            f"table sets differ: stores-only={stores_tables - alembic_tables}, "
            f"alembic-only={alembic_tables - stores_tables}"
        )
        # Ledger 3.1: 18 tables (17 from core/tenant_purge.py's
        # _TABLE_USER_COLUMNS + purge_log, deliberately excluded from purging).
        assert len(stores_tables) == 18, f"expected 18 tables, found {len(stores_tables)}: {sorted(stores_tables)}"

        stores_columns = _columns(conn, _STORES_SCHEMA)
        alembic_columns = {
            k: v
            for k, v in _columns(conn, _ALEMBIC_SCHEMA).items()
            if k[0] not in _ALEMBIC_BOOKKEEPING_TABLES
        }
        assert stores_columns == alembic_columns

        stores_pks = _primary_keys(conn, _STORES_SCHEMA)
        alembic_pks = {
            k: v
            for k, v in _primary_keys(conn, _ALEMBIC_SCHEMA).items()
            if k not in _ALEMBIC_BOOKKEEPING_TABLES
        }
        assert stores_pks == alembic_pks

        stores_indexes = _indexes(conn, _STORES_SCHEMA)
        alembic_indexes = {
            name: ddl
            for name, ddl in _indexes(conn, _ALEMBIC_SCHEMA).items()
            if name not in _ALEMBIC_BOOKKEEPING_INDEXES
        }
        assert stores_indexes == alembic_indexes

        # Explicitly pin down the two partial indexes the ledger calls out by
        # name, so a regression here fails with a readable assertion instead
        # of only showing up in the broader dict-equality check above. This
        # is the first point in this test where these two assertions are
        # actually reached when the table-set check above previously failed
        # (the alembic_version mismatch) — every assertion in this function
        # runs top-to-bottom and stops at the first failure, so until the
        # table-set exclusion above was fixed, these two lines had never
        # actually executed against a live database.
        for idx_name in ("idx_vector_docs_user", "idx_vector_chunks_user"):
            assert "WHERE (NOT deleted)" in stores_indexes[idx_name], stores_indexes[idx_name]
            assert "WHERE (NOT deleted)" in alembic_indexes[idx_name], alembic_indexes[idx_name]


def test_downgrade_to_base_drops_every_table() -> None:
    """`alembic downgrade base` removes all 18 application tables this
    revision created, and leaves exactly `alembic_version` behind — Alembic's
    own revision-bookkeeping table, which downgrading to "base" is defined
    to keep (that row, deleted back to empty rather than the table itself
    being dropped, is how Alembic tells "at base" apart from "unmanaged").

    Asserts the residual set equals _ALEMBIC_BOOKKEEPING_TABLES exactly,
    not "every table except alembic_version is gone" filtered away before
    comparing — so a real table this migration failed to drop still fails
    this assertion instead of being silently swallowed alongside the
    expected one.

    Runs against the "alembic" schema this module already builds up to head
    (rebuilt here to not depend on test execution order), so this is a real
    downgrade against a real Postgres, not a code-review-only claim.
    """
    _build_alembic_schema()
    env = dict(os.environ)
    env["DATABASE_URL_DIRECT"] = _schema_url(_ALEMBIC_SCHEMA)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "base"],
        cwd=str(ROOT_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"alembic downgrade base failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        remaining = _tables(conn, _ALEMBIC_SCHEMA)
    assert remaining == _ALEMBIC_BOOKKEEPING_TABLES, (
        f"expected only Alembic's own bookkeeping table {_ALEMBIC_BOOKKEEPING_TABLES} "
        f"to remain after `alembic downgrade base`, found: {remaining}"
    )
