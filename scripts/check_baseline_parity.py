from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Owner-facing companion to test/cloud_integration/migrations_schema_parity_test.py
# (ledger 3.1's "control that matters most"). That CI test proves the baseline
# matches the STORE CODE; it cannot prove the baseline matches PRODUCTION —
# nobody handed this script real credentials, and it never asks for them. Run
# this by hand, with your own DATABASE_URL_DIRECT, before `alembic stamp 0001`
# against production (the ledger's Phase 3 owner action: "run alembic stamp
# 0001 against production once, after checking the baseline diff").
#
# READ-ONLY against the live database: this script never issues DDL/DML
# against the schema it's diffing. It builds 0001_baseline into a throwaway
# scratch SCHEMA on the same database (via a real `alembic upgrade head`
# subprocess, exactly like the CI parity test), snapshots that, drops the
# scratch schema, and diffs the two snapshots. It cannot alter the schema
# being compared — it can only create-and-drop its own separate, disposable
# schema alongside it.

_SCRATCH_SCHEMA = "wp3a_baseline_parity_scratch"
_LIVE_SCHEMA = "public"


def _schema_dsn(dsn: str, schema: str) -> str:
    sep = "&" if "?" in dsn else "?"
    return f"{dsn}{sep}options=-c%20search_path%3D{schema}%2Cpublic"


def _run_alembic_upgrade(dsn_for_scratch_schema: str) -> None:
    env = dict(os.environ)
    env["DATABASE_URL_DIRECT"] = dsn_for_scratch_schema
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(ROOT_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"alembic upgrade head failed against the scratch schema:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _snapshot(dsn: str, schema: str):
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s AND table_type = 'BASE TABLE' "
                "AND table_name != 'alembic_version'",
                (schema,),
            ).fetchall()
        }
        columns = {}
        for table_name, column_name, data_type, udt_name, is_nullable, column_default in conn.execute(
            "SELECT table_name, column_name, data_type, udt_name, is_nullable, column_default "
            "FROM information_schema.columns WHERE table_schema = %s",
            (schema,),
        ).fetchall():
            if table_name == "alembic_version":
                continue
            default_norm = column_default.replace(f"{schema}.", "") if column_default else column_default
            columns[(table_name, column_name)] = (data_type, udt_name, is_nullable, default_norm)
        indexes = {}
        for indexname, indexdef in conn.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = %s",
            (schema,),
        ).fetchall():
            if indexname.startswith("alembic_version"):
                continue
            indexes[indexname] = indexdef.replace(f"{schema}.", "<schema>.")
    return tables, columns, indexes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only diff of a live production schema against migrations/versions/0001_baseline.py."
    )
    parser.add_argument(
        "--database-url-direct",
        default=os.environ.get("DATABASE_URL_DIRECT", ""),
        help="Direct (non-pooler) Postgres DSN of the database to check. Defaults to $DATABASE_URL_DIRECT.",
    )
    args = parser.parse_args()

    dsn = args.database_url_direct.strip()
    if not dsn:
        print("No DATABASE_URL_DIRECT given (--database-url-direct or the env var). Refusing to guess.")
        return 2

    import psycopg

    print(f"Building 0001_baseline into scratch schema '{_SCRATCH_SCHEMA}' (read-only against '{_LIVE_SCHEMA}')...")
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {_SCRATCH_SCHEMA} CASCADE")
        conn.execute(f"CREATE SCHEMA {_SCRATCH_SCHEMA}")
    try:
        _run_alembic_upgrade(_schema_dsn(dsn, _SCRATCH_SCHEMA))
        base_tables, base_columns, base_indexes = _snapshot(dsn, _SCRATCH_SCHEMA)
        live_tables, live_columns, live_indexes = _snapshot(dsn, _LIVE_SCHEMA)
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {_SCRATCH_SCHEMA} CASCADE")

    ok = True

    only_live = live_tables - base_tables
    only_base = base_tables - live_tables
    if only_live or only_base:
        ok = False
        print("TABLE MISMATCH")
        if only_live:
            print(f"  present live, missing from baseline: {sorted(only_live)}")
        if only_base:
            print(f"  present in baseline, missing live: {sorted(only_base)}")

    for key in sorted(set(live_columns) | set(base_columns)):
        live_val = live_columns.get(key)
        base_val = base_columns.get(key)
        if live_val != base_val:
            ok = False
            print(f"COLUMN MISMATCH {key}: live={live_val} baseline={base_val}")

    for name in sorted(set(live_indexes) | set(base_indexes)):
        live_val = live_indexes.get(name)
        base_val = base_indexes.get(name)
        if live_val != base_val:
            ok = False
            print(f"INDEX MISMATCH {name}:\n  live:     {live_val}\n  baseline: {base_val}")

    if ok:
        print(f"No drift: '{_LIVE_SCHEMA}' schema matches migrations/versions/0001_baseline.py.")
        return 0

    print("\nDrift found above. Do not `alembic stamp 0001` against this database until resolved.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
