"""
test/migrations_env_url_resolution_test.py
--------------------------------------------
Owner action removal: migrations/env.py's get_url() used to require a
hand-copied DATABASE_URL_DIRECT. It now also accepts DATABASE_URL_UNPOOLED
(what Neon's Vercel integration provides automatically), with
DATABASE_URL_DIRECT still winning as an explicit override, and it guards
against a pooled ("-pooler") host reaching Alembic even via that override.
This file exercises get_url()'s resolution order and the pooler guard
directly against the real function in migrations/env.py.

Why the import gymnastics below: migrations/env.py is an Alembic "env.py",
not an ordinary importable module — its top-level code (`config =
context.config`, and the final `if context.is_offline_mode(): ... else:
...` dispatch) only makes sense inside a real Alembic EnvironmentContext,
which `alembic upgrade head` sets up before loading this file. Importing it
plainly (as this repo's other modules are imported in tests) raises
AttributeError on `context.config` before get_url() is ever reached (see
this file's `_import_env_module` docstring for the real traceback this
produces).  Rather than restructure migrations/env.py just to make it
independently importable (out of scope — the brief owns this file's
resolution *logic*, not its Alembic entry-point shape, and the passing offline
`alembic upgrade head --sql` runs recorded in this WP's report already prove
the entry-point shape works end to end), this module fakes just enough of
`alembic.context` — config, is_offline_mode(), configure(), begin_transaction(),
run_migrations() — for migrations/env.py's own top-level `run_migrations_offline()`
call to complete as a no-op, and captures the `url=` kwarg that call receives
straight from get_url(). That is the real get_url() function, in the real
file, under the real module-level control flow — not a re-implementation of
its logic under test.

Requires the `alembic` package (requirements-migrations.txt), which is NOT
part of requirements.txt / the main "tests" CI job's install — only the
"cloud-tests" job and an operator's own migration workflow install it (see
migrations/env.py's own module docstring, and .github/workflows/tests.yml).
`pytest.importorskip` below makes that an honest, clean skip rather than a
collection error when this suite runs without it.
"""
from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

alembic = pytest.importorskip("alembic")
import alembic.context as alembic_context  # noqa: E402  (import after importorskip)

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PY_PATH = ROOT_DIR / "migrations" / "env.py"

_ENV_VARS = ("DATABASE_URL_DIRECT", "DATABASE_URL_UNPOOLED", "DATABASE_URL")


def _import_env_module(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, list]:
    """Load a fresh copy of migrations/env.py with alembic.context faked out
    just enough for its module-level `run_migrations_offline()` call to run
    as a no-op, and return (module_namespace, captured_configure_calls).

    captured_configure_calls records every kwargs dict passed to
    `context.configure(...)` — for the offline path, migrations/env.py calls
    it as `context.configure(url=get_url(), ...)`, so
    captured_configure_calls[0]["url"] is exactly what the real get_url()
    resolved to.

    If get_url() raises (the "neither variable set" / pooled-host-guard
    cases), that exception propagates out of THIS function, because
    migrations/env.py evaluates `url=get_url()` as a call argument before
    context.configure ever runs — i.e. before our no-op stub swallows
    anything.
    """
    captured_configure_calls: list = []

    monkeypatch.setattr(
        alembic_context,
        "config",
        SimpleNamespace(
            config_file_name=None,
            config_ini_section=None,
            get_section=lambda *a, **k: {},
        ),
        raising=False,
    )
    monkeypatch.setattr(alembic_context, "is_offline_mode", lambda: True, raising=False)
    monkeypatch.setattr(
        alembic_context,
        "configure",
        lambda **kw: captured_configure_calls.append(kw),
        raising=False,
    )
    monkeypatch.setattr(
        alembic_context,
        "begin_transaction",
        lambda: contextlib.nullcontext(),
        raising=False,
    )
    monkeypatch.setattr(alembic_context, "run_migrations", lambda: None, raising=False)

    # A unique module name each time (not "migrations.env") sidesteps the
    # sys.modules cache and the fact that migrations/ has no __init__.py —
    # this loads migrations/env.py as a standalone file, exactly as Alembic
    # itself does via importlib under the hood.
    spec = importlib.util.spec_from_file_location(
        "turtle_migrations_env_under_test", ENV_PY_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    return vars(module), captured_configure_calls


def _clear_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class TestGetUrlResolution:
    """The four resolution-order cases from the WP brief."""

    def test_only_database_url_direct_is_used(self, monkeypatch):
        _clear_db_env(monkeypatch)
        monkeypatch.setenv(
            "DATABASE_URL_DIRECT",
            "postgresql://user:pass@ep-fake-direct.c-10.us-east-1.aws.neon.tech/turtle",
        )
        _, calls = _import_env_module(monkeypatch)
        assert len(calls) == 1
        assert calls[0]["url"] == (
            "postgresql+psycopg://user:pass@ep-fake-direct.c-10.us-east-1.aws.neon.tech/turtle"
        )

    def test_only_database_url_unpooled_is_used(self, monkeypatch):
        _clear_db_env(monkeypatch)
        monkeypatch.setenv(
            "DATABASE_URL_UNPOOLED",
            "postgresql://user:pass@ep-fake-unpooled.c-10.us-east-1.aws.neon.tech/turtle",
        )
        _, calls = _import_env_module(monkeypatch)
        assert len(calls) == 1
        assert calls[0]["url"] == (
            "postgresql+psycopg://user:pass@ep-fake-unpooled.c-10.us-east-1.aws.neon.tech/turtle"
        )

    def test_both_set_direct_override_wins(self, monkeypatch):
        _clear_db_env(monkeypatch)
        monkeypatch.setenv(
            "DATABASE_URL_DIRECT",
            "postgresql://user:pass@ep-fake-direct.c-10.us-east-1.aws.neon.tech/turtle",
        )
        monkeypatch.setenv(
            "DATABASE_URL_UNPOOLED",
            "postgresql://user:pass@ep-fake-unpooled.c-10.us-east-1.aws.neon.tech/turtle",
        )
        _, calls = _import_env_module(monkeypatch)
        assert len(calls) == 1
        assert "ep-fake-direct" in calls[0]["url"]
        assert "ep-fake-unpooled" not in calls[0]["url"]

    def test_neither_set_raises_and_names_both_variables(self, monkeypatch):
        _clear_db_env(monkeypatch)
        # A pooled DATABASE_URL being set must NOT be treated as a fallback
        # (this is the whole point of the design) — set it here to prove
        # that too, not just the "nothing at all is set" case.
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://user:pass@ep-fake-pooler.c-10.us-east-1.aws.neon.tech/turtle",
        )
        with pytest.raises(RuntimeError) as excinfo:
            _import_env_module(monkeypatch)
        message = str(excinfo.value)
        assert "DATABASE_URL_DIRECT" in message
        assert "DATABASE_URL_UNPOOLED" in message


class TestPooledHostGuard:
    """The "-pooler" guard, including via the explicit override."""

    def test_pooled_host_via_database_url_direct_is_rejected(self, monkeypatch):
        _clear_db_env(monkeypatch)
        monkeypatch.setenv(
            "DATABASE_URL_DIRECT",
            "postgresql://user:pass@ep-fake-pooler.c-10.us-east-1.aws.neon.tech/turtle",
        )
        with pytest.raises(RuntimeError) as excinfo:
            _import_env_module(monkeypatch)
        message = str(excinfo.value)
        assert "DATABASE_URL_DIRECT" in message
        assert "-pooler" in message or "pooler" in message.lower()

    def test_pooled_host_via_database_url_unpooled_is_also_rejected(self, monkeypatch):
        # Belt-and-suspenders: the guard must not assume only
        # DATABASE_URL_DIRECT can be pasted wrong.
        _clear_db_env(monkeypatch)
        monkeypatch.setenv(
            "DATABASE_URL_UNPOOLED",
            "postgresql://user:pass@ep-fake-pooler.c-10.us-east-1.aws.neon.tech/turtle",
        )
        with pytest.raises(RuntimeError) as excinfo:
            _import_env_module(monkeypatch)
        assert "DATABASE_URL_UNPOOLED" in str(excinfo.value)

    def test_non_pooled_direct_host_is_accepted(self, monkeypatch):
        # Negative control: a direct host that merely CONTAINS "pool" as a
        # substring elsewhere must not false-positive the guard.
        _clear_db_env(monkeypatch)
        monkeypatch.setenv(
            "DATABASE_URL_DIRECT",
            "postgresql://user:pass@ep-fake-direct.c-10.us-east-1.aws.neon.tech/pooled_data",
        )
        _, calls = _import_env_module(monkeypatch)
        assert len(calls) == 1
        assert "ep-fake-direct" in calls[0]["url"]
