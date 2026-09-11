"""
Per-user Google Calendar OAuth wiring tests.

Covers:
  - tools.calendar_tool._load_token_json: per-user file takes priority over
    the legacy global GOOGLE_CALENDAR_TOKEN_JSON env var, with fallback when
    no per-user file exists or no user_id is given.
  - tools.calendar_tool._load_credentials: builds Credentials with token=None
    (forcing a refresh via refresh_token) when a token is resolved.
  - apps.calendar_oauth_routes.token_path_for_user: matches the path
    calendar_tool reads from (personal_memory_dir/google_calendar_token.json)
    so the connect flow and the tool never disagree about the storage location.

apps.calendar_oauth_routes pulls in fastapi, which is not installed in every
dev shell in this repo (a pre-existing gap, unrelated to this change) — those
tests are skipped when the import fails rather than failing the whole file.
"""
from __future__ import annotations

import asyncio
import json
import unittest.mock as mock

import pytest


def test_load_token_json_prefers_per_user_file(tmp_path, monkeypatch):
    import tools.calendar_tool as ct

    fake_settings = mock.MagicMock()
    fake_settings.google_calendar_token_json = '{"refresh_token": "global-legacy"}'
    monkeypatch.setattr(ct, "settings", fake_settings)

    def fake_personal_memory_dir(user_id: str):
        d = tmp_path / user_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr("core.paths.personal_memory_dir", fake_personal_memory_dir)

    user_id = "u123"
    token_path = fake_personal_memory_dir(user_id) / "google_calendar_token.json"
    token_path.write_text(json.dumps({"refresh_token": "per-user-token"}), encoding="utf-8")

    resolved = ct._load_token_json(user_id)
    assert resolved is not None
    assert json.loads(resolved)["refresh_token"] == "per-user-token"


def test_load_token_json_falls_back_to_global_when_no_per_user_file(tmp_path, monkeypatch):
    import tools.calendar_tool as ct

    fake_settings = mock.MagicMock()
    fake_settings.google_calendar_token_json = '{"refresh_token": "global-legacy"}'
    monkeypatch.setattr(ct, "settings", fake_settings)

    def fake_personal_memory_dir(user_id: str):
        d = tmp_path / user_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr("core.paths.personal_memory_dir", fake_personal_memory_dir)

    resolved = ct._load_token_json("no-such-user")
    assert resolved is not None
    assert json.loads(resolved)["refresh_token"] == "global-legacy"


def test_load_token_json_falls_back_when_no_user_id(monkeypatch):
    import tools.calendar_tool as ct

    fake_settings = mock.MagicMock()
    fake_settings.google_calendar_token_json = '{"refresh_token": "global-legacy"}'
    monkeypatch.setattr(ct, "settings", fake_settings)

    resolved = ct._load_token_json(None)
    assert resolved is not None
    assert json.loads(resolved)["refresh_token"] == "global-legacy"


def test_load_credentials_omits_access_token_to_force_refresh(tmp_path, monkeypatch):
    import tools.calendar_tool as ct

    fake_settings = mock.MagicMock()
    fake_settings.google_calendar_credentials_json = json.dumps(
        {"installed": {"client_id": "cid", "client_secret": "csecret"}}
    )
    fake_settings.google_calendar_token_json = json.dumps(
        {"access_token": "short-lived", "refresh_token": "rt-abc"}
    )
    monkeypatch.setattr(ct, "settings", fake_settings)

    creds = ct._load_credentials(None)
    assert creds is not None
    assert creds.token is None
    assert creds.refresh_token == "rt-abc"
    assert creds.client_id == "cid"
    assert creds.client_secret == "csecret"


def test_load_credentials_none_without_creds_json(monkeypatch):
    import tools.calendar_tool as ct

    fake_settings = mock.MagicMock()
    fake_settings.google_calendar_credentials_json = None
    monkeypatch.setattr(ct, "settings", fake_settings)

    assert ct._load_credentials("someone") is None


def test_create_calendar_event_threads_user_id(monkeypatch):
    """create_calendar_event(..., user_id=...) must reach _build_service with that id."""
    import tools.calendar_tool as ct

    fake_settings = mock.MagicMock()
    fake_settings.google_calendar_credentials_json = "{}"
    monkeypatch.setattr(ct, "settings", fake_settings)

    seen: dict = {}

    def fake_build_service(user_id=None):
        seen["user_id"] = user_id
        raise RuntimeError("stop here — we only care what user_id was passed")

    monkeypatch.setattr(ct, "_build_service", fake_build_service)

    args = ct.CalendarCreateArgs(
        title="Test",
        start_iso="2026-06-01T10:00:00+00:00",
        end_iso="2026-06-01T10:30:00+00:00",
    )
    result = asyncio.run(ct.create_calendar_event(args, user_id="u123"))
    assert seen["user_id"] == "u123"
    assert result.status == "invalid"


# ---------------------------------------------------------------------------
# apps.calendar_oauth_routes — skipped if fastapi is not installed in this
# environment (pre-existing gap; core/config.py and other server modules hit
# the same absence and are similarly excluded from fastapi-free test runs).
# Each test guards individually (rather than a module-level importorskip) so
# the fastapi-free tests above still run and report in that environment.
# ---------------------------------------------------------------------------

try:
    import fastapi as _fastapi  # noqa: F401
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

pytestmark_fastapi = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


@pytestmark_fastapi
def test_token_path_matches_calendar_tool_lookup(tmp_path, monkeypatch):
    import core.paths as paths_module

    monkeypatch.setattr(paths_module, "PERSONAL_MEMORY_DIR", tmp_path)
    import importlib
    import apps.calendar_oauth_routes as oauth_routes
    importlib.reload(oauth_routes)

    path = oauth_routes.token_path_for_user("u123")
    assert path.name == "google_calendar_token.json"
    assert path.parent == tmp_path / "u123"


@pytestmark_fastapi
def test_client_config_parses_installed_block(monkeypatch):
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.google_calendar_credentials_json = json.dumps(
        {"installed": {"client_id": "cid", "client_secret": "csecret"}}
    )
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)

    cfg = oauth_routes._client_config()
    assert cfg == {"client_id": "cid", "client_secret": "csecret"}


@pytestmark_fastapi
def test_client_config_missing_raises_503(monkeypatch):
    from fastapi import HTTPException
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.google_calendar_credentials_json = None
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)

    with pytest.raises(HTTPException) as exc_info:
        oauth_routes._client_config()
    assert exc_info.value.status_code == 503


@pytestmark_fastapi
def test_validate_credentials_json_full_client_json_ok():
    import apps.calendar_oauth_routes as oauth_routes

    raw = json.dumps({"web": {"client_id": "cid", "client_secret": "csecret"}})
    ok, message, config = oauth_routes.validate_credentials_json(raw)
    assert ok is True
    assert config == {"client_id": "cid", "client_secret": "csecret"}


@pytestmark_fastapi
def test_validate_credentials_json_bare_secret_string_rejected():
    """Regression test for the real mistake: pasting just the secret, not the JSON."""
    import apps.calendar_oauth_routes as oauth_routes

    ok, message, config = oauth_routes.validate_credentials_json("GOCSPX-fcwQmV_nvkMFTyoTaC0R9ZA")
    assert ok is False
    assert config is None
    assert "not valid JSON" in message
    assert "entire" in message.lower()


@pytestmark_fastapi
def test_validate_credentials_json_missing_client_secret():
    import apps.calendar_oauth_routes as oauth_routes

    raw = json.dumps({"web": {"client_id": "cid"}})
    ok, message, config = oauth_routes.validate_credentials_json(raw)
    assert ok is False
    assert config is None
    assert "client_secret" in message


@pytestmark_fastapi
def test_validate_credentials_json_empty():
    import apps.calendar_oauth_routes as oauth_routes

    ok, message, config = oauth_routes.validate_credentials_json("")
    assert ok is False
    assert config is None
