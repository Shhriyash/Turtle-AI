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
import base64
import json
import unittest.mock as mock

import pytest
from pydantic import SecretStr

# An obviously-fake 32-byte AES-256-GCM key for tests. NEVER a real secret —
# generated deterministically from a fixed fill byte purely so tests are
# reproducible; see core/calendar_token_crypto.py for the real key format
# and generation command.
_TEST_TOKEN_KEY_B64 = base64.urlsafe_b64encode(b"\x11" * 32).decode("ascii")
_TEST_TOKEN_KEY_B64_2 = base64.urlsafe_b64encode(b"\x22" * 32).decode("ascii")


def test_load_token_json_prefers_per_user_file(tmp_path, monkeypatch):
    import tools.calendar_tool as ct

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
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
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
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
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
    fake_settings.google_calendar_token_json = '{"refresh_token": "global-legacy"}'
    monkeypatch.setattr(ct, "settings", fake_settings)

    resolved = ct._load_token_json(None)
    assert resolved is not None
    assert json.loads(resolved)["refresh_token"] == "global-legacy"


def test_load_credentials_omits_access_token_to_force_refresh(tmp_path, monkeypatch):
    import tools.calendar_tool as ct

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
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
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
    fake_settings.google_calendar_credentials_json = None
    monkeypatch.setattr(ct, "settings", fake_settings)

    assert ct._load_credentials("someone") is None


def test_create_calendar_event_threads_user_id(monkeypatch):
    """create_calendar_event(..., user_id=...) must reach _build_service with that id."""
    import tools.calendar_tool as ct

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
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
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
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
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
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


# ---------------------------------------------------------------------------
# Cloud-mode token storage (Vercel migration Phase 1c): _read_token/
# _write_token/_delete_token/_token_exists must route through Postgres
# instead of local disk when settings.is_cloud is True.
# ---------------------------------------------------------------------------

@pytestmark_fastapi
def test_read_token_uses_postgres_in_cloud_mode(monkeypatch):
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = True
    fake_settings.calendar_token_key = None
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)

    with mock.patch(
        "core.storage.cloud.calendar_token_store.get_token_json",
        return_value='{"refresh_token": "cloud-token"}',
    ) as fake_get:
        resolved = asyncio.run(oauth_routes._read_token("usr_a"))
    assert resolved == '{"refresh_token": "cloud-token"}'
    fake_get.assert_called_once_with("usr_a")


@pytestmark_fastapi
def test_write_token_uses_postgres_in_cloud_mode(monkeypatch):
    """Cloud write with CALENDAR_TOKEN_KEY set stores an encrypted envelope
    (not the raw token), and it decrypts back to the original."""
    import apps.calendar_oauth_routes as oauth_routes
    from core.calendar_token_crypto import decrypt_stored, parse_key

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = True
    fake_settings.calendar_token_key = SecretStr(_TEST_TOKEN_KEY_B64)
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)

    with mock.patch(
        "core.storage.cloud.calendar_token_store.put_token_json"
    ) as fake_put:
        asyncio.run(oauth_routes._write_token("usr_a", '{"refresh_token": "x"}'))
    fake_put.assert_called_once()
    stored_user_id, stored_value = fake_put.call_args.args
    assert stored_user_id == "usr_a"
    assert stored_value != '{"refresh_token": "x"}'
    assert "refresh_token" not in stored_value  # not readable as the original token
    plaintext, key_version = decrypt_stored(stored_value, parse_key(_TEST_TOKEN_KEY_B64))
    assert plaintext == '{"refresh_token": "x"}'
    assert key_version == 1


@pytestmark_fastapi
def test_write_token_cloud_without_key_refuses(monkeypatch):
    """CALENDAR_TOKEN_KEY unset in cloud mode: refuse to write plaintext into
    the shared Postgres row (fail closed, legible 503) rather than silently
    storing it unencrypted."""
    from fastapi import HTTPException
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = True
    fake_settings.calendar_token_key = None
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)

    with mock.patch(
        "core.storage.cloud.calendar_token_store.put_token_json"
    ) as fake_put:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(oauth_routes._write_token("usr_a", '{"refresh_token": "x"}'))
    assert exc_info.value.status_code == 503
    fake_put.assert_not_called()


@pytestmark_fastapi
def test_write_token_local_without_key_stores_plaintext(monkeypatch, tmp_path):
    """Local mode with no CALENDAR_TOKEN_KEY: preserves today's plaintext
    behaviour on a single-tenant dev box (a deliberate scope decision, not a
    silent downgrade — see core/calendar_token_crypto.py)."""
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)
    monkeypatch.setattr(
        oauth_routes, "token_path_for_user", lambda uid: tmp_path / f"{uid}.json"
    )

    asyncio.run(oauth_routes._write_token("usr_a", '{"refresh_token": "x"}'))
    assert (tmp_path / "usr_a.json").read_text(encoding="utf-8") == '{"refresh_token": "x"}'


@pytestmark_fastapi
def test_delete_token_uses_postgres_in_cloud_mode(monkeypatch):
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = True
    fake_settings.calendar_token_key = None
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)

    with mock.patch(
        "core.storage.cloud.calendar_token_store.delete_token_json"
    ) as fake_delete:
        asyncio.run(oauth_routes._delete_token("usr_a"))
    fake_delete.assert_called_once_with("usr_a")


@pytestmark_fastapi
def test_token_exists_uses_postgres_in_cloud_mode(monkeypatch):
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = True
    fake_settings.calendar_token_key = None
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)

    with mock.patch(
        "core.storage.cloud.calendar_token_store.token_exists", return_value=True
    ) as fake_exists:
        result = asyncio.run(oauth_routes._token_exists("usr_a"))
    assert result is True
    fake_exists.assert_called_once_with("usr_a")


@pytestmark_fastapi
def test_read_token_uses_local_disk_when_not_cloud(tmp_path, monkeypatch):
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)
    monkeypatch.setattr(
        oauth_routes, "token_path_for_user", lambda uid: tmp_path / f"{uid}.json"
    )

    (tmp_path / "usr_a.json").write_text('{"refresh_token": "local"}', encoding="utf-8")
    resolved = asyncio.run(oauth_routes._read_token("usr_a"))
    assert resolved == '{"refresh_token": "local"}'


# ---------------------------------------------------------------------------
# WP1.E2 (ledger 1b.3): narrowed scope, AES-GCM at rest with a plaintext
# migration path, revoke-before-delete, and the stale-scope reconnect prompt.
# ---------------------------------------------------------------------------

@pytestmark_fastapi
def test_connect_requests_narrow_scope_and_no_include_granted(monkeypatch):
    """The consent URL asks for calendar.events only, and never
    include_granted_scopes (which would silently widen the grant to every
    scope the user ever granted Turtle across other flows)."""
    import asyncio as _asyncio
    from urllib.parse import urlparse, parse_qs
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
    fake_settings.google_calendar_credentials_json = json.dumps(
        {"installed": {"client_id": "cid", "client_secret": "csecret"}}
    )
    fake_settings.public_base_url = "http://127.0.0.1:8765"
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)
    monkeypatch.setattr(oauth_routes, "_require_user", lambda req: "u123")

    fake_req = mock.MagicMock()
    resp = _asyncio.run(oauth_routes.connect(fake_req))
    location = resp.headers["location"]
    query = parse_qs(urlparse(location).query)
    assert query["scope"] == ["https://www.googleapis.com/auth/calendar.events"]
    assert "include_granted_scopes" not in query


@pytestmark_fastapi
def test_plaintext_token_readable_after_deploy_and_reencrypted_on_write(monkeypatch, tmp_path):
    """The core migration guarantee: a token written BEFORE this change (bare
    plaintext JSON, no envelope) is still readable once CALENDAR_TOKEN_KEY is
    configured and deployed — and the next write upgrades it to an encrypted
    envelope."""
    import apps.calendar_oauth_routes as oauth_routes
    from core.calendar_token_crypto import decrypt_stored, parse_key

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = SecretStr(_TEST_TOKEN_KEY_B64)
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)
    monkeypatch.setattr(
        oauth_routes, "token_path_for_user", lambda uid: tmp_path / f"{uid}.json"
    )

    # Simulate a token written before encryption existed: bare plaintext.
    pre_migration_path = tmp_path / "usr_a.json"
    pre_migration_path.write_text('{"refresh_token": "pre-migration"}', encoding="utf-8")

    # 1. Still readable, transparently, after the deploy that adds encryption.
    resolved = asyncio.run(oauth_routes._read_token("usr_a"))
    assert resolved == '{"refresh_token": "pre-migration"}'
    # On-disk content is untouched by the read — still plaintext.
    assert pre_migration_path.read_text(encoding="utf-8") == '{"refresh_token": "pre-migration"}'

    # 2. The next WRITE (e.g. a reconnect) upgrades it to an encrypted envelope.
    asyncio.run(oauth_routes._write_token("usr_a", '{"refresh_token": "post-migration"}'))
    on_disk = pre_migration_path.read_text(encoding="utf-8")
    assert "refresh_token" not in on_disk
    plaintext, key_version = decrypt_stored(on_disk, parse_key(_TEST_TOKEN_KEY_B64))
    assert plaintext == '{"refresh_token": "post-migration"}'
    assert key_version == 1

    # 3. And reading it back afterward still works, now via decryption.
    resolved_after = asyncio.run(oauth_routes._read_token("usr_a"))
    assert resolved_after == '{"refresh_token": "post-migration"}'


@pytestmark_fastapi
def test_disconnect_revokes_before_deleting(monkeypatch):
    """POST /disconnect must call Google's /revoke endpoint with the stored
    refresh_token BEFORE deleting the local/Postgres copy."""
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)
    monkeypatch.setattr(oauth_routes, "_require_user", lambda req: "u123")
    monkeypatch.setattr(
        oauth_routes, "_read_token",
        mock.AsyncMock(return_value='{"refresh_token": "rt-abc", "scope": "https://www.googleapis.com/auth/calendar.events"}'),
    )

    call_order: list[str] = []

    async def fake_revoke(token_json):
        call_order.append("revoke")
        assert "rt-abc" in token_json
        return True

    async def fake_delete(user_id):
        call_order.append("delete")

    monkeypatch.setattr(oauth_routes, "_revoke_at_google", fake_revoke)
    monkeypatch.setattr(oauth_routes, "_delete_token", fake_delete)

    fake_req = mock.MagicMock()
    result = asyncio.run(oauth_routes.disconnect(fake_req))
    assert call_order == ["revoke", "delete"]
    assert result == {"connected": False, "revoked": True}


@pytestmark_fastapi
def test_disconnect_deletes_locally_even_when_revoke_fails(monkeypatch):
    """A revoke failure (Google unreachable / 5xx) must not block the user
    from disconnecting locally — deletion still proceeds, but the response
    reports revoked=False so the caller isn't told the upstream grant is
    definitely gone when it might not be."""
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)
    monkeypatch.setattr(oauth_routes, "_require_user", lambda req: "u123")
    monkeypatch.setattr(
        oauth_routes, "_read_token",
        mock.AsyncMock(return_value='{"refresh_token": "rt-abc"}'),
    )

    deleted = {"called": False}

    async def fake_revoke(token_json):
        return False  # upstream failure

    async def fake_delete(user_id):
        deleted["called"] = True

    monkeypatch.setattr(oauth_routes, "_revoke_at_google", fake_revoke)
    monkeypatch.setattr(oauth_routes, "_delete_token", fake_delete)

    fake_req = mock.MagicMock()
    result = asyncio.run(oauth_routes.disconnect(fake_req))
    assert deleted["called"] is True
    assert result == {"connected": False, "revoked": False}


@pytestmark_fastapi
def test_status_reports_stale_scope_for_old_full_calendar_token(monkeypatch):
    """/status surfaces scope_stale=True for a token minted under the old
    full-access scope, which web/js/calendar.js uses to show a "Reconnect
    Calendar" prompt."""
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)
    monkeypatch.setattr(oauth_routes, "_require_user", lambda req: "u123")
    monkeypatch.setattr(
        oauth_routes, "_read_token",
        mock.AsyncMock(
            return_value=json.dumps(
                {"refresh_token": "rt", "scope": "https://www.googleapis.com/auth/calendar"}
            )
        ),
    )

    fake_req = mock.MagicMock()
    result = asyncio.run(oauth_routes.status(fake_req))
    assert result == {"connected": True, "scope_stale": True}


@pytestmark_fastapi
def test_status_reports_fresh_scope_as_not_stale(monkeypatch):
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)
    monkeypatch.setattr(oauth_routes, "_require_user", lambda req: "u123")
    monkeypatch.setattr(
        oauth_routes, "_read_token",
        mock.AsyncMock(
            return_value=json.dumps(
                {"refresh_token": "rt", "scope": "https://www.googleapis.com/auth/calendar.events"}
            )
        ),
    )

    fake_req = mock.MagicMock()
    result = asyncio.run(oauth_routes.status(fake_req))
    assert result == {"connected": True, "scope_stale": False}


@pytestmark_fastapi
def test_status_not_connected_when_no_token(monkeypatch):
    import apps.calendar_oauth_routes as oauth_routes

    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = None
    monkeypatch.setattr(oauth_routes, "settings", fake_settings)
    monkeypatch.setattr(oauth_routes, "_require_user", lambda req: "u123")
    monkeypatch.setattr(oauth_routes, "_read_token", mock.AsyncMock(return_value=None))

    fake_req = mock.MagicMock()
    result = asyncio.run(oauth_routes.status(fake_req))
    assert result == {"connected": False, "scope_stale": False}


# ---------------------------------------------------------------------------
# core.calendar_token_crypto — key parsing, envelope encrypt/decrypt, nonce
# freshness.
# ---------------------------------------------------------------------------

def test_calendar_token_key_unset_returns_none():
    from core.calendar_token_crypto import parse_key

    assert parse_key(None) is None
    assert parse_key("") is None
    assert parse_key("   ") is None


def test_calendar_token_key_malformed_base64_raises():
    from core.calendar_token_crypto import CalendarTokenKeyError, parse_key

    with pytest.raises(CalendarTokenKeyError):
        parse_key("not-valid-base64!!! ***")


def test_calendar_token_key_wrong_length_raises():
    from core.calendar_token_crypto import CalendarTokenKeyError, parse_key

    short_key = base64.urlsafe_b64encode(b"\x00" * 16).decode("ascii")  # 16 bytes, not 32
    with pytest.raises(CalendarTokenKeyError):
        parse_key(short_key)


def test_calendar_token_key_correct_length_ok():
    from core.calendar_token_crypto import parse_key

    key = parse_key(_TEST_TOKEN_KEY_B64)
    assert key is not None
    assert len(key) == 32


def test_settings_construction_fails_on_malformed_calendar_token_key(monkeypatch):
    """CALENDAR_TOKEN_KEY validation happens at settings-construction time —
    a malformed key must fail process boot, not the first calendar connect."""
    from pydantic import ValidationError
    from core.config import TurtleSettings

    monkeypatch.setenv("CALENDAR_TOKEN_KEY", "not-valid-base64!!!")
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")
    monkeypatch.setenv("COHERE_API_KEY", "fake-test-key")
    with pytest.raises(ValidationError):
        TurtleSettings(_env_file=None)


def test_settings_construction_fails_on_wrong_length_calendar_token_key(monkeypatch):
    from pydantic import ValidationError
    from core.config import TurtleSettings

    short_key = base64.urlsafe_b64encode(b"\x00" * 16).decode("ascii")
    monkeypatch.setenv("CALENDAR_TOKEN_KEY", short_key)
    with pytest.raises(ValidationError):
        TurtleSettings(_env_file=None)


def test_settings_construction_ok_with_valid_calendar_token_key(monkeypatch):
    from core.config import TurtleSettings

    monkeypatch.setenv("CALENDAR_TOKEN_KEY", _TEST_TOKEN_KEY_B64)
    settings = TurtleSettings(_env_file=None)
    assert settings.calendar_token_key.get_secret_value() == _TEST_TOKEN_KEY_B64


def test_settings_construction_ok_with_calendar_token_key_unset(monkeypatch):
    from core.config import TurtleSettings

    monkeypatch.delenv("CALENDAR_TOKEN_KEY", raising=False)
    settings = TurtleSettings(_env_file=None)
    assert settings.calendar_token_key is None


def test_encrypt_for_storage_roundtrips():
    from core.calendar_token_crypto import decrypt_stored, encrypt_for_storage, parse_key

    key = parse_key(_TEST_TOKEN_KEY_B64)
    stored = encrypt_for_storage('{"refresh_token": "abc"}', key, is_cloud=True)
    assert "refresh_token" not in stored
    plaintext, key_version = decrypt_stored(stored, key)
    assert plaintext == '{"refresh_token": "abc"}'
    assert key_version == 1


def test_encrypt_for_storage_cloud_without_key_raises():
    from core.calendar_token_crypto import CalendarTokenKeyRequired, encrypt_for_storage

    with pytest.raises(CalendarTokenKeyRequired):
        encrypt_for_storage('{"refresh_token": "abc"}', None, is_cloud=True)


def test_encrypt_for_storage_local_without_key_is_plaintext():
    from core.calendar_token_crypto import encrypt_for_storage

    stored = encrypt_for_storage('{"refresh_token": "abc"}', None, is_cloud=False)
    assert stored == '{"refresh_token": "abc"}'


def test_decrypt_stored_plaintext_passthrough():
    """A bare (pre-encryption) token is read back unchanged, key_version 0,
    with no key required."""
    from core.calendar_token_crypto import decrypt_stored

    plaintext, key_version = decrypt_stored('{"refresh_token": "abc"}', None)
    assert plaintext == '{"refresh_token": "abc"}'
    assert key_version == 0


def test_decrypt_stored_envelope_without_key_raises():
    from core.calendar_token_crypto import (
        CalendarTokenDecryptError,
        encrypt_for_storage,
        parse_key,
    )

    key = parse_key(_TEST_TOKEN_KEY_B64)
    stored = encrypt_for_storage('{"refresh_token": "abc"}', key, is_cloud=True)
    with pytest.raises(CalendarTokenDecryptError):
        from core.calendar_token_crypto import decrypt_stored

        decrypt_stored(stored, None)


def test_decrypt_stored_envelope_wrong_key_raises():
    from core.calendar_token_crypto import (
        CalendarTokenDecryptError,
        decrypt_stored,
        encrypt_for_storage,
        parse_key,
    )

    key1 = parse_key(_TEST_TOKEN_KEY_B64)
    key2 = parse_key(_TEST_TOKEN_KEY_B64_2)
    stored = encrypt_for_storage('{"refresh_token": "abc"}', key1, is_cloud=True)
    with pytest.raises(CalendarTokenDecryptError):
        decrypt_stored(stored, key2)


def test_encrypting_same_plaintext_twice_produces_different_ciphertext():
    """Proof of nonce freshness: encrypting the identical plaintext twice
    under the same key must never produce the same ciphertext — nonce reuse
    under AES-GCM is catastrophic (recovers the authentication key)."""
    from core.calendar_token_crypto import encrypt_for_storage, parse_key

    key = parse_key(_TEST_TOKEN_KEY_B64)
    stored_1 = encrypt_for_storage('{"refresh_token": "same-plaintext"}', key, is_cloud=True)
    stored_2 = encrypt_for_storage('{"refresh_token": "same-plaintext"}', key, is_cloud=True)
    assert stored_1 != stored_2

    blob_1 = json.loads(stored_1)["blob"]
    blob_2 = json.loads(stored_2)["blob"]
    assert blob_1 != blob_2
    # The leading 12 bytes (base64-decoded) are the nonce — must differ too.
    nonce_1 = base64.b64decode(blob_1)[:12]
    nonce_2 = base64.b64decode(blob_2)[:12]
    assert nonce_1 != nonce_2


# ---------------------------------------------------------------------------
# tools.calendar_tool._load_token_json — decrypts in both cloud and local mode
# ---------------------------------------------------------------------------

def test_load_token_json_decrypts_local_mode(tmp_path, monkeypatch):
    import tools.calendar_tool as ct
    from core.calendar_token_crypto import encrypt_for_storage, parse_key

    key = parse_key(_TEST_TOKEN_KEY_B64)
    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = False
    fake_settings.calendar_token_key = SecretStr(_TEST_TOKEN_KEY_B64)
    fake_settings.google_calendar_token_json = None
    monkeypatch.setattr(ct, "settings", fake_settings)

    def fake_personal_memory_dir(user_id: str):
        d = tmp_path / user_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr("core.paths.personal_memory_dir", fake_personal_memory_dir)

    user_id = "u123"
    token_path = fake_personal_memory_dir(user_id) / "google_calendar_token.json"
    encrypted = encrypt_for_storage(
        json.dumps({"refresh_token": "per-user-token"}), key, is_cloud=False
    )
    assert "refresh_token" not in encrypted
    token_path.write_text(encrypted, encoding="utf-8")

    resolved = ct._load_token_json(user_id)
    assert resolved is not None
    assert json.loads(resolved)["refresh_token"] == "per-user-token"


def test_load_token_json_decrypts_cloud_mode(monkeypatch):
    import tools.calendar_tool as ct
    from core.calendar_token_crypto import encrypt_for_storage, parse_key

    key = parse_key(_TEST_TOKEN_KEY_B64)
    fake_settings = mock.MagicMock()
    fake_settings.is_cloud = True
    fake_settings.calendar_token_key = SecretStr(_TEST_TOKEN_KEY_B64)
    fake_settings.google_calendar_token_json = None
    monkeypatch.setattr(ct, "settings", fake_settings)

    encrypted = encrypt_for_storage(
        json.dumps({"refresh_token": "cloud-token"}), key, is_cloud=True
    )
    with mock.patch(
        "core.storage.cloud.calendar_token_store.get_token_json",
        return_value=encrypted,
    ):
        resolved = ct._load_token_json("u123")
    assert resolved is not None
    assert json.loads(resolved)["refresh_token"] == "cloud-token"
