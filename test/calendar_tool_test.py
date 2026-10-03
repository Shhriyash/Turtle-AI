"""
tools/calendar_tool.py behavior tests.

Migrated from test_tier3_verification.py (TestF4CalendarTool) — the tool-level
behavior only. The two graph-registry tests (test_calendar_graph_registered,
test_calendar_node_kinds_in_enum) are NOT migrated: they pin core.graph, which
is being removed.

Covers: tool + args-model import, CalendarCreateArgs defaults, the invalid
result returned when Google Calendar credentials are absent (create + list),
the contracts-level arg models, and the config fields.
"""
from __future__ import annotations


class _EnvTripwireSettings:
    """Settings stand-in whose google_calendar_token_json raises if ever
    read — proves the legacy env fallback is genuinely never consulted, not
    merely coincidentally unused (WP2.B / ledger 2.8)."""

    def __init__(self, *, is_cloud: bool, google_calendar_credentials_json=None,
                 calendar_token_key=None):
        self.is_cloud = is_cloud
        self.google_calendar_credentials_json = google_calendar_credentials_json
        self.calendar_token_key = calendar_token_key

    @property
    def google_calendar_token_json(self):
        raise AssertionError(
            "legacy env var google_calendar_token_json must not be read "
            "when a user_id is given in cloud mode, nor when it is a "
            "genuine miss with a user_id given in local mode"
        )


class TestF4CalendarTool:
    """Google Calendar tool — args validation + graceful no-creds paths."""

    def test_calendar_tool_importable(self):
        from tools.calendar_tool import (
            create_calendar_event, list_upcoming_events,
            CalendarCreateArgs, CalendarListArgs,
            CalendarEventResult, CalendarEventListResult,
        )
        assert callable(create_calendar_event)
        assert callable(list_upcoming_events)

    def test_calendar_create_args_valid(self):
        from tools.calendar_tool import CalendarCreateArgs
        args = CalendarCreateArgs(
            title="Team Standup",
            start_iso="2026-05-10T09:00:00+00:00",
            end_iso="2026-05-10T09:30:00+00:00",
            attendee_emails=["alice@example.com"],
        )
        assert args.add_google_meet is True

    def test_calendar_create_returns_invalid_without_creds(self):
        import asyncio, unittest.mock as mock
        from tools.calendar_tool import create_calendar_event, CalendarCreateArgs

        async def run():
            import tools.calendar_tool as ct
            from core.config import settings as real_settings
            fake = mock.MagicMock()
            fake.google_calendar_credentials_json = None
            fake.google_calendar_token_json = None
            ct.settings = fake
            args = CalendarCreateArgs(
                title="Test",
                start_iso="2026-05-10T09:00:00+00:00",
                end_iso="2026-05-10T09:30:00+00:00",
            )
            result = await create_calendar_event(args)
            ct.settings = real_settings
            return result

        result = asyncio.run(run())
        assert result.status == "invalid"
        assert "credentials" in result.error_message.lower() or result.error_code == "credentials_missing"

    def test_calendar_list_returns_invalid_without_creds(self):
        import asyncio, unittest.mock as mock
        from tools.calendar_tool import list_upcoming_events, CalendarListArgs

        async def run():
            import tools.calendar_tool as ct
            from core.config import settings as real_settings
            fake_settings = mock.MagicMock()
            fake_settings.google_calendar_credentials_json = None
            fake_settings.google_calendar_token_json = None
            ct.settings = fake_settings
            args = CalendarListArgs(max_results=3)
            result = await list_upcoming_events(args)
            ct.settings = real_settings
            return result

        result = asyncio.run(run())
        assert result.status == "invalid"

    def test_calendar_args_in_contracts(self):
        from tools.contracts import CalendarCreateArgs, CalendarListArgs
        args = CalendarCreateArgs(
            title="Sync",
            start_iso="2026-06-01T10:00:00+00:00",
            end_iso="2026-06-01T10:30:00+00:00",
        )
        assert args.add_google_meet is True
        list_args = CalendarListArgs()
        assert list_args.max_results == 5

    def test_config_has_calendar_fields(self):
        from core.config import TurtleSettings
        fields = TurtleSettings.model_fields
        assert "google_calendar_credentials_json" in fields
        assert "google_calendar_token_json" in fields


class TestCalendarNotifyAttendeesSendUpdates:
    """WP1.E1 (ledger 1b.1): sendUpdates must be "all" ONLY when the user
    explicitly asked to notify attendees — never merely because attendees
    are present. Assert directly on the kwarg passed to
    service.events().insert(), which nothing tested before this WP."""

    @staticmethod
    def _build_service_mock():
        import unittest.mock as mock
        service = mock.MagicMock()
        insert_mock = service.events.return_value.insert
        insert_mock.return_value.execute.return_value = {
            "id": "evt_1",
            "summary": "Sync",
            "start": {"dateTime": "2026-06-01T10:00:00+00:00"},
            "end": {"dateTime": "2026-06-01T10:30:00+00:00"},
            "htmlLink": "https://calendar.google.com/evt_1",
        }
        return service, insert_mock

    def test_send_updates_is_none_with_attendees_but_no_notify_request(self):
        import asyncio
        import unittest.mock as mock
        from tools.calendar_tool import create_calendar_event, CalendarCreateArgs

        async def run():
            import tools.calendar_tool as ct
            fake_settings = mock.MagicMock()
            fake_settings.google_calendar_credentials_json = '{"type": "service_account"}'
            ct.settings = fake_settings
            service, insert_mock = self._build_service_mock()
            with mock.patch.object(ct, "_build_service", return_value=service):
                args = CalendarCreateArgs(
                    title="Sync",
                    start_iso="2026-06-01T10:00:00+00:00",
                    end_iso="2026-06-01T10:30:00+00:00",
                    attendee_emails=["alice@example.com"],
                    notify_attendees=False,
                )
                result = await create_calendar_event(args)
            return result, insert_mock

        result, insert_mock = asyncio.run(run())
        assert result.status == "ok"
        _, kwargs = insert_mock.call_args
        assert kwargs["sendUpdates"] == "none"

    def test_send_updates_is_all_only_when_notify_requested(self):
        import asyncio
        import unittest.mock as mock
        from tools.calendar_tool import create_calendar_event, CalendarCreateArgs

        async def run():
            import tools.calendar_tool as ct
            fake_settings = mock.MagicMock()
            fake_settings.google_calendar_credentials_json = '{"type": "service_account"}'
            ct.settings = fake_settings
            service, insert_mock = self._build_service_mock()
            with mock.patch.object(ct, "_build_service", return_value=service):
                args = CalendarCreateArgs(
                    title="Sync",
                    start_iso="2026-06-01T10:00:00+00:00",
                    end_iso="2026-06-01T10:30:00+00:00",
                    attendee_emails=["alice@example.com"],
                    notify_attendees=True,
                )
                result = await create_calendar_event(args)
            return result, insert_mock

        result, insert_mock = asyncio.run(run())
        assert result.status == "ok"
        _, kwargs = insert_mock.call_args
        assert kwargs["sendUpdates"] == "all"

    def test_send_updates_is_none_by_default_with_no_attendees(self):
        import asyncio
        import unittest.mock as mock
        from tools.calendar_tool import create_calendar_event, CalendarCreateArgs

        async def run():
            import tools.calendar_tool as ct
            fake_settings = mock.MagicMock()
            fake_settings.google_calendar_credentials_json = '{"type": "service_account"}'
            ct.settings = fake_settings
            service, insert_mock = self._build_service_mock()
            with mock.patch.object(ct, "_build_service", return_value=service):
                args = CalendarCreateArgs(
                    title="Sync",
                    start_iso="2026-06-01T10:00:00+00:00",
                    end_iso="2026-06-01T10:30:00+00:00",
                )
                result = await create_calendar_event(args)
            return result, insert_mock

        result, insert_mock = asyncio.run(run())
        assert result.status == "ok"
        _, kwargs = insert_mock.call_args
        assert kwargs["sendUpdates"] == "none"

    def test_notify_attendees_defaults_false_in_both_arg_models(self):
        from tools.calendar_tool import CalendarCreateArgs as ToolArgs
        from tools.contracts import CalendarCreateArgs as ContractArgs

        for cls in (ToolArgs, ContractArgs):
            args = cls(
                title="Sync",
                start_iso="2026-06-01T10:00:00+00:00",
                end_iso="2026-06-01T10:30:00+00:00",
            )
            assert args.notify_attendees is False


class TestRenderCalendarDraft:
    def test_render_includes_legible_fields_for_review(self):
        from tools.calendar_tool import render_calendar_draft, CalendarCreateArgs

        args = CalendarCreateArgs(
            title="Board Sync",
            start_iso="2026-06-01T10:00:00+05:30",
            end_iso="2026-06-01T10:30:00+05:30",
            attendee_emails=["alice@example.com", "bob@example.com"],
            notify_attendees=True,
        )
        rendered = render_calendar_draft(args)
        assert "Board Sync" in rendered
        assert "2026-06-01T10:00:00+05:30" in rendered
        assert "2026-06-01T10:30:00+05:30" in rendered
        assert "alice@example.com" in rendered
        assert "bob@example.com" in rendered
        assert "confirm" in rendered.lower()


class TestCredentialsUnavailableVsMissing:
    """WP2.B (ledger 2.8): a transient DB error / undecryptable token must
    never be conflated with "no token stored", and - critically - must
    never fall back to the legacy single-tenant GOOGLE_CALENDAR_TOKEN_JSON
    env var. Previously all four cases (falsy user_id, any exception on
    lookup, empty/None stored value, any exception on decrypt) collapsed
    into one except-Exception-stored-None and fell through to the env var,
    so a Postgres hiccup for one user could create an event on the
    operator's own personal calendar."""

    _OAUTH_CREDS_JSON = '{"installed": {"client_id": "cid", "client_secret": "csecret"}}'

    def test_cloud_user_id_no_row_is_credentials_missing_env_never_read(self):
        """cloud + user_id + no row -> credentials_missing, env var not read."""
        import asyncio, unittest.mock as mock
        from tools.calendar_tool import create_calendar_event, CalendarCreateArgs

        async def run():
            import tools.calendar_tool as ct
            from core.config import settings as real_settings
            fake_settings = _EnvTripwireSettings(
                is_cloud=True, google_calendar_credentials_json=self._OAUTH_CREDS_JSON
            )
            ct.settings = fake_settings
            try:
                with mock.patch(
                    "core.storage.cloud.calendar_token_store.get_token_json",
                    return_value=None,
                ), mock.patch("googleapiclient.discovery.build") as build_mock:
                    args = CalendarCreateArgs(
                        title="T",
                        start_iso="2026-06-01T10:00:00+00:00",
                        end_iso="2026-06-01T10:30:00+00:00",
                    )
                    result = await create_calendar_event(args, user_id="usr_a")
            finally:
                ct.settings = real_settings
            return result, build_mock

        result, build_mock = asyncio.run(run())
        assert result.status == "invalid"
        assert result.error_code == "credentials_missing"
        build_mock.assert_not_called()  # no client ever built -> no event created

    def test_cloud_user_id_db_error_is_credentials_unavailable_no_event_env_never_read(self):
        """cloud + user_id + get_token_json raising -> credentials_unavailable,
        no event created, env var never read. This is the headline case: a
        transient DB error must not create events on the operator's own
        calendar."""
        import asyncio, unittest.mock as mock
        from tools.calendar_tool import create_calendar_event, CalendarCreateArgs

        async def run():
            import tools.calendar_tool as ct
            from core.config import settings as real_settings
            fake_settings = _EnvTripwireSettings(
                is_cloud=True, google_calendar_credentials_json=self._OAUTH_CREDS_JSON
            )
            ct.settings = fake_settings
            try:
                with mock.patch(
                    "core.storage.cloud.calendar_token_store.get_token_json",
                    side_effect=RuntimeError("connection refused"),
                ), mock.patch("googleapiclient.discovery.build") as build_mock:
                    args = CalendarCreateArgs(
                        title="T",
                        start_iso="2026-06-01T10:00:00+00:00",
                        end_iso="2026-06-01T10:30:00+00:00",
                    )
                    result = await create_calendar_event(args, user_id="usr_a")
            finally:
                ct.settings = real_settings
            return result, build_mock

        result, build_mock = asyncio.run(run())
        assert result.status == "invalid"
        assert result.error_code == "credentials_unavailable"
        # The env var (a raising property here) was never accessed, and no
        # Google API client was ever built, so no event could have been
        # created.
        build_mock.assert_not_called()

    def test_cloud_user_id_list_also_creates_nothing_on_credentials_unavailable(self):
        """Same DB-error scenario via list_upcoming_events: nothing is listed
        (no API call made) and the env var is never read."""
        import asyncio, unittest.mock as mock
        from tools.calendar_tool import list_upcoming_events, CalendarListArgs

        async def run():
            import tools.calendar_tool as ct
            from core.config import settings as real_settings
            fake_settings = _EnvTripwireSettings(
                is_cloud=True, google_calendar_credentials_json=self._OAUTH_CREDS_JSON
            )
            ct.settings = fake_settings
            try:
                with mock.patch(
                    "core.storage.cloud.calendar_token_store.get_token_json",
                    side_effect=RuntimeError("connection refused"),
                ), mock.patch("googleapiclient.discovery.build") as build_mock:
                    args = CalendarListArgs(max_results=3)
                    result = await list_upcoming_events(args, user_id="usr_a")
            finally:
                ct.settings = real_settings
            return result, build_mock

        result, build_mock = asyncio.run(run())
        assert result.status == "invalid"
        assert result.error_code == "credentials_unavailable"
        build_mock.assert_not_called()

    def test_cloud_user_id_decrypt_failure_is_credentials_unavailable_env_never_read(self):
        """cloud + user_id + decryption failure -> credentials_unavailable
        (not credentials_missing): the ciphertext EXISTS but cannot be read
        (wrong key / tampering), which is more alarming than "no token" and
        must not be quietly filed as missing. Env var never read."""
        import asyncio, unittest.mock as mock
        from tools.calendar_tool import create_calendar_event, CalendarCreateArgs

        async def run():
            import tools.calendar_tool as ct
            from core.config import settings as real_settings
            fake_settings = _EnvTripwireSettings(
                is_cloud=True, google_calendar_credentials_json=self._OAUTH_CREDS_JSON
            )
            ct.settings = fake_settings
            try:
                with mock.patch(
                    "core.storage.cloud.calendar_token_store.get_token_json",
                    return_value='{"key_version": 1, "blob": "not-really-valid"}',
                ), mock.patch(
                    "core.calendar_token_crypto.decrypt_stored",
                    side_effect=ValueError("decryption failed: bad tag"),
                ), mock.patch("googleapiclient.discovery.build") as build_mock:
                    args = CalendarCreateArgs(
                        title="T",
                        start_iso="2026-06-01T10:00:00+00:00",
                        end_iso="2026-06-01T10:30:00+00:00",
                    )
                    result = await create_calendar_event(args, user_id="usr_a")
            finally:
                ct.settings = real_settings
            return result, build_mock

        result, build_mock = asyncio.run(run())
        assert result.status == "invalid"
        assert result.error_code == "credentials_unavailable"
        build_mock.assert_not_called()

    def test_local_no_user_id_env_var_still_works_unchanged(self):
        """local + no user_id + env var set -> still works, unchanged."""
        import tools.calendar_tool as ct
        from core.config import settings as real_settings

        fake_settings = _EnvTripwireSettingsWithEnv(
            is_cloud=False, token_json='{"refresh_token": "global-legacy"}'
        )
        ct.settings = fake_settings
        try:
            resolved = ct._load_token_json(None)
        finally:
            ct.settings = real_settings
        assert resolved == '{"refresh_token": "global-legacy"}'

    def test_local_user_id_no_stored_token_is_credentials_missing_env_never_read(self):
        """local + user_id + no stored token -> credentials_missing (not a
        fallback to the env var): the env fallback survives ONLY in local
        mode with NO user_id - a user_id being present at all means the
        per-user path is authoritative, even if nothing was found there."""
        import asyncio, unittest.mock as mock
        from tools.calendar_tool import create_calendar_event, CalendarCreateArgs

        async def run():
            import tools.calendar_tool as ct
            from core.config import settings as real_settings
            fake_settings = _EnvTripwireSettings(
                is_cloud=False, google_calendar_credentials_json=self._OAUTH_CREDS_JSON
            )
            ct.settings = fake_settings

            def fake_personal_memory_dir(user_id):
                import pathlib, tempfile
                d = pathlib.Path(tempfile.mkdtemp()) / user_id
                d.mkdir(parents=True, exist_ok=True)
                return d

            try:
                with mock.patch("core.paths.personal_memory_dir", fake_personal_memory_dir), \
                     mock.patch("googleapiclient.discovery.build") as build_mock:
                    args = CalendarCreateArgs(
                        title="T",
                        start_iso="2026-06-01T10:00:00+00:00",
                        end_iso="2026-06-01T10:30:00+00:00",
                    )
                    result = await create_calendar_event(args, user_id="usr_no_file")
            finally:
                ct.settings = real_settings
            return result, build_mock

        result, build_mock = asyncio.run(run())
        assert result.status == "invalid"
        assert result.error_code == "credentials_missing"
        build_mock.assert_not_called()


class _EnvTripwireSettingsWithEnv:
    """Opposite of _EnvTripwireSettings - used for the one case where the
    env var SHOULD legitimately be read (local mode, no user_id)."""

    def __init__(self, *, is_cloud: bool, token_json: str):
        self.is_cloud = is_cloud
        self.google_calendar_token_json = token_json
