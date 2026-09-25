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
