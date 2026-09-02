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
