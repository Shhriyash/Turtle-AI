"""
test/cloud_gateway_gating_test.py
------------------------------------
Vercel migration Phase 4: verifies the persistent-connection startup hooks
(Discord Gateway, Telegram long-polling, RoutineScheduler) are skipped in
cloud mode — none of them can survive a serverless cold start (see each
hook's own docstring in apps/turtle_server.py for why), and running them
alongside cron-tick / the webhook adapters would duplicate or conflict with
the serverless-shaped replacements.

These hooks are already skipped under pytest for other reasons (the "pytest"
in sys.modules guard, to avoid opening real gateway connections during the
test suite), so each test removes that module temporarily to exercise the
is_cloud branch specifically.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from unittest.mock import patch

import apps.turtle_server as server


def _run(coro):
    return asyncio.run(coro)


class GatewayHookCloudGatingTest(unittest.TestCase):
    def setUp(self) -> None:
        # Temporarily hide the "pytest" sentinel each hook checks first, so
        # the is_cloud branch underneath it actually runs.
        self._pytest_module = sys.modules.pop("pytest", None)

    def tearDown(self) -> None:
        if self._pytest_module is not None:
            sys.modules["pytest"] = self._pytest_module

    def test_discord_gateway_hook_skipped_in_cloud_mode(self) -> None:
        with patch.object(server, "settings") as fake_settings:
            fake_settings.is_cloud = True
            with patch(
                "apps.channels.discord_gateway.start_discord_gateway"
            ) as fake_start:
                _run(server._start_discord_gateway_hook())
        fake_start.assert_not_called()

    def test_discord_gateway_hook_runs_locally(self) -> None:
        with patch.object(server, "settings") as fake_settings:
            fake_settings.is_cloud = False
            with patch(
                "apps.channels.discord_gateway.start_discord_gateway"
            ) as fake_start:
                fake_start.return_value = _completed_future()
                _run(server._start_discord_gateway_hook())
        fake_start.assert_called_once()

    def test_telegram_gateway_hook_skipped_in_cloud_mode(self) -> None:
        with patch.object(server, "settings") as fake_settings:
            fake_settings.is_cloud = True
            with patch(
                "apps.channels.telegram_gateway.start_telegram_gateway"
            ) as fake_start:
                _run(server._start_telegram_gateway_hook())
        fake_start.assert_not_called()

    def test_telegram_gateway_hook_runs_locally(self) -> None:
        with patch.object(server, "settings") as fake_settings:
            fake_settings.is_cloud = False
            with patch(
                "apps.channels.telegram_gateway.start_telegram_gateway"
            ) as fake_start:
                fake_start.return_value = _completed_future()
                _run(server._start_telegram_gateway_hook())
        fake_start.assert_called_once()


def _completed_future():
    async def _noop():
        return None

    return _noop()


class RoutineSchedulerCloudGatingTest(unittest.TestCase):
    def test_routine_scheduler_skipped_in_cloud_mode(self) -> None:
        with patch.object(server, "settings") as fake_settings:
            fake_settings.is_cloud = True
            with patch("core.routine_scheduler.RoutineScheduler") as fake_cls:
                _run(server._start_routine_scheduler())
        fake_cls.assert_not_called()

    def test_routine_scheduler_starts_locally(self) -> None:
        with patch.object(server, "settings") as fake_settings:
            fake_settings.is_cloud = False
            with patch("core.routine_scheduler.RoutineScheduler") as fake_cls:
                _run(server._start_routine_scheduler())
        fake_cls.assert_called_once()
        fake_cls.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
