"""
test/discord_deferred_processing_test.py
--------------------------------------------
Unit coverage for apps/channels/discord.py's deferred-interaction handling
(Vercel migration Phase 4). Discord's 3-second ACK deadline forces a defer;
the real work must run AFTER the response is sent. In cloud mode this can't
safely rely on a bare detached asyncio task (Vercel's documented
post-response background-work API, waitUntil()/after(), is JS-only — see
_kick_off_deferred_processing's docstring), so cloud mode self-invokes
POST /channels/discord/process as an independent request instead. These
tests cover both paths plus the new internal endpoint's auth.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import apps.channels.discord as discord_module
from apps.channels import TurtleResponse
from apps.channels.discord import router


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


_PAYLOAD = {
    "interaction_token": "tok_abc",
    "interaction_id": "int_1",
    "channel_id": "chan_1",
    "discord_user_id": "999",
    "text": "hello",
    "sender_name": "Alice",
    "is_private": True,
}


class KickOffDeferredProcessingTest(unittest.IsolatedAsyncioTestCase):
    async def test_local_mode_uses_in_process_task(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.is_cloud = False
            with patch.object(
                discord_module, "_process_deferred_interaction", new_callable=AsyncMock
            ) as fake_process, patch("asyncio.create_task", wraps=asyncio.create_task) as fake_ct:
                await discord_module._kick_off_deferred_processing(_PAYLOAD)
                await asyncio.sleep(0)  # let the created task actually run

        fake_ct.assert_called_once()
        fake_process.assert_awaited_once_with(_PAYLOAD)

    async def test_cloud_mode_self_invokes_and_sends_full_request(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.cron_shared_secret.get_secret_value.return_value = "shared-secret"
            fake_settings.public_base_url = "https://turtle.example.com"

            fake_client = AsyncMock()
            fake_client.post = AsyncMock(side_effect=httpx.ReadTimeout("expected"))
            fake_client_cm = AsyncMock()
            fake_client_cm.__aenter__.return_value = fake_client
            fake_client_cm.__aexit__.return_value = False

            with patch("httpx.AsyncClient", return_value=fake_client_cm), patch.object(
                discord_module, "_process_deferred_interaction", new_callable=AsyncMock
            ) as fake_process:
                await discord_module._kick_off_deferred_processing(_PAYLOAD)

        # A ReadTimeout is the EXPECTED outcome (we deliberately don't wait
        # for the reply) — must NOT fall back to the in-process task.
        fake_process.assert_not_called()
        fake_client.post.assert_awaited_once()
        call = fake_client.post.await_args
        self.assertEqual(call.args[0], "https://turtle.example.com/channels/discord/process")
        self.assertEqual(call.kwargs["json"], _PAYLOAD)
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer shared-secret")

    async def test_cloud_mode_without_secret_falls_back_to_in_process_task(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.cron_shared_secret = None
            with patch.object(
                discord_module, "_process_deferred_interaction", new_callable=AsyncMock
            ) as fake_process:
                await discord_module._kick_off_deferred_processing(_PAYLOAD)
                await asyncio.sleep(0)

        fake_process.assert_awaited_once_with(_PAYLOAD)

    async def test_cloud_mode_self_invoke_network_error_falls_back(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.cron_shared_secret.get_secret_value.return_value = "shared-secret"
            fake_settings.public_base_url = "https://turtle.example.com"

            fake_client = AsyncMock()
            fake_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            fake_client_cm = AsyncMock()
            fake_client_cm.__aenter__.return_value = fake_client
            fake_client_cm.__aexit__.return_value = False

            with patch("httpx.AsyncClient", return_value=fake_client_cm), patch.object(
                discord_module, "_process_deferred_interaction", new_callable=AsyncMock
            ) as fake_process:
                await discord_module._kick_off_deferred_processing(_PAYLOAD)
                await asyncio.sleep(0)

        fake_process.assert_awaited_once_with(_PAYLOAD)


class DiscordProcessEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _make_client()

    def test_missing_auth_rejected(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "real-secret"
            resp = self.client.post("/channels/discord/process", json=_PAYLOAD)
        self.assertEqual(resp.status_code, 401)

    def test_wrong_secret_rejected(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "real-secret"
            resp = self.client.post(
                "/channels/discord/process",
                json=_PAYLOAD,
                headers={"Authorization": "Bearer wrong"},
            )
        self.assertEqual(resp.status_code, 401)

    def test_no_secret_configured_rejected(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.cron_shared_secret = None
            resp = self.client.post(
                "/channels/discord/process",
                json=_PAYLOAD,
                headers={"Authorization": "Bearer anything"},
            )
        self.assertEqual(resp.status_code, 401)

    def test_correct_secret_processes_and_returns_ok(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "real-secret"
            with patch.object(
                discord_module, "_process_deferred_interaction", new_callable=AsyncMock
            ) as fake_process:
                resp = self.client.post(
                    "/channels/discord/process",
                    json=_PAYLOAD,
                    headers={"Authorization": "Bearer real-secret"},
                )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True})
        fake_process.assert_awaited_once_with(_PAYLOAD)

    def test_missing_required_field_rejected(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "real-secret"
            bad_payload = dict(_PAYLOAD)
            del bad_payload["text"]
            resp = self.client.post(
                "/channels/discord/process",
                json=bad_payload,
                headers={"Authorization": "Bearer real-secret"},
            )
        self.assertEqual(resp.status_code, 400)


class ProcessDeferredInteractionTest(unittest.IsolatedAsyncioTestCase):
    async def test_success_dispatches_and_sends_followup(self) -> None:
        with patch.object(
            discord_module, "identity_manager"
        ) as fake_identity, patch.object(
            discord_module, "dispatch_event", new_callable=AsyncMock
        ) as fake_dispatch, patch.object(
            discord_module, "_send_followup", new_callable=AsyncMock
        ) as fake_send:
            fake_identity.resolve_user = AsyncMock(return_value="usr_a")
            fake_dispatch.return_value = TurtleResponse(
                content="reply text", channel="discord", user_id="usr_a"
            )
            await discord_module._process_deferred_interaction(_PAYLOAD)

        fake_dispatch.assert_awaited_once()
        event = fake_dispatch.call_args[0][0]
        self.assertEqual(event.content, "hello")
        self.assertTrue(event.is_private)
        fake_send.assert_awaited_once_with("tok_abc", "reply text")

    async def test_failure_sends_graceful_error_followup(self) -> None:
        with patch.object(
            discord_module, "identity_manager"
        ) as fake_identity, patch.object(
            discord_module, "dispatch_event", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ), patch.object(
            discord_module, "_send_followup", new_callable=AsyncMock
        ) as fake_send:
            fake_identity.resolve_user = AsyncMock(return_value="usr_a")
            await discord_module._process_deferred_interaction(_PAYLOAD)

        fake_send.assert_awaited_once()
        self.assertIn("wrong", fake_send.call_args[0][1].lower())


if __name__ == "__main__":
    unittest.main()
