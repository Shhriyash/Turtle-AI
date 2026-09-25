"""
test/discord_deferred_processing_test.py
--------------------------------------------
Unit coverage for apps/channels/discord.py's deferred-interaction handling
(Vercel migration Phase 4). Discord's 3-second ACK deadline forces a defer;
the real work must run AFTER the response is sent. In cloud mode this can't
safely rely on a bare detached asyncio task (Vercel's documented
post-response background-work API, waitUntil()/after(), is JS-only — see
_kick_off_deferred_processing's docstring), so cloud mode self-invokes
POST /channels/discord/process as an independent request instead.

WP 1.B (ledger 1a.3 / S-7.3): that self-invoke now authenticates with
INTERNAL_JOB_SECRET (not the retired CRON_SHARED_SECRET), signs the request
(timestamp + nonce + body HMAC, core/internal_auth.py), and sends only an
opaque job id — the real payload (including discord_user_id) is stashed in
Redis first and read-and-deleted by the endpoint, never trusted from the
wire. Uses fakeredis (matches test/redis_backends_test.py's pattern).
"""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import fakeredis
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import apps.channels.discord as discord_module
from apps.channels import TurtleResponse
from apps.channels.discord import router
from core.internal_auth import sign_request, store_job_payload, take_job_payload


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _fake_redis_patch(fake_client):
    return patch(
        "core.internal_auth.get_redis_client",
        new=AsyncMock(return_value=fake_client),
    )


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
    def setUp(self) -> None:
        self.fake_redis = fakeredis.FakeAsyncRedis(decode_responses=True)
        self.redis_patcher = _fake_redis_patch(self.fake_redis)
        self.redis_patcher.start()
        self.addCleanup(self.redis_patcher.stop)

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

    async def test_cloud_mode_self_invokes_and_sends_only_a_job_id(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.internal_job_secret.get_secret_value.return_value = "job-secret"
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

        # The impersonation surface is closed: the wire body carries only an
        # opaque job id, never discord_user_id directly.
        sent_body = json.loads(call.kwargs["content"])
        self.assertEqual(set(sent_body), {"job_id"})

        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer job-secret")
        self.assertIn("X-Turtle-Timestamp", call.kwargs["headers"])
        self.assertIn("X-Turtle-Nonce", call.kwargs["headers"])
        self.assertIn("X-Turtle-Signature", call.kwargs["headers"])

        stored = await take_job_payload(sent_body["job_id"])
        self.assertEqual(stored, _PAYLOAD)

    async def test_cloud_mode_without_secret_falls_back_to_in_process_task(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.internal_job_secret = None
            with patch.object(
                discord_module, "_process_deferred_interaction", new_callable=AsyncMock
            ) as fake_process:
                await discord_module._kick_off_deferred_processing(_PAYLOAD)
                await asyncio.sleep(0)

        fake_process.assert_awaited_once_with(_PAYLOAD)

    async def test_cloud_mode_self_invoke_network_error_falls_back(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.internal_job_secret.get_secret_value.return_value = "job-secret"
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

    async def test_cloud_mode_redis_unavailable_falls_back_to_in_process(self) -> None:
        from core.storage.cloud import CloudBackendUnavailable

        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.internal_job_secret.get_secret_value.return_value = "job-secret"
            with patch(
                "core.internal_auth.get_redis_client",
                new=AsyncMock(side_effect=CloudBackendUnavailable("no REDIS_URL")),
            ), patch.object(
                discord_module, "_process_deferred_interaction", new_callable=AsyncMock
            ) as fake_process:
                await discord_module._kick_off_deferred_processing(_PAYLOAD)
                await asyncio.sleep(0)

        fake_process.assert_awaited_once_with(_PAYLOAD)

    async def test_cloud_mode_redis_command_fails_falls_back_to_in_process(self) -> None:
        """Coordinator fail-fix: unset-URL is not the only failure mode. A
        client object that EXISTS but whose command raises the redis-py
        driver's own exception (here, a real redis.exceptions.ConnectionError
        — not a mocked CloudBackendUnavailable) must degrade the same way,
        not surface as an unhandled 500 out of the Discord request handler.
        """
        import redis.exceptions

        fake_redis_client = AsyncMock()
        fake_redis_client.set = AsyncMock(
            side_effect=redis.exceptions.ConnectionError("connection refused")
        )

        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.internal_job_secret.get_secret_value.return_value = "job-secret"
            with patch(
                "core.internal_auth.get_redis_client",
                new=AsyncMock(return_value=fake_redis_client),
            ), patch.object(
                discord_module, "_process_deferred_interaction", new_callable=AsyncMock
            ) as fake_process:
                await discord_module._kick_off_deferred_processing(_PAYLOAD)
                await asyncio.sleep(0)

        fake_process.assert_awaited_once_with(_PAYLOAD)


class DiscordProcessEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _make_client()
        self.fake_redis = fakeredis.FakeAsyncRedis(decode_responses=True)
        self.redis_patcher = _fake_redis_patch(self.fake_redis)
        self.redis_patcher.start()
        self.addCleanup(self.redis_patcher.stop)

    def _signed_post(self, secret: str, body_dict: dict, *, auth: str | None = None):
        body = json.dumps(body_dict).encode("utf-8")
        envelope = sign_request(secret, body)
        headers = {"Authorization": auth if auth is not None else f"Bearer {secret}"}
        headers.update(envelope.headers())
        return self.client.post("/channels/discord/process", content=body, headers=headers)

    def _store_job(self, payload: dict) -> str:
        return asyncio.run(store_job_payload(payload))

    def test_missing_auth_rejected(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            resp = self.client.post("/channels/discord/process", json={"job_id": "whatever"})
        self.assertEqual(resp.status_code, 401)

    def test_wrong_secret_rejected(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            resp = self._signed_post("real-secret", {"job_id": "whatever"}, auth="Bearer wrong")
        self.assertEqual(resp.status_code, 401)

    def test_no_secret_configured_rejected(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.internal_job_secret = None
            resp = self.client.post(
                "/channels/discord/process",
                json={"job_id": "whatever"},
                headers={"Authorization": "Bearer anything"},
            )
        self.assertEqual(resp.status_code, 401)

    def test_cron_tick_secret_is_rejected_here(self) -> None:
        # The two secrets are NOT interchangeable.
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "job-secret"
            resp = self.client.post(
                "/channels/discord/process",
                json={"job_id": "whatever"},
                headers={"Authorization": "Bearer cron-tick-secret-value"},
            )
        self.assertEqual(resp.status_code, 401)

    def test_forged_signature_rejected(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            body = json.dumps({"job_id": "whatever"}).encode("utf-8")
            envelope = sign_request("real-secret", body)
            resp = self.client.post(
                "/channels/discord/process",
                content=body,
                headers={
                    "Authorization": "Bearer real-secret",
                    "X-Turtle-Timestamp": envelope.timestamp,
                    "X-Turtle-Nonce": envelope.nonce,
                    "X-Turtle-Signature": "0" * 64,
                },
            )
        self.assertEqual(resp.status_code, 401)

    def test_fabricated_body_with_no_valid_job_id_rejected(self) -> None:
        """WP 1.B / S-7.3 acceptance criterion: a correctly-signed request
        whose job_id doesn't point at anything real is rejected — a caller
        can't just invent a body (even one with a plausible-looking
        discord_user_id) and have it processed.
        """
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            with patch.object(
                discord_module, "_process_deferred_interaction", new_callable=AsyncMock
            ) as fake_process:
                resp = self._signed_post(
                    "real-secret",
                    {"job_id": "fabricated-id", "discord_user_id": "999"},
                )
        self.assertEqual(resp.status_code, 401)
        fake_process.assert_not_called()

    def test_unreachable_redis_rejects_with_401_not_500(self) -> None:
        """Coordinator fail-fix: verify_request's nonce claim hitting a real
        redis-py driver exception (not a mocked CloudBackendUnavailable)
        must produce a 401, never an unhandled 500.
        """
        import redis.exceptions

        fake_redis_client = AsyncMock()
        fake_redis_client.set = AsyncMock(
            side_effect=redis.exceptions.ConnectionError("connection refused")
        )
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            with patch(
                "core.internal_auth.get_redis_client",
                new=AsyncMock(return_value=fake_redis_client),
            ):
                resp = self._signed_post("real-secret", {"job_id": "whatever"})
        self.assertEqual(resp.status_code, 401)

    def test_replayed_request_rejected(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            job_id = self._store_job(_PAYLOAD)
            body = json.dumps({"job_id": job_id}).encode("utf-8")
            envelope = sign_request("real-secret", body)
            headers = {"Authorization": "Bearer real-secret", **envelope.headers()}
            with patch.object(
                discord_module, "_process_deferred_interaction", new_callable=AsyncMock
            ):
                first = self.client.post(
                    "/channels/discord/process", content=body, headers=headers
                )
                second = self.client.post(
                    "/channels/discord/process", content=body, headers=headers
                )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 401)

    def test_correct_secret_processes_the_stored_payload(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            job_id = self._store_job(_PAYLOAD)
            with patch.object(
                discord_module, "_process_deferred_interaction", new_callable=AsyncMock
            ) as fake_process:
                resp = self._signed_post("real-secret", {"job_id": job_id})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True})
        fake_process.assert_awaited_once_with(_PAYLOAD)

    def test_missing_required_field_in_stored_payload_rejected(self) -> None:
        with patch.object(discord_module, "settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            bad_payload = dict(_PAYLOAD)
            del bad_payload["text"]
            job_id = self._store_job(bad_payload)
            resp = self._signed_post("real-secret", {"job_id": job_id})
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
