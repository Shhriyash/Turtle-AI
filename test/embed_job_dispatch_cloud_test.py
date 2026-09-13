"""
test/embed_job_dispatch_cloud_test.py
-----------------------------------------
Unit coverage for core/worker.py::dispatch_embed_personal_memory_job's
cloud-mode self-invoke path and apps/cron_tick_routes.py's
POST /internal/embed-personal-memory endpoint (fix for the last open issue
from the post-migration audit: write_topic()'s embed-job enqueue was a bare
detached asyncio.create_task carrying the same unconfirmed-survival risk
already fixed for Discord's deferred processing).
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.worker as worker
from apps.cron_tick_routes import router


class DispatchEmbedJobLocalModeTest(unittest.IsolatedAsyncioTestCase):
    async def test_local_mode_enqueues_via_queue_service(self) -> None:
        with patch.object(worker, "settings") as fake_settings:
            fake_settings.is_cloud = False
            with patch.object(worker.queue_service, "enqueue", new_callable=AsyncMock) as fake_enqueue:
                worker.dispatch_embed_personal_memory_job("usr_a", "identity", ["- Name: Alice"])
                await asyncio.sleep(0)  # let the created task run

        fake_enqueue.assert_awaited_once_with(
            "embed_personal_memory", user_id="usr_a", topic_name="identity", lines=["- Name: Alice"]
        )

    def test_no_running_loop_is_a_silent_noop(self) -> None:
        # Called from a plain sync context (no event loop) — must not raise.
        with patch.object(worker, "settings") as fake_settings:
            fake_settings.is_cloud = False
            worker.dispatch_embed_personal_memory_job("usr_a", "identity", ["- x"])


class DispatchEmbedJobCloudModeTest(unittest.IsolatedAsyncioTestCase):
    async def test_cloud_mode_self_invokes_and_sends_full_request(self) -> None:
        with patch.object(worker, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.cron_shared_secret.get_secret_value.return_value = "shared-secret"
            fake_settings.public_base_url = "https://turtle.example.com"

            fake_client = AsyncMock()
            fake_client.post = AsyncMock(side_effect=httpx.ReadTimeout("expected"))
            fake_client_cm = AsyncMock()
            fake_client_cm.__aenter__.return_value = fake_client
            fake_client_cm.__aexit__.return_value = False

            with patch("httpx.AsyncClient", return_value=fake_client_cm):
                worker.dispatch_embed_personal_memory_job("usr_a", "identity", ["- Name: Alice"])
                await asyncio.sleep(0)
                await asyncio.sleep(0)

        fake_client.post.assert_awaited_once()
        call = fake_client.post.await_args
        self.assertEqual(call.args[0], "https://turtle.example.com/internal/embed-personal-memory")
        self.assertEqual(
            call.kwargs["json"], {"user_id": "usr_a", "topic_name": "identity", "lines": ["- Name: Alice"]}
        )
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer shared-secret")

    async def test_cloud_mode_without_secret_runs_job_in_process(self) -> None:
        with patch.object(worker, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.cron_shared_secret = None
            with patch.dict(
                worker._REGISTRY, {"embed_personal_memory": AsyncMock()}, clear=False
            ) as _:
                fake_job = worker._REGISTRY["embed_personal_memory"]
                worker.dispatch_embed_personal_memory_job("usr_a", "identity", ["- x"])
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                fake_job.assert_awaited_once_with(user_id="usr_a", topic_name="identity", lines=["- x"])

    async def test_cloud_mode_network_error_falls_back_to_in_process(self) -> None:
        with patch.object(worker, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.cron_shared_secret.get_secret_value.return_value = "shared-secret"
            fake_settings.public_base_url = "https://turtle.example.com"

            fake_client = AsyncMock()
            fake_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            fake_client_cm = AsyncMock()
            fake_client_cm.__aenter__.return_value = fake_client
            fake_client_cm.__aexit__.return_value = False

            with patch("httpx.AsyncClient", return_value=fake_client_cm), patch.dict(
                worker._REGISTRY, {"embed_personal_memory": AsyncMock()}, clear=False
            ):
                fake_job = worker._REGISTRY["embed_personal_memory"]
                worker.dispatch_embed_personal_memory_job("usr_a", "identity", ["- x"])
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                fake_job.assert_awaited_once_with(user_id="usr_a", topic_name="identity", lines=["- x"])


class EmbedPersonalMemoryEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_missing_auth_rejected(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "real-secret"
            resp = self.client.post(
                "/internal/embed-personal-memory",
                json={"user_id": "usr_a", "topic_name": "identity", "lines": ["- x"]},
            )
        self.assertEqual(resp.status_code, 401)

    def test_no_secret_configured_returns_503(self) -> None:
        # Matches _check_auth's documented posture (same as admin_routes.py):
        # an unset secret fails loud with 503 rather than leaving the
        # endpoint silently open.
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_shared_secret = None
            resp = self.client.post(
                "/internal/embed-personal-memory",
                json={"user_id": "usr_a", "topic_name": "identity", "lines": ["- x"]},
                headers={"Authorization": "Bearer anything"},
            )
        self.assertEqual(resp.status_code, 503)

    def test_correct_auth_runs_the_job(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "real-secret"
            with patch("core.background_tasks.embed_personal_memory", new_callable=AsyncMock) as fake_job:
                resp = self.client.post(
                    "/internal/embed-personal-memory",
                    json={"user_id": "usr_a", "topic_name": "identity", "lines": ["- Name: Alice"]},
                    headers={"Authorization": "Bearer real-secret"},
                )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True})
        fake_job.assert_awaited_once_with(user_id="usr_a", topic_name="identity", lines=["- Name: Alice"])

    def test_missing_required_field_rejected(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "real-secret"
            resp = self.client.post(
                "/internal/embed-personal-memory",
                json={"user_id": "usr_a"},
                headers={"Authorization": "Bearer real-secret"},
            )
        self.assertEqual(resp.status_code, 400)

    def test_job_failure_returns_500(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.cron_shared_secret.get_secret_value.return_value = "real-secret"
            with patch(
                "core.background_tasks.embed_personal_memory",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ):
                resp = self.client.post(
                    "/internal/embed-personal-memory",
                    json={"user_id": "usr_a", "topic_name": "identity", "lines": ["- x"]},
                    headers={"Authorization": "Bearer real-secret"},
                )
        self.assertEqual(resp.status_code, 500)


if __name__ == "__main__":
    unittest.main()
