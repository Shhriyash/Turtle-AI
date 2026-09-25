"""
test/embed_job_dispatch_cloud_test.py
-----------------------------------------
Unit coverage for core/worker.py::dispatch_embed_personal_memory_job's
cloud-mode self-invoke path and apps/cron_tick_routes.py's
POST /internal/embed-personal-memory endpoint.

WP 1.B (ledger 1a.3 / S-7.3): the self-invoke now authenticates with
INTERNAL_JOB_SECRET (not the retired CRON_SHARED_SECRET), signs the request
(timestamp + nonce + body HMAC, core/internal_auth.py), and sends only an
opaque job id — the real payload (user_id, topic_name, lines) is stashed in
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

import core.worker as worker
from apps.cron_tick_routes import router
from core.internal_auth import sign_request, take_job_payload


def _fake_redis_patch(fake_client):
    """Patch core.internal_auth's Redis accessor with a shared fakeredis
    instance — both the caller (store_job_payload) and callee
    (take_job_payload / the nonce claim) go through this one module-level
    accessor, so a single fake client is enough to exercise the whole
    round trip in-process.
    """
    return patch(
        "core.internal_auth.get_redis_client",
        new=AsyncMock(return_value=fake_client),
    )


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
    def setUp(self) -> None:
        self.fake_redis = fakeredis.FakeAsyncRedis(decode_responses=True)
        self.redis_patcher = _fake_redis_patch(self.fake_redis)
        self.redis_patcher.start()
        self.addCleanup(self.redis_patcher.stop)

    async def test_cloud_mode_self_invokes_and_sends_only_a_job_id(self) -> None:
        with patch.object(worker, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.internal_job_secret.get_secret_value.return_value = "job-secret"
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

        # The impersonation surface is closed: the wire body carries only an
        # opaque job id, never user_id/topic_name/lines directly.
        sent_body = json.loads(call.kwargs["content"])
        self.assertEqual(set(sent_body), {"job_id"})
        self.assertNotIn("user_id", sent_body)

        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer job-secret")
        self.assertIn("X-Turtle-Timestamp", call.kwargs["headers"])
        self.assertIn("X-Turtle-Nonce", call.kwargs["headers"])
        self.assertIn("X-Turtle-Signature", call.kwargs["headers"])

        # The real payload landed in the job store under that id, intact.
        stored = await take_job_payload(sent_body["job_id"])
        self.assertEqual(
            stored, {"user_id": "usr_a", "topic_name": "identity", "lines": ["- Name: Alice"]}
        )

    async def test_cloud_mode_without_secret_runs_job_in_process(self) -> None:
        with patch.object(worker, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.internal_job_secret = None
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
            fake_settings.internal_job_secret.get_secret_value.return_value = "job-secret"
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

    async def test_cloud_mode_redis_unavailable_falls_back_to_in_process(self) -> None:
        # Payload-by-reference needs Redis to store the payload; if it's
        # down, degrade the same way as "no secret configured" rather than
        # dropping the embed. Overrides setUp's fakeredis patch for the
        # duration of this `with` block only; addCleanup still tears down
        # the outer fakeredis patcher normally afterwards.
        from core.storage.cloud import CloudBackendUnavailable

        with patch.object(worker, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.internal_job_secret.get_secret_value.return_value = "job-secret"
            with patch(
                "core.internal_auth.get_redis_client",
                new=AsyncMock(side_effect=CloudBackendUnavailable("no REDIS_URL")),
            ), patch.dict(
                worker._REGISTRY, {"embed_personal_memory": AsyncMock()}, clear=False
            ):
                fake_job = worker._REGISTRY["embed_personal_memory"]
                worker.dispatch_embed_personal_memory_job("usr_a", "identity", ["- x"])
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                fake_job.assert_awaited_once_with(user_id="usr_a", topic_name="identity", lines=["- x"])

    async def test_cloud_mode_redis_command_fails_falls_back_to_in_process_and_runs(
        self,
    ) -> None:
        """Coordinator fail-fix: unset-URL is not the only failure mode. A
        client object that EXISTS but whose command raises the redis-py
        driver's own exception (here, a real redis.exceptions.ConnectionError
        — not a mocked CloudBackendUnavailable) must still degrade to running
        the job in-process, not silently drop it.
        """
        import redis.exceptions

        fake_redis_client = AsyncMock()
        fake_redis_client.set = AsyncMock(
            side_effect=redis.exceptions.ConnectionError("connection refused")
        )

        with patch.object(worker, "settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.internal_job_secret.get_secret_value.return_value = "job-secret"
            with patch(
                "core.internal_auth.get_redis_client",
                new=AsyncMock(return_value=fake_redis_client),
            ), patch.dict(
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
        self.fake_redis = fakeredis.FakeAsyncRedis(decode_responses=True)
        self.redis_patcher = _fake_redis_patch(self.fake_redis)
        self.redis_patcher.start()
        self.addCleanup(self.redis_patcher.stop)

    def _signed_post(self, secret: str, body_dict: dict, *, auth: str | None = None):
        body = json.dumps(body_dict).encode("utf-8")
        envelope = sign_request(secret, body)
        headers = {"Authorization": auth if auth is not None else f"Bearer {secret}"}
        headers.update(envelope.headers())
        return self.client.post(
            "/internal/embed-personal-memory", content=body, headers=headers
        )

    def _store_job(self, payload: dict) -> str:
        from core.internal_auth import store_job_payload

        return asyncio.run(store_job_payload(payload))

    def test_missing_auth_rejected(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            resp = self.client.post(
                "/internal/embed-personal-memory",
                json={"job_id": "whatever"},
            )
        self.assertEqual(resp.status_code, 401)

    def test_no_secret_configured_returns_503(self) -> None:
        # Matches the documented posture (same as admin_routes.py): an unset
        # secret fails loud with 503 rather than leaving the endpoint open.
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.internal_job_secret = None
            resp = self.client.post(
                "/internal/embed-personal-memory",
                json={"job_id": "whatever"},
                headers={"Authorization": "Bearer anything"},
            )
        self.assertEqual(resp.status_code, 503)

    def test_cron_tick_secret_is_rejected_here(self) -> None:
        # The two secrets are NOT interchangeable.
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "job-secret"
            resp = self.client.post(
                "/internal/embed-personal-memory",
                json={"job_id": "whatever"},
                headers={"Authorization": "Bearer cron-tick-secret-value"},
            )
        self.assertEqual(resp.status_code, 401)

    def test_forged_signature_rejected(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            body = json.dumps({"job_id": "whatever"}).encode("utf-8")
            envelope = sign_request("real-secret", body)
            resp = self.client.post(
                "/internal/embed-personal-memory",
                content=body,
                headers={
                    "Authorization": "Bearer real-secret",
                    "X-Turtle-Timestamp": envelope.timestamp,
                    "X-Turtle-Nonce": envelope.nonce,
                    "X-Turtle-Signature": "0" * 64,
                },
            )
        self.assertEqual(resp.status_code, 401)

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
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            with patch(
                "core.internal_auth.get_redis_client",
                new=AsyncMock(return_value=fake_redis_client),
            ):
                resp = self._signed_post("real-secret", {"job_id": "whatever"})
        self.assertEqual(resp.status_code, 401)

    def test_replayed_request_rejected(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            job_id = self._store_job(
                {"user_id": "usr_a", "topic_name": "identity", "lines": ["- x"]}
            )
            body = json.dumps({"job_id": job_id}).encode("utf-8")
            envelope = sign_request("real-secret", body)
            headers = {"Authorization": "Bearer real-secret", **envelope.headers()}
            with patch("core.background_tasks.embed_personal_memory", new_callable=AsyncMock):
                first = self.client.post(
                    "/internal/embed-personal-memory", content=body, headers=headers
                )
                second = self.client.post(
                    "/internal/embed-personal-memory", content=body, headers=headers
                )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 401)

    def test_correct_auth_runs_the_job_from_the_stored_payload(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            job_id = self._store_job(
                {"user_id": "usr_a", "topic_name": "identity", "lines": ["- Name: Alice"]}
            )
            with patch("core.background_tasks.embed_personal_memory", new_callable=AsyncMock) as fake_job:
                resp = self._signed_post("real-secret", {"job_id": job_id})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True})
        fake_job.assert_awaited_once_with(user_id="usr_a", topic_name="identity", lines=["- Name: Alice"])

    def test_unknown_job_id_rejected(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            resp = self._signed_post("real-secret", {"job_id": "does-not-exist"})
        self.assertEqual(resp.status_code, 401)

    def test_missing_job_id_rejected(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            resp = self._signed_post("real-secret", {})
        self.assertEqual(resp.status_code, 400)

    def test_body_user_id_cannot_impersonate_without_a_valid_job_id(self) -> None:
        """WP 1.B / S-7.3 acceptance criterion: even a caller holding the
        correct secret (and able to sign correctly) cannot make this
        endpoint write for an arbitrary user_id by putting it directly in
        the body — the endpoint no longer reads user_id from the request at
        all, only from a payload the CALLER previously stored server-side.
        """
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            with patch("core.background_tasks.embed_personal_memory", new_callable=AsyncMock) as fake_job:
                resp = self._signed_post(
                    "real-secret",
                    {"user_id": "usr_victim", "topic_name": "hacked", "lines": ["- pwned"]},
                )
        self.assertEqual(resp.status_code, 400)  # no job_id -> rejected before any lookup
        fake_job.assert_not_called()

    def test_missing_required_field_in_stored_payload_rejected(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            job_id = self._store_job({"user_id": "usr_a"})  # missing topic_name/lines
            resp = self._signed_post("real-secret", {"job_id": job_id})
        self.assertEqual(resp.status_code, 400)

    def test_job_failure_returns_500(self) -> None:
        with patch("apps.cron_tick_routes.settings") as fake_settings:
            fake_settings.internal_job_secret.get_secret_value.return_value = "real-secret"
            job_id = self._store_job(
                {"user_id": "usr_a", "topic_name": "identity", "lines": ["- x"]}
            )
            with patch(
                "core.background_tasks.embed_personal_memory",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ):
                resp = self._signed_post("real-secret", {"job_id": job_id})
        self.assertEqual(resp.status_code, 500)


if __name__ == "__main__":
    unittest.main()
