"""
test/internal_auth_test.py
------------------------------
Unit coverage for core/internal_auth.py (WP 1.B / ledger 1a.3, S-7.3): the
bearer-token constant-time check, the signed-request envelope (timestamp +
nonce + body HMAC, replay-checked via a Redis nonce claim), and the
payload-by-reference job store.

Uses fakeredis (matches test/redis_backends_test.py's existing pattern) —
no live Redis reachable in this environment.
"""
from __future__ import annotations

import json
import time
import unittest
from unittest.mock import patch

import fakeredis

from core.internal_auth import (
    SignatureError,
    check_bearer,
    require_secret,
    sign_request,
    store_job_payload,
    take_job_payload,
    verify_request,
)
from core.storage.cloud import CloudBackendUnavailable


class CheckBearerTest(unittest.TestCase):
    def test_correct_token_accepted(self) -> None:
        self.assertTrue(check_bearer("s3cret", "Bearer s3cret"))

    def test_wrong_token_rejected(self) -> None:
        self.assertFalse(check_bearer("s3cret", "Bearer wrong"))

    def test_missing_header_rejected(self) -> None:
        self.assertFalse(check_bearer("s3cret", None))

    def test_unset_secret_never_passes(self) -> None:
        # Even an empty Authorization header must not compare-equal to an
        # empty/unset expected secret.
        self.assertFalse(check_bearer("", ""))
        self.assertFalse(check_bearer(None, "Bearer anything"))

    def test_case_insensitive_bearer_prefix(self) -> None:
        self.assertTrue(check_bearer("s3cret", "bearer s3cret"))


class RequireSecretTest(unittest.TestCase):
    def test_unset_raises_value_error_naming_the_var(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            require_secret(None, "SOME_VAR")
        self.assertEqual(str(ctx.exception), "SOME_VAR")

    def test_set_returns_value(self) -> None:
        class _Fake:
            def get_secret_value(self):
                return "abc"

        self.assertEqual(require_secret(_Fake(), "SOME_VAR"), "abc")


class SignVerifyRoundTripTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.fake = fakeredis.FakeAsyncRedis(decode_responses=True)
        self.patcher = patch(
            "core.internal_auth.get_redis_client", new=self._get_fake_client
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    async def _get_fake_client(self):
        return self.fake

    async def test_valid_signature_is_accepted(self) -> None:
        body = b'{"job_id": "abc123"}'
        envelope = sign_request("s3cret", body)
        await verify_request(
            "s3cret", envelope.timestamp, envelope.nonce, envelope.signature, body
        )  # must not raise

    async def test_forged_signature_rejected(self) -> None:
        body = b'{"job_id": "abc123"}'
        envelope = sign_request("s3cret", body)
        with self.assertRaises(SignatureError):
            await verify_request(
                "s3cret", envelope.timestamp, envelope.nonce, "deadbeef" * 8, body
            )

    async def test_wrong_secret_rejected(self) -> None:
        body = b'{"job_id": "abc123"}'
        envelope = sign_request("s3cret", body)
        with self.assertRaises(SignatureError):
            await verify_request(
                "different-secret", envelope.timestamp, envelope.nonce, envelope.signature, body
            )

    async def test_body_swapped_after_signing_rejected(self) -> None:
        original_body = b'{"job_id": "abc123"}'
        swapped_body = b'{"job_id": "victim-job"}'
        envelope = sign_request("s3cret", original_body)
        with self.assertRaises(SignatureError):
            await verify_request(
                "s3cret", envelope.timestamp, envelope.nonce, envelope.signature, swapped_body
            )

    async def test_replayed_nonce_rejected(self) -> None:
        body = b'{"job_id": "abc123"}'
        envelope = sign_request("s3cret", body)
        await verify_request(
            "s3cret", envelope.timestamp, envelope.nonce, envelope.signature, body
        )
        with self.assertRaises(SignatureError):
            await verify_request(
                "s3cret", envelope.timestamp, envelope.nonce, envelope.signature, body
            )

    async def test_stale_timestamp_rejected(self) -> None:
        body = b"{}"
        stale_ts = str(int(time.time()) - 301)
        nonce = "n1"
        from core.internal_auth import _hmac_hex

        sig = _hmac_hex("s3cret", stale_ts, nonce, body)
        with self.assertRaises(SignatureError):
            await verify_request("s3cret", stale_ts, nonce, sig, body)

    async def test_future_timestamp_rejected(self) -> None:
        body = b"{}"
        future_ts = str(int(time.time()) + 301)
        nonce = "n2"
        from core.internal_auth import _hmac_hex

        sig = _hmac_hex("s3cret", future_ts, nonce, body)
        with self.assertRaises(SignatureError):
            await verify_request("s3cret", future_ts, nonce, sig, body)

    async def test_missing_envelope_fields_rejected(self) -> None:
        with self.assertRaises(SignatureError):
            await verify_request("s3cret", None, None, None, b"{}")

    async def test_malformed_timestamp_rejected(self) -> None:
        with self.assertRaises(SignatureError):
            await verify_request("s3cret", "not-a-number", "n3", "whatever", b"{}")

    async def test_redis_unavailable_fails_closed(self) -> None:
        body = b"{}"
        envelope = sign_request("s3cret", body)

        async def _raise_unavailable():
            raise CloudBackendUnavailable("no REDIS_URL")

        with patch("core.internal_auth.get_redis_client", new=_raise_unavailable):
            with self.assertRaises(SignatureError):
                await verify_request(
                    "s3cret", envelope.timestamp, envelope.nonce, envelope.signature, body
                )


class JobPayloadStoreTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.fake = fakeredis.FakeAsyncRedis(decode_responses=True)
        self.patcher = patch(
            "core.internal_auth.get_redis_client", new=self._get_fake_client
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    async def _get_fake_client(self):
        return self.fake

    async def test_store_then_take_round_trips_the_payload(self) -> None:
        payload = {"user_id": "usr_a", "topic_name": "identity", "lines": ["- x"]}
        job_id = await store_job_payload(payload)
        self.assertTrue(job_id)
        got = await take_job_payload(job_id)
        self.assertEqual(got, payload)

    async def test_take_deletes_so_a_replay_finds_nothing(self) -> None:
        job_id = await store_job_payload({"a": 1})
        first = await take_job_payload(job_id)
        second = await take_job_payload(job_id)
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    async def test_unknown_job_id_returns_none(self) -> None:
        got = await take_job_payload("does-not-exist")
        self.assertIsNone(got)

    async def test_stored_key_uses_turtle_job_prefix(self) -> None:
        job_id = await store_job_payload({"a": 1})
        raw = await self.fake.get(f"turtle:job:{job_id}")
        self.assertEqual(json.loads(raw), {"a": 1})


if __name__ == "__main__":
    unittest.main()
