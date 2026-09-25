"""
test/redis_backends_test.py
-----------------------------
Unit coverage for the Redis-backed cloud replacements added in the Vercel
migration's Phase 1b (core/storage/cloud/redis_backends.py):

- RedisWebSocketRateLimiter -> core/guardrails.py::WebSocketRateLimiter
- RedisChannelGateBuffer    -> core/channel_gate.py::ChannelGateBuffer
- redis_is_duplicate_invocation / redis_record_invocation
      -> tools/idempotency.py's SQLite-backed functions

Uses fakeredis (a real in-memory Redis protocol implementation, not a
hand-rolled mock) so sorted-set / TTL semantics are exercised faithfully. No
live Redis is reachable in this environment; an Upstash integration pass is
still needed once one is provisioned (same caveat as test/cloud_storage_test.py
for Postgres).
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import fakeredis

from core.guardrails import WebSocketRateLimitExceeded
from core.storage.cloud.redis_backends import (
    RedisChannelGateBuffer,
    RedisWebSocketRateLimiter,
    redis_is_duplicate_invocation,
    redis_record_invocation,
)


def _patch_redis_client(fake_client):
    return patch(
        "core.storage.cloud.redis_backends.get_redis_sync_client", return_value=fake_client
    )


class RedisWebSocketRateLimiterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = fakeredis.FakeStrictRedis(decode_responses=True)
        self.patcher = _patch_redis_client(self.fake)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_allows_under_limit(self) -> None:
        limiter = RedisWebSocketRateLimiter(per_hour=5, per_day=100)
        for _ in range(5):
            limiter.check_and_record("usr_a")  # must not raise

    def test_raises_when_hourly_limit_reached(self) -> None:
        limiter = RedisWebSocketRateLimiter(per_hour=2, per_day=100)
        limiter.check_and_record("usr_a")
        limiter.check_and_record("usr_a")
        with self.assertRaises(WebSocketRateLimitExceeded) as ctx:
            limiter.check_and_record("usr_a")
        self.assertEqual(ctx.exception.window, "hour")

    def test_raises_when_daily_limit_reached(self) -> None:
        limiter = RedisWebSocketRateLimiter(per_hour=1000, per_day=2)
        limiter.check_and_record("usr_a")
        limiter.check_and_record("usr_a")
        with self.assertRaises(WebSocketRateLimitExceeded) as ctx:
            limiter.check_and_record("usr_a")
        self.assertEqual(ctx.exception.window, "day")

    def test_different_users_have_independent_counters(self) -> None:
        limiter = RedisWebSocketRateLimiter(per_hour=1, per_day=100)
        limiter.check_and_record("usr_a")
        limiter.check_and_record("usr_b")  # must not raise — separate key

    def test_zero_disables_the_limit(self) -> None:
        limiter = RedisWebSocketRateLimiter(per_hour=0, per_day=0)
        for _ in range(10):
            limiter.check_and_record("usr_a")

    def test_empty_user_id_is_a_noop(self) -> None:
        limiter = RedisWebSocketRateLimiter(per_hour=1, per_day=1)
        limiter.check_and_record("")
        limiter.check_and_record("")  # would raise if it were tracked


class RedisChannelGateBufferTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = fakeredis.FakeStrictRedis(decode_responses=True)
        self.patcher = _patch_redis_client(self.fake)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.buffer = RedisChannelGateBuffer(ttl_seconds=300)

    def test_no_outstanding_prompt_returns_none(self) -> None:
        self.assertIsNone(self.buffer.try_consume_answer(("u1", "telegram"), "yes"))

    def test_answers_within_ttl(self) -> None:
        key = ("u1", "telegram")
        self.buffer.note_prompt(key, ("ev1", "ev2"))
        result = self.buffer.try_consume_answer(key, "yes")
        self.assertEqual(result, (True, ("ev1", "ev2")))

    def test_reject_answer_returns_false_verdict(self) -> None:
        key = ("u1", "telegram")
        self.buffer.note_prompt(key, ("ev1",))
        result = self.buffer.try_consume_answer(key, "no")
        self.assertEqual(result, (False, ("ev1",)))

    def test_non_answer_leaves_prompt_outstanding(self) -> None:
        key = ("u1", "telegram")
        self.buffer.note_prompt(key, ("ev1",))
        self.assertIsNone(self.buffer.try_consume_answer(key, "what time is it"))
        # Still outstanding — a real answer afterward succeeds.
        result = self.buffer.try_consume_answer(key, "yes")
        self.assertEqual(result, (True, ("ev1",)))

    def test_answering_clears_the_prompt(self) -> None:
        key = ("u1", "telegram")
        self.buffer.note_prompt(key, ("ev1",))
        first = self.buffer.try_consume_answer(key, "yes")
        self.assertEqual(first, (True, ("ev1",)))
        second = self.buffer.try_consume_answer(key, "yes")
        self.assertIsNone(second)

    def test_different_channels_are_independent(self) -> None:
        self.buffer.note_prompt(("u1", "telegram"), ("ev1",))
        self.assertIsNone(self.buffer.try_consume_answer(("u1", "discord"), "yes"))
        self.assertEqual(
            self.buffer.try_consume_answer(("u1", "telegram"), "yes"), (True, ("ev1",))
        )

    def test_note_prompt_ignores_empty_event_ids(self) -> None:
        key = ("u1", "telegram")
        self.buffer.note_prompt(key, ())
        self.assertIsNone(self.buffer.try_consume_answer(key, "yes"))

    def test_has_outstanding(self) -> None:
        key = ("u1", "telegram")
        self.assertFalse(self.buffer.has_outstanding(key))
        self.buffer.note_prompt(key, ("ev1",))
        self.assertTrue(self.buffer.has_outstanding(key))

    def test_clear_removes_outstanding(self) -> None:
        key = ("u1", "telegram")
        self.buffer.note_prompt(key, ("ev1",))
        self.buffer.clear(key)
        self.assertIsNone(self.buffer.try_consume_answer(key, "yes"))

    def test_expiry_via_redis_ttl(self) -> None:
        short_buffer = RedisChannelGateBuffer(ttl_seconds=1)
        key = ("u1", "telegram")
        short_buffer.note_prompt(key, ("ev1",))
        # fakeredis honors real wall-clock TTL; advance past it.
        self.fake.pexpire(short_buffer._redis_key(key), 1)
        time.sleep(0.05)
        self.assertIsNone(short_buffer.try_consume_answer(key, "yes"))


class RedisIdempotencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = fakeredis.FakeStrictRedis(decode_responses=True)
        self.patcher = patch(
            "core.storage.cloud.redis_backends.get_redis_sync_client", return_value=self.fake
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_new_key_is_not_a_duplicate(self) -> None:
        self.assertIsNone(redis_is_duplicate_invocation("k1"))

    def test_only_successful_sends_are_cached(self) -> None:
        redis_record_invocation("k1", "Failed to send email: boom")
        self.assertIsNone(redis_is_duplicate_invocation("k1"))

        redis_record_invocation("k2", "Email sent successfully! message id 123")
        self.assertEqual(
            redis_is_duplicate_invocation("k2"), "Email sent successfully! message id 123"
        )

    def test_distinct_keys_are_independent(self) -> None:
        redis_record_invocation("k1", "Email sent successfully! a")
        self.assertIsNone(redis_is_duplicate_invocation("k2"))

    def test_same_key_twice_before_finalize_is_still_in_flight(self) -> None:
        """Reservation semantics: a second call before record_invocation must
        NOT get a fresh None (that would let a concurrent send fire) and must
        not silently claim a result that doesn't exist yet."""
        from core.storage.cloud.redis_backends import _PENDING_MESSAGE

        self.assertIsNone(redis_is_duplicate_invocation("k1"))
        self.assertEqual(redis_is_duplicate_invocation("k1"), _PENDING_MESSAGE)

        redis_record_invocation("k1", "Email sent successfully! done")
        self.assertEqual(
            redis_is_duplicate_invocation("k1"), "Email sent successfully! done"
        )

    def test_failed_send_deletes_reservation_allows_retry(self) -> None:
        self.assertIsNone(redis_is_duplicate_invocation("k1"))
        redis_record_invocation("k1", "Failed to send email: smtp boom")
        # Retry is a fresh reservation, not a duplicate.
        self.assertIsNone(redis_is_duplicate_invocation("k1"))

    def test_two_concurrent_reservations_exactly_one_proceeds(self) -> None:
        """Simulated concurrency: two callers racing on the same key via the
        same atomic SET NX — exactly one must get the reservation."""
        from core.storage.cloud.redis_backends import _PENDING_MESSAGE

        first = redis_is_duplicate_invocation("k1")
        second = redis_is_duplicate_invocation("k1")
        results = [first, second]
        self.assertEqual(results.count(None), 1)
        self.assertEqual(results.count(_PENDING_MESSAGE), 1)

    def test_redis_unavailable_raises_and_is_fail_closed(self) -> None:
        """Fail-closed flip: the previous behaviour caught the exception and
        returned None (fail open — treat as new, proceed to send). Now it
        must raise so the caller refuses the send instead."""
        from core.storage.cloud.redis_backends import IdempotencyReservationError

        with patch(
            "core.storage.cloud.redis_backends.get_redis_sync_client",
            side_effect=RuntimeError("redis unreachable"),
        ):
            with self.assertRaises(IdempotencyReservationError):
                redis_is_duplicate_invocation("k1")

    def test_success_kwarg_caches_non_email_shaped_result(self) -> None:
        """WP1.E1: redis_record_invocation's `success` kwarg (added so
        calendar_confirm's result — never "Email sent successfully" — can
        still be cached) must override the default string-sniff."""
        redis_record_invocation("k1", "Event created: Board Sync", success=True)
        self.assertEqual(redis_is_duplicate_invocation("k1"), "Event created: Board Sync")

    def test_success_kwarg_false_releases_reservation(self) -> None:
        redis_record_invocation("k1", "some non-email result text", success=False)
        self.assertIsNone(redis_is_duplicate_invocation("k1"))

    def test_success_kwarg_omitted_keeps_email_sniff_behaviour(self) -> None:
        """Back-compat: existing email call sites don't pass `success`, so
        the string-sniff must still apply exactly as before."""
        redis_record_invocation("k1", "Failed to send email: boom")
        self.assertIsNone(redis_is_duplicate_invocation("k1"))
        redis_record_invocation("k2", "Email sent successfully! message id 123")
        self.assertEqual(
            redis_is_duplicate_invocation("k2"), "Email sent successfully! message id 123"
        )

    def test_cross_tenant_keys_built_via_key_builder_do_not_collide(self) -> None:
        """Ledger acceptance criterion, exercised through the real key
        builder: two different user_ids sending the byte-identical email
        within 60s both get a fresh reservation (both send)."""
        from tools.idempotency import build_email_idempotency_key

        key_a = build_email_idempotency_key(
            "usr_a", recipients=["shared@example.com"], subject="Hi", body="Same body", cc=[], bcc=[],
        )
        key_b = build_email_idempotency_key(
            "usr_b", recipients=["shared@example.com"], subject="Hi", body="Same body", cc=[], bcc=[],
        )
        self.assertNotEqual(key_a, key_b)
        self.assertIsNone(redis_is_duplicate_invocation(key_a))
        self.assertIsNone(redis_is_duplicate_invocation(key_b))


if __name__ == "__main__":
    unittest.main()
