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


if __name__ == "__main__":
    unittest.main()
