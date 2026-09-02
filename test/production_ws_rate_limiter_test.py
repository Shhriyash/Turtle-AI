"""
Phase 6 — WebSocket per-user inbound rate limiter.

The sliding-window counter is the only thing standing between a hostile
cookie and Turtle's LLM bill, so it gets explicit assertions for both the
hourly and daily caps + cross-user isolation.
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from core.guardrails import WebSocketRateLimiter, WebSocketRateLimitExceeded


class WebSocketRateLimiterTests(unittest.TestCase):
    def test_under_limit_does_not_raise(self) -> None:
        limiter = WebSocketRateLimiter(per_hour=3, per_day=10)
        for _ in range(3):
            limiter.check_and_record("usr_alice")

    def test_exceeding_hourly_cap_raises_with_window(self) -> None:
        limiter = WebSocketRateLimiter(per_hour=2, per_day=100)
        limiter.check_and_record("usr_alice")
        limiter.check_and_record("usr_alice")
        with self.assertRaises(WebSocketRateLimitExceeded) as ctx:
            limiter.check_and_record("usr_alice")
        self.assertEqual(ctx.exception.window, "hour")
        self.assertEqual(ctx.exception.limit, 2)

    def test_exceeding_daily_cap_raises_with_day_window(self) -> None:
        limiter = WebSocketRateLimiter(per_hour=10**6, per_day=2)
        limiter.check_and_record("usr_alice")
        limiter.check_and_record("usr_alice")
        with self.assertRaises(WebSocketRateLimitExceeded) as ctx:
            limiter.check_and_record("usr_alice")
        self.assertEqual(ctx.exception.window, "day")

    def test_per_user_isolation(self) -> None:
        # Alice exhausts her bucket; Bob is unaffected.
        limiter = WebSocketRateLimiter(per_hour=1, per_day=100)
        limiter.check_and_record("usr_alice")
        with self.assertRaises(WebSocketRateLimitExceeded):
            limiter.check_and_record("usr_alice")
        limiter.check_and_record("usr_bob")  # must not raise

    def test_hourly_window_slides_forward(self) -> None:
        # Use a manipulable clock so we don't sleep for an hour.
        limiter = WebSocketRateLimiter(per_hour=1, per_day=100)
        base = 1_000_000.0
        with patch("core.guardrails.time.time", return_value=base):
            limiter.check_and_record("usr_alice")
            with self.assertRaises(WebSocketRateLimitExceeded):
                limiter.check_and_record("usr_alice")
        # Advance 61 minutes; the first hit falls out of the hourly window.
        with patch("core.guardrails.time.time", return_value=base + 3661):
            limiter.check_and_record("usr_alice")  # must succeed

    def test_empty_user_id_is_no_op(self) -> None:
        limiter = WebSocketRateLimiter(per_hour=1, per_day=1)
        # Anonymous frames (no user_id resolved) must not blow up the limiter.
        for _ in range(5):
            limiter.check_and_record("")

    def test_zero_limits_disable_enforcement(self) -> None:
        limiter = WebSocketRateLimiter(per_hour=0, per_day=0)
        for _ in range(1000):
            limiter.check_and_record("usr_alice")


if __name__ == "__main__":
    unittest.main()
