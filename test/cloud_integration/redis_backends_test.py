"""
Real-Redis roundtrip test for core/storage/cloud/redis_backends.py: the
rate limiter, channel-gate buffer, and idempotency helpers, all against a
real Redis (redis:7) rather than fakeredis.
"""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.cloud


def test_redis_rate_limiter_gate_buffer_and_idempotency_roundtrip() -> None:
    from core.guardrails import WebSocketRateLimitExceeded
    from core.storage.cloud.redis_backends import (
        RedisChannelGateBuffer,
        RedisWebSocketRateLimiter,
        redis_is_duplicate_invocation,
        redis_record_invocation,
    )
    from core.storage.cloud import get_redis_sync_client

    user_id = f"usr_{uuid.uuid4().hex[:12]}"

    # --- Rate limiter --------------------------------------------------
    limiter = RedisWebSocketRateLimiter(per_hour=2, per_day=10)
    limiter.check_and_record(user_id)
    limiter.check_and_record(user_id)
    with pytest.raises(WebSocketRateLimitExceeded):
        limiter.check_and_record(user_id)

    # --- Channel gate buffer --------------------------------------------
    gate = RedisChannelGateBuffer(ttl_seconds=30)
    key = (user_id, "web_email")
    assert gate.has_outstanding(key) is False

    gate.note_prompt(key, ("evt_a", "evt_b"))
    assert gate.has_outstanding(key) is True

    result = gate.try_consume_answer(key, "yes")
    assert result is not None
    verdict, event_ids = result
    assert verdict is True
    assert event_ids == ("evt_a", "evt_b")
    assert gate.has_outstanding(key) is False

    # --- Idempotency ------------------------------------------------------
    idem_key = f"idem_{uuid.uuid4().hex[:12]}"
    assert redis_is_duplicate_invocation(idem_key) is None
    redis_record_invocation(idem_key, "Email sent successfully to a@example.invalid")
    assert redis_is_duplicate_invocation(idem_key) == "Email sent successfully to a@example.invalid"

    # Cleanup.
    client = get_redis_sync_client()
    client.delete(f"turtle:ws_rate:{user_id}")
    client.delete(f"turtle:idem:{idem_key}")
