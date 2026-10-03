"""
Real-Redis roundtrip for core/health_tracker.py's shared provider cooldowns
(WP5-A2 / ledger 5.5): SET turtle:cooldown:{bucket} 1 EX <s>, MGET once per
cascade, TTL expiry, DEL on success. Runs against a real redis:7 in the
cloud-tests CI job; self-skips without REDIS_URL/DATABASE_URL (see conftest).
"""
from __future__ import annotations

import time
import uuid

import pytest

from test.cloud_integration.conftest import run_async

pytestmark = pytest.mark.cloud


def _make_agent():
    # Unique model name per test so concurrent CI runs / reruns never collide
    # and the key can be cleaned up exactly.
    class _Probe:
        model_name = f"probe-{uuid.uuid4().hex[:10]}"

    class _Agent:
        model = _Probe()

    return _Agent()


def _http_error(status: int):
    from pydantic_ai.exceptions import ModelHTTPError

    try:
        return ModelHTTPError(status_code=status, model_name="m", body="requires more credits")
    except TypeError:
        return ModelHTTPError(status, "m", "requires more credits")


@pytest.fixture(autouse=True)
def _cloud_mode_and_clean_state(monkeypatch):
    from core import health_tracker
    from core.config import settings

    monkeypatch.setattr(settings, "deploy_mode", "cloud")
    with health_tracker._lock:
        health_tracker._cooldown_until.clear()
        health_tracker._shared_until.clear()
    health_tracker._redis_down_until = 0.0
    yield
    with health_tracker._lock:
        health_tracker._cooldown_until.clear()
        health_tracker._shared_until.clear()


async def _drain(health_tracker) -> None:
    import asyncio

    if health_tracker._pending:
        await asyncio.gather(*list(health_tracker._pending))


def test_402_roundtrips_through_real_redis_to_a_cold_instance() -> None:
    from core import health_tracker
    from core.storage.cloud import get_redis_client

    agent = _make_agent()
    key = health_tracker._SHARED_PREFIX + health_tracker._bucket_id(agent)

    async def scenario():
        client = await get_redis_client()
        try:
            # Warm instance: 402 -> mirrored.
            health_tracker.mark_failure(agent, _http_error(402))
            await _drain(health_tracker)
            assert abs(float(await client.get(key)) - (time.time() + 300)) < 30
            ttl = await client.ttl(key)
            assert 0 < ttl <= 300

            # Cold instance: empty process-local state, one MGET.
            with health_tracker._lock:
                health_tracker._cooldown_until.clear()
                health_tracker._shared_until.clear()
            assert health_tracker.is_cooling(agent) is False
            await health_tracker.refresh_shared([agent])
            assert health_tracker.is_cooling(agent) is True

            # Success clears the shared key.
            health_tracker.mark_success(agent)
            await _drain(health_tracker)
            assert await client.exists(key) == 0
        finally:
            await client.delete(key)

    run_async(scenario())


def test_shared_cooldown_expires_with_the_redis_ttl() -> None:
    from core import health_tracker
    from core.storage.cloud import get_redis_client

    agent = _make_agent()
    bid = health_tracker._bucket_id(agent)
    key = health_tracker._SHARED_PREFIX + bid

    async def scenario():
        client = await get_redis_client()
        try:
            health_tracker._mirror_set(bid, 1)  # EX 1
            await _drain(health_tracker)
            await health_tracker.refresh_shared([agent])
            assert health_tracker.is_cooling(agent) is True
            time.sleep(1.3)
            await health_tracker.refresh_shared([agent])
            assert health_tracker.is_cooling(agent) is False
        finally:
            await client.delete(key)

    run_async(scenario())


def test_rung_scope_429_is_not_written_to_real_redis() -> None:
    from core import health_tracker
    from core.storage.cloud import get_redis_client

    agent = _make_agent()
    key = health_tracker._SHARED_PREFIX + health_tracker._bucket_id(agent)

    async def scenario():
        client = await get_redis_client()
        health_tracker.mark_failure(agent, _http_error(429))
        await _drain(health_tracker)
        assert await client.exists(key) == 0

    run_async(scenario())
