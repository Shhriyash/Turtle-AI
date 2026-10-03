"""
WP5-A2 (ledger 5.5) -- provider cooldowns shared across instances via Redis.

Mechanism under test: core/health_tracker._cooldown_until is a process-local
dict, so a freshly started instance does not know a provider is out of credit
(402) until it burns a call on it. In cloud mode bucket-scope cooldowns are now
mirrored as ``SET turtle:cooldown:{bucket_id} 1 EX <s>`` and read once per
cascade with MGET.

"Two instances" is modelled by one fakeredis server (the shared Redis) plus
wiping the module's process-local dicts between "instances" -- the dicts are the
only per-process state. This proves the mechanism, not a production frequency.
"""
from __future__ import annotations

import asyncio
import time

import fakeredis
import pytest
import redis
import redis.asyncio as aredis

from core import health_tracker
from core.config import settings
from core.llm_client import run_agent_with_fallbacks
from pydantic_ai.exceptions import ModelHTTPError


def _http_error(status_code: int, message: str = ""):
    body = message or f"HTTP {status_code}"
    try:
        return ModelHTTPError(status_code=status_code, model_name="m", body=body)
    except TypeError:
        return ModelHTTPError(status_code, "m", body)


class _OpenRouterModel:
    model_name = "google/gemini-2.5-flash"


class _GroqModel:
    model_name = "openai/gpt-oss-20b"


class _FakeAgent:
    def __init__(self, model, *, raises: Exception | None = None, result=None):
        self.model = model
        self._raises = raises
        self._result = result
        self.calls = 0

    async def run(self, *a, **k):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._result


OR_KEY = "turtle:cooldown:_OpenRouterModel:google/gemini-2.5-flash"


def _reset_process_state() -> None:
    """What a brand-new process looks like to health_tracker."""
    with health_tracker._lock:
        health_tracker._cooldown_until.clear()
        getattr(health_tracker, "_shared_until", {}).clear()
    health_tracker._redis_down_until = 0.0


async def _drain() -> None:
    if getattr(health_tracker, "_pending", None):
        await asyncio.gather(*list(health_tracker._pending))


@pytest.fixture(autouse=True)
def _clean():
    _reset_process_state()
    yield
    _reset_process_state()


@pytest.fixture
def cloud_fake_redis(monkeypatch):
    """Cloud mode + one shared fakeredis, returned to health_tracker through the
    same seams it uses in production (get_redis_client / get_redis_sync_client)."""
    from core.storage import cloud

    server = fakeredis.FakeServer()
    aclient = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
    sclient = fakeredis.FakeRedis(server=server, decode_responses=True)

    async def _get():
        return aclient

    monkeypatch.setattr(settings, "deploy_mode", "cloud")
    monkeypatch.setattr(cloud, "get_redis_client", _get)
    monkeypatch.setattr(cloud, "get_redis_sync_client", lambda: sclient)
    aclient._test_server = server
    return aclient


def test_cold_instance_does_not_reburn_a_402_provider(cloud_fake_redis):
    """Acceptance: one instance sees a 402, a fresh instance that never saw the
    failure skips that provider and reaches the working one."""

    async def scenario():
        # Instance A (warm): burns a call on OpenRouter, gets 402.
        or_a = _FakeAgent(_OpenRouterModel(), raises=_http_error(402, "requires more credits"))
        groq_a = _FakeAgent(_GroqModel(), result="from-A")
        assert await run_agent_with_fallbacks(or_a, [groq_a]) == "from-A"
        assert or_a.calls == 1
        await _drain()

        # Instance B (cold): empty process-local state, same Redis.
        _reset_process_state()
        or_b = _FakeAgent(_OpenRouterModel(), raises=_http_error(402, "requires more credits"))
        groq_b = _FakeAgent(_GroqModel(), result="from-B")
        assert await run_agent_with_fallbacks(or_b, [groq_b]) == "from-B"
        return or_b.calls

    assert asyncio.run(scenario()) == 0  # B never touched the dead provider


def test_mirror_uses_ledger_key_value_and_ttl(cloud_fake_redis):
    async def scenario():
        health_tracker.mark_failure(_FakeAgent(_OpenRouterModel()), _http_error(402, "credits"))
        await _drain()
        value = await cloud_fake_redis.get(OR_KEY)
        ttl = await cloud_fake_redis.ttl(OR_KEY)
        return value, ttl

    value, ttl = asyncio.run(scenario())
    assert abs(float(value) - (time.time() + 300)) < 30  # absolute wall-clock deadline
    assert 0 < ttl <= 300  # the TTL carries the deadline; no monotonic value stored


def test_rung_scope_cooldown_is_not_mirrored(cloud_fake_redis):
    async def scenario():
        health_tracker.mark_failure(_FakeAgent(_OpenRouterModel()), _http_error(429, "slow down"))
        await _drain()
        return await cloud_fake_redis.keys("turtle:cooldown:*")

    assert asyncio.run(scenario()) == []


def test_one_mget_per_cascade_not_one_per_rung(cloud_fake_redis, monkeypatch):
    calls = {"mget": 0, "get": 0}
    real_mget = cloud_fake_redis.mget

    async def counting_mget(*a, **k):
        calls["mget"] += 1
        return await real_mget(*a, **k)

    monkeypatch.setattr(cloud_fake_redis, "mget", counting_mget)

    async def scenario():
        agents = [_FakeAgent(_OpenRouterModel(), raises=_http_error(500)) for _ in range(3)]
        groq = _FakeAgent(_GroqModel(), result="ok")
        await run_agent_with_fallbacks(agents[0], agents[1:] + [groq])

    asyncio.run(scenario())
    assert calls["mget"] == 1


def test_mark_success_clears_the_shared_key(cloud_fake_redis):
    async def scenario():
        agent = _FakeAgent(_OpenRouterModel())
        health_tracker.mark_failure(agent, _http_error(402, "credits"))
        await _drain()
        assert await cloud_fake_redis.exists(OR_KEY) == 1
        health_tracker.mark_success(agent)
        await _drain()
        return await cloud_fake_redis.exists(OR_KEY)

    assert asyncio.run(scenario()) == 0


def test_expired_key_is_no_longer_cooling(cloud_fake_redis):
    async def scenario():
        await cloud_fake_redis.set(OR_KEY, repr(time.time() + 0.05), px=50)
        agent = _FakeAgent(_OpenRouterModel())
        await health_tracker.refresh_shared([agent])
        assert health_tracker.is_cooling(agent) is True
        await asyncio.sleep(0.1)
        await health_tracker.refresh_shared([agent])
        return health_tracker.is_cooling(agent)

    assert asyncio.run(scenario()) is False


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: redis.exceptions.ConnectionError("Error 111 connecting to redis"),
        lambda: redis.exceptions.TimeoutError("Timeout reading from socket"),
    ],
    ids=["ConnectionError", "TimeoutError"],
)
def test_fails_open_on_the_drivers_real_errors(cloud_fake_redis, monkeypatch, make_error):
    """Redis raising the driver's own error types must not break the cascade."""

    async def boom(*a, **k):
        raise make_error()

    monkeypatch.setattr(cloud_fake_redis, "mget", boom)
    monkeypatch.setattr(cloud_fake_redis, "set", boom)

    async def scenario():
        primary = _FakeAgent(_OpenRouterModel(), raises=_http_error(402, "credits"))
        groq = _FakeAgent(_GroqModel(), result="served")
        out = await run_agent_with_fallbacks(primary, [groq])
        await _drain()
        return out, primary.calls, health_tracker.is_cooling(primary)

    out, primary_calls, cooling = asyncio.run(scenario())
    assert out == "served"
    assert primary_calls == 1
    assert cooling is True  # local state still works while Redis is down


def test_fails_open_against_a_genuinely_unreachable_redis(monkeypatch):
    """No fake at all: a real redis.asyncio client pointed at a closed port
    raises the driver's real connection error from a real socket attempt."""
    from core.storage import cloud

    dead = aredis.from_url(
        "redis://127.0.0.1:1", decode_responses=True,
        socket_connect_timeout=0.2, socket_timeout=0.2,
    )

    async def _get():
        return dead

    monkeypatch.setattr(settings, "deploy_mode", "cloud")
    monkeypatch.setattr(cloud, "get_redis_client", _get)

    async def scenario():
        primary = _FakeAgent(_OpenRouterModel(), result="fine")
        out = await run_agent_with_fallbacks(primary, [])
        await dead.aclose()
        return out

    assert asyncio.run(scenario()) == "fine"
    assert health_tracker._redis_down_until > 0  # it did try, fail, and back off


def test_backoff_skips_redis_after_a_failure(cloud_fake_redis, monkeypatch):
    calls = {"n": 0}

    async def boom(*a, **k):
        calls["n"] += 1
        raise redis.exceptions.ConnectionError("down")

    monkeypatch.setattr(cloud_fake_redis, "mget", boom)

    async def scenario():
        agent = _FakeAgent(_GroqModel())
        await health_tracker.refresh_shared([agent])
        await health_tracker.refresh_shared([agent])
        await health_tracker.refresh_shared([agent])

    asyncio.run(scenario())
    assert calls["n"] == 1


def test_local_mode_never_touches_redis(monkeypatch):
    from core.storage import cloud

    assert settings.is_cloud is False

    async def forbidden():
        raise AssertionError("local mode must not create a Redis client")

    def forbidden_sync():
        raise AssertionError("local mode must not create a Redis client")

    monkeypatch.setattr(cloud, "get_redis_client", forbidden)
    monkeypatch.setattr(cloud, "get_redis_sync_client", forbidden_sync)

    async def scenario():
        primary = _FakeAgent(_OpenRouterModel(), raises=_http_error(402, "credits"))
        groq = _FakeAgent(_GroqModel(), result="local-ok")
        out = await run_agent_with_fallbacks(primary, [groq])
        return out, health_tracker.is_cooling(primary)

    out, cooling = asyncio.run(scenario())
    assert out == "local-ok"
    assert cooling is True  # in-process cooldown unchanged in local mode
    assert health_tracker._redis_down_until == 0.0


def test_sync_cascade_reads_shared_cooldown(cloud_fake_redis):
    """run_agent_sync_with_fallbacks uses the sync client (offline CLI)."""
    from core.llm_client import run_agent_sync_with_fallbacks

    class _SyncAgent:
        def __init__(self, model, result):
            self.model = model
            self._result = result
            self.calls = 0

        def run_sync(self, *a, **k):
            self.calls += 1
            return self._result

    sync_client = fakeredis.FakeRedis(server=cloud_fake_redis._test_server, decode_responses=True)
    sync_client.set(OR_KEY, repr(time.time() + 120), ex=120)

    dead = _SyncAgent(_OpenRouterModel(), "dead")
    live = _SyncAgent(_GroqModel(), "live")
    assert run_agent_sync_with_fallbacks(dead, [live]) == "live"
    assert dead.calls == 0


# ---------------------------------------------------------------------------
# Shared view must outlive the first rung's LLM call (review follow-up)
# ---------------------------------------------------------------------------
class _Clock:
    """Stand-in for health_tracker's `time` module with an advanceable offset."""

    def __init__(self) -> None:
        self.offset = 0.0

    def monotonic(self) -> float:
        return time.monotonic() + self.offset

    def time(self) -> float:
        return time.time() + self.offset


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(health_tracker, "time", clock)
    return clock


def test_shared_view_outlives_a_slow_first_rung(cloud_fake_redis, fake_clock):
    """All rungs shared-cooled -> the cascade bypasses the eligibility filter
    and tries rung 0. That attempt takes 30s (> the old 10s window); the in-loop
    is_cooling re-check for the sibling must STILL see the shared cooldown."""

    class _SlowDead(_FakeAgent):
        async def run(self, *a, **k):
            self.calls += 1
            fake_clock.offset += 30.0  # the first rung's LLM call consumed 30s
            raise self._raises

    async def scenario():
        await cloud_fake_redis.set(OR_KEY, repr(time.time() + 300), ex=300)
        first = _SlowDead(_OpenRouterModel(), raises=_http_error(500))
        sibling = _FakeAgent(_OpenRouterModel(), result="should-not-run")
        with pytest.raises(ModelHTTPError):
            await run_agent_with_fallbacks(first, [sibling])
        return first.calls, sibling.calls

    first_calls, sibling_calls = asyncio.run(scenario())
    assert first_calls == 1       # idx 0 is always tried
    assert sibling_calls == 0     # dead family's sibling key still skipped


def test_view_expires_with_the_real_deadline_not_before(cloud_fake_redis, fake_clock):
    async def scenario():
        agent = _FakeAgent(_OpenRouterModel())
        await cloud_fake_redis.set(OR_KEY, repr(time.time() + 100), ex=100)
        await health_tracker.refresh_shared([agent])
        fake_clock.offset = 60.0
        still = health_tracker.is_cooling(agent)
        fake_clock.offset = 101.0
        gone = health_tracker.is_cooling(agent)
        return still, gone

    assert asyncio.run(scenario()) == (True, False)


@pytest.mark.parametrize("legacy", ["1", "garbage", "nan"], ids=["legacy-1", "garbage", "nan"])
def test_legacy_or_unparseable_value_still_cools_briefly(cloud_fake_redis, legacy):
    """Rollover: a key written by the previous deploy holds the literal "1"."""

    async def scenario():
        agent = _FakeAgent(_OpenRouterModel())
        await cloud_fake_redis.set(OR_KEY, legacy, ex=100)
        await health_tracker.refresh_shared([agent])
        return health_tracker.is_cooling(agent)

    assert asyncio.run(scenario()) is True


# ---------------------------------------------------------------------------
# Stream cascade: the call site needs its own test (controls wired at one end)
# ---------------------------------------------------------------------------
class _StreamResult:
    def __init__(self, text):
        self._text = text

    async def stream_text(self, delta=True):
        yield self._text

    async def get_output(self):
        return self._text

    def new_messages(self):
        return []


class _StreamCtx:
    def __init__(self, agent):
        self._agent = agent

    async def __aenter__(self):
        self._agent.calls += 1
        if self._agent._raises is not None:
            raise self._agent._raises
        return _StreamResult(self._agent._result)

    async def __aexit__(self, *exc):
        return False


class _StreamAgent(_FakeAgent):
    def run_stream(self, *a, **k):
        return _StreamCtx(self)


def test_stream_cascade_reads_shared_cooldown_with_one_mget(cloud_fake_redis, monkeypatch):
    from core.llm_client import StreamCollector, stream_agent_text_with_fallbacks

    calls = {"mget": 0}
    real_mget = cloud_fake_redis.mget

    async def counting_mget(*a, **k):
        calls["mget"] += 1
        return await real_mget(*a, **k)

    monkeypatch.setattr(cloud_fake_redis, "mget", counting_mget)

    async def scenario():
        await cloud_fake_redis.set(OR_KEY, repr(time.time() + 300), ex=300)
        dead = _StreamAgent(_OpenRouterModel(), raises=_http_error(402, "credits"))
        live = _StreamAgent(_GroqModel(), result="hello")
        collector = StreamCollector()
        out = []
        async for d in stream_agent_text_with_fallbacks(dead, [live], collector=collector):
            out.append(d)
        return "".join(out), dead.calls, live.calls

    text, dead_calls, live_calls = asyncio.run(scenario())
    assert (text, dead_calls, live_calls) == ("hello", 0, 1)
    assert calls["mget"] == 1


def test_sync_cascade_end_to_end_uses_one_sync_mget(cloud_fake_redis, monkeypatch):
    """Pins the sync runner's MGET count (dead rung gets 0 calls, 1 MGET)."""
    from core.llm_client import run_agent_sync_with_fallbacks
    from core.storage import cloud as cloud_mod

    sync_client = fakeredis.FakeRedis(server=cloud_fake_redis._test_server, decode_responses=True)
    sync_client.set(OR_KEY, repr(time.time() + 120), ex=120)
    n = {"mget": 0}
    real = sync_client.mget

    def counting(*a, **k):
        n["mget"] += 1
        return real(*a, **k)

    monkeypatch.setattr(sync_client, "mget", counting)
    monkeypatch.setattr(cloud_mod, "get_redis_sync_client", lambda: sync_client)

    class _S:
        def __init__(self, model, result):
            self.model, self._r, self.calls = model, result, 0

        def run_sync(self, *a, **k):
            self.calls += 1
            return self._r

    dead, live = _S(_OpenRouterModel(), "dead"), _S(_GroqModel(), "live")
    assert run_agent_sync_with_fallbacks(dead, [live]) == "live"
    assert (dead.calls, n["mget"]) == (0, 1)
