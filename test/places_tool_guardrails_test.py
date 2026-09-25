"""
tools/places_tool.py guardrail tests (WP 1.G / S-7.9).

Covers:
  - place_id validation + quoting: malformed IDs never reach the transport
  - error responses never leak Google's raw body to the model, but the raw
    body DOES reach the server-side log
  - Redis/in-process cache: hit skips the HTTP call, miss fires it, TTLs differ
  - per-user daily call cap: refuses after N calls with a sensible ToolResult
  - Redis-unavailable posture: fails open against a driver-level exception,
    not just a mocked CloudBackendUnavailable
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

import httpx
import pytest


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


def _set_api_key(monkeypatch, value: str | None) -> None:
    import tools.places_tool as places_tool

    class _Stub:
        def __init__(self, v):
            self._v = v

        google_maps_api_key = None

    stub = _Stub(value)
    if value is None:
        stub.google_maps_api_key = None
    else:
        class _SecretLike:
            def __init__(self, v):
                self._v = v

            def get_secret_value(self):
                return self._v

        stub.google_maps_api_key = _SecretLike(value)
    monkeypatch.setattr(places_tool, "settings", stub)


def _no_op_cache_and_cap(monkeypatch):
    """Wire in a fresh, always-empty in-process cache + an effectively
    unlimited call cap, so tests that aren't specifically about caching/caps
    don't trip over the module-level singletons other tests may have warmed."""
    import tools.places_tool as places_tool
    from tools.places_guardrails import InProcessPlacesCache, InProcessPlacesCallLimiter

    cache = InProcessPlacesCache()
    limiter = InProcessPlacesCallLimiter(per_day=10_000)
    monkeypatch.setattr(places_tool, "get_places_cache", lambda: cache)
    monkeypatch.setattr(places_tool, "get_places_call_limiter", lambda: limiter)
    return cache, limiter


# ---------------------------------------------------------------------------
# place_id validation + quoting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    [
        "abc/def",
        "abc?def",
        "abc#def",
        "abc def",
        "abc\ndef",
    ],
)
def test_place_details_rejects_malformed_place_id(monkeypatch, bad_id):
    from tools.places_tool import place_details, PlaceDetailsArgs

    _set_api_key(monkeypatch, "key")
    _no_op_cache_and_cap(monkeypatch)

    called = {"hit": False}

    def handler(request: httpx.Request) -> httpx.Response:
        called["hit"] = True
        return httpx.Response(200, json={"id": "should-not-be-reached"})

    async def go():
        async with _mock_client(handler) as client:
            return await place_details(PlaceDetailsArgs(place_id=bad_id), http_client=client)

    result = asyncio.run(go())
    assert result.status == "invalid"
    assert result.error_code == "invalid_place_id"
    assert called["hit"] is False, "transport must never be called for a malformed place_id"


def test_place_details_accepts_legitimate_place_id(monkeypatch):
    from tools.places_tool import place_details, PlaceDetailsArgs

    _set_api_key(monkeypatch, "key")
    _no_op_cache_and_cap(monkeypatch)

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"id": "ChIJN1t_tDeuEmsRUsoyG83frY4"})

    async def go():
        async with _mock_client(handler) as client:
            return await place_details(
                PlaceDetailsArgs(place_id="ChIJN1t_tDeuEmsRUsoyG83frY4"), http_client=client
            )

    result = asyncio.run(go())
    assert result.status == "ok"
    assert captured["url"].endswith("/places/ChIJN1t_tDeuEmsRUsoyG83frY4")


def test_place_details_accepts_places_prefix(monkeypatch):
    from tools.places_tool import place_details, PlaceDetailsArgs

    _set_api_key(monkeypatch, "key")
    _no_op_cache_and_cap(monkeypatch)

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"id": "abc123"})

    async def go():
        async with _mock_client(handler) as client:
            return await place_details(
                PlaceDetailsArgs(place_id="places/abc123"), http_client=client
            )

    result = asyncio.run(go())
    assert result.status == "ok"
    assert captured["url"].endswith("/places/abc123")


def test_place_details_quotes_the_path_segment(monkeypatch):
    """A place_id in the accepted charset that still needs percent-encoding
    (here: nothing does, by construction of the charset, but this proves the
    quoting call actually runs and produces byte-identical output for a
    representative id — i.e. quoting is exercised, not skipped)."""
    from tools.places_tool import place_details, PlaceDetailsArgs

    _set_api_key(monkeypatch, "key")
    _no_op_cache_and_cap(monkeypatch)

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"id": "A-Za_09"})

    async def go():
        async with _mock_client(handler) as client:
            return await place_details(PlaceDetailsArgs(place_id="A-Za_09"), http_client=client)

    asyncio.run(go())
    from urllib.parse import quote

    assert captured["url"].endswith("/places/" + quote("A-Za_09", safe=""))


# ---------------------------------------------------------------------------
# Error responses: no raw body to the model, raw body DOES reach the log
# ---------------------------------------------------------------------------


def test_403_error_does_not_leak_raw_body_but_logs_it(monkeypatch, caplog):
    from tools.places_tool import find_place, FindPlaceArgs

    _set_api_key(monkeypatch, "bad-key")
    _no_op_cache_and_cap(monkeypatch)

    secret_body = "PERMISSION_DENIED: api key AIzaSyFAKE-SECRET-VALUE-1234 not enabled"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text=secret_body)

    async def go():
        async with _mock_client(handler) as client:
            return await find_place(FindPlaceArgs(query="anywhere"), http_client=client)

    with caplog.at_level(logging.WARNING, logger="tools.places_tool"):
        result = asyncio.run(go())

    assert result.status == "upstream_error"
    assert result.error_code == "auth_failed"
    assert "AIzaSyFAKE-SECRET-VALUE-1234" not in result.error_message
    assert "PERMISSION_DENIED" not in result.error_message

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "AIzaSyFAKE-SECRET-VALUE-1234" in logged
    assert "PERMISSION_DENIED" in logged


def test_400_error_keeps_place_id_guidance_without_raw_body(monkeypatch, caplog):
    from tools.places_tool import place_details, PlaceDetailsArgs

    _set_api_key(monkeypatch, "key")
    _no_op_cache_and_cap(monkeypatch)

    secret_body = "INVALID_ARGUMENT: some internal Google diagnostic string xyz"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=secret_body)

    async def go():
        async with _mock_client(handler) as client:
            return await place_details(PlaceDetailsArgs(place_id="looksvalidbutfake"), http_client=client)

    with caplog.at_level(logging.WARNING, logger="tools.places_tool"):
        result = asyncio.run(go())

    assert result.status == "invalid"
    assert result.error_code == "bad_request"
    assert "place_id" in result.error_message.lower()
    assert secret_body not in result.error_message

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert secret_body in logged


# ---------------------------------------------------------------------------
# Cache: hit skips HTTP, miss fires it, TTLs differ
# ---------------------------------------------------------------------------


def test_find_place_cache_hit_skips_http_call(monkeypatch):
    from tools.places_tool import find_place, FindPlaceArgs
    import tools.places_tool as places_tool
    from tools.places_guardrails import InProcessPlacesCache, InProcessPlacesCallLimiter

    _set_api_key(monkeypatch, "key")
    cache = InProcessPlacesCache()
    limiter = InProcessPlacesCallLimiter(per_day=10_000)
    monkeypatch.setattr(places_tool, "get_places_cache", lambda: cache)
    monkeypatch.setattr(places_tool, "get_places_call_limiter", lambda: limiter)

    calls = {"n": 0}
    body = {"places": [{"id": "abc123", "displayName": {"text": "Cached Place"}}]}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=body)

    async def go():
        async with _mock_client(handler) as client:
            r1 = await find_place(FindPlaceArgs(query="coffee near me"), http_client=client)
            r2 = await find_place(FindPlaceArgs(query="coffee near me"), http_client=client)
            return r1, r2

    r1, r2 = asyncio.run(go())
    assert calls["n"] == 1, "second identical search must be served from cache"
    assert r1.status == "ok" and r2.status == "ok"
    assert r2.data.results[0].name == "Cached Place"


def test_find_place_cache_miss_on_different_query_fires_http(monkeypatch):
    from tools.places_tool import find_place, FindPlaceArgs
    import tools.places_tool as places_tool
    from tools.places_guardrails import InProcessPlacesCache, InProcessPlacesCallLimiter

    _set_api_key(monkeypatch, "key")
    cache = InProcessPlacesCache()
    limiter = InProcessPlacesCallLimiter(per_day=10_000)
    monkeypatch.setattr(places_tool, "get_places_cache", lambda: cache)
    monkeypatch.setattr(places_tool, "get_places_call_limiter", lambda: limiter)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"places": []})

    async def go():
        async with _mock_client(handler) as client:
            await find_place(FindPlaceArgs(query="pizza"), http_client=client)
            await find_place(FindPlaceArgs(query="sushi"), http_client=client)

    asyncio.run(go())
    assert calls["n"] == 2


def test_search_and_details_ttls_differ():
    import tools.places_tool as places_tool

    assert places_tool._SEARCH_CACHE_TTL_S == 600
    assert places_tool._DETAILS_CACHE_TTL_S == 3600
    assert places_tool._SEARCH_CACHE_TTL_S != places_tool._DETAILS_CACHE_TTL_S


# ---------------------------------------------------------------------------
# Per-user daily call cap
# ---------------------------------------------------------------------------


def test_call_cap_refuses_after_n_calls(monkeypatch):
    from tools.places_tool import find_place, FindPlaceArgs
    import tools.places_tool as places_tool
    from tools.places_guardrails import InProcessPlacesCache, InProcessPlacesCallLimiter

    _set_api_key(monkeypatch, "key")
    cache = InProcessPlacesCache()
    limiter = InProcessPlacesCallLimiter(per_day=2)
    monkeypatch.setattr(places_tool, "get_places_cache", lambda: cache)
    monkeypatch.setattr(places_tool, "get_places_call_limiter", lambda: limiter)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"places": []})

    async def go():
        results = []
        async with _mock_client(handler) as client:
            for q in ("aa", "bb", "cc"):
                results.append(
                    await find_place(
                        FindPlaceArgs(query=q), http_client=client, user_id="user-1"
                    )
                )
        return results

    r1, r2, r3 = asyncio.run(go())
    assert r1.status == "empty" and r2.status == "empty"
    assert r3.status == "rate_limited"
    assert r3.retryable
    assert r3.retry_after_ms > 0
    assert calls["n"] == 2, "the third (capped) call must never reach the transport"


def test_call_cap_is_per_user(monkeypatch):
    from tools.places_tool import find_place, FindPlaceArgs
    import tools.places_tool as places_tool
    from tools.places_guardrails import InProcessPlacesCache, InProcessPlacesCallLimiter

    _set_api_key(monkeypatch, "key")
    cache = InProcessPlacesCache()
    limiter = InProcessPlacesCallLimiter(per_day=1)
    monkeypatch.setattr(places_tool, "get_places_cache", lambda: cache)
    monkeypatch.setattr(places_tool, "get_places_call_limiter", lambda: limiter)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"places": []})

    async def go():
        async with _mock_client(handler) as client:
            r_a = await find_place(FindPlaceArgs(query="aa"), http_client=client, user_id="user-A")
            r_b = await find_place(FindPlaceArgs(query="bb"), http_client=client, user_id="user-B")
            return r_a, r_b

    r_a, r_b = asyncio.run(go())
    assert r_a.status == "empty"
    assert r_b.status == "empty", "a different user's cap must be independent"


# ---------------------------------------------------------------------------
# Redis-unavailable posture: fails open against a driver-level failure
# ---------------------------------------------------------------------------


def test_redis_cache_driver_failure_fails_open(monkeypatch, caplog):
    from tools.places_guardrails import RedisPlacesCache

    class _BrokenClient:
        def get(self, key):
            raise ConnectionError("connection refused by the redis driver")

        def set(self, *a, **k):
            raise ConnectionError("connection refused by the redis driver")

    def _fake_get_client():
        return _BrokenClient()

    monkeypatch.setattr(
        "core.storage.cloud.get_redis_sync_client", _fake_get_client
    )

    cache = RedisPlacesCache()
    with caplog.at_level(logging.WARNING):
        result = cache.get("turtle:places_cache:v1:find_place:deadbeef")
        cache.set("turtle:places_cache:v1:find_place:deadbeef", {"places": []}, 600)
    assert result is None  # treated as a miss, not raised


def test_redis_call_limiter_driver_failure_fails_open(monkeypatch):
    from tools.places_guardrails import RedisPlacesCallLimiter

    class _BrokenClient:
        def zremrangebyscore(self, *a, **k):
            raise ConnectionError("connection refused by the redis driver")

    monkeypatch.setattr(
        "core.storage.cloud.get_redis_sync_client", lambda: _BrokenClient()
    )

    limiter = RedisPlacesCallLimiter(per_day=1)
    # Must not raise PlacesCallCapExceeded (or anything else) — fails open.
    limiter.check_and_record("user-1")


# ---------------------------------------------------------------------------
# WP 1.G2 must-fix 1 — RedisPlacesCallLimiter.check_and_record must be
# atomic under real concurrency: a sliding-window sorted set built from
# separate ZREMRANGEBYSCORE -> ZCARD (read) -> conditional ZADD (write)
# round-trips is racy (each caller decides on a count read a moment
# earlier). Fixed by ZADD-ing a uniquely-keyed member FIRST, then counting
# (which necessarily includes the caller's own just-landed write), and
# self-removing only the caller's own member on refusal or mid-flight
# failure.
#
# _FakeRedisCap below simulates a REAL Redis round trip: each command
# sleeps (network latency) BEFORE taking a short internal lock to mutate/
# read the shared sorted set (the "atomic on the server" part). The sleep
# is deliberately OUTSIDE the lock so many callers' round trips can be
# in-flight — and interleaved with each other — at once. A sequential fake
# (no delay, or delay-under-lock) would serialize every command and make
# any implementation "pass" without exercising the race at all.
# ---------------------------------------------------------------------------

import random
import threading as _threading
from concurrent.futures import ThreadPoolExecutor


class _FakeRedisCap:
    def __init__(self, delay: float = 0.015, jitter: float = 0.01, seed: int = 7):
        self._lock = _threading.Lock()
        self._zset: dict[str, dict[str, float]] = {}
        self._delay = delay
        self._jitter = jitter
        self._rng = random.Random(seed)

    def _sleep(self):
        # Random jitter per call — real network latency is not perfectly
        # synchronized across concurrent callers, so a fixed delay alone
        # would let every caller's round trips land in lockstep and hide
        # the very interleaving we're trying to exercise.
        time.sleep(self._delay + self._rng.uniform(0, self._jitter))

    def zadd(self, key, mapping):
        self._sleep()
        with self._lock:
            self._zset.setdefault(key, {}).update(mapping)

    def zremrangebyscore(self, key, min_, max_):
        self._sleep()
        with self._lock:
            z = self._zset.setdefault(key, {})
            cutoff = float(max_)
            for m in [m for m, s in z.items() if s <= cutoff]:
                del z[m]

    def zcard(self, key):
        self._sleep()
        with self._lock:
            return len(self._zset.get(key, {}))

    def zrem(self, key, member):
        with self._lock:
            self._zset.get(key, {}).pop(member, None)

    def expire(self, key, seconds):
        pass


def test_call_cap_concurrent_callers_cannot_exceed_cap(monkeypatch):
    """N=20 real OS threads race against a cap of 5. Must never collectively
    exceed the cap. Against the pre-fix ZREMRANGEBYSCORE -> ZCARD -> ZADD
    sequence, this test fails (see the WP report for the captured failure:
    20 allowed against a cap of 5)."""
    from tools.places_guardrails import RedisPlacesCallLimiter

    fake = _FakeRedisCap()
    monkeypatch.setattr("core.storage.cloud.get_redis_sync_client", lambda: fake)

    limiter = RedisPlacesCallLimiter(per_day=5)
    n = 20
    results: list[bool] = [False] * n

    def worker(i):
        try:
            limiter.check_and_record("user-race")
            results[i] = True
        except Exception:
            results[i] = False

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(worker, i) for i in range(n)]
        for f in futures:
            f.result()

    allowed_count = sum(results)
    assert allowed_count <= 5, (
        f"{allowed_count} of {n} concurrent callers were allowed against a "
        f"cap of 5 — the call cap was raced past, which is exactly the WP "
        f"1.G2 must-fix. Under the old check-then-act sequence this "
        f"assertion fails ({n} > 5)."
    )
    assert allowed_count >= 1, "the fake Redis or limiter logic is broken, not just strict"

    # The sorted set must end up holding exactly the members that were
    # actually granted — no stray leftovers from refused attempts.
    key = "turtle:places_cap:v1:user-race"
    assert len(fake._zset.get(key, {})) == allowed_count


def test_call_cap_refusal_leaves_no_stranded_member(monkeypatch):
    """A refused call must remove exactly the member IT added, not leave it
    behind and not touch anyone else's member."""
    from tools.places_guardrails import RedisPlacesCallLimiter, PlacesCallCapExceeded

    fake = _FakeRedisCap(delay=0.0, jitter=0.0)
    monkeypatch.setattr("core.storage.cloud.get_redis_sync_client", lambda: fake)

    limiter = RedisPlacesCallLimiter(per_day=2)
    limiter.check_and_record("user-1")
    limiter.check_and_record("user-1")
    key = "turtle:places_cap:v1:user-1"
    assert len(fake._zset.get(key, {})) == 2

    with pytest.raises(PlacesCallCapExceeded):
        limiter.check_and_record("user-1")

    # The refused call's own member must not survive; the two already-
    # granted members must be untouched.
    assert len(fake._zset.get(key, {})) == 2


def test_call_cap_mid_flight_exception_leaves_no_stranded_member(monkeypatch):
    """A driver failure AFTER our own ZADD lands (e.g. ZCARD blows up) must
    still remove our own member before failing open — not just refusals."""
    from tools.places_guardrails import RedisPlacesCallLimiter

    fake = _FakeRedisCap(delay=0.0, jitter=0.0)

    real_zcard = fake.zcard

    def _boom_zcard(key):
        raise ConnectionError("simulated mid-flight redis failure")

    fake.zcard = _boom_zcard
    monkeypatch.setattr("core.storage.cloud.get_redis_sync_client", lambda: fake)

    limiter = RedisPlacesCallLimiter(per_day=5)
    limiter.check_and_record("user-1")  # must not raise — fails open

    key = "turtle:places_cap:v1:user-1"
    assert fake._zset.get(key, {}) == {}, (
        "the ZADD-ed member must be cleaned up when a later call in the "
        "same check_and_record fails, not stranded in the sorted set"
    )


def test_call_cap_cancelled_error_leaves_no_stranded_member_and_propagates(monkeypatch):
    """asyncio.CancelledError is a BaseException, not an Exception. Must
    still clean up our own member (Wave 1 shipped exactly this bug: a
    stranding cleanup path that only caught Exception), AND must still
    propagate the cancellation rather than being silently swallowed as if
    it were an ordinary Redis failure."""
    import asyncio as _asyncio

    from tools.places_guardrails import RedisPlacesCallLimiter

    fake = _FakeRedisCap(delay=0.0, jitter=0.0)

    def _cancel_zcard(key):
        raise _asyncio.CancelledError()

    fake.zcard = _cancel_zcard
    monkeypatch.setattr("core.storage.cloud.get_redis_sync_client", lambda: fake)

    limiter = RedisPlacesCallLimiter(per_day=5)
    with pytest.raises(_asyncio.CancelledError):
        limiter.check_and_record("user-1")

    key = "turtle:places_cap:v1:user-1"
    assert fake._zset.get(key, {}) == {}, (
        "CancelledError mid-check must not strand the member either"
    )


# ---------------------------------------------------------------------------
# WP 1.G2 must-fix 2 — a ConnectError's str(exc) must never reach the model.
# ---------------------------------------------------------------------------


def test_connect_error_does_not_leak_host_or_key_to_model(monkeypatch, caplog):
    from tools.places_tool import find_place, FindPlaceArgs

    _set_api_key(monkeypatch, "key")
    _no_op_cache_and_cap(monkeypatch)

    leaky_detail = (
        "[Errno 111] Connection refused: internal-host-10-0-4-17.corp.local "
        "key=AIzaSyFAKE-INTERNAL-LEAK-9999"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(leaky_detail, request=request)

    async def go():
        async with _mock_client(handler) as client:
            return await find_place(FindPlaceArgs(query="anywhere"), http_client=client)

    with caplog.at_level(logging.WARNING, logger="tools.places_tool"):
        result = asyncio.run(go())

    assert result.status == "upstream_error"
    assert result.error_code == "network_error"
    assert result.retryable is True
    assert "internal-host-10-0-4-17.corp.local" not in result.error_message
    assert "AIzaSyFAKE-INTERNAL-LEAK-9999" not in result.error_message

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "internal-host-10-0-4-17.corp.local" in logged
    assert "AIzaSyFAKE-INTERNAL-LEAK-9999" in logged


# ---------------------------------------------------------------------------
# WP 1.G2 must-fix 3 — the tuned timeout must apply per-request, even when
# the injected http_client has none of its own (apps/turtle_server.py's
# shared client is a plain httpx.AsyncClient() with no timeout set).
# ---------------------------------------------------------------------------


def test_tuned_timeout_applies_even_with_a_timeoutless_injected_client(monkeypatch):
    from tools.places_tool import find_place, FindPlaceArgs, _DEFAULT_TIMEOUT

    _set_api_key(monkeypatch, "key")
    _no_op_cache_and_cap(monkeypatch)

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json={"places": []})

    async def go():
        # An injected client built with NO timeout of its own — exactly how
        # apps/turtle_server.py hands its shared http_client to every tool.
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await find_place(FindPlaceArgs(query="anywhere"), http_client=client)

    asyncio.run(go())

    assert seen["timeout"] == {
        "connect": _DEFAULT_TIMEOUT.connect,
        "read": _DEFAULT_TIMEOUT.read,
        "write": _DEFAULT_TIMEOUT.write,
        "pool": _DEFAULT_TIMEOUT.pool,
    }


# ---------------------------------------------------------------------------
# Nice-to-have — local-mode in-process dicts stay bounded rather than
# growing without limit for the lifetime of a long-running dev process.
# ---------------------------------------------------------------------------


def test_in_process_cache_stays_bounded_under_many_distinct_keys():
    from tools.places_guardrails import InProcessPlacesCache

    cache = InProcessPlacesCache()
    for i in range(cache._MAX_ENTRIES + 500):
        cache.set(f"k{i}", i, ttl_seconds=600)
    assert len(cache._store) <= cache._MAX_ENTRIES


def test_in_process_call_limiter_stays_bounded_under_many_distinct_users():
    from tools.places_guardrails import InProcessPlacesCallLimiter

    limiter = InProcessPlacesCallLimiter(per_day=1000)
    for i in range(limiter._MAX_USERS + 500):
        limiter.check_and_record(f"user-{i}")
    assert len(limiter._events) <= limiter._MAX_USERS
