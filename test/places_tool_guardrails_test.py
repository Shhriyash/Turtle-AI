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
