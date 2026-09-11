"""
tools/places_tool.py behavior tests.

Covers:
  - find_place: ok / empty / auth_failed / rate_limited / credentials_missing
  - place_details: ok + credentials_missing
  - get_directions: ok + invalid travel_mode + empty routes
  - render_* helpers produce non-empty, sane strings
  - contracts arg schemas are re-exported so pydantic-ai can pick them up

The tests use httpx.MockTransport so no network calls fire.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


def _set_api_key(monkeypatch, value: str | None) -> None:
    """Patch the settings singleton so places_tool._api_key() sees the value."""
    import tools.places_tool as places_tool

    class _Stub:
        def __init__(self, v):
            self._v = v

        google_maps_api_key = None  # populated below

    stub = _Stub(value)
    if value is None:
        stub.google_maps_api_key = None
    else:
        # Emulate SecretStr's .get_secret_value(); places_tool falls back to str().
        class _SecretLike:
            def __init__(self, v):
                self._v = v

            def get_secret_value(self):
                return self._v

        stub.google_maps_api_key = _SecretLike(value)
    monkeypatch.setattr(places_tool, "settings", stub)


# ---------------------------------------------------------------------------
# find_place
# ---------------------------------------------------------------------------


def test_find_place_credentials_missing(monkeypatch):
    from tools.places_tool import find_place, FindPlaceArgs

    _set_api_key(monkeypatch, None)

    async def go():
        return await find_place(FindPlaceArgs(query="Mercure Hotel Dubai"))

    result = asyncio.run(go())
    assert result.status == "upstream_error"
    assert result.error_code == "credentials_missing"


def test_find_place_ok(monkeypatch):
    from tools.places_tool import find_place, FindPlaceArgs, render_find_place

    _set_api_key(monkeypatch, "test-key")

    body = {
        "places": [
            {
                "id": "abc123",
                "displayName": {"text": "Mercure Hotel Dubai Barsha"},
                "formattedAddress": "Sheikh Zayed Road, Al Barsha 1, Dubai, UAE",
                "shortFormattedAddress": "Al Barsha 1, Dubai",
                "location": {"latitude": 25.11, "longitude": 55.20},
                "googleMapsUri": "https://maps.google.com/?cid=1",
                "websiteUri": "https://all.accor.com/hotel/mercure-dubai",
                "internationalPhoneNumber": "+971 4 000 0000",
                "rating": 4.2,
                "userRatingCount": 1234,
                "priceLevel": "PRICE_LEVEL_MODERATE",
                "currentOpeningHours": {
                    "openNow": True,
                    "weekdayDescriptions": [
                        "Monday: Open 24 hours",
                        "Tuesday: Open 24 hours",
                    ],
                },
                "primaryTypeDisplayName": {"text": "Hotel"},
                "types": ["lodging", "point_of_interest"],
                "businessStatus": "OPERATIONAL",
            }
        ]
    }

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content or b"{}")
        return httpx.Response(200, json=body)

    async def go():
        async with _mock_client(handler) as client:
            return await find_place(
                FindPlaceArgs(query="Mercure Hotel Dubai"),
                http_client=client,
            )

    result = asyncio.run(go())
    assert result.status == "ok"
    assert result.data is not None
    assert len(result.data.results) == 1
    top = result.data.results[0]
    assert top.place_id == "abc123"
    assert top.maps_url.startswith("https://maps.google.com/")
    assert top.rating == 4.2
    assert top.open_now is True
    assert top.category == "Hotel"

    # Verify request shape sent to Google.
    assert captured["method"] == "POST"
    assert captured["url"].startswith("https://places.googleapis.com/v1/places:searchText")
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert "places.googleMapsUri" in captured["headers"]["x-goog-fieldmask"]
    assert captured["json"]["textQuery"] == "Mercure Hotel Dubai"
    assert captured["json"]["pageSize"] == 5

    # Render helper should surface the Maps URL for the LLM.
    rendered = render_find_place(result.data)
    assert "Maps: https://maps.google.com/?cid=1" in rendered
    assert "Mercure Hotel Dubai Barsha" in rendered


def test_find_place_empty(monkeypatch):
    from tools.places_tool import find_place, FindPlaceArgs

    _set_api_key(monkeypatch, "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"places": []})

    async def go():
        async with _mock_client(handler) as client:
            return await find_place(
                FindPlaceArgs(query="asdlkjasdlkjasdlkj"),
                http_client=client,
            )

    result = asyncio.run(go())
    assert result.status == "empty"


def test_find_place_auth_failed(monkeypatch):
    from tools.places_tool import find_place, FindPlaceArgs

    _set_api_key(monkeypatch, "bad-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="PERMISSION_DENIED: places api not enabled")

    async def go():
        async with _mock_client(handler) as client:
            return await find_place(
                FindPlaceArgs(query="anywhere"),
                http_client=client,
            )

    result = asyncio.run(go())
    assert result.status == "upstream_error"
    assert result.error_code == "auth_failed"
    assert not result.retryable


def test_find_place_rate_limited(monkeypatch):
    from tools.places_tool import find_place, FindPlaceArgs

    _set_api_key(monkeypatch, "key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Too many requests")

    async def go():
        async with _mock_client(handler) as client:
            return await find_place(FindPlaceArgs(query="xx"), http_client=client)

    result = asyncio.run(go())
    assert result.status == "rate_limited"
    assert result.retryable
    assert result.retry_after_ms > 0


def test_find_place_lat_lng_bias(monkeypatch):
    from tools.places_tool import find_place, FindPlaceArgs

    _set_api_key(monkeypatch, "key")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content or b"{}")
        return httpx.Response(200, json={"places": []})

    async def go():
        async with _mock_client(handler) as client:
            return await find_place(
                FindPlaceArgs(query="coffee", location_bias="12.97, 77.59"),
                http_client=client,
            )

    asyncio.run(go())
    assert "locationBias" in captured["json"]
    circle = captured["json"]["locationBias"]["circle"]
    assert circle["center"]["latitude"] == 12.97
    assert circle["center"]["longitude"] == 77.59


# ---------------------------------------------------------------------------
# place_details
# ---------------------------------------------------------------------------


def test_place_details_ok(monkeypatch):
    from tools.places_tool import place_details, PlaceDetailsArgs, render_place_details

    _set_api_key(monkeypatch, "key")

    body = {
        "id": "abc123",
        "displayName": {"text": "Mercure Hotel Dubai Barsha"},
        "formattedAddress": "Sheikh Zayed Road, Al Barsha 1, Dubai, UAE",
        "googleMapsUri": "https://maps.google.com/?cid=1",
        "currentOpeningHours": {
            "openNow": True,
            "weekdayDescriptions": [
                "Monday: Open 24 hours",
                "Tuesday: Open 24 hours",
                "Wednesday: Open 24 hours",
            ],
        },
        "editorialSummary": {"text": "4-star hotel with pool and gym."},
    }
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        return httpx.Response(200, json=body)

    async def go():
        async with _mock_client(handler) as client:
            return await place_details(
                PlaceDetailsArgs(place_id="abc123"), http_client=client
            )

    result = asyncio.run(go())
    assert result.status == "ok"
    assert result.data is not None
    assert len(result.data.weekly_hours) == 3
    assert "4-star hotel" in result.data.summary

    assert captured["method"] == "GET"
    assert captured["url"].endswith("/places/abc123")

    rendered = render_place_details(result.data)
    assert "Weekly hours" in rendered
    assert "4-star hotel" in rendered


def test_place_details_credentials_missing(monkeypatch):
    from tools.places_tool import place_details, PlaceDetailsArgs

    _set_api_key(monkeypatch, None)

    async def go():
        return await place_details(PlaceDetailsArgs(place_id="whatever"))

    result = asyncio.run(go())
    assert result.status == "upstream_error"
    assert result.error_code == "credentials_missing"


# ---------------------------------------------------------------------------
# get_directions
# ---------------------------------------------------------------------------


def test_get_directions_ok(monkeypatch):
    from tools.places_tool import (
        get_directions,
        GetDirectionsArgs,
        render_directions,
    )

    _set_api_key(monkeypatch, "key")

    body = {
        "routes": [
            {
                "distanceMeters": 152_400,
                "duration": "10800s",
                "warnings": ["Toll road ahead"],
            }
        ]
    }
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content or b"{}")
        return httpx.Response(200, json=body)

    async def go():
        async with _mock_client(handler) as client:
            return await get_directions(
                GetDirectionsArgs(
                    origin="Bengaluru",
                    destination="Mysore",
                    travel_mode="DRIVE",
                ),
                http_client=client,
            )

    result = asyncio.run(go())
    assert result.status == "ok"
    assert result.data is not None
    assert result.data.distance_meters == 152_400
    assert result.data.duration_seconds == 10800
    assert "152 km" in result.data.distance_text
    assert result.data.duration_text == "3h"
    assert "google.com/maps/dir" in result.data.maps_url
    assert "travelmode=driving" in result.data.maps_url

    # Traffic-aware routing enabled for DRIVE.
    assert captured["json"]["routingPreference"] == "TRAFFIC_AWARE"
    # Address waypoints (not lat/lng) for named origins.
    assert captured["json"]["origin"] == {"address": "Bengaluru"}

    rendered = render_directions(result.data)
    assert "Route (drive)" in rendered
    assert "152 km" in rendered
    assert "Toll road ahead" in rendered


def test_get_directions_lat_lng_waypoint(monkeypatch):
    from tools.places_tool import get_directions, GetDirectionsArgs

    _set_api_key(monkeypatch, "key")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content or b"{}")
        return httpx.Response(200, json={"routes": [{"distanceMeters": 1, "duration": "60s"}]})

    async def go():
        async with _mock_client(handler) as client:
            return await get_directions(
                GetDirectionsArgs(
                    origin="12.9716,77.5946",
                    destination="Mysore",
                    travel_mode="WALK",
                ),
                http_client=client,
            )

    asyncio.run(go())
    origin = captured["json"]["origin"]
    assert origin == {"location": {"latLng": {"latitude": 12.9716, "longitude": 77.5946}}}
    # WALK must NOT set routingPreference (Routes API rejects that combo).
    assert "routingPreference" not in captured["json"]


def test_get_directions_invalid_mode(monkeypatch):
    from tools.places_tool import get_directions, GetDirectionsArgs

    _set_api_key(monkeypatch, "key")

    async def go():
        return await get_directions(
            GetDirectionsArgs(origin="AA", destination="BB", travel_mode="TELEPORT")
        )

    result = asyncio.run(go())
    assert result.status == "invalid"


def test_get_directions_empty(monkeypatch):
    from tools.places_tool import get_directions, GetDirectionsArgs

    _set_api_key(monkeypatch, "key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"routes": []})

    async def go():
        async with _mock_client(handler) as client:
            return await get_directions(
                GetDirectionsArgs(origin="Nowhere", destination="Elsewhere"),
                http_client=client,
            )

    result = asyncio.run(go())
    assert result.status == "empty"


# ---------------------------------------------------------------------------
# Contracts re-exports (pydantic-ai wires arg schemas from tools.contracts)
# ---------------------------------------------------------------------------


def test_contracts_reexports_places_args():
    from tools.contracts import FindPlaceArgs, PlaceDetailsArgs, GetDirectionsArgs

    a = FindPlaceArgs(query="hi")
    assert a.max_results == 5
    b = PlaceDetailsArgs(place_id="x")
    assert b.place_id == "x"
    c = GetDirectionsArgs(origin="AA", destination="BB")
    assert c.travel_mode == "DRIVE"


# ---------------------------------------------------------------------------
# Config field wired through
# ---------------------------------------------------------------------------


def test_settings_has_google_maps_api_key():
    from core.config import settings

    # Attribute must exist even if unset in the current env.
    assert hasattr(settings, "google_maps_api_key")
