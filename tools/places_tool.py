"""
tools/places_tool.py
--------------------
Google Maps Platform tool — Places API (New) + Routes API.

Capabilities:
  - find_place():        text search, returns top matches with Google Maps links
  - place_details():     full details for a specific place_id
  - get_directions():    driving / walking / transit route between two locations
                         with distance and duration

Auth:
  Single API key via GOOGLE_MAPS_API_KEY. Enable "Places API (New)" and
  "Routes API" in the Cloud project; restrict the key to those two APIs.

The key is sent as an X-Goog-Api-Key header (not a URL query parameter) so it
never lands in server access logs. See:
  https://developers.google.com/maps/documentation/places/web-service/text-search
  https://developers.google.com/maps/documentation/routes/compute_route_directions
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from core.config import settings
from core.storage.factory import get_places_cache, get_places_call_limiter
from tools.contracts import ToolResult
from tools.places_guardrails import PlacesCallCapExceeded, cache_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint constants
# ---------------------------------------------------------------------------

_PLACES_BASE = "https://places.googleapis.com/v1"
_ROUTES_ENDPOINT = "https://routes.googleapis.com/directions/v2:computeRoutes"

# Explicit per-phase timeout rather than one flat float. Both Google surfaces
# normally reply well under 2 seconds:
#   connect=3.0 — DNS + TCP + TLS handshake; a cold path but should never
#                 legitimately take longer than this against Google's edge.
#   write=3.0   — uploading our own (small, <2KB) JSON payload; generous
#                 headroom for a slow uplink without masking a hung socket.
#   read=8.0    — waiting on Google's response body; the dominant phase on a
#                 slow-DNS or cold-start day, so it gets the largest share.
#   pool=3.0    — waiting for a free connection out of httpx's pool.
# Worst case ~12s end-to-end, matching the previous flat ceiling, but each
# phase now fails on its own schedule instead of one undifferentiated timer.
#
# Applied PER-REQUEST (as the `timeout=` kwarg on each client.post/get call
# in _post_json/_get_json), not just as the fallback httpx.AsyncClient's
# constructor default. apps/turtle_server.py always injects a shared
# http_client built as a plain httpx.AsyncClient() with no timeout of its
# own, so relying on the constructor-time default alone would make these
# tuned values dead code in the deployed app — every Places/Routes call
# would silently fall back to httpx's generic default instead. httpx
# supports overriding a client's timeout per request, which works whether or
# not a client was injected, so that's what's used here.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=3.0, read=8.0, write=3.0, pool=3.0)

# Cache TTLs (S-7.9): search results churn faster (open/closed, new listings)
# than a single place's static details, so search gets a shorter window.
_SEARCH_CACHE_TTL_S = 600  # 10 minutes
_DETAILS_CACHE_TTL_S = 3600  # 1 hour

# Google's Place IDs are opaque, base64url-style tokens — see
# https://developers.google.com/maps/documentation/places/web-service/place-id
# Every documented/example ID (e.g. "ChIJN1t_tDeuEmsRUsoyG83frY4") uses only
# ASCII letters, digits, underscore and hyphen. We accept exactly that set
# (plus the "places/" resource-name prefix, stripped before this check) and
# reject everything else — in particular "/", "?", "#", and whitespace,
# which would otherwise let a model-supplied value redirect the request to a
# different path or inject extra query/fragment components once
# interpolated into a URL.
_PLACE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,255}$")

# Field masks control cost and payload size. Keep these tight — Google bills
# text search by SKU tier, and asking for atmosphere fields on every hit
# escalates every reply.
_TEXT_SEARCH_FIELD_MASK = (
    "places.id,"
    "places.displayName,"
    "places.formattedAddress,"
    "places.shortFormattedAddress,"
    "places.location,"
    "places.googleMapsUri,"
    "places.websiteUri,"
    "places.internationalPhoneNumber,"
    "places.nationalPhoneNumber,"
    "places.rating,"
    "places.userRatingCount,"
    "places.priceLevel,"
    "places.currentOpeningHours.openNow,"
    "places.currentOpeningHours.weekdayDescriptions,"
    "places.primaryTypeDisplayName,"
    "places.types,"
    "places.businessStatus"
)

_PLACE_DETAILS_FIELD_MASK = (
    "id,"
    "displayName,"
    "formattedAddress,"
    "shortFormattedAddress,"
    "location,"
    "googleMapsUri,"
    "websiteUri,"
    "internationalPhoneNumber,"
    "nationalPhoneNumber,"
    "rating,"
    "userRatingCount,"
    "priceLevel,"
    "currentOpeningHours.openNow,"
    "currentOpeningHours.weekdayDescriptions,"
    "regularOpeningHours.weekdayDescriptions,"
    "primaryTypeDisplayName,"
    "types,"
    "businessStatus,"
    "editorialSummary"
)

_ROUTES_FIELD_MASK = (
    "routes.distanceMeters,"
    "routes.duration,"
    "routes.staticDuration,"
    "routes.polyline.encodedPolyline,"
    "routes.legs.startLocation,"
    "routes.legs.endLocation,"
    "routes.legs.startAddress,"
    "routes.legs.endAddress,"
    "routes.description,"
    "routes.warnings"
)

_SUPPORTED_TRAVEL_MODES = {"DRIVE", "WALK", "BICYCLE", "TRANSIT", "TWO_WHEELER"}


# ---------------------------------------------------------------------------
# Arg schemas (also declared in tools/contracts.py for pydantic-ai registration)
# ---------------------------------------------------------------------------


class FindPlaceArgs(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=300,
        description=(
            "Natural-language place query, e.g. 'Mercure Hotel Dubai', "
            "'best sushi near Shibuya', 'Kempegowda International Airport'. "
            "Include the city or landmark for disambiguation when the user gave one."
        ),
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="How many candidate places to return (1-10).",
    )
    location_bias: str = Field(
        default="",
        description=(
            "Optional biasing hint — a city, neighbourhood, or 'lat,lng' pair "
            "the search should prefer results near. Leave empty for a global search."
        ),
    )


class PlaceDetailsArgs(BaseModel):
    place_id: str = Field(
        min_length=1,
        description=(
            "Google Places place_id, exactly as returned by find_place. "
            "Do NOT invent IDs — only use one you've seen in a prior tool result."
        ),
    )


class GetDirectionsArgs(BaseModel):
    origin: str = Field(
        min_length=2,
        description=(
            "Starting address, place name, or 'lat,lng' pair. "
            "E.g. 'Bengaluru Airport', 'Times Square NYC', '12.9716,77.5946'."
        ),
    )
    destination: str = Field(
        min_length=2,
        description="Ending address, place name, or 'lat,lng' pair.",
    )
    travel_mode: str = Field(
        default="DRIVE",
        description=(
            "One of DRIVE, WALK, BICYCLE, TRANSIT, TWO_WHEELER. Default DRIVE."
        ),
    )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class PlaceSummary(BaseModel):
    place_id: str
    name: str
    address: str
    maps_url: str
    website: str = ""
    phone: str = ""
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    price_level: str = ""
    open_now: Optional[bool] = None
    hours_today: str = ""
    category: str = ""
    business_status: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class FindPlaceResult(BaseModel):
    query: str
    results: list[PlaceSummary]


class PlaceDetailsResult(BaseModel):
    place: PlaceSummary
    weekly_hours: list[str] = Field(default_factory=list)
    summary: str = ""


class DirectionsResult(BaseModel):
    origin: str
    destination: str
    travel_mode: str
    distance_meters: int
    distance_text: str
    duration_seconds: int
    duration_text: str
    maps_url: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _api_key() -> Optional[str]:
    """Read the configured Google Maps API key, unwrapping the SecretStr."""
    key = settings.google_maps_api_key
    if key is None:
        return None
    try:
        return key.get_secret_value().strip() or None
    except Exception:
        # Some test paths pass a plain string.
        text = str(key).strip()
        return text or None


def _credentials_missing_result(kind: str) -> ToolResult[Any]:
    return ToolResult.upstream_error(
        message=(
            "Google Maps API key is not configured. "
            "Set GOOGLE_MAPS_API_KEY in the environment and enable the "
            "Places API (New) plus Routes API in Google Cloud."
        ),
        code="credentials_missing",
        retryable=False,
    )


def _format_distance(meters: int) -> str:
    if meters < 1000:
        return f"{meters} m"
    km = meters / 1000.0
    if km < 10:
        return f"{km:.1f} km"
    return f"{int(round(km))} km"


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    return f"{minutes}m"


def _parse_iso_duration(value: str) -> int:
    """Routes API returns durations as ISO-8601 style '123s'. Parse the seconds int."""
    if not value:
        return 0
    text = value.strip().rstrip("s")
    try:
        return int(float(text))
    except ValueError:
        return 0


def _price_level_label(level: str) -> str:
    mapping = {
        "PRICE_LEVEL_FREE": "free",
        "PRICE_LEVEL_INEXPENSIVE": "$",
        "PRICE_LEVEL_MODERATE": "$$",
        "PRICE_LEVEL_EXPENSIVE": "$$$",
        "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
    }
    return mapping.get(level, "")


def _summarize_place(raw: dict) -> PlaceSummary:
    """Reduce a Places-API record to the trimmed shape we surface to the LLM."""
    name = ((raw.get("displayName") or {}).get("text") or "").strip()
    address = (
        raw.get("shortFormattedAddress")
        or raw.get("formattedAddress")
        or ""
    ).strip()
    hours_today = ""
    open_now: Optional[bool] = None
    hours = raw.get("currentOpeningHours") or {}
    if isinstance(hours, dict):
        open_now = hours.get("openNow")
        weekday = hours.get("weekdayDescriptions") or []
        # Show today's line only — the LLM can call place_details for the week.
        if weekday:
            hours_today = weekday[0]
    location = raw.get("location") or {}
    lat = location.get("latitude") if isinstance(location, dict) else None
    lng = location.get("longitude") if isinstance(location, dict) else None
    category = ((raw.get("primaryTypeDisplayName") or {}).get("text") or "").strip()
    if not category:
        types = raw.get("types") or []
        if types:
            category = types[0].replace("_", " ")
    return PlaceSummary(
        place_id=raw.get("id") or "",
        name=name,
        address=address,
        maps_url=(raw.get("googleMapsUri") or "").strip(),
        website=(raw.get("websiteUri") or "").strip(),
        phone=(
            raw.get("internationalPhoneNumber")
            or raw.get("nationalPhoneNumber")
            or ""
        ).strip(),
        rating=raw.get("rating"),
        rating_count=raw.get("userRatingCount"),
        price_level=_price_level_label(raw.get("priceLevel") or ""),
        open_now=open_now,
        hours_today=hours_today,
        category=category,
        business_status=(raw.get("businessStatus") or "").strip(),
        latitude=lat,
        longitude=lng,
    )


def _render_place_summary_line(place: PlaceSummary) -> str:
    """Render one place as a compact human-readable block for the LLM."""
    lines: list[str] = [f"{place.name}"]
    if place.category:
        lines[0] += f"  ({place.category})"
    if place.address:
        lines.append(f"  Address: {place.address}")
    if place.maps_url:
        lines.append(f"  Maps: {place.maps_url}")
    if place.website:
        lines.append(f"  Website: {place.website}")
    if place.phone:
        lines.append(f"  Phone: {place.phone}")
    if place.rating is not None:
        count = place.rating_count or 0
        lines.append(f"  Rating: {place.rating} ({count} reviews)")
    if place.price_level:
        lines.append(f"  Price: {place.price_level}")
    if place.open_now is True:
        lines.append("  Open now: yes")
    elif place.open_now is False:
        lines.append("  Open now: no")
    if place.hours_today:
        lines.append(f"  Hours: {place.hours_today}")
    if place.business_status and place.business_status != "OPERATIONAL":
        lines.append(f"  Status: {place.business_status}")
    return "\n".join(lines)


def _http_error_result(
    status_code: int,
    body_text: str,
    context: str,
) -> ToolResult[Any]:
    """Map an HTTP error from Google to our ToolResult envelope.

    Fixed, non-leaky messages go to the model; Google's raw response body
    goes ONLY to the server-side log (status + which call + trimmed body),
    never into anything returned up the call stack. Existing ToolResult
    codes and retryable flags are unchanged — the calling layer depends on
    both.
    """
    trimmed_for_log = (body_text or "").strip()
    if len(trimmed_for_log) > 2000:
        trimmed_for_log = trimmed_for_log[:2000] + "…(truncated)"
    logger.warning(
        "Google Places/Routes API error: call=%s status=%s body=%s",
        context,
        status_code,
        trimmed_for_log,
    )
    if status_code == 429:
        return ToolResult.rate_limited(
            retry_after_ms=60_000,
            message=(
                f"{context} is being rate-limited by Google right now. "
                "Try again in about a minute."
            ),
        )
    if status_code in (401, 403):
        return ToolResult.upstream_error(
            message=(
                f"{context} rejected the API credentials (HTTP {status_code}). "
                "This is a server configuration issue, not something the user "
                "can fix — check GOOGLE_MAPS_API_KEY, the enabled APIs, and key "
                "restrictions."
            ),
            code="auth_failed",
            retryable=False,
        )
    if status_code == 400:
        return ToolResult.invalid(
            message=(
                f"{context} rejected the request as malformed (HTTP 400). If "
                "this was a place_details call, the place_id was probably "
                "invalid or invented — only use place_ids returned by "
                "find_place."
            ),
            code="bad_request",
        )
    return ToolResult.upstream_error(
        message=(
            f"{context} request failed (HTTP {status_code}). This is usually "
            "transient."
        ),
        code=f"http_{status_code}",
        retryable=status_code >= 500,
    )


def _validate_and_quote_place_id(raw: str) -> Optional[str]:
    """Validate a model-supplied place_id and URL-quote it for the request
    PATH (unlike find_place/get_directions, whose model-controlled inputs
    only ever land in a JSON body, never a URL). Returns the quoted path
    segment, or None if *raw* fails validation — the caller must reject
    before issuing any request in that case. Validation is not a substitute
    for quoting and quoting is not a substitute for validation; both run.
    """
    pid = (raw or "").strip()
    if pid.startswith("places/"):
        pid = pid[len("places/"):]
    if not pid or not _PLACE_ID_RE.match(pid):
        return None
    return quote(pid, safe="")


def _ms_until_next_utc_midnight() -> int:
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(1, int((tomorrow - now).total_seconds() * 1000))


def _call_cap_result(context: str) -> ToolResult[Any]:
    """Fixed refusal returned when a user has exhausted their daily Places/
    Routes call cap. Modeled as rate_limited (retryable, with a retry
    window) rather than upstream_error — this is a soft, time-boxed budget,
    not a hard failure — matching how a 429 from Google itself is surfaced.
    """
    return ToolResult.rate_limited(
        retry_after_ms=_ms_until_next_utc_midnight(),
        message=(
            f"The daily limit for Google Places/Directions lookups ({context}) "
            "has been reached for this account. Try again after midnight UTC."
        ),
    )


async def _post_json(
    url: str,
    payload: dict,
    field_mask: str,
    *,
    context: str,
    http_client: Optional[httpx.AsyncClient] = None,
) -> tuple[Optional[dict], Optional[ToolResult[Any]]]:
    """POST to a Google API. Returns (parsed_json, None) on success, or (None, error)."""
    api_key = _api_key()
    if not api_key:
        return None, _credentials_missing_result("places")
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": field_mask,
    }
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient()
    try:
        resp = await client.post(
            url, json=payload, headers=headers, timeout=_DEFAULT_TIMEOUT
        )
    except httpx.TimeoutException:
        return None, ToolResult.upstream_error(
            message="Google API timed out.", code="timeout", retryable=True
        )
    except httpx.HTTPError as exc:
        # Never surface str(exc) to the model — an httpx.ConnectError etc.
        # can carry an internal hostname or an API-key-shaped string in its
        # message. Fixed, non-leaky message to the model; full detail to the
        # server-side log only, mirroring _http_error_result's HTTP-status
        # path.
        logger.warning("Google API network error: call=%s error=%s", context, exc)
        return None, ToolResult.upstream_error(
            message="Google API network error.", code="network_error", retryable=True
        )
    finally:
        if owns_client:
            await client.aclose()

    if resp.status_code >= 400:
        return None, _http_error_result(resp.status_code, resp.text, context)
    try:
        return resp.json(), None
    except ValueError:
        return None, ToolResult.upstream_error(
            message="Google API returned non-JSON response.", code="bad_response",
        )


async def _get_json(
    url: str,
    field_mask: str,
    *,
    context: str,
    http_client: Optional[httpx.AsyncClient] = None,
) -> tuple[Optional[dict], Optional[ToolResult[Any]]]:
    api_key = _api_key()
    if not api_key:
        return None, _credentials_missing_result("places")
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": field_mask,
    }
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient()
    try:
        resp = await client.get(url, headers=headers, timeout=_DEFAULT_TIMEOUT)
    except httpx.TimeoutException:
        return None, ToolResult.upstream_error(
            message="Google API timed out.", code="timeout", retryable=True
        )
    except httpx.HTTPError as exc:
        # See _post_json's identical branch: str(exc) is never model-safe.
        logger.warning("Google API network error: call=%s error=%s", context, exc)
        return None, ToolResult.upstream_error(
            message="Google API network error.", code="network_error", retryable=True
        )
    finally:
        if owns_client:
            await client.aclose()

    if resp.status_code >= 400:
        return None, _http_error_result(resp.status_code, resp.text, context)
    try:
        return resp.json(), None
    except ValueError:
        return None, ToolResult.upstream_error(
            message="Google API returned non-JSON response.", code="bad_response",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def find_place(
    args: FindPlaceArgs,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
    user_id: Optional[str] = None,
) -> ToolResult[FindPlaceResult]:
    """Text search: return the top matching places for a natural-language query.

    user_id scopes the per-user daily call cap only (tools/places_guardrails.py)
    — the response cache itself is intentionally global, not per-user. Wired
    through from apps/turtle_server.py's tool-registration closure
    (user_id=ctx.deps.user_id), so the cap keys on the real caller in
    production. The "anonymous" fallback below only fires for a caller that
    genuinely has no user id (e.g. a direct call with user_id left unset,
    such as in tests) — it is not the default production path.
    """
    query = args.query.strip()
    if not query:
        return ToolResult.invalid("query must not be empty")

    bias = (args.location_bias or "").strip()
    cache = get_places_cache()
    key = cache_key(
        "find_place",
        {"query": query.lower(), "max_results": args.max_results, "location_bias": bias.lower()},
    )
    cached = cache.get(key)
    if cached is not None:
        places_raw = cached.get("places") or []
        if not places_raw:
            return ToolResult.empty("No places matched that query.")
        summaries = [_summarize_place(p) for p in places_raw]
        return ToolResult.ok(FindPlaceResult(query=query, results=summaries))

    limiter = get_places_call_limiter()
    try:
        limiter.check_and_record(user_id or "anonymous")
    except PlacesCallCapExceeded:
        return _call_cap_result("find_place")

    payload: dict[str, Any] = {
        "textQuery": query,
        "pageSize": args.max_results,
    }
    if bias:
        # Callers can pass either 'lat,lng' or a plain place name. We append
        # the bias into the free-text query in the plain-name case (Google's
        # circular biasing needs numeric coordinates + a radius, which is not
        # what LLM callers typically have).
        parts = [p.strip() for p in bias.split(",")]
        numeric = len(parts) == 2 and all(_looks_numeric(p) for p in parts)
        if numeric:
            lat, lng = float(parts[0]), float(parts[1])
            payload["locationBias"] = {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": 20_000.0,
                }
            }
        else:
            payload["textQuery"] = f"{query} near {bias}"

    data, err = await _post_json(
        f"{_PLACES_BASE}/places:searchText",
        payload,
        _TEXT_SEARCH_FIELD_MASK,
        context="find_place",
        http_client=http_client,
    )
    if err is not None:
        return err  # type: ignore[return-value]
    cache.set(key, data or {}, _SEARCH_CACHE_TTL_S)
    places_raw = (data or {}).get("places") or []
    if not places_raw:
        return ToolResult.empty("No places matched that query.")
    summaries = [_summarize_place(p) for p in places_raw]
    return ToolResult.ok(FindPlaceResult(query=query, results=summaries))


async def place_details(
    args: PlaceDetailsArgs,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
    user_id: Optional[str] = None,
) -> ToolResult[PlaceDetailsResult]:
    """Fetch full details for a specific place_id from Places API (New).

    See find_place's docstring for the user_id note — same wiring, same
    "anonymous" fallback.
    """
    pid = args.place_id.strip()
    if not pid:
        return ToolResult.invalid("place_id must not be empty")

    # Validate BEFORE quoting and before any request fires — quoting alone
    # would still let a rejected character set through percent-encoded;
    # validation alone would still leave an unescaped value in the URL path.
    # Places API (New) accepts either the bare id or 'places/<id>'; the
    # 'places/' prefix is stripped inside the validator so a second '/'
    # can't sneak through as part of the id itself.
    quoted_path_id = _validate_and_quote_place_id(pid)
    if quoted_path_id is None:
        return ToolResult.invalid(
            "place_id contains characters that are never part of a real "
            "Google place_id (only letters, digits, '_' and '-' are valid, "
            "with an optional 'places/' prefix). This usually means the "
            "place_id was invented rather than copied from a prior "
            "find_place result.",
            code="invalid_place_id",
        )

    cache = get_places_cache()
    ckey = cache_key("place_details", {"place_id": quoted_path_id})
    cached = cache.get(ckey)
    if cached is not None:
        data = cached
    else:
        limiter = get_places_call_limiter()
        try:
            limiter.check_and_record(user_id or "anonymous")
        except PlacesCallCapExceeded:
            return _call_cap_result("place_details")

        data, err = await _get_json(
            f"{_PLACES_BASE}/places/{quoted_path_id}",
            _PLACE_DETAILS_FIELD_MASK,
            context="place_details",
            http_client=http_client,
        )
        if err is not None:
            return err  # type: ignore[return-value]
        cache.set(ckey, data or {}, _DETAILS_CACHE_TTL_S)

    if not data:
        return ToolResult.empty("No details returned for that place_id.")
    summary = _summarize_place(data)
    weekly: list[str] = []
    for hours_key in ("currentOpeningHours", "regularOpeningHours"):
        block = data.get(hours_key)
        if isinstance(block, dict):
            weekday = block.get("weekdayDescriptions") or []
            if weekday and not weekly:
                weekly = list(weekday)
    editorial = ""
    ed_block = data.get("editorialSummary")
    if isinstance(ed_block, dict):
        editorial = (ed_block.get("text") or "").strip()
    return ToolResult.ok(
        PlaceDetailsResult(place=summary, weekly_hours=weekly, summary=editorial)
    )


async def get_directions(
    args: GetDirectionsArgs,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
    user_id: Optional[str] = None,
) -> ToolResult[DirectionsResult]:
    """Compute a route between two locations via the Routes API.

    Not cached (routes are traffic-sensitive and the brief specifies TTLs
    only for search/details), but still counted against the per-user daily
    call cap — it is the same billed Google surface as the other two tools.
    See find_place's docstring for the user_id caveat.
    """
    origin = args.origin.strip()
    destination = args.destination.strip()
    if not origin or not destination:
        return ToolResult.invalid("origin and destination must not be empty")
    mode = (args.travel_mode or "DRIVE").strip().upper()
    if mode not in _SUPPORTED_TRAVEL_MODES:
        return ToolResult.invalid(
            f"travel_mode must be one of: {', '.join(sorted(_SUPPORTED_TRAVEL_MODES))}"
        )

    limiter = get_places_call_limiter()
    try:
        limiter.check_and_record(user_id or "anonymous")
    except PlacesCallCapExceeded:
        return _call_cap_result("get_directions")

    payload: dict[str, Any] = {
        "origin": _routes_waypoint(origin),
        "destination": _routes_waypoint(destination),
        "travelMode": mode,
        "computeAlternativeRoutes": False,
        "languageCode": "en",
        "units": "METRIC",
    }
    # Traffic-aware routing is only meaningful for driving; the API rejects
    # this preference for WALK/BICYCLE etc.
    if mode in {"DRIVE", "TWO_WHEELER"}:
        payload["routingPreference"] = "TRAFFIC_AWARE"

    data, err = await _post_json(
        _ROUTES_ENDPOINT,
        payload,
        _ROUTES_FIELD_MASK,
        context="get_directions",
        http_client=http_client,
    )
    if err is not None:
        return err  # type: ignore[return-value]
    routes = (data or {}).get("routes") or []
    if not routes:
        return ToolResult.empty(
            "No route found between those points. Check spelling or try a nearer landmark."
        )
    route = routes[0]
    meters = int(route.get("distanceMeters") or 0)
    duration_s = _parse_iso_duration(route.get("duration") or "")
    warnings = list(route.get("warnings") or [])
    maps_url = _build_directions_maps_url(origin, destination, mode)
    return ToolResult.ok(
        DirectionsResult(
            origin=origin,
            destination=destination,
            travel_mode=mode,
            distance_meters=meters,
            distance_text=_format_distance(meters),
            duration_seconds=duration_s,
            duration_text=_format_duration(duration_s),
            maps_url=maps_url,
            warnings=warnings,
        )
    )


# ---------------------------------------------------------------------------
# Rendering helpers used by the tool-registration layer
# ---------------------------------------------------------------------------


def render_find_place(result: FindPlaceResult) -> str:
    blocks = [_render_place_summary_line(p) for p in result.results]
    header = f"Top results for {result.query!r}:"
    return header + "\n\n" + "\n\n".join(blocks)


def render_place_details(result: PlaceDetailsResult) -> str:
    lines = [_render_place_summary_line(result.place)]
    if result.weekly_hours:
        lines.append("  Weekly hours:")
        for line in result.weekly_hours:
            lines.append(f"    - {line}")
    if result.summary:
        lines.append(f"  Summary: {result.summary}")
    return "\n".join(lines)


def render_directions(result: DirectionsResult) -> str:
    lines = [
        f"Route ({result.travel_mode.lower()}): {result.origin} -> {result.destination}",
        f"  Distance: {result.distance_text} ({result.distance_meters} m)",
        f"  Duration: {result.duration_text}",
        f"  Maps: {result.maps_url}",
    ]
    if result.warnings:
        for w in result.warnings:
            lines.append(f"  Warning: {w}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _routes_waypoint(value: str) -> dict:
    """Build a Routes API waypoint from a free-text address or 'lat,lng'."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) == 2 and all(_looks_numeric(p) for p in parts):
        lat, lng = float(parts[0]), float(parts[1])
        return {"location": {"latLng": {"latitude": lat, "longitude": lng}}}
    return {"address": value}


def _build_directions_maps_url(origin: str, destination: str, mode: str) -> str:
    """Compose a clickable Google Maps directions URL for the pair.

    Uses the public search endpoint, so the URL works for any recipient (no
    account required) and honours whatever travel mode Maps derives from the
    'travelmode' query param.
    """
    from urllib.parse import quote_plus

    mode_map = {
        "DRIVE": "driving",
        "WALK": "walking",
        "BICYCLE": "bicycling",
        "TRANSIT": "transit",
        "TWO_WHEELER": "driving",
    }
    tm = mode_map.get(mode, "driving")
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={quote_plus(origin)}"
        f"&destination={quote_plus(destination)}"
        f"&travelmode={tm}"
    )
