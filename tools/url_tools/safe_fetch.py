"""
tools/url_tools/safe_fetch.py

SSRF hardening for every model-initiated outbound fetch (ledger 1a.1 / S-7.1).

`validate_public_url` refuses any URL whose scheme, port or resolved
address is not a plain, public, globally-routable http(s) endpoint.
`safe_get` layers a redirect-safe, body-capped fetch on top of it for
direct httpx callers: `follow_redirects=False`, with up to
MAX_REDIRECTS hops followed manually, re-validating the full URL on
every hop.

DNS rebinding — the resolved address changing between validation and the
actual TCP connect httpx makes internally — is an accepted residual risk
per the ledger decision. We deliberately do not pin the validated IP for
the socket connect (that would need a custom httpx transport); a future
hardening pass could add one, but it is out of scope here.
"""
from __future__ import annotations

import ipaddress
import socket
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import httpx

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {80, 443}
MAX_REDIRECTS = 3
MAX_BODY_BYTES = 256 * 1024  # 256 KB

# Fixed, non-leaky message surfaced to the model/caller. Never interpolate
# the resolved IP, hostname or validation reason into this string.
REFUSAL_MESSAGE = (
    "This URL could not be fetched: it points to a restricted or "
    "non-public network address."
)


class UnsafeUrlError(Exception):
    """Raised when a URL fails SSRF validation (scheme, port or address)."""


def _is_disallowed_address(addr) -> bool:
    """
    Return True if `addr` (an ipaddress.IPv4Address/IPv6Address) must be
    refused.

    We check `is_global` *and* the individual loopback/link-local/private/
    multicast/reserved/unspecified flags explicitly, rather than trusting
    `is_global` alone. `is_global` is defined as "not global if it matches
    any special-purpose range", so in principle the individual checks are
    redundant with it — but that definition lives in a registry that has
    grown across Python versions, and a category we haven't anticipated
    (or a future stdlib change) could leave a gap between "not global" and
    "actually safe". Checking both is belt-and-braces so a regression in
    either definition doesn't silently reopen the SSRF hole.
    """
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        # IPv4-mapped IPv6 (::ffff:127.0.0.1 etc.) — validate the embedded
        # IPv4 address, otherwise these slip through IPv6-only checks.
        addr = addr.ipv4_mapped
    return (
        not addr.is_global
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_private
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def _validate_host_addresses(host: str) -> None:
    """Resolve `host` (or parse it as an IP literal) and refuse any bad address."""
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        # Host is already an IP literal — validate directly, no DNS round trip.
        if _is_disallowed_address(literal):
            raise UnsafeUrlError("host resolves to a non-public address")
        return

    try:
        # AF_UNSPEC resolves both A (IPv4) and AAAA (IPv6) records.
        infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"DNS resolution failed: {exc}") from exc

    if not infos:
        raise UnsafeUrlError("DNS resolution returned no addresses")

    # Validate *every* returned address, not just the first — a hostname
    # that resolves to a mix of public and private addresses is refused.
    for _family, _type, _proto, _canonname, sockaddr in infos:
        addr = ipaddress.ip_address(sockaddr[0])
        if _is_disallowed_address(addr):
            raise UnsafeUrlError("host resolves to a non-public address")


def validate_public_url(url: str) -> None:
    """
    Validate that `url` is http(s), on port 80/443 (explicit or implied),
    and resolves only to public/global addresses.

    Raises UnsafeUrlError on any violation. Callers must not surface the
    exception message to the model/user — use REFUSAL_MESSAGE instead.
    """
    parsed = urllib.parse.urlsplit(url)

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"scheme not allowed: {parsed.scheme!r}")

    try:
        host = parsed.hostname
    except ValueError as exc:
        raise UnsafeUrlError(f"unparseable host: {exc}") from exc
    if not host:
        raise UnsafeUrlError("URL has no host")

    try:
        port = parsed.port
    except ValueError as exc:
        raise UnsafeUrlError(f"unparseable port: {exc}") from exc
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80
    if port not in ALLOWED_PORTS:
        raise UnsafeUrlError(f"port not allowed: {port}")

    _validate_host_addresses(host)


@dataclass
class SafeResponse:
    """Minimal response shape returned by safe_get — not a real httpx.Response."""
    status_code: int
    headers: httpx.Headers
    content: bytes
    url: str
    truncated: bool

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


async def _read_capped(response: httpx.Response) -> tuple[bytes, bool]:
    """
    Read a streamed response body up to MAX_BODY_BYTES.

    Enforced during streaming (via aiter_bytes), not by reading the whole
    body into `response.text`/`response.content` and truncating afterwards
    — a cap applied post-hoc would already have pulled an arbitrarily large
    body into memory, which is exactly what the cap exists to prevent.

    On exceeding the cap we truncate-and-continue: we keep the bytes read
    so far, stop pulling in any more, and mark the result `truncated=True`
    rather than refusing the whole fetch. A legitimate page that happens to
    be slightly larger than the cap is still useful content for the
    caller; refusing outright would turn "large" into "unusable" for no
    security benefit, since we've already stopped reading further bytes
    the moment the cap is hit.
    """
    chunks: list[bytes] = []
    total = 0
    truncated = False
    async for chunk in response.aiter_bytes():
        remaining = MAX_BODY_BYTES - total
        if remaining <= 0:
            truncated = True
            break
        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            total += remaining
            truncated = True
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks), truncated


async def safe_get(
    client: httpx.AsyncClient,
    url: str,
    headers: Optional[dict] = None,
    timeout: float = 20.0,
) -> SafeResponse:
    """
    Validate `url`, fetch it with redirects disabled, and manually follow
    up to MAX_REDIRECTS hops, re-validating the full URL (scheme, port and
    resolved address) before following each one. A relative `Location` is
    resolved against the previous URL before validation. Exceeding
    MAX_REDIRECTS hops is a refusal (UnsafeUrlError), not a silent stop.

    Body is capped at MAX_BODY_BYTES, enforced while streaming.
    """
    current_url = url
    for hop in range(MAX_REDIRECTS + 1):
        validate_public_url(current_url)

        async with client.stream(
            "GET", current_url, headers=headers, timeout=timeout, follow_redirects=False
        ) as response:
            location = response.headers.get("location")
            is_redirect = getattr(response, "is_redirect", None)
            if is_redirect is None:
                is_redirect = 300 <= response.status_code < 400 and bool(location)

            if is_redirect and location:
                if hop >= MAX_REDIRECTS:
                    raise UnsafeUrlError("too many redirects")
                current_url = urllib.parse.urljoin(current_url, location)
                continue

            content, truncated = await _read_capped(response)
            return SafeResponse(
                status_code=response.status_code,
                headers=response.headers,
                content=content,
                url=current_url,
                truncated=truncated,
            )

    raise UnsafeUrlError("too many redirects")
