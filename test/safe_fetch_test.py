"""
test/safe_fetch_test.py

SSRF-hardening tests for tools/url_tools/safe_fetch.py (WP1.A / S-7.1 /
ledger 1a.1).

Fully offline: IP-literal hosts are validated without any DNS call, and
every hostname-based case monkeypatches socket.getaddrinfo with a fixed
fake result — nothing here touches the real network or real DNS.
"""
from __future__ import annotations

import asyncio
import socket

import pytest

from tools.url_tools.safe_fetch import (
    MAX_BODY_BYTES,
    REFUSAL_MESSAGE,
    UnsafeUrlError,
    safe_get,
    validate_public_url,
)


def _addrinfo(*ips):
    """Build a fake socket.getaddrinfo() return value for the given IP strings."""
    result = []
    for ip in ips:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        result.append((family, socket.SOCK_STREAM, 6, "", (ip, 0)))
    return result


# ── literal-IP hosts (no DNS involved at all) ────────────────────────────

class TestLiteralIpRejection:
    def test_aws_metadata_ip_refused(self):
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://169.254.169.254/latest/meta-data/")

    def test_loopback_refused(self):
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://127.0.0.1:8765/")

    def test_ipv6_loopback_literal_refused(self):
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://[::1]/")

    def test_private_10_refused(self):
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://10.0.0.1/")

    def test_private_192_refused(self):
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://192.168.1.1/")

    def test_private_172_refused(self):
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://172.16.0.1/")

    def test_ipv4_mapped_ipv6_loopback_refused(self):
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://[::ffff:127.0.0.1]/")


# ── scheme and port ───────────────────────────────────────────────────────

class TestSchemeAndPort:
    def test_file_scheme_refused(self):
        with pytest.raises(UnsafeUrlError):
            validate_public_url("file:///etc/passwd")

    def test_gopher_scheme_refused(self):
        with pytest.raises(UnsafeUrlError):
            validate_public_url("gopher://x/")

    def test_explicit_disallowed_port_refused(self, monkeypatch):
        monkeypatch.setattr(
            socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34")
        )
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://example.com:22/")


# ── control characters (must be rejected before any parsing/DNS) ─────────

class TestControlCharacterRejection:
    def test_null_byte_refused(self):
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://example.com\x00.evil.com/")

    def test_newline_refused(self):
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://example.com/\nattack")

    def test_tab_refused(self):
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://example.com/\tattack")

    def test_carriage_return_refused(self):
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://example.com/\rattack")


# ── malformed URLs must not leak a raw exception string ──────────────────

class TestMalformedUrlRefusal:
    def test_unbalanced_ipv6_bracket_refused_with_fixed_message(self):
        with pytest.raises(UnsafeUrlError) as exc_info:
            validate_public_url("http://[::1/")
        assert str(exc_info.value) == REFUSAL_MESSAGE

    def test_bare_open_bracket_refused_with_fixed_message(self):
        with pytest.raises(UnsafeUrlError) as exc_info:
            validate_public_url("http://[")
        assert str(exc_info.value) == REFUSAL_MESSAGE


# ── hostname resolution (DNS mocked) ──────────────────────────────────────

class TestHostnameResolution:
    def test_localhost_refused(self, monkeypatch):
        monkeypatch.setattr(
            socket, "getaddrinfo", lambda *a, **k: _addrinfo("127.0.0.1")
        )
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://localhost/")

    def test_mixed_public_private_addresses_refused(self, monkeypatch):
        """A hostname resolving to several addresses where any one is
        private must be refused — not just checked against the first."""
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *a, **k: _addrinfo("93.184.216.34", "10.1.2.3"),
        )
        with pytest.raises(UnsafeUrlError):
            validate_public_url("http://mixed.example.com/")

    def test_ordinary_public_url_allowed(self, monkeypatch):
        monkeypatch.setattr(
            socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34")
        )
        validate_public_url("http://example.com/")  # must not raise


# ── redirect chain (safe_get) ─────────────────────────────────────────────

class _FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _FakeResponse:
    def __init__(self, status_code, headers, body=b""):
        self.status_code = status_code
        self.headers = headers
        self._body = body
        self.is_redirect = 300 <= status_code < 400 and "location" in headers
        self.bytes_yielded = 0

    async def aiter_bytes(self):
        chunk_size = 8192
        for i in range(0, len(self._body), chunk_size):
            chunk = self._body[i:i + chunk_size]
            self.bytes_yielded += len(chunk)
            yield chunk


class _FakeClient:
    """Stand-in for httpx.AsyncClient exposing only .stream(), matching
    what safe_get actually calls — no real network involved."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requested_urls = []

    def stream(self, method, url, headers=None, timeout=None, follow_redirects=None):
        assert follow_redirects is False, "safe_get must pass follow_redirects=False"
        self.requested_urls.append(url)
        resp = self._responses.pop(0)
        return _FakeStreamCtx(resp)


class TestRedirectChain:
    def test_three_hops_allowed(self, monkeypatch):
        monkeypatch.setattr(
            socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34")
        )
        responses = [
            _FakeResponse(302, {"location": "http://example.com/2"}),
            _FakeResponse(302, {"location": "http://example.com/3"}),
            _FakeResponse(302, {"location": "http://example.com/4"}),
            _FakeResponse(200, {"content-type": "text/html"}, b"final content"),
        ]
        client = _FakeClient(responses)
        result = asyncio.run(safe_get(client, "http://example.com/1"))
        assert result.status_code == 200
        assert result.content == b"final content"
        assert len(client.requested_urls) == 4

    def test_fourth_hop_refused(self, monkeypatch):
        monkeypatch.setattr(
            socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34")
        )
        responses = [
            _FakeResponse(302, {"location": "http://example.com/2"}),
            _FakeResponse(302, {"location": "http://example.com/3"}),
            _FakeResponse(302, {"location": "http://example.com/4"}),
            _FakeResponse(302, {"location": "http://example.com/5"}),
        ]
        client = _FakeClient(responses)
        with pytest.raises(UnsafeUrlError):
            asyncio.run(safe_get(client, "http://example.com/1"))

    def test_redirect_to_loopback_refused_at_hop(self, monkeypatch):
        def fake_getaddrinfo(host, *a, **k):
            assert host == "example.com"
            return _addrinfo("93.184.216.34")

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        responses = [
            _FakeResponse(302, {"location": "http://127.0.0.1/admin"}),
        ]
        client = _FakeClient(responses)
        with pytest.raises(UnsafeUrlError):
            asyncio.run(safe_get(client, "http://example.com/1"))

    def test_relative_redirect_location_resolved_before_validation(self, monkeypatch):
        monkeypatch.setattr(
            socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34")
        )
        responses = [
            _FakeResponse(302, {"location": "/next-page"}),
            _FakeResponse(200, {"content-type": "text/html"}, b"ok"),
        ]
        client = _FakeClient(responses)
        result = asyncio.run(safe_get(client, "http://example.com/1"))
        assert result.status_code == 200
        assert client.requested_urls[1] == "http://example.com/next-page"


# ── body cap enforced during streaming ────────────────────────────────────

class TestBodyCap:
    def test_body_capped_during_streaming(self, monkeypatch):
        monkeypatch.setattr(
            socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34")
        )
        big_body = b"x" * (MAX_BODY_BYTES + 50_000)
        responses = [_FakeResponse(200, {"content-type": "text/plain"}, big_body)]
        client = _FakeClient(responses)

        result = asyncio.run(safe_get(client, "http://example.com/big"))

        assert len(result.content) == MAX_BODY_BYTES
        assert result.truncated is True
        # Proves the cap is enforced *during* streaming, not after buffering
        # the whole body: the fake response never yielded all of its bytes
        # because iteration stopped as soon as the cap was hit.
        fake_resp = responses[0]
        assert fake_resp.bytes_yielded < len(big_body)

    def test_small_body_not_truncated(self, monkeypatch):
        monkeypatch.setattr(
            socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34")
        )
        small_body = b"hello world"
        responses = [_FakeResponse(200, {"content-type": "text/plain"}, small_body)]
        client = _FakeClient(responses)

        result = asyncio.run(safe_get(client, "http://example.com/small"))

        assert result.content == small_body
        assert result.truncated is False
