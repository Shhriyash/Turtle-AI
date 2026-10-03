"""httpx's INFO request log must never carry a URL credential.

Telegram puts the bot token in the URL path (/bot<TOKEN>/getMe) and httpx logs
the full URL as 'HTTP Request: GET <url> "HTTP/1.1 200 OK"'. In production the
host runtime sets the root logger to INFO, so the token reached the log stream.
These tests use the real installed httpx against a real local socket.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx

import apps.turtle_server as srv  # noqa: F401  (import installs the log scrubber)

FAKE_TOKEN = "123456789:AAFAKE-not-a-real-token_xyz"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _real_request(path: str) -> list[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cap = _Capture()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(cap)
    root.setLevel(logging.INFO)  # what the production host does
    try:
        async def go():
            async with httpx.AsyncClient() as c:
                return await c.get(f"http://127.0.0.1:{server.server_port}{path}")
        r = asyncio.run(go())
        assert r.status_code == 200
    finally:
        root.removeHandler(cap)
        root.setLevel(old_level)
        server.shutdown()
    return cap.lines


def test_httpx_request_log_does_not_contain_path_credential():
    lines = _real_request(f"/bot{FAKE_TOKEN}/getMe")
    joined = "\n".join(lines)
    assert "HTTP Request" in joined, f"no httpx request line captured: {lines}"
    assert FAKE_TOKEN not in joined
    assert "AAFAKE" not in joined


def test_httpx_request_log_does_not_contain_query_credential():
    lines = _real_request("/v1/x?api_key=SECRETKEY999&key=SECRETKEY999")
    joined = "\n".join(lines)
    assert "HTTP Request" in joined
    assert "SECRETKEY999" not in joined


def test_request_is_still_observable_host_method_status():
    lines = _real_request(f"/bot{FAKE_TOKEN}/getMe")
    line = next(l for l in lines if "HTTP Request" in l)
    assert "GET" in line and "127.0.0.1" in line and "200 OK" in line


def test_redact_urls_helper():
    out = srv._redact_urls(
        f'HTTP Request: GET https://user:pw@api.telegram.org/bot{FAKE_TOKEN}/getMe?x=1 "HTTP/1.1 200 OK"'
    )
    assert FAKE_TOKEN not in out and "pw@" not in out and "x=1" not in out
    assert "api.telegram.org" in out and "200 OK" in out
