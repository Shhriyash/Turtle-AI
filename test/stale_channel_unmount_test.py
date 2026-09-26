"""
Tier 1 stale-surface cleanup — proves the blast radius of unmounting the
three closed channel routers (WhatsApp/Twilio, iMessage, Slack) by testing
*reachability* against the real `apps.turtle_server.app`, over HTTP.

The owner has stated only three channels are open: Telegram, Discord and the
web UI. WhatsApp, iMessage and Slack are closed by policy (not deprecated,
not removed) — their routers are commented out of `include_router` in
apps/turtle_server.py but the modules and imports remain in the tree.

This test is the control that makes the unmount safe to repeat: if any of
the three routers is ever remounted, this test fails, forcing that decision
to be deliberate rather than accidental.

Why HTTP reachability and not route-table introspection: an earlier version
of this test built its assertions by walking `app.routes` and reading a
`.path` attribute off each entry (`{getattr(r, "path", "") for r in
app.routes}`). Whether that comprehension actually sees a mounted sub-router's
paths depends on whether `FastAPI.include_router` flattens the child route
objects directly into `app.routes`, versus appending some other object for
the mount and leaving the children reachable only by walking into it. That is
an internal storage detail, not something FastAPI's public API promises, and
it is not guaranteed to be the same across FastAPI versions — a version where
the top-level `app.routes` comprehension can't see an included router's paths
at all turns `test_closed_channels_are_not_mounted` into a test that passes
whether or not WhatsApp/iMessage/Slack are mounted (and simultaneously fails
`test_open_channels_are_mounted`, which needs to see Telegram's and Discord's
real paths). Whether a request to a path is actually served (vs. 404s) is
the property this control is supposed to guarantee, is stable regardless of
how `include_router` happens to store things internally, and is exactly what
an accidental remount would change. So we ask the app the same way a real
caller (or an attacker) would: over `TestClient`, by HTTP status code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.turtle_server import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # `with TestClient(app)` runs the app's startup/shutdown lifespan.
    # apps/turtle_server.py's Discord/Telegram gateway startup hooks already
    # no-op under pytest (`if "pytest" in sys.modules: return`) specifically
    # so this doesn't open a real gateway connection every time a test enters
    # the app's lifespan — see the comments on _start_discord_gateway_hook /
    # _start_telegram_gateway_hook.
    with TestClient(app) as c:
        yield c


def test_closed_channels_are_not_mounted(client: TestClient):
    # Closed by policy: owner does not run these channels. An unmounted route
    # 404s — FastAPI has nothing registered at the path at all.
    assert client.post("/channels/whatsapp", json={}).status_code == 404
    assert client.post("/channels/imessage", json={}).status_code == 404
    assert client.post("/channels/slack/events", json={}).status_code == 404


def test_open_channels_are_mounted(client: TestClient):
    # Mounted routes are reachable: the request is routed to real handler
    # code, which then rejects an unsigned/unauthenticated request. A 404
    # here would mean nothing is registered at the path; a non-404 proves
    # the router is mounted and its auth check is the thing running.
    telegram_resp = client.post("/channels/telegram/webhook", json={})
    assert telegram_resp.status_code != 404
    assert telegram_resp.status_code == 401  # missing/wrong webhook secret token

    discord_resp = client.post("/channels/discord", json={})
    assert discord_resp.status_code != 404
    assert discord_resp.status_code == 401  # missing Ed25519 signature headers

    # Web surface.
    assert client.get("/").status_code == 200
    with client.websocket_connect("/ws"):
        pass
