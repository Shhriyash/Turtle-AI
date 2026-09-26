"""
Tier 1 stale-surface cleanup — proves the blast radius of unmounting the
three closed channel routers (WhatsApp/Twilio, iMessage, Slack) by
enumerating the *real* `apps.turtle_server.app` route table.

The owner has stated only three channels are open: Telegram, Discord and the
web UI. WhatsApp, iMessage and Slack are closed by policy (not deprecated,
not removed) — their routers are commented out of `include_router` in
apps/turtle_server.py but the modules and imports remain in the tree.

This test is the control that makes the unmount safe to repeat: if any of
the three routers is ever remounted, this test fails, forcing that decision
to be deliberate rather than accidental.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.turtle_server import app  # noqa: E402


def _route_paths() -> set[str]:
    return {getattr(r, "path", "") for r in app.routes}


def test_open_channels_are_mounted():
    paths = _route_paths()
    # Telegram webhook
    assert "/channels/telegram/webhook" in paths
    # Discord's two routes
    assert "/channels/discord" in paths
    assert "/channels/discord/process" in paths
    # Web surface
    assert "/" in paths
    assert "/ws" in paths


def test_closed_channels_are_not_mounted():
    paths = _route_paths()
    # Closed by policy: owner does not run these channels.
    assert "/channels/whatsapp" not in paths
    assert "/channels/imessage" not in paths
    assert "/channels/slack/events" not in paths
