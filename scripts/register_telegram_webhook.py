"""One-shot: register Turtle's Telegram webhook URL with the Bot API.

Reads TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET, and TURTLE_PUBLIC_BASE_URL
from your environment / .env (via core.config.settings) and calls setWebhook.
Idempotent — safe to run repeatedly, and re-run whenever TURTLE_PUBLIC_BASE_URL
changes (e.g. after a new production deploy).

    python scripts/register_telegram_webhook.py

This points Telegram at the webhook (apps/channels/telegram_webhook.py)
instead of the local long-polling gateway (apps/channels/telegram_gateway.py)
— the two are mutually exclusive per Telegram bot (setWebhook implicitly
disables getUpdates/long-polling for the same bot token). Only run this
against a deployment that actually serves the webhook route (cloud mode).
"""
from __future__ import annotations

import asyncio
import sys

# Ensure the repo root is importable when run as `python scripts/...`.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.channels.telegram_webhook import register_telegram_webhook  # noqa: E402
from core.config import settings  # noqa: E402


def main() -> int:
    if not settings.telegram_bot_token:
        print("ERROR: TELEGRAM_BOT_TOKEN must be set (in your environment or .env).")
        return 1
    if not settings.telegram_webhook_secret:
        print(
            "ERROR: TELEGRAM_WEBHOOK_SECRET must be set — generate one with:\n"
            "  python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
        return 1
    url = f"{settings.public_base_url.rstrip('/')}/channels/telegram/webhook"
    print(f"Registering Telegram webhook at {url} ...")
    asyncio.run(register_telegram_webhook())
    print("Done. Check the output above for 'registered' or an error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
