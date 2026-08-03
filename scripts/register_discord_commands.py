"""One-shot: register Turtle's Discord slash + message commands.

Reads DISCORD_BOT_TOKEN and DISCORD_APPLICATION_ID from your environment / .env
(via core.config.settings) and PUTs the command set to Discord. Idempotent —
safe to run repeatedly. Run this ONCE after you create the bot application (and
again only if you change the command definitions in apps/channels/discord.py).

    python scripts/register_discord_commands.py

Global commands can take up to ~1 hour to appear in every server the first time;
they show up almost immediately in a server the bot was already in.
"""
from __future__ import annotations

import asyncio
import sys

# Ensure the repo root is importable when run as `python scripts/...`.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.channels.discord import register_slash_commands  # noqa: E402
from core.config import settings  # noqa: E402


def main() -> int:
    if not settings.discord_bot_token or not settings.discord_application_id:
        print(
            "ERROR: DISCORD_BOT_TOKEN and DISCORD_APPLICATION_ID must be set "
            "(in your environment or .env) before registering commands."
        )
        return 1
    print("Registering Discord commands for application "
          f"{settings.discord_application_id} ...")
    asyncio.run(register_slash_commands())
    print("Done. Check the output above for 'registered' or an error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
