from __future__ import annotations

from dotenv import load_dotenv

from .paths import ROOT_DIR


def load_env(override: bool = True) -> None:
    """Load environment variables from the repo-level .env file."""
    load_dotenv(ROOT_DIR / ".env", override=override)

