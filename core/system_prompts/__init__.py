from __future__ import annotations

from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """Load a system prompt from core/system_prompts/<name>.txt."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")
