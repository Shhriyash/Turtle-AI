"""Guest-side helpers, bind-mounted read-only into the sandbox container.

Nothing in this package may import from `core.*` — it runs against whatever
Python the sandbox image ships, not Turtle's interpreter.
"""
from __future__ import annotations

from pathlib import Path

FSOPS_HOST_PATH = Path(__file__).resolve().parent / "fsops.py"
