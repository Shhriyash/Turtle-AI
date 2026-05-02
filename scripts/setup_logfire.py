"""
One-click Logfire setup for Turtle.

Run once:
    python scripts/setup_logfire.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGFIRE_DIR = ROOT / ".logfire"


def run(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False).returncode


def main():
    print("=== Turtle × Logfire setup ===\n")

    if LOGFIRE_DIR.exists():
        tokens = list(LOGFIRE_DIR.glob("*.toml")) + list(LOGFIRE_DIR.glob("*.json"))
        if tokens:
            print(f"Token found in {LOGFIRE_DIR} — skipping auth.")
            print("Run  logfire projects use  to switch projects.\n")
            print("Done. Start Turtle normally and traces will flow to Logfire.")
            return

    print("Step 1: authenticate with Logfire (opens browser)...")
    rc = run([sys.executable, "-m", "logfire", "auth"])
    if rc != 0:
        print("Auth failed. Make sure logfire is installed:  pip install logfire")
        sys.exit(rc)

    print("\nStep 2: create / select a project...")
    rc = run([sys.executable, "-m", "logfire", "projects", "new"])
    if rc != 0:
        # Fall back to interactive project list
        run([sys.executable, "-m", "logfire", "projects", "use"])

    print("\nDone. Start Turtle normally — traces will flow to Logfire automatically.")
    print("Dashboard: https://logfire.pydantic.dev/")


if __name__ == "__main__":
    main()
