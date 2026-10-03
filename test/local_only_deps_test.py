"""
test/local_only_deps_test.py
----------------------------
Ledger 5.6: `scipy` and `sounddevice` live in requirements-local.txt, not
requirements.txt, so the Vercel build never installs them. That is only safe
while nothing on the server's import chain imports them.

Two controls, each checked from both ends:

  1. Consumer side: a FRESH interpreter imports the real app (cloud and local
     mode) and we assert neither package ended up in sys.modules. A fresh
     process is required -- in-process, another test may already have imported
     them. A positive control proves the probe really detects an import.
  2. Declaration side: requirements.txt must not list them (otherwise the move
     was undone and control 1 would be guarding nothing), and
     requirements-local.txt must.

Plus: tools.tts.tts.stream_tts() raises an actionable error when sounddevice is
genuinely un-importable (import blocked, not the raising function mocked).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_ONLY = ("scipy", "sounddevice")

_PROBE = (
    "import sys, json\n"
    "{pre}\n"
    "import apps.turtle_server\n"
    "print('PROBE:' + json.dumps([m for m in {mods!r} if m in sys.modules]))\n"
)


def _loaded_after_boot(deploy: str, pre: str = "") -> list[str]:
    env = dict(os.environ)
    for key in (
        "GROQ_API_KEY", "GEMINI_API_KEY", "OPEN_ROUTER_API_KEY_1",
        "COHERE_API_KEY", "TAVILY_API_KEY", "DEEPGRAM_API_KEY", "AUTH_SECRET_KEY",
    ):
        env.setdefault(key, "ci-dummy")
    env["TURTLE_DEPLOY"] = deploy
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(pre=pre, mods=LOCAL_ONLY)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE:"):
            return json.loads(line[len("PROBE:"):])
    raise AssertionError(
        f"probe did not run (rc={proc.returncode}):\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    )


def _declared(path: str) -> set[str]:
    names = set()
    for line in (REPO_ROOT / path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[=<>\[;~ ]", line)[0].lower().replace("_", "-"))
    return names


class AppBootDoesNotImportLocalOnlyDeps(unittest.TestCase):
    def test_cloud_boot_imports_neither_package(self) -> None:
        self.assertEqual(_loaded_after_boot("cloud"), [])

    def test_local_boot_imports_neither_package(self) -> None:
        self.assertEqual(_loaded_after_boot("local"), [])

    def test_probe_detects_an_import(self) -> None:
        # Positive control: if the probe cannot see a real top-level import it
        # would pass vacuously. Skip (not pass) when the package is absent.
        for mod in LOCAL_ONLY:
            if importlib.util.find_spec(mod) is None:
                self.skipTest(f"{mod} not importable here; cannot run positive control")
        self.assertEqual(
            sorted(_loaded_after_boot("cloud", pre="import scipy, sounddevice")),
            sorted(LOCAL_ONLY),
        )


class RequirementsDeclaration(unittest.TestCase):
    def test_not_in_cloud_requirements(self) -> None:
        self.assertEqual(_declared("requirements.txt") & set(LOCAL_ONLY), set())

    def test_in_local_requirements(self) -> None:
        self.assertEqual(_declared("requirements-local.txt") & set(LOCAL_ONLY), set(LOCAL_ONLY))


class StreamTtsMissingSounddevice(unittest.TestCase):
    def test_clear_error_when_sounddevice_missing(self) -> None:
        from tools.tts import tts

        # `None` in sys.modules makes `import sounddevice` raise ImportError
        # exactly as a genuinely absent package does.
        with mock.patch.dict(sys.modules, {"sounddevice": None}):
            with self.assertRaises(ImportError) as ctx:
                tts.stream_tts("hello")
        msg = str(ctx.exception)
        self.assertIn("sounddevice", msg)
        self.assertIn("requirements-local.txt", msg)
        self.assertIn("local", msg.lower())
        self.assertIsInstance(ctx.exception.__cause__, ImportError)


if __name__ == "__main__":
    unittest.main()
