"""Guards against Vercel's Git integration re-opening the untested-deploy race.

See .github/workflows/deploy-vercel.yml's header for the full story: that
workflow is meant to be the ONLY path anything reaches Vercel through, but
Vercel's dashboard Git integration auto-deploys (and auto-promotes) on every
push independently of it unless `vercel.json` sets
`git.deploymentEnabled: false`. This test fails loudly if that key is ever
dropped or narrowed to only cover specific branches.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERCEL_JSON = REPO_ROOT / "vercel.json"


def _load_config() -> dict:
    with VERCEL_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_git_deployment_enabled_is_false_for_all_branches():
    config = _load_config()
    deployment_enabled = config.get("git", {}).get("deploymentEnabled", "MISSING")

    assert deployment_enabled is False, (
        "vercel.json must set git.deploymentEnabled to the boolean `false` "
        f"(found {deployment_enabled!r}). Without this, Vercel's dashboard "
        "Git integration will auto-deploy AND auto-promote every push to "
        "production on its own, bypassing the test gate in "
        ".github/workflows/deploy-vercel.yml and letting an untested commit "
        "reach production. A per-branch object like {'main': false} is not "
        "sufficient either, since it leaves other branches auto-deploying."
    )


def test_functions_max_duration_unchanged():
    config = _load_config()
    max_duration = config.get("functions", {}).get("main.py", {}).get("maxDuration")

    assert max_duration == 300, (
        "vercel.json's functions['main.py'].maxDuration changed from the "
        f"expected 300 (found {max_duration!r}). This likely means the file "
        "was carelessly rewritten; double check the git.deploymentEnabled "
        "guard above is still intact too."
    )
