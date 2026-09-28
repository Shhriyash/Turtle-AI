"""
test/deploy_workflow_env_extraction_test.py
----------------------------------------------
.github/workflows/deploy-vercel.yml's "Run database migrations" step reads
DATABASE_URL_UNPOOLED out of a file `vercel env pull --environment=production`
wrote, without ever executing that file as shell. This file proves why that
matters and that the workflow's actual extraction command (read straight out
of deploy-vercel.yml, not reimplemented here) behaves correctly against a
realistic hostile-looking value.

An earlier version of this step used `set -a; source "$env_file"`, which has
two real, verified-not-just-reasoned-about hazards against a pulled dotenv
file:

  1. Silent corruption: inside a double-quoted value, bash expands `$name`
     references DURING `source`. A DSN whose password legitimately contains
     a "$" (nothing stops Neon from generating one on a rotation) is silently
     truncated/altered rather than rejected — the failure only surfaces
     later as a confusing authentication error during a production deploy.
     Backticks/`$(...)` are worse: `source` runs the file as bash, so a
     value containing command substitution executes arbitrary commands on
     the runner.
  2. Secret-fanout: `source`-ing the file exports EVERY variable in it
     (every other production secret Vercel returns for that environment)
     into the step's process environment, not just the one variable the
     step needed. A later crash traceback, a verbose tool, or a debug flag
     someone adds in six months would then dump all of them into a CI log —
     `rm -f` on the file afterwards does not undo an export that already
     happened.

The fixture below reproduces both: a DATABASE_URL_UNPOOLED value containing
"$", a backtick, and "&", plus an unrelated GEMINI_API_KEY in the same file.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "deploy-vercel.yml"


def _bash_path(path: Path) -> str:
    """Render `path` the way THIS environment's `bash` (resolved via
    subprocess.run, not necessarily the same bash an interactive shell
    would find) expects it.

    On the real target of deploy-vercel.yml (a plain Linux GitHub Actions
    runner) this is a no-op: paths are already POSIX, no drive letters.
    Locally on Windows it matters and is environment-dependent: `bash`
    resolved via subprocess.run in this sandbox is WSL's bash (confirmed
    via `uname -a` -> "Linux ... microsoft-standard-WSL2"), which needs
    "/mnt/<drive>/..." rather than a raw "C:\\..." or even "C:/..." path —
    passing the latter produced a literal "No such file or directory" for
    the whole concatenated (separator-stripped) string.
    """
    posix = str(path).replace("\\", "/")
    if os.name == "nt" and len(posix) >= 2 and posix[1] == ":":
        drive = posix[0].lower()
        rest = posix[2:]
        return f"/mnt/{drive}{rest}"
    return posix


# Deliberately hostile to a naive `source`: a literal "$cd" (bash expands
# this to whatever $cd happens to resolve to, silently, mid-DSN), a
# backtick command substitution, and a bare "&" (a real, unremarkable
# character in any DSN's query string, e.g. "&channel_binding=require").
_TRICKY_VALUE = (
    "postgresql://u:npg_ab$cd`whoami`@host/db"
    "?sslmode=require&channel_binding=require"
)
_UNRELATED_SECRET = "sk-fake-should-never-leave-this-step-0000000000"


def _write_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "vercel-production.env"
    fixture.write_text(
        'DATABASE_URL="postgresql://u:p@ep-fake-pooler.c-10.us-east-1.aws.neon.tech/db"\n'
        f'DATABASE_URL_UNPOOLED="{_TRICKY_VALUE}"\n'
        f'GEMINI_API_KEY="{_UNRELATED_SECRET}"\n',
        encoding="utf-8",
    )
    return fixture


def _extraction_command_from_workflow() -> str:
    """Pull the real `extracted_unpooled=...` line straight out of
    deploy-vercel.yml, so this test exercises the actual deployed command,
    not a hand-copied restatement of it that could silently drift from what
    the workflow really runs.
    """
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    match = re.search(r"^\s*extracted_unpooled=.*$", text, re.MULTILINE)
    assert match is not None, (
        "deploy-vercel.yml's DATABASE_URL_UNPOOLED extraction line "
        "(the `extracted_unpooled=...` assignment in the 'Run database "
        "migrations' step) was not found — it moved or was renamed, and "
        "this test needs updating to match."
    )
    return match.group(0).strip()


def _run_extraction(fixture: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    # Written to a real .sh FILE and run as `bash <path>` (a single simple
    # argv element) rather than `bash -c "<script>"`. The script text below
    # contains many nested single/double quotes (it embeds the workflow's
    # own `python3 -c '...'` one-liner verbatim) — passing that as one
    # `-c` argument through Python's subprocess on Windows re-encodes it via
    # MSVCRT/Win32 command-line quoting rules (list2cmdline), which the
    # resolved bash then re-parses with ITS OWN, different quoting rules,
    # corrupting the nested quotes before bash ever sees the intended
    # script (verified: doing exactly that produced a Python SyntaxError
    # from mangled `\"` sequences that were never in the source string). A
    # file has no such problem: bash reads its literal bytes.
    script = (
        "set -euo pipefail\n"
        f"env_file={shlex.quote(_bash_path(fixture))}\n"
        f"{_extraction_command_from_workflow()}\n"
        'printf "%s\\n" "$extracted_unpooled"\n'
        'printf "GEMINI_API_KEY=[%s]\\n" "${GEMINI_API_KEY:-<not set>}"\n'
    )
    script_path = tmp_path / "run_extraction.sh"
    # newline="": write literal "\n" only — Path.write_text's default
    # universal-newline translation turns "\n" into "\r\n" on Windows,
    # which bash reads as a stray "\r" glued onto the previous token
    # (verified: produced "set: pipefail\r: invalid option name" from a
    # plain "set -euo pipefail" line).
    with open(script_path, "w", encoding="utf-8", newline="") as f:
        f.write(script)
    return subprocess.run(
        ["bash", _bash_path(script_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
    )


def test_workflow_extraction_recovers_value_byte_identical_and_does_not_leak_other_keys(
    tmp_path,
):
    fixture = _write_fixture(tmp_path)
    result = _run_extraction(fixture, tmp_path)
    assert result.returncode == 0, (
        f"extraction command exited non-zero.\nstdout={result.stdout!r}\n"
        f"stderr={result.stderr!r}"
    )
    stdout_lines = result.stdout.splitlines()
    assert len(stdout_lines) == 2, result.stdout
    recovered_value, leaked_line = stdout_lines

    # The dollar-sign / backtick / ampersand all survive untouched — nothing
    # was expanded or executed.
    assert recovered_value == _TRICKY_VALUE

    # The unrelated GEMINI_API_KEY in the same pulled file must never enter
    # this step's environment.
    assert leaked_line == "GEMINI_API_KEY=[<not set>]"
