"""
tools/idempotency.py
--------------------
B5: Idempotency keys on every side-effecting tool.

Email idempotency:
  key = sha1(sorted_recipients + subject + body_first_100_chars + minute_bucket)
  Stored in SQLite (local) tool_invocations table.
  Re-attempts within 60 s become no-ops returning the cached result.

Usage::

    from tools.idempotency import is_duplicate_invocation, record_invocation

    key = build_email_idempotency_key(recipients, subject, body)
    cached = is_duplicate_invocation(key)
    if cached is not None:
        return cached          # no-op: return prior result
    result = send_email_now(...)
    record_invocation(key, result)
    return result
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_IDEMPOTENCY_WINDOW_S = 60          # Re-attempts within this window are no-ops
_DB_PATH: Path | None = None        # Resolved lazily from env / default


def _get_db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        # Anchor to settings.data_dir (repo-root default, honors TURTLE_DATA_DIR).
        # The old bare Path("data") was CWD-relative: launching the server from
        # another directory silently created a fresh tool_invocations.db beside
        # that CWD instead of under <repo>/data — the exact hazard core/config.py
        # (data_dir field + _anchor_data_dir) fixed for everything else. Reusing
        # settings.data_dir keeps idempotency state co-located with the rest of
        # the data volume. Safe from import cycles: core.config is a leaf module
        # (stdlib + pydantic only) and tools/ already imports it (calendar_tool).
        from core.config import settings
        base = settings.data_dir
        base.mkdir(parents=True, exist_ok=True)
        # Log the resolved location once: deployments that previously launched
        # from a non-repo CWD had a stray CWD-relative DB; the log makes the
        # anchor change visible instead of silently "losing" old entries.
        print(f"LOG: idempotency DB at {base / 'tool_invocations.db'}")
        _DB_PATH = base / "tool_invocations.db"
    return _DB_PATH


# ---------------------------------------------------------------------------
# DB setup (creates table once on first call)
# ---------------------------------------------------------------------------

_DB_INITIALIZED = False


def _ensure_db() -> sqlite3.Connection:
    global _DB_INITIALIZED
    conn = sqlite3.connect(str(_get_db_path()), timeout=5)
    if not _DB_INITIALIZED:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_invocations (
                idempotency_key TEXT PRIMARY KEY,
                result          TEXT NOT NULL,
                created_at_s    REAL NOT NULL
            )
        """)
        # Prune old rows on startup (older than 1 hour keeps the table lean)
        conn.execute(
            "DELETE FROM tool_invocations WHERE created_at_s < ?",
            (time.time() - 3600,),
        )
        conn.commit()
        _DB_INITIALIZED = True
    return conn


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------

def build_email_idempotency_key(
    recipients: list[str],
    subject: str,
    body: str,
    *,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
) -> str:
    """Build a stable idempotency key for an email send.

    The created_at_s window in is_duplicate_invocation already bounds the
    dedup horizon; a wall-clock bucket in the key made dedup fail exactly when
    retries straddled a minute boundary.
    """
    sorted_recipients = sorted(r.lower().strip() for r in recipients)
    sorted_cc = sorted(r.lower().strip() for r in (cc or []))
    sorted_bcc = sorted(r.lower().strip() for r in (bcc or []))
    body_prefix = (body or "")[:100]
    canonical = (
        f"to:{','.join(sorted_recipients)}"
        f"|cc:{','.join(sorted_cc)}"
        f"|bcc:{','.join(sorted_bcc)}"
        f"|sub:{(subject or '').strip()}"
        f"|body:{body_prefix}"
    )
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Idempotency check + record
# ---------------------------------------------------------------------------

def is_duplicate_invocation(idempotency_key: str) -> Optional[str]:
    """Return the cached result string if this key was seen in the last 60 s.

    Returns None if the invocation is new (caller should proceed).
    """
    try:
        conn = _ensure_db()
        cutoff = time.time() - _IDEMPOTENCY_WINDOW_S
        row = conn.execute(
            "SELECT result FROM tool_invocations WHERE idempotency_key = ? AND created_at_s >= ?",
            (idempotency_key, cutoff),
        ).fetchone()
        conn.close()
        if row:
            return str(row[0])
        return None
    except Exception as exc:
        print(f"LOG: Idempotency check failed ({exc}), treating as new invocation")
        return None


def record_invocation(idempotency_key: str, result: str) -> None:
    """Persist the result of a completed tool invocation."""
    if not str(result).startswith("Email sent successfully"):
        # Only successful sends are idempotency-cached; a failure must not
        # no-op the user's retry.
        return
    try:
        conn = _ensure_db()
        conn.execute(
            "INSERT OR REPLACE INTO tool_invocations (idempotency_key, result, created_at_s) VALUES (?, ?, ?)",
            (idempotency_key, result, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"LOG: Idempotency record failed ({exc}) — continuing without idempotency")
