"""
core/account_linking.py
-----------------------
Cross-channel account linking: one human, one memory.

THE PROBLEM. ``resolve_user(channel, external_id)`` mints a fresh user_id per
channel binding, so the same person on web and on Discord is two Turtle users
with two disjoint memories.

WHY NOT JUST MATCH ON EMAIL. The tempting fix — "when a Discord user says
'my email is X', link them to the web account with email X" — is an account
TAKEOVER vector: anyone who knows your email address could type it into Discord
and inherit your entire memory. A self-claimed identifier proves nothing.

THE DESIGN (claim code, redeemed on an authenticated surface):

  1. On the channel, the user asks to link. Turtle issues a short-lived,
     single-use CLAIM CODE bound to (channel, external_id) — NOT to any target
     account. Holding the code proves only "I control this Discord account".
  2. The user signs in to Turtle on the WEB (where the turtle_uid cookie
     authenticates them as a specific user_id) and redeems the code.
  3. Redemption is the moment both sides are proven: the code proves control of
     the channel identity, the authenticated session proves ownership of the
     target account. Only then is the mapping re-pointed and memory merged.

The code is deliberately useless on its own: it names no account, and redeeming
it requires an already-authenticated session. A leaked code lets an attacker
attach THEIR OWN channel handle to their own account — not read anyone's data.

Codes are single-use, TTL-bounded, and stored server-side in users.sqlite.
"""
from __future__ import annotations

import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Unambiguous alphabet: no 0/O, 1/I/L — these get read aloud and retyped.
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_CODE_LENGTH = 8
LINK_CODE_TTL_MINUTES = 15

_TABLE = "link_codes"


def _normalize_code(code: str) -> str:
    return (code or "").strip().upper().replace(" ", "").replace("-", "")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


@dataclass(frozen=True)
class LinkCode:
    code: str
    channel: str
    channel_user_id: str
    source_user_id: str
    expires_at: str


class LinkCodeStore:
    """Server-side store for pending link claims (sqlite, alongside identities)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connect(self):
        """Short-lived connection that is always CLOSED.

        `with sqlite3.connect(...)` only commits/rolls back — it does NOT close,
        so using it directly leaks a file handle per call (and on Windows pins
        the db file open). Linking is infrequent, so open/close per operation is
        the right trade for not holding handles.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    code TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    channel_user_id TEXT NOT NULL,
                    source_user_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT
                )
                """
            )

    def issue(self, *, channel: str, channel_user_id: str, source_user_id: str) -> LinkCode:
        """Mint a fresh claim code for a channel identity.

        Any previous unconsumed code for the same channel identity is dropped, so
        a user who asks twice can't leave a stale code redeemable.
        """
        code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))
        expires = _iso(_utc_now() + timedelta(minutes=LINK_CODE_TTL_MINUTES))
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM {_TABLE} WHERE channel = ? AND channel_user_id = ? "
                f"AND consumed_at IS NULL",
                (channel, channel_user_id),
            )
            conn.execute(
                f"INSERT INTO {_TABLE} (code, channel, channel_user_id, source_user_id, expires_at) "
                f"VALUES (?, ?, ?, ?, ?)",
                (code, channel, channel_user_id, source_user_id, expires),
            )
        return LinkCode(code, channel, channel_user_id, source_user_id, expires)

    def peek(self, code: str) -> LinkCode | None:
        """Look up a code WITHOUT marking it consumed.

        Used by the redemption route to validate the code, then do the risky
        merge work with the code still redeemable — so a merge failure leaves
        the user with a working code to retry, instead of a burned code and an
        orphaned source journal. Only expired/unknown/already-consumed return None.
        """
        normalized = _normalize_code(code)
        if not normalized:
            return None
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE code = ?", (normalized,)
            ).fetchone()
        if row is None or row["consumed_at"] is not None:
            return None
        try:
            expires = datetime.fromisoformat(row["expires_at"])
        except Exception:
            return None
        if expires <= _utc_now():
            return None
        return LinkCode(
            normalized, row["channel"], row["channel_user_id"],
            row["source_user_id"], row["expires_at"],
        )

    def consume(self, code: str) -> LinkCode | None:
        """Atomically redeem a code. Returns None if unknown, expired, or reused.

        The UPDATE ... WHERE consumed_at IS NULL is what makes this single-use
        even if two redemptions race: exactly one gets rowcount 1.
        """
        normalized = _normalize_code(code)
        if not normalized:
            return None
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE code = ?", (normalized,)
            ).fetchone()
            if row is None or row["consumed_at"] is not None:
                return None
            try:
                expires = datetime.fromisoformat(row["expires_at"])
            except Exception:
                return None
            if expires <= _utc_now():
                return None
            cursor = conn.execute(
                f"UPDATE {_TABLE} SET consumed_at = ? WHERE code = ? AND consumed_at IS NULL",
                (_iso(_utc_now()), normalized),
            )
            if (cursor.rowcount or 0) != 1:
                return None  # lost the race
        return LinkCode(
            normalized,
            row["channel"],
            row["channel_user_id"],
            row["source_user_id"],
            row["expires_at"],
        )

    def purge_expired(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                f"DELETE FROM {_TABLE} WHERE expires_at <= ?", (_iso(_utc_now()),)
            )
            return cursor.rowcount or 0


# Module-level thin wrappers so the redemption route can call these through
# asyncio.to_thread against a bound `store` instance without lambdas.
def peek(store: LinkCodeStore, code: str) -> LinkCode | None:
    return store.peek(code)


def mark_consumed(store: LinkCodeStore, code: str) -> bool:
    """Atomically mark a code consumed. Returns True on the first consume, False
    if it was already consumed / doesn't exist (peek+consume race lost)."""
    return store.consume(code) is not None


def merge_memory(source_user_id: str, target_user_id: str) -> dict[str, Any]:
    """Fold the source user's memory into the target's.

    The journal is the source of truth, so merging means replaying the source's
    events into the target's journal and rebuilding the target's projections.
    Everything derived (markdown topics, sqlite read model) is regenerated from
    the merged journal, so no bespoke migration is needed for those.

    Non-destructive: the source journal is left on disk. Linking is rare and
    irreversible-looking to the user; keeping the original means a bad merge can
    be investigated rather than mourned.
    """
    result: dict[str, Any] = {"events_copied": 0, "replayed": False, "ok": True, "error": ""}
    if not source_user_id or not target_user_id or source_user_id == target_user_id:
        return result

    from core.memory_journal import JournalStore
    from core.memory_replayer import replay
    from core.personal_memory_store import PersonalMemoryStore

    source_journal = JournalStore(user_id=source_user_id)
    target_journal = JournalStore(user_id=target_user_id)

    try:
        events = source_journal.load_all()
    except Exception as exc:
        # Was silently returning 200 to the caller. Now the redemption route
        # inspects `ok` and refuses to commit the link on failure.
        result["ok"] = False
        result["error"] = f"read source journal: {exc}"
        print(f"LOG: link merge could not read source journal {source_user_id}: {exc}")
        return result
    if not events:
        return result

    # Skip events the target already has (a re-link, or the same fact learned on
    # both surfaces). event_id is stable, so it is the natural dedup key.
    try:
        existing = {e.event_id for e in target_journal.load_all()}
    except Exception:
        existing = set()
    fresh = [e for e in events if e.event_id not in existing]
    if fresh:
        try:
            target_journal.append_many(fresh)
            result["events_copied"] = len(fresh)
        except Exception as exc:
            result["ok"] = False
            result["error"] = f"append: {exc}"
            print(f"LOG: link merge append failed: {exc}")
            return result

    try:
        replay(target_journal.load_all(), store=PersonalMemoryStore(user_id=target_user_id))
        result["replayed"] = True
    except Exception as exc:
        # The events landed in the journal — the source of truth — but the
        # projection did not rebuild. That's recoverable (next replay heals it),
        # but the caller deserves to know rather than get a false 200.
        result["ok"] = False
        result["error"] = f"replay: {exc}"
        print(f"LOG: link merge replay failed for {target_user_id}: {exc}")
    return result
