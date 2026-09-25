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

TWO-SIDED BINDING (ledger 1b.6). The paragraph above was aspirational, not
accurate: `apps/turtle_server.py`'s `link_account` tool docstring already
documents the real gap it left open — ANY authenticated web session, not just
the one the channel user intends, could redeem a leaked/intercepted code and
walk away with THAT SENDER's memory merged into a stranger's account. The
missing piece was that the code, while it names no *target* account, also
named no *expected redeemer* — so "an already-authenticated session" was
sufficient, when it should have needed to be a SPECIFIC one.

`issue()` now takes an optional `expected_email`: what the channel-side user
says they will sign in to the web with. This is NOT trusted as an
authorization credential — a self-claimed email proves nothing, same as
ever — it is only ever used to REFUSE a redeemer whose authenticated
account's email doesn't match. Authentication (proof of account ownership)
still gates the merge exactly as before; the binding is an *additional*
constraint on top, never a substitute for it. Redemption compares
`expected_email` against the target account's own `users.primary_email` —
an attribute the *web sign-in flow* established, not something the redeemer
types into this endpoint — so there is no read-then-trust step for the
redeemer to spoof.

What a leaked code gets an attacker now: if the code carries an
`expected_email`, nothing — redemption 400s exactly like an invalid code
unless the attacker is ALSO already authenticated as the one specific account
matching that email, and if they already control that account they had no
need of the code to begin with (linking that account was the intended
outcome). Compare to before this change: any authenticated account, chosen
by the attacker, could redeem — which is strictly worse. A code minted
before this change shipped (no `expected_email` on the row) still redeems
under the old, unbound rule until it expires (codes are TTL-bounded to 15
minutes, so this window is short and self-closing — see `reserve()`).
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
# A reservation blocks OTHER targets from redeeming this code. It ages out so a
# redeemer that crashed mid-merge doesn't lock the code out until expiry — the
# SAME target can retry immediately; a DIFFERENT target waits this long.
RESERVATION_TTL_SECONDS = 60

_TABLE = "link_codes"


def _normalize_code(code: str) -> str:
    return (code or "").strip().upper().replace(" ", "").replace("-", "")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _normalize_email(email: str | None) -> str:
    """Mirrors core.identity.normalize_email's policy exactly (strip+lower,
    no plus-tag/dot folding) but is deliberately a local copy rather than an
    import: core/identity.py belongs to a parallel WP and this module must
    not take on a load-bearing dependency on it. The normalization itself is
    a one-line policy, not the kind of logic that drifting duplication would
    meaningfully risk (same trade the two LinkCodeStore backends already make
    for the code alphabet / TTL constants)."""
    return (email or "").strip().lower()


@dataclass(frozen=True)
class LinkCode:
    code: str
    channel: str
    channel_user_id: str
    source_user_id: str
    expires_at: str
    # Two-sided binding (ledger 1b.6): what the channel-side issuer says the
    # redeemer's web account email will be. None on codes issued before this
    # field existed, or if the issuer didn't supply one — see reserve()/the
    # redemption endpoint for what that means at redemption time.
    expected_email: str | None = None


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
                    consumed_at TEXT,
                    -- Codex finding: peek+merge without reservation lets TWO
                    -- authenticated redeemers with DIFFERENT targets each pass
                    -- peek and each copy the source memory into their own
                    -- account, then last-writer-wins the mapping. Reserving
                    -- the code atomically to ONE target closes it: the second
                    -- caller sees a mismatched reserved_for and is rejected.
                    reserved_for TEXT,
                    reserved_at TEXT,
                    expected_email TEXT
                )
                """
            )
            # Add columns to a pre-reservation / pre-binding table. A live
            # production row with no expected_email (i.e. issued before this
            # migration ran) reads back as NULL -> LinkCode.expected_email is
            # None -> redemption treats it as unbound (see module docstring
            # for why that's the deliberate, bounded-by-TTL choice here).
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({_TABLE})")}
            if "reserved_for" not in cols:
                conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN reserved_for TEXT")
            if "reserved_at" not in cols:
                conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN reserved_at TEXT")
            if "expected_email" not in cols:
                conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN expected_email TEXT")

    def issue(
        self,
        *,
        channel: str,
        channel_user_id: str,
        source_user_id: str,
        expected_email: str | None = None,
    ) -> LinkCode:
        """Mint a fresh claim code for a channel identity.

        Any previous unconsumed code for the same channel identity is dropped, so
        a user who asks twice can't leave a stale code redeemable.

        ``expected_email`` (two-sided binding, ledger 1b.6): what the issuer
        says their web account email is. Normalized and stored verbatim; None
        means "no binding" (see module docstring for what that means at
        redemption). Never validated against anything at issue time — it is
        a self-claimed assertion that only ever narrows who may redeem, never
        who is trusted.
        """
        code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))
        expires = _iso(_utc_now() + timedelta(minutes=LINK_CODE_TTL_MINUTES))
        normalized_email = _normalize_email(expected_email) or None
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM {_TABLE} WHERE channel = ? AND channel_user_id = ? "
                f"AND consumed_at IS NULL",
                (channel, channel_user_id),
            )
            conn.execute(
                f"INSERT INTO {_TABLE} "
                f"(code, channel, channel_user_id, source_user_id, expires_at, expected_email) "
                f"VALUES (?, ?, ?, ?, ?, ?)",
                (code, channel, channel_user_id, source_user_id, expires, normalized_email),
            )
        return LinkCode(code, channel, channel_user_id, source_user_id, expires, normalized_email)

    def peek(self, code: str) -> LinkCode | None:
        """Read-only lookup used by tests. Endpoint code should call ``reserve``
        so two concurrent authenticated redeemers can't both pass validation."""
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
            row["source_user_id"], row["expires_at"], row["expected_email"],
        )

    def reserve(self, code: str, target_user_id: str) -> tuple[str, LinkCode | None]:
        """Atomically claim a code for one target account for the merge window.

        Returns a status string:
            "ok"        — reservation acquired (or refreshed for same target)
            "invalid"   — code unknown / expired / already consumed
            "locked"    — reserved for a DIFFERENT target within TTL — a race,
                          reject this redeemer

        Guarantees, via ONE conditional UPDATE:
            * a fresh code with no reservation gets reserved to this target;
            * an existing reservation for the same target is refreshed
              (idempotent retry inside TTL);
            * an existing reservation for another target within TTL blocks —
              this is the property that closes the two-target race Codex found.

        The reservation itself is not the burn — mark_consumed is still the
        one-shot commit. If merge fails, the reservation ages out in
        RESERVATION_TTL_SECONDS and the source account can retry from the SAME
        channel/DM (which resolves to the same target).
        """
        normalized = _normalize_code(code)
        if not normalized or not target_user_id:
            return ("invalid", None)
        now = _iso(_utc_now())
        cutoff = _iso(_utc_now() - timedelta(seconds=RESERVATION_TTL_SECONDS))
        with self._connect() as conn:
            # One statement: reserve iff (unconsumed) AND (not-expired) AND
            # (unreserved OR expired reservation OR same target).
            cursor = conn.execute(
                f"""
                UPDATE {_TABLE}
                   SET reserved_for = ?, reserved_at = ?
                 WHERE code = ?
                   AND consumed_at IS NULL
                   AND expires_at > ?
                   AND (reserved_for IS NULL
                        OR reserved_for = ?
                        OR reserved_at IS NULL
                        OR reserved_at < ?)
                """,
                (target_user_id, now, normalized, now, target_user_id, cutoff),
            )
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE code = ?", (normalized,)
            ).fetchone()
        if row is None or row["consumed_at"] is not None:
            return ("invalid", None)
        try:
            expires = datetime.fromisoformat(row["expires_at"])
        except Exception:
            return ("invalid", None)
        if expires <= _utc_now():
            return ("invalid", None)
        if (cursor.rowcount or 0) == 0:
            # Update matched nothing → an active reservation for a different
            # target is holding the code. Do NOT reveal to the loser who is
            # holding it or that the code exists at all beyond "not for you".
            return ("locked", None)
        claim = LinkCode(
            normalized, row["channel"], row["channel_user_id"],
            row["source_user_id"], row["expires_at"], row["expected_email"],
        )
        return ("ok", claim)

    def release_reservation(self, code: str, target_user_id: str) -> None:
        """Drop THIS target's reservation on failure so retry is not blocked.

        Reservations age out on their own, but releasing eagerly means a
        transient merge failure retries immediately instead of after TTL.
        Only the target that HOLDS the reservation may release it.
        """
        normalized = _normalize_code(code)
        if not normalized or not target_user_id:
            return
        with self._connect() as conn:
            conn.execute(
                f"UPDATE {_TABLE} SET reserved_for = NULL, reserved_at = NULL "
                f"WHERE code = ? AND reserved_for = ? AND consumed_at IS NULL",
                (normalized, target_user_id),
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
            row["expected_email"],
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


def reserve(store: LinkCodeStore, code: str, target_user_id: str) -> tuple[str, LinkCode | None]:
    return store.reserve(code, target_user_id)


def release_reservation(store: LinkCodeStore, code: str, target_user_id: str) -> None:
    store.release_reservation(code, target_user_id)


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
