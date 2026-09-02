"""
core/identity.py
----------------
F5: Per-tenant identity mapping.
Resolves channel-specific IDs (e.g., WhatsApp number, Slack ID) to a canonical internal UserId.

W2 (identity stability across resets): the email->user_id binding used to live
ONLY in data/users.sqlite. A manual reset that deleted that DB while the user's
data/memory/personal/<id>/ dir survived caused the next onboarding to mint a
fresh id and orphan all memory ("3 user_ids burned"). We now also drop a durable
per-user account.json marker inside the memory dir and self-heal from it on a
mapping miss, so a verified email always rebinds to its original identity.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
from pydantic import BaseModel

from core.config import settings
from core.io_atomic import atomic_write_json

# The web onboarding channel whose channel_user_id is an email address. Only
# this channel gets email normalization, primary_email population, and marker
# rebind — other channels (slack, whatsapp, ...) keep opaque handles verbatim.
WEB_EMAIL_CHANNEL = "web_email"

ACCOUNT_MARKER_FILENAME = "account.json"


def normalize_email(email: str) -> str:
    """Canonical form for email lookups: strip surrounding whitespace, lowercase.

    This is the ONE place email normalization happens; /onboarding/start,
    /forget-me, resolve_user(), and the marker rebind scan all route through it
    so a single policy governs whether two strings name the same account.

    Policy is deliberately EXACT-MATCH on the local part: we do NOT strip
    plus-tags ("a+tag@x.com") or dots in Gmail-style addresses. Two people (or a
    real provider) can treat those as distinct mailboxes, and collapsing them
    would let one user rebind onto another's memory. Casing/whitespace are the
    only accidental differences we fold away. Broadening this is out of scope.
    """
    return (email or "").strip().lower()


def write_account_marker(user_id: str, email: str, verified: bool) -> Path:
    """Atomically write data/memory/personal/<user_id>/account.json.

    This marker is the durable email->user_id binding that lets resolve_user()
    self-heal after a users.sqlite reset. It lives *inside* the user's memory
    dir on purpose: a /forget-me purge rmtrees that dir and the marker dies with
    it, so a deleted user can never be silently resurrected via rebind.

    ``verified`` records whether email ownership was proven (magic-link /claim
    click) or merely asserted (dev /start fast-path). Cloud rebind requires
    verified==True. Returns the marker path.

    core.paths is imported lazily so tests that monkeypatch
    core.paths.PERSONAL_MEMORY_DIR are honored, and to keep the import graph
    acyclic.
    """
    from core import paths  # lazy: honor monkeypatched PERSONAL_MEMORY_DIR

    normalized = normalize_email(email)
    marker_path = paths.personal_memory_dir(user_id) / ACCOUNT_MARKER_FILENAME

    # Preserve the original created_at across re-writes (dev /start writes the
    # marker unverified, then /claim rewrites it verified — the account wasn't
    # "created" twice).
    created_at: Optional[str] = None
    if marker_path.exists():
        try:
            prior = json.loads(marker_path.read_text(encoding="utf-8"))
            prior_created = prior.get("created_at")
            if isinstance(prior_created, str) and prior_created:
                created_at = prior_created
        except (OSError, ValueError):
            created_at = None
    if not created_at:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    payload = {
        "user_id": user_id,
        "email": normalized,
        "email_verified": bool(verified),
        "created_at": created_at,
        "channel": WEB_EMAIL_CHANNEL,
    }
    atomic_write_json(marker_path, payload)
    return marker_path


class UserIdentity(BaseModel):
    user_id: str
    primary_email: Optional[str] = None
    created_at: str

class IdentityManager:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (settings.data_dir / "users.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                '''CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    primary_email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )'''
            )
            await db.execute(
                '''CREATE TABLE IF NOT EXISTS channel_mappings (
                    channel TEXT,
                    channel_user_id TEXT,
                    user_id TEXT,
                    PRIMARY KEY (channel, channel_user_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )'''
            )
            await db.execute(
                '''CREATE TABLE IF NOT EXISTS claimed_tokens (
                    jti TEXT PRIMARY KEY,
                    user_id TEXT,
                    claimed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )'''
            )
            await db.commit()

    async def mark_token_claimed(self, jti: str, user_id: str) -> bool:
        """Insert jti into claimed_tokens. Returns True on first claim, False if already present."""
        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute(
                    "INSERT INTO claimed_tokens (jti, user_id) VALUES (?, ?)",
                    (jti, user_id),
                )
                await db.commit()
                return True
            except aiosqlite.IntegrityError:
                return False

    async def link_channel(
        self, *, user_id: str, channel: str, channel_user_id: str
    ) -> str | None:
        """Re-point a channel identity at an existing user_id. Returns the prior
        user_id it was bound to (or None if it was unbound).

        This is the write half of account linking — one human, one memory. It is
        deliberately NOT reachable from resolve_user: re-pointing an identity
        must only ever happen after BOTH sides are proven (a claim code proving
        control of the channel identity, redeemed inside an authenticated
        session proving ownership of the target account). See
        core/account_linking.py for the flow and why matching on a self-claimed
        email would be an account-takeover vector.
        """
        target = (channel_user_id or "").strip()
        if not user_id or not channel or not target:
            return None
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT user_id FROM channel_mappings WHERE channel = ? AND channel_user_id = ?",
                (channel, target),
            )
            row = await cursor.fetchone()
            previous = row[0] if row else None
            if previous == user_id:
                return previous  # already linked; nothing to do
            await db.execute(
                "INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,)
            )
            await db.execute(
                "INSERT OR REPLACE INTO channel_mappings (channel, channel_user_id, user_id) "
                "VALUES (?, ?, ?)",
                (channel, target, user_id),
            )
            await db.commit()
        print(
            f"LOG: linked {channel}/{target} -> {user_id}"
            + (f" (was {previous})" if previous else "")
        )
        return previous

    async def resolve_user(self, channel: str, channel_user_id: str) -> str:
        """Resolve a channel user ID to a canonical internal UserId. Creates one if missing.

        For the web_email channel the channel_user_id is an email address, so we:
          * normalize it (strip+lower) so casing/whitespace never mints a duplicate,
          * populate users.primary_email (previously always left NULL), and
          * on a mapping MISS, attempt a self-healing rebind from surviving
            account.json markers BEFORE minting a fresh id — this is what keeps a
            users.sqlite reset from orphaning the user's memory dir (W2).
        Other channels keep their opaque handles verbatim and never rebind.
        """
        is_email = channel == WEB_EMAIL_CHANNEL
        lookup_id = normalize_email(channel_user_id) if is_email else channel_user_id

        # 1) Existing mapping wins.
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT user_id FROM channel_mappings WHERE channel = ? AND channel_user_id = ?",
                (channel, lookup_id)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    if is_email:
                        # Backfill primary_email for rows minted before the
                        # column was populated; only touches NULLs so this is
                        # a one-time repair per user, not a per-login write.
                        await db.execute(
                            "UPDATE users SET primary_email = ? WHERE user_id = ?"
                            " AND primary_email IS NULL",
                            (lookup_id, row[0]),
                        )
                        await db.commit()
                    return row[0]

        # 2) Mapping MISS. For web_email, try to rebind from a surviving marker
        #    before minting — a manual reset that dropped users.sqlite while the
        #    memory dir survived must NOT burn a new id.
        if is_email:
            rebound = await self._rebind_from_markers(channel, lookup_id)
            if rebound is not None:
                return rebound

        # 3) Genuinely new user — mint.
        async with aiosqlite.connect(self.db_path) as db:
            new_user_id = f"usr_{uuid.uuid4().hex[:12]}"
            await db.execute("INSERT INTO users (user_id) VALUES (?)", (new_user_id,))
            if is_email:
                # Populate primary_email so /admin/users stops showing NULL and
                # the column can back future email->id lookups.
                await db.execute(
                    "UPDATE users SET primary_email = ? WHERE user_id = ?",
                    (lookup_id, new_user_id),
                )
            try:
                await db.execute(
                    "INSERT INTO channel_mappings (channel, channel_user_id, user_id) VALUES (?, ?, ?)",
                    (channel, lookup_id, new_user_id)
                )
                await db.commit()
            except aiosqlite.IntegrityError:
                # Two concurrent first-logins for the same handle both missed
                # the mapping and raced to mint; the PK on (channel,
                # channel_user_id) makes exactly one win. Yield to the winner
                # instead of surfacing a 500 (Codex review R3#3). Our losing
                # users row stays behind as an unreferenced orphan — harmless.
                async with db.execute(
                    "SELECT user_id FROM channel_mappings WHERE channel = ? AND channel_user_id = ?",
                    (channel, lookup_id),
                ) as cursor:
                    winner = await cursor.fetchone()
                if winner:
                    print(
                        f"LOG: identity mint race for {channel}:{lookup_id} — "
                        f"yielding to existing {winner[0]}"
                    )
                    return winner[0]
                raise
            return new_user_id

    async def _rebind_from_markers(self, channel: str, email: str) -> Optional[str]:
        """Scan personal-memory account.json markers for a surviving identity.

        Called on a web_email mapping miss. If a marker's normalized email
        matches ``email`` and (in cloud mode) is email_verified, we re-INSERT the
        users row + channel mapping and return the existing user_id — rebinding
        the caller to their original identity instead of minting a fresh one.
        Returns None when no eligible marker exists (genuinely new user, or a
        purged user whose marker died with its dir).

        Dev mode accepts unverified markers (the /start fast-path grants a cookie
        without proving email ownership anyway); cloud mode requires proof.

        core.paths is imported lazily so a monkeypatched PERSONAL_MEMORY_DIR
        (tests) is honored and the import graph stays acyclic.
        """
        from core import paths  # lazy: honor monkeypatched PERSONAL_MEMORY_DIR

        target = normalize_email(email)
        require_verified = settings.is_cloud
        try:
            marker_paths = sorted(paths.PERSONAL_MEMORY_DIR.glob(f"*/{ACCOUNT_MARKER_FILENAME}"))
        except OSError:
            return None

        for marker_path in marker_paths:
            try:
                data = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if normalize_email(str(data.get("email", ""))) != target or not target:
                continue
            # Strict boolean check: a marker carrying email_verified as the
            # STRING "false" must not pass (bool("false") is True).
            verified = data.get("email_verified") is True
            if require_verified and not verified:
                # Cloud mode requires proof of ownership; an unverified marker
                # (dev fast-path residue) must not silently rebind a stranger.
                print(
                    f"LOG: identity rebind skipped (unverified marker in cloud) "
                    f"email={target} dir={marker_path.parent.name}"
                )
                continue
            # The marker's LOCATION is the identity, not its payload: the dir
            # name is what the memory tree is actually keyed by, and trusting an
            # embedded user_id would let a marker written inside one user's dir
            # bind their email onto a DIFFERENT user's memory (Codex review
            # R3#1). A mismatched embedded id marks a tampered/corrupt marker.
            dir_user_id = marker_path.parent.name
            embedded = str(data.get("user_id") or "")
            if embedded and embedded != dir_user_id:
                print(
                    f"LOG: identity rebind skipped (marker user_id {embedded!r} "
                    f"!= dir {dir_user_id!r}) email={target}"
                )
                continue
            user_id = dir_user_id
            # Re-materialize the identity rows the reset destroyed.
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,)
                )
                await db.execute(
                    "UPDATE users SET primary_email = ? WHERE user_id = ?",
                    (target, user_id),
                )
                await db.execute(
                    "INSERT OR REPLACE INTO channel_mappings (channel, channel_user_id, user_id) "
                    "VALUES (?, ?, ?)",
                    (channel, target, user_id),
                )
                await db.commit()
            print(
                f"LOG: identity rebound from marker email={target} "
                f"user_id={user_id} verified={verified}"
            )
            return user_id
        return None

identity_manager = IdentityManager()
