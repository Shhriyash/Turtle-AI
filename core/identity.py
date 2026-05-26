"""
core/identity.py
----------------
F5: Per-tenant identity mapping.
Resolves channel-specific IDs (e.g., WhatsApp number, Slack ID) to a canonical internal UserId.
"""
import uuid
from typing import Optional
import aiosqlite
from pathlib import Path
from pydantic import BaseModel
from core.config import settings

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

    async def resolve_user(self, channel: str, channel_user_id: str) -> str:
        """Resolve a channel user ID to a canonical internal UserId. Creates one if missing."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT user_id FROM channel_mappings WHERE channel = ? AND channel_user_id = ?", 
                (channel, channel_user_id)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return row[0]
            
            # Not found, create new user
            new_user_id = f"usr_{uuid.uuid4().hex[:12]}"
            await db.execute("INSERT INTO users (user_id) VALUES (?)", (new_user_id,))
            await db.execute(
                "INSERT INTO channel_mappings (channel, channel_user_id, user_id) VALUES (?, ?, ?)",
                (channel, channel_user_id, new_user_id)
            )
            await db.commit()
            return new_user_id

identity_manager = IdentityManager()
