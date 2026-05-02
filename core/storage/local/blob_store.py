"""
core/storage/local/blob_store.py
---------------------------------
G1: Local filesystem BlobStore for local mode.

Blobs are stored under data/blobs/<key> on disk.
The key may contain forward slashes; they are preserved as subdirectory structure.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from core.config import settings
from core.storage import BlobStore


class LocalBlobStore(BlobStore):
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (settings.data_dir / "blobs")
        self.root.mkdir(parents=True, exist_ok=True)

    async def put(self, key: str, data: bytes) -> str:
        """Write *data* to the blob store under *key*. Returns the resolved path as a URI."""
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest.as_uri()

    async def get(self, key: str) -> bytes | None:
        """Read blob bytes by key. Returns None if not found."""
        dest = self.root / key
        if not dest.exists():
            return None
        return dest.read_bytes()

    async def delete(self, key: str) -> bool:
        """Delete a blob. Returns True if it existed."""
        dest = self.root / key
        if dest.exists():
            dest.unlink()
            return True
        return False

    @staticmethod
    def content_key(data: bytes, prefix: str = "blobs") -> str:
        """Derive a content-addressed key from data bytes (SHA-256, first 32 hex chars)."""
        digest = hashlib.sha256(data).hexdigest()[:32]
        return f"{prefix}/{digest[:2]}/{digest}"
