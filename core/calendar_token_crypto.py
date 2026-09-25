"""
core/calendar_token_crypto.py
------------------------------
AES-256-GCM encryption-at-rest for stored Google Calendar OAuth tokens.

Used by apps/calendar_oauth_routes.py (writes/reads the per-user token),
tools/calendar_tool.py (reads it to build API credentials), and
core/storage/cloud/calendar_token_store.py's caller (the Postgres row is
just a TEXT column holding whatever this module hands back — the store
itself is encryption-agnostic).

Key format
----------
CALENDAR_TOKEN_KEY is a urlsafe-base64-encoded 32-byte (256-bit) AES key.
Generate one with:

    python -c "import secrets, base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"

parse_key() is called from core/config.py's field_validator on every process
boot that has CALENDAR_TOKEN_KEY set, so a malformed or wrong-length key
fails the deploy immediately instead of the first time someone connects a
calendar.

On-disk / on-row envelope
--------------------------
A stored value is one of:

  - key_version 0 (implicit): the raw Google token-endpoint JSON response,
    verbatim, UNENCRYPTED. This is the format every token wrote before this
    module existed used, and it stays a supported READ format forever —
    every currently-connected user's token is in exactly this shape the
    moment this code ships, and there is no way to encrypt it without the
    key the deploy is being configured with right now. It is read
    transparently (decrypt_stored treats "not our envelope" as key_version
    0) and re-encrypted the next time it is written (apps/
    calendar_oauth_routes.py's _write_token always calls encrypt_for_storage,
    which never emits key_version 0).

  - key_version >= 1: a JSON envelope ``{"key_version": 1, "blob": "..."}``
    where blob is base64(nonce[12 bytes] || AES-256-GCM(ciphertext+tag)) of
    the original raw token JSON string. A fresh random nonce is drawn with
    os.urandom for every encryption — reusing a nonce under the same GCM key
    is catastrophic (it lets an attacker recover the authentication key and
    breaks confidentiality for every message sharing it), so nonces are never
    memoized, derived, or reused across calls.

CALENDAR_TOKEN_KEY unset
-------------------------
Local mode: falls back to storing plaintext (key_version 0), exactly
today's behaviour. This is a deliberate scope decision for a single-tenant
dev box, not an oversight — see apps/calendar_oauth_routes.py's _write_token.

Cloud mode: encrypt_for_storage refuses (raises CalendarTokenKeyRequired)
rather than silently writing plaintext into a shared multi-tenant Postgres
row. The caller (apps/calendar_oauth_routes.py) turns that into a legible
503 instead of a silent downgrade.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

CURRENT_KEY_VERSION = 1
_NONCE_LEN = 12  # 96-bit nonce: the size AES-GCM is designed/recommended for
_KEY_LEN = 32  # AES-256


class CalendarTokenKeyError(ValueError):
    """CALENDAR_TOKEN_KEY is set but malformed (bad base64 or wrong length).

    Subclasses ValueError (not RuntimeError) deliberately: pydantic v2 only
    converts ValueError/TypeError/AssertionError raised inside a
    field_validator into a proper pydantic.ValidationError at settings
    construction (core/config.py) — any other exception type propagates
    raw and would look like an unhandled crash instead of a normal
    validation failure at boot.
    """


class CalendarTokenKeyRequired(RuntimeError):
    """CALENDAR_TOKEN_KEY is unset where it is required (cloud writes)."""


class CalendarTokenDecryptError(RuntimeError):
    """An encrypted envelope was found but could not be decrypted (missing/
    wrong key, or the ciphertext/tag was tampered with)."""


def _pad_b64(s: str) -> str:
    """Tolerate a key copy-pasted without its trailing '=' padding."""
    return s + "=" * (-len(s) % 4)


def parse_key(raw: Optional[str]) -> Optional[bytes]:
    """Decode CALENDAR_TOKEN_KEY into 32 raw key bytes.

    Returns None for an unset/empty value. Raises CalendarTokenKeyError for a
    non-empty value that is not valid urlsafe-base64 or does not decode to
    exactly 32 bytes. Meant to be called at settings-construction time
    (core/config.py) so a malformed key surfaces at process boot.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        key_bytes = base64.urlsafe_b64decode(_pad_b64(stripped))
    except Exception as exc:
        raise CalendarTokenKeyError(
            f"CALENDAR_TOKEN_KEY is not valid urlsafe-base64: {exc}"
        ) from exc
    if len(key_bytes) != _KEY_LEN:
        raise CalendarTokenKeyError(
            "CALENDAR_TOKEN_KEY must decode to exactly 32 bytes (AES-256), "
            f"got {len(key_bytes)}. Generate one with: python -c \"import "
            "secrets, base64; print(base64.urlsafe_b64encode(secrets."
            "token_bytes(32)).decode())\""
        )
    return key_bytes


def _encrypt_blob(plaintext: str, key: bytes) -> str:
    nonce = os.urandom(_NONCE_LEN)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def _decrypt_blob(blob_b64: str, key: bytes) -> str:
    try:
        raw = base64.b64decode(blob_b64)
        nonce, ciphertext = raw[:_NONCE_LEN], raw[_NONCE_LEN:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")
    except InvalidTag as exc:
        raise CalendarTokenDecryptError(
            "Calendar token ciphertext failed authentication — wrong "
            "CALENDAR_TOKEN_KEY, or the stored value was tampered with."
        ) from exc
    except Exception as exc:
        raise CalendarTokenDecryptError(
            f"Calendar token ciphertext could not be decrypted: {exc}"
        ) from exc


def encrypt_for_storage(token_json: str, key: Optional[bytes], *, is_cloud: bool) -> str:
    """Turn a raw token JSON string into what gets written to disk/Postgres.

    key=None:
      - local (is_cloud=False): returns token_json unchanged (key_version 0,
        plaintext) — today's behaviour, preserved for a single-tenant dev box.
      - cloud (is_cloud=True): raises CalendarTokenKeyRequired. Storing
        plaintext in a shared Postgres row silently defeats the point of this
        change, so cloud writes refuse instead.

    key set: always returns the key_version>=1 JSON envelope, regardless of
    deploy mode.
    """
    if key is None:
        if is_cloud:
            raise CalendarTokenKeyRequired(
                "CALENDAR_TOKEN_KEY is not set. Calendar tokens cannot be "
                "stored in cloud mode without it — set CALENDAR_TOKEN_KEY "
                "and try connecting again."
            )
        return token_json
    envelope = {
        "key_version": CURRENT_KEY_VERSION,
        "blob": _encrypt_blob(token_json, key),
    }
    return json.dumps(envelope)


def decrypt_stored(stored: str, key: Optional[bytes]) -> tuple[str, int]:
    """Inverse of encrypt_for_storage. Returns (token_json, key_version_found).

    Transparently reads BOTH formats:
      - a key_version envelope (requires `key` to decrypt; raises
        CalendarTokenDecryptError if `key` is None or wrong)
      - a bare plaintext token JSON blob written before this module existed
        (key_version 0 — no key needed, no error, ever, for this shape)

    This is the crux of the migration path: an old plaintext token must stay
    readable after CALENDAR_TOKEN_KEY is configured and deployed, not break
    the instant encryption exists. Callers (apps/calendar_oauth_routes.py)
    re-encrypt on next write, at which point the row/file finally moves to
    key_version 1.
    """
    envelope: Optional[dict] = None
    try:
        parsed = json.loads(stored)
        if isinstance(parsed, dict) and "key_version" in parsed and "blob" in parsed:
            envelope = parsed
    except (json.JSONDecodeError, TypeError):
        envelope = None

    if envelope is None:
        # Not our envelope shape -> treat as a pre-encryption plaintext token.
        return stored, 0

    version = envelope.get("key_version")
    if not isinstance(version, int) or version < 1:
        return stored, 0

    if key is None:
        raise CalendarTokenDecryptError(
            "A calendar token is encrypted (key_version="
            f"{version}) but CALENDAR_TOKEN_KEY is not set — cannot decrypt."
        )
    plaintext = _decrypt_blob(envelope["blob"], key)
    return plaintext, version
