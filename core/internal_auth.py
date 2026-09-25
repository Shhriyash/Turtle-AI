"""
core/internal_auth.py
----------------------
WP 1.B (ledger 1a.3 / S-7.3): auth primitives for Turtle's internal-only
self-calls — apps/channels/discord.py's deferred-interaction self-invoke and
core/worker.py's embed-personal-memory self-invoke, both landing on their
matching endpoint in apps/cron_tick_routes.py / apps/channels/discord.py.

Before this module, ALL internal automation (GitHub Actions' cron-tick AND
Turtle's own self-calls) shared one bearer secret, compared with a plain
``!=``, and the receiving endpoints trusted identity fields (``user_id``,
``discord_user_id``) straight out of the request body. A leaked secret was
enough to both impersonate any user and replay a captured request forever.

This module closes that by giving every internal caller/callee three
primitives:

  1. ``check_bearer`` — constant-time "Authorization: Bearer <secret>" check.
     Used as-is by the simple, non-self-call case (GitHub Actions calling
     /internal/cron-tick with CRON_TICK_SECRET).
  2. ``sign_request`` / ``verify_request`` — an HMAC envelope (timestamp +
     nonce + the exact request body bytes) for the two Turtle-to-Turtle
     self-calls, replay-checked via a Redis nonce claim. Used ON TOP OF a
     bearer check for those two endpoints (apps/cron_tick_routes.py's
     /internal/embed-personal-memory, apps/channels/discord.py's
     /channels/discord/process).
  3. ``store_job_payload`` / ``take_job_payload`` — payload-by-reference.
     The self-call site stores the real job body (containing the identity
     fields) in Redis and sends only an opaque job id; the receiving
     endpoint reads-and-deletes it server-side. The wire body a caller with
     just the secret could forge therefore carries no identity field at
     all — the impersonation surface is closed even if the secret leaks.

Fail posture when Redis is unavailable: FAIL CLOSED (refuse the call). This
module's Redis-touching functions only run in cloud mode, on the two
self-call paths — both of which already fall back to running the job
in-process (see apps/channels/discord.py::_kick_off_deferred_processing and
core/worker.py::_self_invoke_embed_job) whenever anything about the internal
-automation round trip can't be completed, INCLUDING Redis being down. So
"fail closed" here doesn't strand cloud mode: the caller-side fallback
degrades to in-process execution (less robust, but not silently broken), and
local mode never reaches this module at all (it has no Redis and its callers
check ``settings.is_cloud`` first). Fail-OPEN on a down replay store would be
the unsafe choice here — it would silently disable replay protection at
exactly the moment the check depending on it can't run.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass

from core.storage.cloud import CloudBackendUnavailable, get_redis_client

logger = logging.getLogger(__name__)

# Ledger S-7.3: "rejected when older than 300 s [...] or when the nonce was
# seen". Applied SYMMETRICALLY — a timestamp more than CLOCK_SKEW_S in the
# FUTURE is rejected too. An allowance in only the past direction is a common
# miss: it would let an attacker pre-mint a signature timestamped far ahead
# and have it stay "fresh" indefinitely.
CLOCK_SKEW_S = 300

# Nonce replay-claim TTL. Matches CLOCK_SKEW_S: once a signature is too old
# to pass the timestamp check anyway, remembering its nonce for longer buys
# nothing.
NONCE_TTL_S = 300

# Payload-by-reference TTL. Generous relative to the fire-and-forget self-
# call's own short read timeout — the receiving endpoint runs synchronously
# right after, this just bounds how long an unclaimed payload lingers if the
# receiving request never arrives at all.
JOB_PAYLOAD_TTL_S = 900

# `turtle:` prefix matches the rest of the codebase's Redis key convention
# (see core/storage/cloud/redis_backends.py, tools/idempotency.py).
_NONCE_KEY_PREFIX = "turtle:nonce:"
_JOB_KEY_PREFIX = "turtle:job:"


class SignatureError(Exception):
    """An internal self-call's auth envelope (secret, signature, timestamp,
    or nonce) failed verification. Callers translate this into a 401."""


def check_bearer(expected: str | None, authorization: str | None) -> bool:
    """Constant-time check that ``authorization`` is ``Bearer <expected>``.

    Replaces the plain ``token != expected`` comparisons that used to live
    inline in apps/cron_tick_routes.py and apps/channels/discord.py — one of
    those two call sites already used hmac.compare_digest, the other didn't;
    this is the single implementation both now share. Returns False (never
    raises) for an unset/empty ``expected`` so callers can't accidentally
    treat "no secret configured" as "anything passes".
    """
    if not expected:
        return False
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[len("bearer "):].strip()
    return hmac.compare_digest(token, expected)


def require_secret(secret: object, env_var_name: str) -> str:
    """Pull the value out of a pydantic SecretStr | None settings field,
    raising ValueError(env_var_name) when it's unset — callers turn that into
    their own "endpoint disabled, name the missing var" HTTPException.
    """
    value = secret.get_secret_value() if secret is not None else None
    if not value:
        raise ValueError(env_var_name)
    return value


def _hmac_hex(secret: str, timestamp: str, nonce: str, body: bytes) -> str:
    """The MAC covers timestamp, nonce AND the body bytes — all three, in a
    fixed, unambiguous framing (length-implicit via the "." separators being
    outside the possible header charset for these fields is not relied on;
    what matters is that all three inputs are mixed into the digest, so a
    caller can't swap the body, the nonce, or the timestamp post-signing
    without invalidating the signature).
    """
    mac = hmac.new(secret.encode("utf-8"), digestmod=hashlib.sha256)
    mac.update(timestamp.encode("utf-8"))
    mac.update(b"\n")
    mac.update(nonce.encode("utf-8"))
    mac.update(b"\n")
    mac.update(body)
    return mac.hexdigest()


@dataclass(frozen=True)
class SignedEnvelope:
    timestamp: str
    nonce: str
    signature: str

    def headers(self) -> dict[str, str]:
        return {
            "X-Turtle-Timestamp": self.timestamp,
            "X-Turtle-Nonce": self.nonce,
            "X-Turtle-Signature": self.signature,
        }


def sign_request(secret: str, body: bytes) -> SignedEnvelope:
    """Build the signature envelope for an outgoing self-call.

    ``body`` MUST be the EXACT bytes that will be sent on the wire (e.g. fed
    to httpx's ``content=``, not ``json=``, which could re-serialise and
    change the bytes) — see ``verify_request`` for why the two sides must
    agree on literal bytes rather than a re-serialised dict.
    """
    timestamp = str(int(time.time()))
    nonce = uuid.uuid4().hex
    signature = _hmac_hex(secret, timestamp, nonce, body)
    return SignedEnvelope(timestamp=timestamp, nonce=nonce, signature=signature)


async def verify_request(
    secret: str,
    timestamp: str | None,
    nonce: str | None,
    signature: str | None,
    body: bytes,
) -> None:
    """Verify a self-call's signature envelope against ``body`` — the RAW
    request bytes as received (``await request.body()``), never a
    re-serialised dict. Re-serialising lets a JSON formatting difference
    (key order, whitespace, float formatting) break a legitimate request, or
    worse, lets two semantically-different bodies canonicalise to the same
    JSON text, silently defeating the "body is covered by the signature"
    guarantee. Hashing the literal bytes closes both holes.

    Raises SignatureError on any failure: missing envelope fields, malformed
    timestamp, timestamp outside +/-CLOCK_SKEW_S, signature mismatch, replay
    store unavailable, or nonce already claimed.
    """
    if not timestamp or not nonce or not signature:
        raise SignatureError("missing signature envelope")

    try:
        ts = int(timestamp)
    except ValueError as exc:
        raise SignatureError("malformed timestamp") from exc

    now = int(time.time())
    if abs(now - ts) > CLOCK_SKEW_S:
        raise SignatureError("timestamp outside allowed window")

    expected_signature = _hmac_hex(secret, timestamp, nonce, body)
    if not hmac.compare_digest(expected_signature, signature):
        raise SignatureError("signature mismatch")

    # Replay claim: SET key value NX EX — first caller to claim a given
    # nonce wins; the SET returns falsy for anyone replaying it (even a
    # captured, otherwise-valid request, even inside the window), and they
    # are rejected. This is the first atomic Redis claim in the codebase
    # (the house style elsewhere for atomic claims is Postgres
    # INSERT...ON CONFLICT DO NOTHING, see
    # core/storage/cloud/routine_last_fired_store.py::try_claim_fire) — no
    # existing Redis idiom to match, so this establishes one.
    try:
        client = await get_redis_client()
        claimed = await client.set(
            f"{_NONCE_KEY_PREFIX}{nonce}", "1", nx=True, ex=NONCE_TTL_S
        )
    except CloudBackendUnavailable as exc:
        # Fail CLOSED — see this module's docstring for the reasoning.
        raise SignatureError(f"replay store unavailable: {exc}") from exc
    if not claimed:
        raise SignatureError("nonce already used")


async def store_job_payload(payload: dict) -> str:
    """Payload-by-reference, caller side: stash the real job body (the one
    carrying identity fields like user_id/discord_user_id) in Redis under an
    opaque id and hand back just that id. The self-call then sends ONLY the
    id — a forged wire body carrying a fabricated user_id has nothing to
    point at.

    Raises CloudBackendUnavailable if Redis isn't configured; callers catch
    that and fall back to running the job in-process (see this module's
    docstring).
    """
    job_id = uuid.uuid4().hex
    client = await get_redis_client()
    await client.set(f"{_JOB_KEY_PREFIX}{job_id}", json.dumps(payload), ex=JOB_PAYLOAD_TTL_S)
    return job_id


async def take_job_payload(job_id: str) -> dict | None:
    """Payload-by-reference, callee side: atomically read-and-delete the
    stored payload so a replay of the outer request (even inside the 300s
    signature window) finds nothing to act on. Returns None if the id is
    unknown or already consumed/expired.

    Uses Redis GETDEL (atomic single command — confirmed present on both the
    sync and async clients in the pinned redis-py 5.2.x; no pipeline/
    transaction fallback needed).
    """
    client = await get_redis_client()
    raw = await client.getdel(f"{_JOB_KEY_PREFIX}{job_id}")
    if raw is None:
        return None
    return json.loads(raw)
