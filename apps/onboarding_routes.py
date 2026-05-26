"""
apps/onboarding_routes.py
-------------------------
Phase 4 — magic-link onboarding.

Endpoints:
    POST /onboarding/start   -> mints a short-lived JWT, emails the user a
                                claim link via the existing Gmail SMTP bot.
    GET  /onboarding/claim   -> verifies the JWT, seeds identity.md with the
                                form facts, sets a signed turtle_uid cookie,
                                and redirects to /.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, EmailStr, Field

from core.config import settings
from core.identity import identity_manager
from core.personal_memory_store import PersonalMemoryStore
from core.telemetry import emit as emit_event


ALGORITHM = "HS256"

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onboarding", tags=["onboarding"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DEV_FALLBACK_SECRET = "dev-fallback-secret-do-not-use-in-production-32b"


def _secret() -> str:
    raw = (
        settings.auth_secret_key.get_secret_value()
        if settings.auth_secret_key is not None
        else ""
    )
    if not raw:
        # Cloud deploys must set AUTH_SECRET_KEY explicitly. Locally, fall back
        # to a >=32-byte literal so PyJWT doesn't warn about short HMAC keys.
        if settings.is_cloud:
            raise RuntimeError(
                "AUTH_SECRET_KEY is required in cloud mode but is empty or unset."
            )
        return _DEV_FALLBACK_SECRET
    return raw


def _now() -> datetime:
    return datetime.now(UTC)


# Per-IP onboarding-start rate limiter (in-process; swap for Redis later).
_recent_starts: dict[str, list[float]] = {}


def _check_rate_limit(ip: str) -> None:
    """Allow at most settings.onboarding_rate_limit_per_hour starts per IP."""
    if not ip:
        return
    limit = max(1, int(settings.onboarding_rate_limit_per_hour))
    now = time.time()
    cutoff = now - 3600
    history = [t for t in _recent_starts.get(ip, []) if t > cutoff]
    if len(history) >= limit:
        raise HTTPException(
            status_code=429,
            detail="Too many onboarding requests from this address. Try again later.",
        )
    history.append(now)
    _recent_starts[ip] = history


def _is_valid_email(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value or ""))


def issue_session_cookie_value(user_id: str) -> str:
    """Mint the long-lived JWT carried in the turtle_uid cookie."""
    expire = _now() + timedelta(days=settings.session_cookie_ttl_days)
    return jwt.encode(
        {"sub": user_id, "channel": "web", "exp": expire},
        _secret(),
        algorithm=ALGORITHM,
    )


def verify_session_cookie(token: str) -> str | None:
    """Return the user_id encoded in a turtle_uid cookie, or None if invalid."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
        sub = payload.get("sub")
        return sub if isinstance(sub, str) and sub else None
    except jwt.PyJWTError as exc:
        logger.warning(
            "session-cookie verify failed: %s (token_len=%d, secret_len=%d)",
            type(exc).__name__, len(token), len(_secret()),
        )
        return None


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class OnboardingStartRequest(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=80)
    timezone: str = Field(default="UTC", max_length=80)


# ---------------------------------------------------------------------------
# Email body
# ---------------------------------------------------------------------------


def _claim_link(token: str) -> str:
    base = settings.public_base_url.rstrip("/")
    return f"{base}/onboarding/claim?token={token}"


def _email_html(name: str, link: str, ttl_minutes: int) -> str:
    safe_name = (name or "there").replace("<", "&lt;").replace(">", "&gt;")
    return f"""\
<!doctype html>
<html><body style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f5f7fa;padding:32px;">
  <div style="max-width:480px;margin:0 auto;background:#fff;border-radius:12px;padding:32px;border:1px solid #e3e6eb;">
    <h2 style="margin:0 0 12px;color:#1c1f26;">Hi {safe_name},</h2>
    <p style="color:#4b5160;line-height:1.5;">Click the button below to start chatting with Turtle. The link expires in {ttl_minutes} minutes.</p>
    <p style="margin:24px 0;">
      <a href="{link}" style="display:inline-block;background:#4fd1c5;color:#08221f;text-decoration:none;padding:12px 20px;border-radius:8px;font-weight:600;">Start chatting</a>
    </p>
    <p style="color:#8a93a6;font-size:12px;">If the button doesn't work, paste this URL into your browser:<br />{link}</p>
  </div>
</body></html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/start")
async def onboarding_start(req: Request, body: OnboardingStartRequest) -> JSONResponse:
    """Mint a magic-link JWT and email it to the user."""
    client_ip = req.client.host if req.client else ""
    _check_rate_limit(client_ip)

    email = body.email.strip().lower()
    name = body.name.strip()
    timezone = (body.timezone or "UTC").strip() or "UTC"

    if not _is_valid_email(email):
        raise HTTPException(status_code=400, detail="Invalid email address.")

    await identity_manager.init_db()
    user_id = await identity_manager.resolve_user("web_email", email)

    # Seed identity.md immediately on form submission so the user's name is
    # known even before they click the magic link. The /claim handler will
    # re-seed (idempotent) on first successful claim.
    seed_lines: list[str] = []
    if name:
        seed_lines.append(f"- Name: {name}")
    if email:
        seed_lines.append(f"- Primary email: {email}")
    if timezone:
        seed_lines.append(f"- Timezone: {timezone}")
    if seed_lines:
        try:
            PersonalMemoryStore(user_id=user_id).write_topic(
                "identity", seed_lines, metadata={"source": "explicit"}
            )
        except Exception:
            logger.exception(
                "onboarding/start: identity seed write failed for user_id=%s", user_id
            )
        # Also journal the identity facts as applied events so the replayer
        # (which rebuilds identity.md from journal on session end) preserves
        # them. Without this, replay sees zero applied identity events and
        # unlinks the seeded file.
        try:
            from core.memory_journal import JournalStore, make_event

            journal = JournalStore(user_id=user_id)
            ord_idx = 0
            if name:
                journal.append(make_event(
                    kind="fact", topic="identity", key="identity.name",
                    value={"name": name}, confidence=1.0,
                    source="explicit", extractor="deterministic",
                    session_id="onboarding", turn_id=f"onboarding_{ord_idx}",
                    applied=True,
                ))
                ord_idx += 1
            if email:
                journal.append(make_event(
                    kind="fact", topic="identity", key="identity.primary_email",
                    value={"primary_email": email}, confidence=1.0,
                    source="explicit", extractor="deterministic",
                    session_id="onboarding", turn_id=f"onboarding_{ord_idx}",
                    applied=True,
                ))
                ord_idx += 1
            if timezone:
                journal.append(make_event(
                    kind="fact", topic="identity", key="identity.timezone",
                    value={"timezone": timezone}, confidence=1.0,
                    source="explicit", extractor="deterministic",
                    session_id="onboarding", turn_id=f"onboarding_{ord_idx}",
                    applied=True,
                ))
        except Exception:
            logger.exception(
                "onboarding/start: identity journal-event write failed for user_id=%s",
                user_id,
            )

    ttl = max(1, int(settings.magic_link_jwt_ttl_minutes))
    expire = _now() + timedelta(minutes=ttl)
    token = jwt.encode(
        {
            "sub": user_id,
            "kind": "onboarding_claim",
            "email": email,
            "name": name,
            "tz": timezone,
            "jti": uuid.uuid4().hex,
            "exp": expire,
        },
        _secret(),
        algorithm=ALGORITHM,
    )

    # Send the email via Turtle's own outbound channel.
    from tools.email_tools.config import create_email_tool_from_env

    email_tool = create_email_tool_from_env()
    if email_tool is None:
        raise HTTPException(
            status_code=503,
            detail="Email sending is not configured on this server.",
        )

    link = _claim_link(token)
    try:
        result = email_tool.send_email(
            receiver=email,
            subject="Your link to start with Turtle",
            body=_email_html(name, link, ttl),
            content_type="html",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to send email: {e!s}")

    if isinstance(result, str) and result.lower().startswith("error"):
        raise HTTPException(status_code=502, detail=result)

    emit_event("onboarding_start", user_id=user_id, channel="web_email")

    response = JSONResponse({"status": "sent"})

    # In local/dev mode, set the session cookie immediately so the user is
    # authenticated as `user_id` when they open the chat tab — no need to click
    # the magic-link email to start using Turtle. In cloud/prod we still
    # require the /claim round-trip to prove email ownership.
    if not settings.is_cloud:
        cookie_value = issue_session_cookie_value(user_id)
        response.set_cookie(
            key="turtle_uid",
            value=cookie_value,
            max_age=settings.session_cookie_ttl_days * 24 * 60 * 60,
            httponly=True,
            samesite="lax",
            secure=False,
            path="/",
        )
        logger.info(
            "onboarding/start: dev-mode session cookie issued for user_id=%s "
            "(skipping magic-link click)", user_id
        )

    return response


@router.get("/claim")
async def onboarding_claim(token: str) -> Any:
    """Verify the magic-link JWT, seed identity.md, set cookie, redirect to /."""
    try:
        payload = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="Link expired. Request a new one.")
    except jwt.PyJWTError:
        raise HTTPException(status_code=400, detail="Invalid link.")

    if payload.get("kind") != "onboarding_claim":
        raise HTTPException(status_code=400, detail="Invalid link.")

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=400, detail="Invalid link.")

    jti = payload.get("jti")
    if not isinstance(jti, str) or not jti:
        # Legacy tokens minted before jti was added — fall back to user_id+iat shape.
        jti = f"legacy:{user_id}:{payload.get('exp')}"

    await identity_manager.init_db()
    first_claim = await identity_manager.mark_token_claimed(jti, user_id)
    if not first_claim:
        raise HTTPException(status_code=410, detail="Link already used.")

    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip()
    tz = (payload.get("tz") or "UTC").strip() or "UTC"

    # Seed identity.md directly (source=explicit, bypass confirmation gate).
    # Single-use jti guarantees this only runs once per token, so the Cohere
    # embed pass for identity rows fires at most once per onboarding.
    store = PersonalMemoryStore(user_id=user_id)
    seed_lines: list[str] = []
    if name:
        seed_lines.append(f"- Name: {name}")
    if email:
        seed_lines.append(f"- Primary email: {email}")
    if tz:
        seed_lines.append(f"- Timezone: {tz}")
    if seed_lines:
        try:
            store.write_topic("identity", seed_lines, metadata={"source": "explicit"})
        except Exception:
            # Don't block the redirect if seeding fails — identity can still be
            # filled in by the regular extraction pipeline on the first turn.
            logger.exception("onboarding: identity seed write failed for user_id=%s", user_id)

    emit_event("onboarding_complete", user_id=user_id, channel="web_email")
    cookie_value = issue_session_cookie_value(user_id)
    # Use a 200 HTML response with JS navigation instead of a 303 redirect.
    # Some email clients (Gmail's link rewriter, preview prefetchers) and some
    # browsers don't auto-follow the Location header after click-throughs from
    # external origins, which leaves the user staring at the stale onboarding
    # tab. A real HTML page in the user's actual tab guarantees navigation.
    html = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Signing you in…</title>"
        "<meta http-equiv=\"refresh\" content=\"0; url=/\">"
        "<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
        "background:#f5f7fa;color:#1c1f26;display:flex;align-items:center;"
        "justify-content:center;height:100vh;margin:0}</style></head>"
        "<body><p>Signing you in…</p>"
        "<script>window.location.replace('/');</script></body></html>"
    )
    response = HTMLResponse(content=html, status_code=200)
    response.set_cookie(
        key="turtle_uid",
        value=cookie_value,
        max_age=settings.session_cookie_ttl_days * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
        secure=settings.is_cloud,
        path="/",
    )
    return response
