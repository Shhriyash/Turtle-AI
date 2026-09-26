"""
apps/admin_routes.py
--------------------
Phase 7 — operational endpoints.

Endpoints:
    GET  /admin/users          — list users with storage + activity stats
    POST /forget-me            — request GDPR-style deletion (sends magic link)
    GET  /forget-me/confirm    — verify magic link and delete user data

Auth model:
    * Admin endpoints require header ``X-Admin-Token: <settings.admin_token>``.
      When ``admin_token`` is unset, the route returns 503 so a misconfigured
      cloud deploy fails loud instead of leaking data.
    * /forget-me does not require admin auth — it relies on possession of the
      user's email inbox (the same trust boundary as the magic-link login).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jwt
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, EmailStr

from core.config import settings
from core.identity import identity_manager, normalize_email
from core.paths import PERSONAL_MEMORY_DIR, RAG_DATA_DIR
from core.telemetry import emit as emit_event
from core.tenant_purge import purge_user as _purge_user


ALGORITHM = "HS256"
router = APIRouter(tags=["admin"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _secret() -> str:
    """Delegate to the centralised auth secret — see apps/onboarding_routes."""
    from core.auth_secret import auth_secret
    return auth_secret()


def _require_admin(token: str | None) -> None:
    expected = (
        settings.admin_token.get_secret_value()
        if settings.admin_token is not None
        else None
    )
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Admin endpoints are disabled (TURTLE_ADMIN_TOKEN not set).",
        )
    if not token or token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized.")


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _journal_event_count(user_dir: Path) -> int:
    journal_dir = user_dir / "journal"
    if not journal_dir.exists():
        return 0
    count = 0
    for shard in journal_dir.glob("*/events.jsonl"):
        try:
            with shard.open("r", encoding="utf-8") as fh:
                for _ in fh:
                    count += 1
        except OSError:
            pass
    return count


def _last_seen(user_dir: Path) -> str | None:
    latest: float = 0.0
    if not user_dir.exists():
        return None
    try:
        for entry in user_dir.rglob("*"):
            if entry.is_file():
                try:
                    mtime = entry.stat().st_mtime
                    if mtime > latest:
                        latest = mtime
                except OSError:
                    pass
    except OSError:
        return None
    if latest <= 0:
        return None
    return datetime.fromtimestamp(latest, tz=UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# /admin/users
# ---------------------------------------------------------------------------


@router.get("/admin/users")
async def admin_users(x_admin_token: str | None = Header(default=None)) -> JSONResponse:
    """Return a list of users with rough usage metrics. Read-only.

    Goes through identity_manager.list_users() (the async surface both
    IdentityManager and PostgresIdentityManager expose) instead of a raw
    ``aiosqlite.connect(identity_manager.db_path)`` query — that attribute
    only exists on the local manager, so the old direct-SQLite version raised
    AttributeError on every call in cloud mode.

    The per-user filesystem stats (storage_bytes/rag_bytes/journal_events/
    last_seen) only mean anything against local disk — cloud mode has no
    such filesystem, so those fields are reported as null there rather than
    walking a directory tree that doesn't exist for that deploy.
    """
    _require_admin(x_admin_token)

    await identity_manager.init_db()
    rows = await identity_manager.list_users()

    users: list[dict[str, Any]] = []
    for row in rows:
        user_id = row["user_id"]
        entry: dict[str, Any] = {
            "user_id": user_id,
            "primary_email": row["primary_email"],
            "created_at": row["created_at"],
        }
        if settings.is_cloud:
            entry.update({
                "storage_bytes": None,
                "rag_bytes": None,
                "journal_events": None,
                "last_seen": None,
            })
        else:
            user_dir = PERSONAL_MEMORY_DIR / user_id
            entry.update({
                "storage_bytes": _dir_size_bytes(user_dir),
                "rag_bytes": _dir_size_bytes(RAG_DATA_DIR / user_id),
                "journal_events": _journal_event_count(user_dir),
                "last_seen": _last_seen(user_dir),
            })
        users.append(entry)

    return JSONResponse({
        "users": users,
        "count": len(users),
        "storage_cap_mb": settings.user_storage_cap_mb,
    })


# ---------------------------------------------------------------------------
# /forget-me  (GDPR delete)
# ---------------------------------------------------------------------------


class ForgetMeRequest(BaseModel):
    email: EmailStr


def _forget_link(token: str) -> str:
    return f"{settings.public_base_url.rstrip('/')}/forget-me/confirm?token={token}"


def _forget_email_html(link: str, ttl_minutes: int) -> str:
    return f"""\
<!doctype html>
<html><body style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f5f7fa;padding:32px;">
  <div style="max-width:480px;margin:0 auto;background:#fff;border-radius:12px;padding:32px;border:1px solid #e3e6eb;">
    <h2 style="margin:0 0 12px;color:#1c1f26;">Delete your Turtle data?</h2>
    <p style="color:#4b5160;line-height:1.5;">If you requested deletion, click the button below. This permanently removes your memory, RAG index, and account. The link expires in {ttl_minutes} minutes.</p>
    <p style="margin:24px 0;">
      <a href="{link}" style="display:inline-block;background:#e05a5a;color:#fff;text-decoration:none;padding:12px 20px;border-radius:8px;font-weight:600;">Delete my data</a>
    </p>
    <p style="color:#8a93a6;font-size:12px;">If you did NOT request this, ignore this email. Your data stays put.</p>
  </div>
</body></html>
"""


@router.post("/forget-me")
async def forget_me_start(req: Request, body: ForgetMeRequest) -> JSONResponse:
    """Send a confirmation link to the user's primary email."""
    # Normalize through the shared helper so the lookup matches the exact key
    # resolve_user() stored in channel_mappings (same casing/whitespace policy).
    email = normalize_email(body.email)
    await identity_manager.init_db()

    # Resolve without creating — only act when the user exists. lookup_user
    # is the shared non-minting async surface both managers expose (see
    # core/identity.py); this used to be a raw
    # aiosqlite.connect(identity_manager.db_path) query, which raised
    # AttributeError in cloud mode (PostgresIdentityManager has no db_path).
    user_id = await identity_manager.lookup_user("web_email", email)
    # Always return 200 so existence of the email is not leaked.
    if not user_id:
        return JSONResponse({"status": "sent"})

    ttl = max(1, int(settings.magic_link_jwt_ttl_minutes))
    expire = datetime.now(UTC) + timedelta(minutes=ttl)
    token = jwt.encode(
        {
            "sub": user_id,
            "kind": "forget_me",
            "email": email,
            "exp": expire,
        },
        _secret(),
        algorithm=ALGORITHM,
    )

    from tools.email_tools.config import create_email_tool_from_env

    email_tool = create_email_tool_from_env()
    if email_tool is None:
        raise HTTPException(
            status_code=503,
            detail="Email sending is not configured on this server.",
        )

    link = _forget_link(token)
    try:
        result = email_tool.send_email(
            receiver=email,
            subject="Confirm deletion of your Turtle data",
            body=_forget_email_html(link, ttl),
            content_type="html",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to send email: {e!s}")
    if isinstance(result, str) and result.lower().startswith("error"):
        raise HTTPException(status_code=502, detail=result)

    emit_event("forget_me_requested", user_id=user_id)
    return JSONResponse({"status": "sent"})


@router.get("/forget-me/confirm")
async def forget_me_confirm(token: str) -> HTMLResponse:
    """Verify the deletion JWT and purge the user."""
    try:
        payload = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="Link expired. Request a new one.")
    except jwt.PyJWTError:
        raise HTTPException(status_code=400, detail="Invalid link.")

    if payload.get("kind") != "forget_me":
        raise HTTPException(status_code=400, detail="Invalid link.")

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=400, detail="Invalid link.")

    await identity_manager.init_db()
    removed = await _purge_user(user_id)
    emit_event("forget_me_completed", user_id=user_id, removed=removed)

    body = """\
<!doctype html>
<html><body style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0f1115;color:#e8eaf0;padding:48px;">
  <div style="max-width:480px;margin:0 auto;background:#181b22;border:1px solid #262a33;border-radius:14px;padding:32px;">
    <h2 style="margin:0 0 12px;">Your data has been deleted.</h2>
    <p style="color:#8a93a6;line-height:1.5;">All memory, RAG embeddings, and account rows tied to your email are gone. You can sign up again any time.</p>
  </div>
</body></html>
"""
    response = HTMLResponse(body)
    # Best-effort: clear the session cookie if present in the same browser.
    response.delete_cookie("turtle_uid", path="/")
    return response
