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

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jwt
import aiosqlite
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, EmailStr

from core.config import settings
from core.identity import identity_manager
from core.paths import PERSONAL_MEMORY_DIR, RAG_DATA_DIR
from core.telemetry import emit as emit_event


ALGORITHM = "HS256"
router = APIRouter(tags=["admin"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _secret() -> str:
    if settings.auth_secret_key is None:
        return "dev-fallback-secret"
    return settings.auth_secret_key.get_secret_value()


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
    """Return a list of users with rough usage metrics. Read-only."""
    _require_admin(x_admin_token)

    await identity_manager.init_db()

    users: list[dict[str, Any]] = []
    async with aiosqlite.connect(identity_manager.db_path) as db:
        async with db.execute(
            "SELECT user_id, primary_email, created_at FROM users ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
    for user_id, primary_email, created_at in rows:
        user_dir = PERSONAL_MEMORY_DIR / user_id
        users.append({
            "user_id": user_id,
            "primary_email": primary_email,
            "created_at": created_at,
            "storage_bytes": _dir_size_bytes(user_dir),
            "rag_bytes": _dir_size_bytes(RAG_DATA_DIR / user_id),
            "journal_events": _journal_event_count(user_dir),
            "last_seen": _last_seen(user_dir),
        })

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
    email = body.email.strip().lower()
    await identity_manager.init_db()

    # Resolve without creating — only act when the user exists.
    async with aiosqlite.connect(identity_manager.db_path) as db:
        async with db.execute(
            "SELECT user_id FROM channel_mappings WHERE channel = ? AND channel_user_id = ?",
            ("web_email", email),
        ) as cursor:
            row = await cursor.fetchone()
    # Always return 200 so existence of the email is not leaked.
    if not row:
        return JSONResponse({"status": "sent"})
    user_id = row[0]

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


async def _purge_user(user_id: str) -> dict[str, Any]:
    """Hard-delete every artifact tied to a user_id."""
    removed: dict[str, Any] = {"memory": False, "rag": False, "rows": 0}

    memory_dir = PERSONAL_MEMORY_DIR / user_id
    if memory_dir.exists():
        shutil.rmtree(memory_dir, ignore_errors=True)
        removed["memory"] = not memory_dir.exists()

    rag_dir = RAG_DATA_DIR / user_id
    if rag_dir.exists():
        shutil.rmtree(rag_dir, ignore_errors=True)
        removed["rag"] = not rag_dir.exists()

    async with aiosqlite.connect(identity_manager.db_path) as db:
        cursor = await db.execute(
            "DELETE FROM channel_mappings WHERE user_id = ?", (user_id,)
        )
        removed["rows"] += cursor.rowcount or 0
        cursor = await db.execute(
            "DELETE FROM users WHERE user_id = ?", (user_id,)
        )
        removed["rows"] += cursor.rowcount or 0
        await db.commit()

    return removed


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
