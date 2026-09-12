"""
apps/calendar_oauth_routes.py
------------------------------
Per-user Google Calendar OAuth2 connect flow.

Endpoints:
    GET /integrations/google_calendar/connect   -> redirect to Google's consent screen
    GET /integrations/google_calendar/callback  -> exchange the auth code for tokens,
                                                    persist a per-user token file

The OAuth CLIENT itself (client_id/client_secret) is shared across every
Turtle user — one Google Cloud OAuth client registered via
GOOGLE_CALENDAR_CREDENTIALS_JSON — but each signed-in user authorizes it
individually against their own Google account, and their resulting refresh
token is stored under their own personal memory directory
(personal_memory_dir(user_id)/google_calendar_token.json), never in a shared
env var. tools/calendar_tool.py reads that per-user file first, falling back
to the legacy global GOOGLE_CALENDAR_TOKEN_JSON env var for single-tenant /
dev deployments that never used this flow.

This mirrors the magic-link pattern in apps/onboarding_routes.py: JWTs signed
with the shared auth secret, short TTL, single-purpose "kind" claim so a token
minted for one flow can't be replayed into another.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any
from urllib.parse import urlencode

import httpx
import jwt
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from core.config import settings
from core.paths import personal_memory_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integrations/google_calendar", tags=["google_calendar_oauth"])

ALGORITHM = "HS256"
_STATE_TTL_SECONDS = 600  # 10 minutes is plenty for a consent-screen round trip
_SCOPE = "https://www.googleapis.com/auth/calendar"
_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

_TOKEN_FILENAME = "google_calendar_token.json"


def _secret() -> str:
    """Same process-wide auth secret every other Turtle JWT is signed with."""
    from core.auth_secret import auth_secret
    return auth_secret()


def token_path_for_user(user_id: str):
    """Where a connected user's Calendar OAuth token lives on disk (local
    mode only — cloud mode stores it in Postgres, see _read_token/_write_token/
    _delete_token/_token_exists below)."""
    return personal_memory_dir(user_id) / _TOKEN_FILENAME


async def _read_token(user_id: str) -> str | None:
    """Local disk locally; Postgres in cloud mode (survives a cold start,
    unlike the local file — see core/storage/cloud/calendar_token_store.py).
    """
    if settings.is_cloud:
        from core.storage.cloud.calendar_token_store import get_token_json

        return await asyncio.to_thread(get_token_json, user_id)
    path = token_path_for_user(user_id)
    return path.read_text(encoding="utf-8") if path.exists() else None


async def _write_token(user_id: str, token_json: str) -> None:
    if settings.is_cloud:
        from core.storage.cloud.calendar_token_store import put_token_json

        await asyncio.to_thread(put_token_json, user_id, token_json)
        return
    path = token_path_for_user(user_id)
    path.write_text(token_json, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass  # best-effort on platforms without POSIX chmod semantics (Windows)


async def _delete_token(user_id: str) -> None:
    if settings.is_cloud:
        from core.storage.cloud.calendar_token_store import delete_token_json

        await asyncio.to_thread(delete_token_json, user_id)
        return
    path = token_path_for_user(user_id)
    if path.exists():
        path.unlink()


async def _token_exists(user_id: str) -> bool:
    if settings.is_cloud:
        from core.storage.cloud.calendar_token_store import token_exists

        return await asyncio.to_thread(token_exists, user_id)
    return token_path_for_user(user_id).exists()


def validate_credentials_json(raw: str) -> tuple[bool, str, dict[str, str] | None]:
    """Pure validator for GOOGLE_CALENDAR_CREDENTIALS_JSON.

    Returns (ok, message, client_config). client_config is {"client_id":...,
    "client_secret":...} on success, None on failure. Shared by _client_config
    (raises HTTPException on a live request) and the app-startup check in
    apps/turtle_server.py (logs a warning instead of crashing boot, since a
    misconfigured Calendar integration shouldn't take down the whole app) so
    the two never drift on what "valid" means.

    A real-world failure mode this catches: pasting only the bare client
    secret string (e.g. "GOCSPX-...") instead of the full OAuth client JSON
    Google Cloud Console downloads — that fails json.loads immediately with a
    clear message instead of surfacing as an opaque 503 on first calendar use.
    """
    if not raw or not raw.strip():
        return (
            False,
            "GOOGLE_CALENDAR_CREDENTIALS_JSON is empty. Set it to the OAuth "
            "client JSON downloaded from Google Cloud Console "
            "(APIs & Services > Credentials).",
            None,
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return (
            False,
            f"GOOGLE_CALENDAR_CREDENTIALS_JSON is not valid JSON ({exc}). "
            "Paste the ENTIRE downloaded client JSON file, not just the "
            "client secret string.",
            None,
        )
    if not isinstance(data, dict):
        return (
            False,
            "GOOGLE_CALENDAR_CREDENTIALS_JSON must be a JSON object, "
            f"got {type(data).__name__}.",
            None,
        )
    block = data.get("installed") or data.get("web") or data
    if not isinstance(block, dict):
        return (
            False,
            "GOOGLE_CALENDAR_CREDENTIALS_JSON's 'installed'/'web' block is not an object.",
            None,
        )
    client_id = block.get("client_id")
    client_secret = block.get("client_secret")
    missing = [
        name for name, value in (("client_id", client_id), ("client_secret", client_secret))
        if not value
    ]
    if missing:
        return (
            False,
            f"GOOGLE_CALENDAR_CREDENTIALS_JSON is missing: {', '.join(missing)}. "
            "Paste the full client JSON from Cloud Console, not a partial value.",
            None,
        )
    return True, "ok", {"client_id": client_id, "client_secret": client_secret}


def _client_config() -> dict[str, str]:
    """Parse GOOGLE_CALENDAR_CREDENTIALS_JSON into {client_id, client_secret}.

    Accepts either the raw Google Cloud "Desktop app" / "Web app" client JSON
    (which nests fields under "installed" or "web") or a flat
    {"client_id": ..., "client_secret": ...} object.
    """
    ok, message, config = validate_credentials_json(settings.google_calendar_credentials_json or "")
    if not ok or config is None:
        raise HTTPException(status_code=503, detail=message)
    return config


def _redirect_uri() -> str:
    base = settings.public_base_url.rstrip("/")
    return f"{base}/integrations/google_calendar/callback"


def _require_user(req: Request) -> str:
    """Resolve the signed-in user from the turtle_uid session cookie.

    Reuses the same cookie/verifier the rest of the web app already trusts
    (apps/onboarding_routes.verify_session_cookie) — Calendar connection is
    not its own auth system, it rides on the existing one.
    """
    from apps.onboarding_routes import verify_session_cookie

    cookie_token = req.cookies.get("turtle_uid", "")
    user_id = verify_session_cookie(cookie_token) if cookie_token else None
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Sign in to Turtle first, then connect Google Calendar.",
        )
    return user_id


def _result_page(message: str, *, ok: bool) -> HTMLResponse:
    color = "#1c7a4d" if ok else "#b3261e"
    safe_message = message.replace("<", "&lt;").replace(">", "&gt;")
    html = f"""\
<!doctype html>
<html><head><meta charset="utf-8"><title>Google Calendar</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f5f7fa;
       color:#1c1f26;display:flex;align-items:center;justify-content:center;
       height:100vh;margin:0}}
  .card{{max-width:420px;background:#fff;border-radius:12px;padding:32px;
        border:1px solid #e3e6eb;text-align:center}}
  .status{{color:{color};font-weight:600;margin-bottom:8px}}
</style></head>
<body><div class="card">
  <div class="status">{"Connected" if ok else "Not connected"}</div>
  <p>{safe_message}</p>
</div></body></html>
"""
    return HTMLResponse(content=html, status_code=200 if ok else 400)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/connect")
async def connect(req: Request) -> RedirectResponse:
    """Start the OAuth dance: redirect the signed-in user to Google's consent screen."""
    user_id = _require_user(req)
    client = _client_config()

    # state carries the user_id so the callback knows whose token this is,
    # and is signed + short-lived so it can't be forged or replayed stale.
    state_token = jwt.encode(
        {
            "sub": user_id,
            "kind": "calendar_oauth_state",
            "jti": uuid.uuid4().hex,
            "exp": time.time() + _STATE_TTL_SECONDS,
        },
        _secret(),
        algorithm=ALGORITHM,
    )

    params = {
        "client_id": client["client_id"],
        "redirect_uri": _redirect_uri(),
        "response_type": "code",
        "scope": _SCOPE,
        "access_type": "offline",
        # Force Google to reissue a refresh_token even if this user already
        # granted consent before — without this, a reconnect after revoking
        # access silently comes back with no refresh_token at all.
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state_token,
    }
    return RedirectResponse(f"{_AUTH_ENDPOINT}?{urlencode(params)}", status_code=302)


@router.get("/callback")
async def callback(req: Request, code: str = "", state: str = "", error: str = "") -> HTMLResponse:
    """Google redirects here with an auth code; exchange it and persist the token."""
    if error:
        return _result_page(f"Google Calendar connection was not completed ({error}).", ok=False)
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state parameter.")

    try:
        payload = jwt.decode(state, _secret(), algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        return _result_page(
            "This connect link expired before you finished. Please try connecting again.",
            ok=False,
        )
    except jwt.PyJWTError:
        raise HTTPException(status_code=400, detail="Invalid state parameter.")

    if payload.get("kind") != "calendar_oauth_state":
        raise HTTPException(status_code=400, detail="Invalid state parameter.")
    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=400, detail="Invalid state parameter.")

    # The state's signature + expiry already stop forgery and replay-after-
    # expiry. This extra check stops a *live* state value leaking (e.g. via a
    # shared link or referrer) and being completed from a different signed-in
    # browser than the one that started the flow.
    session_user_id = _require_user(req)
    if session_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="This connect link belongs to a different signed-in account.",
        )

    client = _client_config()
    token_payload = {
        "code": code,
        "client_id": client["client_id"],
        "client_secret": client["client_secret"],
        "redirect_uri": _redirect_uri(),
        "grant_type": "authorization_code",
    }

    # The auth code is single-use, so a bare connection blip must not burn it
    # without a fight — retry once on a transient connection-level failure
    # (never on an HTTP error response, which is a real rejection from Google
    # and retrying it would just waste the code a second time for nothing).
    resp: httpx.Response | None = None
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=20.0) as http_client:
                resp = await http_client.post(_TOKEN_ENDPOINT, data=token_payload)
            break
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            last_exc = exc
            logger.warning(
                "calendar oauth: token exchange attempt %d/2 failed: %s: %s",
                attempt + 1, type(exc).__name__, exc,
            )
            continue
        except httpx.HTTPError as exc:
            logger.exception("calendar oauth: token exchange network error")
            return _result_page(
                f"Could not reach Google ({type(exc).__name__}: {exc or 'no details'}). "
                "Please try connecting again.",
                ok=False,
            )

    if resp is None:
        logger.error("calendar oauth: token exchange failed after retry: %s", last_exc)
        detail = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown error"
        return _result_page(
            f"Could not reach Google after retrying ({detail}). "
            "Check your network connection and try connecting again.",
            ok=False,
        )

    if resp.status_code >= 400:
        logger.error(
            "calendar oauth: token exchange failed status=%s body=%s",
            resp.status_code, resp.text[:400],
        )
        return _result_page(
            "Google rejected the connection request. Please try again.", ok=False
        )

    token_data: dict[str, Any] = resp.json()
    if "refresh_token" not in token_data:
        # We always send prompt=consent, so Google should always include one;
        # log loudly if it doesn't, since silent-degrade here means the
        # calendar tool will work until the short-lived access token expires
        # and then fail with no way to self-heal.
        logger.warning(
            "calendar oauth: no refresh_token in Google's response for user_id=%s "
            "— reconnect will be required once the access token expires", user_id,
        )

    await _write_token(user_id, json.dumps(token_data, indent=2))

    logger.info("calendar oauth: connected Google Calendar for user_id=%s", user_id)
    return _result_page(
        "Google Calendar is connected. You can close this tab and go back to Turtle.",
        ok=True,
    )


@router.get("/status")
async def status(req: Request) -> dict[str, bool]:
    """Whether the signed-in user currently has a Calendar token stored."""
    user_id = _require_user(req)
    return {"connected": await _token_exists(user_id)}


@router.post("/disconnect")
async def disconnect(req: Request) -> dict[str, bool]:
    """Delete the signed-in user's stored Calendar token."""
    user_id = _require_user(req)
    await _delete_token(user_id)
    return {"connected": False}
