"""
apps/auth.py
------------
G4: Authentication and session token validation.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import WebSocket, WebSocketException, status

from core.config import settings

ALGORITHM = "HS256"


def create_session_token(user_id: str, channel: str = "web") -> str:
    """Create a short-lived JWT token for WebSocket connection."""
    expire = datetime.now(UTC) + timedelta(hours=24)
    to_encode = {"sub": user_id, "channel": channel, "exp": expire}
    
    from core.auth_secret import auth_secret
    secret = auth_secret()
    encoded_jwt = jwt.encode(to_encode, secret, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify the JWT token and return payload."""
    from core.auth_secret import auth_secret
    secret = auth_secret()
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")


async def authenticate_websocket(websocket: WebSocket) -> str:
    """
    Resolve the WebSocket caller's user_id.

    Resolution order:
      1. ``turtle_uid`` cookie minted by the magic-link onboarding flow
         (apps/onboarding_routes.py).
      2. ``?token=...`` query param (legacy native-client path).
      3. ``Authorization: Bearer ...`` header.
      4. Dev-only fallback to a shared local_dev_user when TURTLE_DEV_ANON=1.

    Closes the connection with 1008 if none resolve.
    """
    # 1. Cookie set by /onboarding/claim.
    cookie_token = websocket.cookies.get("turtle_uid")
    if cookie_token:
        from apps.onboarding_routes import verify_session_cookie

        user_id = verify_session_cookie(cookie_token)
        if user_id:
            return user_id

    # 2/3. Query-param or Authorization header (legacy).
    token = websocket.query_params.get("token")
    if not token and "Authorization" in websocket.headers:
        auth_header = websocket.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

    # 4. Dev-only escape hatch.
    if not token and settings.dev_anon and not settings.is_cloud:
        from core.identity import identity_manager
        await identity_manager.init_db()
        print(
            "WARN: auth falling back to local_dev_user (TURTLE_DEV_ANON=1). "
            "No identity.md is seeded for this user — Turtle will ask for the name. "
            "Go through /onboarding/start to use a real identity."
        )
        return await identity_manager.resolve_user("web", "local_dev_user")

    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing authentication token")
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Missing authentication token")

    try:
        payload = verify_token(token)
        return payload["sub"]
    except ValueError as e:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(e))
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason=str(e))
