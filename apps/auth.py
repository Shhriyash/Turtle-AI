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
    
    secret = settings.auth_secret_key.get_secret_value() if settings.auth_secret_key else "dev-fallback-secret"
    encoded_jwt = jwt.encode(to_encode, secret, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify the JWT token and return payload."""
    secret = settings.auth_secret_key.get_secret_value() if settings.auth_secret_key else "dev-fallback-secret"
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")


async def authenticate_websocket(websocket: WebSocket) -> str:
    """
    Extracts token from query parameters ?token=... or headers.
    Returns user_id. Closes connection if invalid.
    """
    token = websocket.query_params.get("token")
    if not token and "Authorization" in websocket.headers:
        auth_header = websocket.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

    # In local mode, if no token is provided, fallback to a default local user to ease dev
    if not token and not settings.is_cloud:
        from core.identity import identity_manager
        await identity_manager.init_db()
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
