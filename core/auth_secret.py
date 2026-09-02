"""
core/auth_secret.py
-------------------
The ONE source of truth for the JWT signing secret used across apps/auth.py,
apps/onboarding_routes.py, and apps/admin_routes.py.

Codex adversarial review, 2026-08-09: the previous per-module ``_secret()``
helpers each fell back to the *literal* string ``"dev-fallback-secret"`` (or
``"dev-fallback-secret-do-not-use-in-production-32b"``) when ``AUTH_SECRET_KEY``
was unset. Two paths were network-reachable with that fallback:

  * ``apps/auth.py.verify_token`` — used no cloud gate at all, so a cloud
    deployment that forgot to set ``AUTH_SECRET_KEY`` accepted tokens signed
    with the literal.
  * ``apps/onboarding_routes.py._secret`` — did gate on cloud, but a *tunneled*
    local deploy is equally reachable and used the literal too.

A public deployment therefore accepted forged tokens for any ``user_id`` on
every memory endpoint, WebSocket, confirmation action, and the new
``/api/account/link`` route.

This module replaces every static fallback with a **process-local random**
secret when ``AUTH_SECRET_KEY`` is unset:

  * secrets.token_urlsafe(48) is generated once at import time
  * shared across all three modules so tokens issued by one path verify in
    another (previously they didn't, because two different literals were used)
  * printed to the log as a plainly-scary warning
  * regenerated every process restart, invalidating any leaked token

Cloud still hard-fails startup without an explicit ``AUTH_SECRET_KEY``. Any
non-loopback bind should as well; that check lives at the FastAPI startup hook
that reads this module.
"""
from __future__ import annotations

import secrets
import threading

from core.config import settings

_lock = threading.Lock()
_cached: str | None = None


def auth_secret() -> str:
    """Return the JWT signing secret.

    Raises RuntimeError in cloud mode when ``AUTH_SECRET_KEY`` is not set —
    fail fast at first request rather than silently accept forged tokens.
    """
    configured = (
        settings.auth_secret_key.get_secret_value()
        if settings.auth_secret_key is not None
        else ""
    )
    if configured:
        return configured

    if settings.is_cloud:
        raise RuntimeError(
            "AUTH_SECRET_KEY is required in cloud mode but is empty or unset."
        )

    global _cached
    if _cached is None:
        with _lock:
            if _cached is None:
                _cached = secrets.token_urlsafe(48)
                print(
                    "WARN: AUTH_SECRET_KEY is unset — generated a PROCESS-RANDOM "
                    "development secret. Tokens will be invalidated on restart. "
                    "Set AUTH_SECRET_KEY for a stable secret (required in cloud).",
                    flush=True,
                )
    return _cached


def is_using_fallback_secret() -> bool:
    """Diagnostic: True when we are using a randomly-generated dev secret.

    Used by the startup binding check so a non-loopback dev bind refuses to
    start silently — see apps/turtle_server.py startup hook.
    """
    configured = (
        settings.auth_secret_key.get_secret_value()
        if settings.auth_secret_key is not None
        else ""
    )
    return not bool(configured)


def _reset_for_tests() -> None:
    """Test-only: force the next auth_secret() call to regenerate."""
    global _cached
    with _lock:
        _cached = None
