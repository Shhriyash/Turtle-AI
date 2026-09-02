"""
scripts/testbench/client.py
---------------------------
TurtleClient — drives a LIVE Turtle server (WS + REST) for the synthetic-human
test harness.

Responsibilities
  * Onboarding + cookie capture (dev mode: /onboarding/start sets turtle_uid
    directly, no magic-link click), with reuse from persona_session.json so
    re-runs keep the SAME cookie -> the same on-disk memory dir.
  * An async WebSocket `send_turn(text) -> TurnResult` that measures TTFR
    (wall-clock ms to the first `done` frame), captures server-side timings,
    and records every frame seen.
  * Auto-confirmation of pending memory. Two surfaces are exercised, exactly
    as a real user would:
        1. inline: when a `confirmation_prompt` frame arrives mid-turn, POST
           /api/memory/confirm for each event_id.
        2. batch: `drain_pending()` polls /api/memory/pending and confirms
           everything queued by the async extractor after the turn settles.

The confirm/pending endpoints resolve the caller via `Authorization: Bearer
<token>`. The turtle_uid cookie value *is* a plain HS256 JWT with `sub`=user_id
signed with the same AUTH_SECRET_KEY the server uses, so we pass the cookie
value straight through as the bearer token — it resolves to the persona's real
user_id and matches the active WS session state.

Robust to timeouts throughout: a missing `timing`/`done` frame never hangs the
harness; the turn just returns with whatever it collected plus an error note.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import websockets

BASE_URL = "http://127.0.0.1:8765"
WS_URL = "ws://127.0.0.1:8765/ws"

TESTBENCH_DIR = Path(__file__).resolve().parent
SESSION_FILE = TESTBENCH_DIR / "persona_session.json"
OUT_DIR = TESTBENCH_DIR / "out"


# ---------------------------------------------------------------------------
# JWT helper (decode only — never verify; we just want the `sub` claim)
# ---------------------------------------------------------------------------
def decode_jwt_sub(token: str) -> str | None:
    """Base64-decode a JWT payload segment and return its `sub`, unverified."""
    try:
        payload_seg = token.split(".")[1]
        pad = "=" * (-len(payload_seg) % 4)
        raw = base64.urlsafe_b64decode(payload_seg + pad)
        return json.loads(raw).get("sub")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class TurnResult:
    reply: str | None = None
    ttfr_ms: float | None = None
    server_total_ms: float | None = None
    server_llm_ms: float | None = None
    frames: list[dict[str, Any]] = field(default_factory=list)
    confirmations_seen: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reply": self.reply,
            "ttfr_ms": self.ttfr_ms,
            "server_total_ms": self.server_total_ms,
            "server_llm_ms": self.server_llm_ms,
            "confirmations_seen": self.confirmations_seen,
            "error": self.error,
            "frames": self.frames,
        }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class TurtleClient:
    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        ws_url: str = WS_URL,
        session_file: Path = SESSION_FILE,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.ws_url = ws_url
        self.session_file = Path(session_file)
        self.email: str | None = None
        self.cookie: str | None = None
        self.user_id: str | None = None
        self.ws: Any = None
        self._http: httpx.AsyncClient | None = None

    # -- persistence of (email -> cookie, user_id) --------------------------
    def _load_sessions(self) -> dict[str, Any]:
        if self.session_file.exists():
            try:
                return json.loads(self.session_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_session(self) -> None:
        data = self._load_sessions()
        data[self.email] = {"cookie": self.cookie, "user_id": self.user_id}
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -- onboarding ---------------------------------------------------------
    def onboard(
        self,
        email: str,
        name: str,
        timezone: str = "America/New_York",
        *,
        reuse: bool = True,
    ) -> str:
        """Ensure a turtle_uid cookie for `email`. Reuse a persisted one if present.

        Returns the cookie value. Synchronous (one-shot httpx call).
        """
        self.email = email
        if reuse:
            existing = self._load_sessions().get(email)
            if existing and existing.get("cookie"):
                self.cookie = existing["cookie"]
                self.user_id = existing.get("user_id") or decode_jwt_sub(self.cookie)
                print(f"[client] reusing cookie for {email} -> user_id={self.user_id}")
                return self.cookie

        with httpx.Client(base_url=self.base_url, timeout=30.0) as c:
            resp = c.post(
                "/onboarding/start",
                json={"email": email, "name": name, "timezone": timezone},
            )
        cookie_val = resp.cookies.get("turtle_uid")
        if not cookie_val:
            # Fall back to parsing the raw Set-Cookie header.
            raw = resp.headers.get("set-cookie", "")
            for part in raw.split(";"):
                if part.strip().startswith("turtle_uid="):
                    cookie_val = part.strip().split("=", 1)[1]
                    break
        if not cookie_val:
            raise RuntimeError(
                f"onboarding/start did not return a turtle_uid cookie "
                f"(status={resp.status_code}, body={resp.text[:200]})"
            )
        self.cookie = cookie_val
        self.user_id = decode_jwt_sub(cookie_val)
        self._save_session()
        print(f"[client] onboarded {email} -> user_id={self.user_id}")
        return self.cookie

    # -- websocket lifecycle ------------------------------------------------
    async def connect(self, *, drain_s: float = 3.0) -> list[dict[str, Any]]:
        if not self.cookie:
            raise RuntimeError("onboard() must run before connect()")
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        headers = {"Cookie": f"turtle_uid={self.cookie}"}
        self.ws = await websockets.connect(self.ws_url, additional_headers=headers)
        ready = await self._collect(drain_s, stop_on_timing=False)
        return ready

    async def close(self) -> None:
        try:
            if self.ws is not None:
                await self.ws.close()
        except Exception:
            pass
        try:
            if self._http is not None:
                await self._http.aclose()
        except Exception:
            pass

    async def __aenter__(self) -> "TurtleClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # -- frame collection ---------------------------------------------------
    def _parse(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, (bytes, bytearray)):
            return {"type": "<binary>", "len": len(raw)}
        try:
            return json.loads(raw)
        except Exception:
            return {"type": "<text>", "raw": str(raw)[:200]}

    async def _collect(
        self, budget_s: float, *, stop_on_timing: bool = True
    ) -> list[dict[str, Any]]:
        """Drain frames for up to budget_s seconds (or until a timing frame)."""
        out: list[dict[str, Any]] = []
        deadline = time.monotonic() + budget_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            except websockets.ConnectionClosed:
                break
            frame = self._parse(raw)
            out.append(frame)
            if stop_on_timing and frame.get("type") == "timing":
                break
        return out

    # -- send one turn ------------------------------------------------------
    async def send_turn(self, text: str, *, timeout: float = 75.0) -> TurnResult:
        """Send a user turn; collect frames until the `timing` frame or timeout.

        TTFR is measured wall-clock from ws.send to the first `done` frame.
        Any `confirmation_prompt` frame seen inline is immediately confirmed.
        """
        res = TurnResult()
        t0 = time.monotonic()
        try:
            await self.ws.send(json.dumps({"type": "text", "content": text}))
        except Exception as e:
            res.error = {"code": "send_failed", "message": str(e)}
            return res

        deadline = t0 + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if res.error is None:
                    res.error = {"code": "timeout", "message": f"no timing frame in {timeout}s"}
                break
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                if res.error is None:
                    res.error = {"code": "timeout", "message": f"recv timeout ({timeout}s)"}
                break
            except websockets.ConnectionClosed as e:
                res.error = {"code": "ws_closed", "message": str(e)}
                break

            frame = self._parse(raw)
            res.frames.append(frame)
            ftype = frame.get("type")

            if ftype == "confirmation_prompt":
                res.confirmations_seen.append(frame)
                for eid in frame.get("event_ids", []) or []:
                    await self.confirm_event(eid)
            elif ftype == "done" and res.reply is None:
                res.reply = frame.get("content")
                res.ttfr_ms = round((time.monotonic() - t0) * 1000, 1)
            elif ftype == "timing":
                res.server_total_ms = frame.get("total_ms")
                res.server_llm_ms = frame.get("llm_ms")
                break
            elif ftype == "error":
                res.error = {
                    "code": frame.get("code"),
                    "message": frame.get("message"),
                }
                # keep collecting briefly in case a done/timing still arrives

        return res

    # -- confirmation REST --------------------------------------------------
    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.cookie}"}

    async def fetch_pending(self) -> list[dict[str, Any]]:
        if self._http is None:
            return []
        try:
            r = await self._http.get(
                "/api/memory/pending", headers=self._auth_headers()
            )
            if r.status_code != 200:
                return []
            return r.json().get("pending", []) or []
        except Exception:
            return []

    async def confirm_event(self, event_id: str, *, accepted: bool = True) -> bool:
        if self._http is None:
            return False
        try:
            r = await self._http.post(
                "/api/memory/confirm",
                json={"event_id": event_id, "accepted": accepted},
                headers=self._auth_headers(),
            )
            return r.status_code == 200
        except Exception:
            return False

    async def drain_pending(
        self, *, rounds: int = 4, delay: float = 1.0
    ) -> list[dict[str, Any]]:
        """Poll pending candidates and confirm them all.

        The extractor queues candidates on a background task, so we poll a few
        times with a short delay to catch late arrivals. Returns the list of
        confirmed items (event_id/topic/key/question).
        """
        confirmed: list[dict[str, Any]] = []
        seen: set[str] = set()
        empty_streak = 0
        for _ in range(rounds):
            pending = await self.fetch_pending()
            if not pending:
                empty_streak += 1
                # Stop early only after we've already confirmed something AND
                # got a clean empty read (nothing left queued).
                if confirmed and empty_streak >= 1:
                    break
                await asyncio.sleep(delay)
                continue
            empty_streak = 0
            for item in pending:
                eid = item.get("event_id")
                if not eid or eid in seen:
                    continue
                if await self.confirm_event(eid):
                    seen.add(eid)
                    confirmed.append(item)
            await asyncio.sleep(delay)
        return confirmed
