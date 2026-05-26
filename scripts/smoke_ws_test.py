"""Smoke test: connect as Shriyash, send 4 turns, capture all WS messages.

Usage: python scripts/smoke_ws_test.py <cookie_value>
"""
from __future__ import annotations

import asyncio
import json
import sys

import websockets


COOKIE = sys.argv[1] if len(sys.argv) > 1 else ""
URI = "ws://127.0.0.1:8765/ws"
HEADERS = {"Cookie": f"turtle_uid={COOKIE}"} if COOKIE else {}


TURNS = [
    "Hi, who am I?",
    "Every morning at 8am please send me a daily news brief.",
    "yes",
    "Remember that my best friend is Aarav.",
]


async def collect(ws, deadline_s: float) -> list[dict]:
    out: list[dict] = []
    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline_s)
            except asyncio.TimeoutError:
                return out
            if isinstance(raw, bytes):
                out.append({"type": "<binary>", "len": len(raw)})
            else:
                try:
                    out.append(json.loads(raw))
                except Exception:
                    out.append({"type": "<text>", "raw": raw[:200]})
    except websockets.ConnectionClosed:
        return out


def summarize(msgs: list[dict]) -> str:
    parts = []
    for m in msgs:
        t = m.get("type", "?")
        if t == "text":
            parts.append(f"[text] {str(m.get('content',''))[:140]}")
        elif t == "confirmation_prompt":
            parts.append(f"[confirm topic={m.get('topic')} key={m.get('key')} msg={str(m.get('message',''))[:120]}]")
        elif t == "error":
            parts.append(f"[error code={m.get('code')} msg={str(m.get('message',''))[:120]}]")
        elif t == "status":
            parts.append(f"[status {m.get('status')}]")
        elif t == "timing":
            parts.append(f"[timing total={m.get('total_ms')}ms]")
        else:
            parts.append(f"[{t}] {json.dumps(m, default=str)[:120]}")
    return "\n  ".join(parts)


async def main() -> None:
    async with websockets.connect(URI, additional_headers=HEADERS) as ws:
        print("connected")
        # drain ready/status messages
        ready = await collect(ws, 3.0)
        print(f"on-connect:\n  {summarize(ready)}\n")

        for i, turn in enumerate(TURNS, 1):
            print(f"--- turn {i}: {turn!r}")
            await ws.send(json.dumps({"type": "text", "content": turn}))
            msgs = await collect(ws, 25.0)
            print(f"  {summarize(msgs)}\n")
            await asyncio.sleep(2.0)  # let async extractor settle


if __name__ == "__main__":
    asyncio.run(main())
