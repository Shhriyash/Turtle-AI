"""
scripts/testbench/run_session.py
--------------------------------
Run ONE session (1..4) of the Maya Chen scenario end-to-end against the LIVE
Turtle server at ws://127.0.0.1:8765/ws.

  python scripts/testbench/run_session.py <n>            # verbatim (default)
  python scripts/testbench/run_session.py <n> --llm      # Groq-naturalized turns

Each turn:
  * naturalize the beat (verbatim by default for reproducibility)
  * send it over the WS, capture reply + TTFR + server timings + frames
  * settle ~2s for the async extractor, then drain+confirm pending memory
  * read ground-truth tool calls newly persisted this turn

Writes scripts/testbench/out/session_<n>.json and prints a compact summary.
The cookie is reused from persona_session.json so memory is persistent across
sessions/re-runs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

# Model replies contain Unicode (curly quotes, narrow no-break spaces); the
# Windows console defaults to cp1252 and crashes on them. Force UTF-8 output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from client import TurtleClient, OUT_DIR
from human import naturalize
from scenario import PERSONA, get_session
from evaluate import read_persona_tool_calls

SETTLE_S = 2.0


def _new_tools(prev: Counter, now: Counter) -> list[str]:
    """Tool names whose call count increased since the previous read."""
    out: list[str] = []
    for name, n in now.items():
        if n > prev.get(name, 0):
            out.append(name)
    return sorted(out)


async def run(session_n: int, *, verbatim: bool) -> dict:
    beats = get_session(session_n)
    client = TurtleClient()
    client.onboard(PERSONA["email"], PERSONA["name"], PERSONA["timezone"])
    user_id = client.user_id or "unknown"

    ready = await client.connect(drain_s=3.0)
    print(f"[run] connected user_id={user_id}; on-connect frames: "
          f"{[f.get('type') or f.get('status') for f in ready]}")

    # Baseline persisted tool calls so we can attribute new ones per turn.
    prev_tools = Counter(c["tool_name"] for c in read_persona_tool_calls(user_id))

    turns: list[dict] = []
    last_reply: str | None = None
    for beat in beats:
        text = naturalize(
            PERSONA["persona_card"],
            beat.canonical_text,
            intent=beat.intent,
            required_facts=beat.required_facts,
            last_reply=last_reply,
            verbatim=verbatim,
        )
        print(f"\n[run] --- {beat.id}: {text!r}")
        result = await client.send_turn(text)
        # Let the async extractor queue candidates, then confirm them all.
        await asyncio.sleep(SETTLE_S)
        confirmed = await client.drain_pending()

        now_tools = Counter(c["tool_name"] for c in read_persona_tool_calls(user_id))
        tools_detected = _new_tools(prev_tools, now_tools)
        prev_tools = now_tools

        last_reply = result.reply

        turn_rec = {
            "beat_id": beat.id,
            "intent": beat.intent,
            "expect": beat.expect,
            "sent": text,
            "reply": result.reply,
            "ttfr_ms": result.ttfr_ms,
            "server_total_ms": result.server_total_ms,
            "server_llm_ms": result.server_llm_ms,
            "tools_detected": tools_detected,
            "confirmations_seen": result.confirmations_seen,
            "confirmed_pending": confirmed,
            "error": result.error,
            "frames": result.frames,
        }
        turns.append(turn_rec)

        reply_preview = (result.reply or "").replace("\n", " ")[:110]
        print(f"[run] ttfr={result.ttfr_ms}ms server_total={result.server_total_ms}ms "
              f"llm={result.server_llm_ms}ms tools={tools_detected} "
              f"confirm_seen={len(result.confirmations_seen)} "
              f"confirmed={len(confirmed)} err={result.error}")
        print(f"[run] reply: {reply_preview}")

    await client.close()

    out = {
        "session": session_n,
        "user_id": user_id,
        "email": PERSONA["email"],
        "verbatim": verbatim,
        "turns": turns,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"session_{session_n}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[run] wrote {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one Turtle scenario session")
    ap.add_argument("session", type=int, help="session number 1..4")
    ap.add_argument("--llm", action="store_true",
                    help="naturalize turns via Groq (default: verbatim canonical text)")
    args = ap.parse_args()
    asyncio.run(run(args.session, verbatim=not args.llm))


if __name__ == "__main__":
    main()
