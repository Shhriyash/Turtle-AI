"""
scripts/testbench/human.py
--------------------------
Persona "naturalizer". Given a persona card + an intent beat + Turtle's last
reply, produce ONE natural next user message.

Two modes:
  * verbatim (DEFAULT): return the beat's canonical text unchanged. This is the
    reproducible path used by run_session.py — it keeps the planted facts exact
    so ground-truth assertions are deterministic.
  * llm: paraphrase the beat via Groq (llama-3.1-8b-instant) in the persona's
    voice, while preserving the required facts. Useful for stress-testing that
    extraction/recall survive natural phrasing.

Groq key: os.getenv('GROQ_API_KEY') after `import core.env` (+ load_env()).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the repo root importable when run as a script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import core.env  # noqa: E402  (side-effect import per protocol)

try:
    core.env.load_env(override=False)
except Exception:
    pass

_MODEL = "llama-3.1-8b-instant"


def _build_system(persona_card: str) -> str:
    return (
        f"{persona_card}\n\n"
        "You are the human user talking to an AI assistant named Turtle. "
        "Write ONLY your next message to Turtle, casually, in first person. "
        "No quotation marks, no preamble, no stage directions, one short message."
    )


def naturalize(
    persona_card: str,
    canonical_text: str,
    *,
    intent: str = "",
    required_facts: list[str] | None = None,
    last_reply: str | None = None,
    verbatim: bool = True,
) -> str:
    """Produce the next user message for a beat.

    verbatim=True (default) returns canonical_text verbatim.
    """
    required_facts = required_facts or []
    if verbatim:
        return canonical_text

    key = os.getenv("GROQ_API_KEY")
    if not key:
        # No key -> deterministic fallback so the harness never hard-fails.
        return canonical_text

    try:
        from groq import Groq

        client = Groq(api_key=key)
        facts_line = (
            f" You MUST state these facts explicitly: {', '.join(required_facts)}."
            if required_facts
            else ""
        )
        user_prompt = (
            f"Intent for your next message: {intent}."
            f"{facts_line}\n"
            f"The exact idea to convey (rephrase in your own casual voice): "
            f"\"{canonical_text}\"\n"
        )
        if last_reply:
            user_prompt += f"\nTurtle just said: \"{last_reply[:400]}\"\n"
        resp = client.chat.completions.create(
            model=_MODEL,
            temperature=0.7,
            max_tokens=80,
            messages=[
                {"role": "system", "content": _build_system(persona_card)},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip().strip('"').strip()
        return text or canonical_text
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[human] Groq naturalize failed ({e}); falling back to verbatim")
        return canonical_text


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Persona naturalizer (test harness)")
    ap.add_argument("--beat-text", required=True, help="canonical text of the beat")
    ap.add_argument("--intent", default="", help="semantic intent")
    ap.add_argument("--facts", default="", help="comma-separated required facts")
    ap.add_argument("--last-reply", default="", help="Turtle's previous reply")
    ap.add_argument("--llm", action="store_true", help="use Groq (default: verbatim)")
    args = ap.parse_args()

    from scenario import PERSONA  # local import; scenario is a sibling module

    facts = [f.strip() for f in args.facts.split(",") if f.strip()]
    out = naturalize(
        PERSONA["persona_card"],
        args.beat_text,
        intent=args.intent,
        required_facts=facts,
        last_reply=args.last_reply or None,
        verbatim=not args.llm,
    )
    print(out)


if __name__ == "__main__":
    # allow `python human.py` from inside scripts/testbench/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _cli()
