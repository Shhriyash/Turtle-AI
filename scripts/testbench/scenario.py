"""
scripts/testbench/scenario.py
-----------------------------
The persona (Maya Chen) and the 4-session beat script the harness drives.

Each beat carries:
  * id            — stable turn id (session prefix)
  * canonical_text— the verbatim message sent in verbatim mode
  * intent        — the semantic goal (for the LLM naturalizer / readability)
  * required_facts— facts the message must state (planted for later recall)
  * expect        — what ground-truth behavior we assert for this beat:
        tool     : a tool we expect to fire (e.g. "search_web")
        recall   : substrings the reply should contain (memory-conditioned)
        memory   : (topic, needle) that should be persisted to disk
        routine  : True if this beat should create a workflow routine
        flipback : True if this is the H1 flip-back "latest wins" check
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PERSONA: dict[str, Any] = {
    "name": "Maya Chen",
    "email": "maya.chen.turtle@example.com",
    "timezone": "America/New_York",
    "persona_card": (
        "You are Maya Chen, a product manager in New York. You are friendly, "
        "busy, and to-the-point. You are vegetarian, you have a dog named Pixel, "
        "your best friend is Aarav, and you are leading a project codenamed "
        "Atlas. You like concise replies."
    ),
}


@dataclass
class Beat:
    id: str
    canonical_text: str
    intent: str
    required_facts: list[str] = field(default_factory=list)
    expect: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Session scripts
# ---------------------------------------------------------------------------
SESSIONS: dict[int, list[Beat]] = {
    # Session 1 — plant facts
    1: [
        Beat(
            id="s1a",
            canonical_text="I'm vegetarian, please remember that.",
            intent="state a dietary preference to be remembered",
            required_facts=["vegetarian"],
            expect={"memory": ("preferences", "vegetarian")},
        ),
        Beat(
            id="s1b",
            canonical_text="I prefer concise, to-the-point replies.",
            intent="state a communication-style preference",
            required_facts=["concise"],
            expect={"memory": ("preferences", "concise")},
        ),
        Beat(
            id="s1c",
            canonical_text="My dog's name is Pixel.",
            intent="state a personal fact about a pet",
            required_facts=["Pixel"],
            expect={"memory": ("relations", "Pixel")},
        ),
        Beat(
            id="s1d",
            canonical_text="I'm working on a project codenamed Atlas.",
            intent="state a project fact",
            required_facts=["Atlas"],
            expect={"memory": ("projects", "Atlas")},
        ),
        Beat(
            id="s1e",
            canonical_text="My best friend is Aarav.",
            intent="state a relationship fact",
            required_facts=["Aarav"],
            expect={"memory": ("relations", "Aarav")},
        ),
    ],
    # Session 2 — recall + web tool
    2: [
        Beat(
            id="s2a",
            canonical_text="What do you remember about me?",
            intent="ask for a memory recall",
            expect={"recall": ["vegetarian", "Pixel", "Atlas", "concise"]},
        ),
        Beat(
            id="s2b",
            canonical_text="Give me one quick high-protein breakfast idea.",
            intent="ask for advice that should respect the vegetarian preference",
            expect={"recall_soft": ["vegetarian"]},
        ),
        Beat(
            id="s2c",
            canonical_text="What's the latest news about AI coding agents?",
            intent="ask for fresh news (should trigger a web search)",
            expect={"tool": "search_web"},
        ),
    ],
    # Session 3 — flip-back (H1)
    3: [
        Beat(
            id="s3a",
            canonical_text="Actually, I now prefer detailed, thorough replies.",
            intent="update the communication-style preference to detailed",
            required_facts=["detailed"],
            expect={"memory": ("preferences", "detailed")},
        ),
        Beat(
            id="s3b",
            canonical_text="On second thought, keep replies concise like before.",
            intent="flip the preference back to concise",
            required_facts=["concise"],
            expect={"memory": ("preferences", "concise")},
        ),
        Beat(
            id="s3c",
            canonical_text="How do you prefer replying to me?",
            intent="ask which reply style is current (expect concise, the latest)",
            expect={"flipback": True, "recall": ["concise"]},
        ),
    ],
    # Session 4 — routine + deep recall
    4: [
        Beat(
            id="s4a",
            canonical_text="Every weekday at 8am, remind me to post my standup.",
            intent="create a recurring weekday routine",
            expect={"routine": True},
        ),
        Beat(
            id="s4b",
            canonical_text="Remind me — what's my dog's name and who's my best friend?",
            intent="deep recall of two planted facts",
            expect={"recall": ["Pixel", "Aarav"]},
        ),
    ],
}


def get_session(n: int) -> list[Beat]:
    if n not in SESSIONS:
        raise SystemExit(f"No such session {n}; valid sessions: {sorted(SESSIONS)}")
    return SESSIONS[n]
