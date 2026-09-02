"""
core/sandbox/tripwire.py
------------------------
The v1 deny-list, demoted.

READ THIS BEFORE ADDING A RULE. This module is NOT a security control and must
never become one. It is a **prompt-injection detector**. It matches, it logs, it
stamps the audit record, and by default it returns and lets the call proceed —
because the container is the boundary (docs/sandbox_design_v2.md §2).

Why not use it to block? Because a deny-list over a Turing-complete execution
surface loses, always. v1 blocked `rm -rf` and every one of these sails past it:

    rm -r -f /                       flags unfused
    rm --recursive --force /         long flags
    python3 -c "shutil.rmtree('/')"  no rm token at all
    printf 'rm -rf /' > s; . s       badness is data, then sourced
    make deploy                      badness is in a Makefile
    git commit                       badness is in .git/hooks/pre-commit
    $'\\x72\\x6d' -rf /              shell expansion happens after our regex

You cannot enumerate badness in a language that has eval. The far worse failure
is second-order: a deny-list makes the *system* look safe, so the container gets
treated as optional, and then one bypass is a full compromise.

So the rules here are tuned for SIGNAL, not coverage. A hit should mean "a model
turn almost certainly went hostile" — which is exactly the record you want in
the log after an incident. False positives are expensive here in a way they are
not for a blocker, because a noisy tripwire gets ignored.

`ShellPolicy.tripwire_blocks` can turn hits into denials. It buys approximately
nothing (see the table above) and exists only for operators who want the
belt-and-braces. It is off by default and documented as ineffective.
"""
from __future__ import annotations

import re
from typing import NamedTuple, Sequence


class TripwireRule(NamedTuple):
    rule_id: str
    pattern: re.Pattern[str]
    note: str


def _c(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


# High-signal only. Each entry answers: "if this fires on a normal coding task,
# is that surprising?" If the answer is no, the rule does not belong here.
_RULES: list[TripwireRule] = [
    # ── Exfiltration: the payload of essentially every real injection ────────
    TripwireRule(
        "credential-path",
        _c(r"(?:\.ssh/|id_rsa|id_ed25519|\.aws/credentials|\.netrc|"
           r"\.env\b|credentials\.json|token\.json|users\.sqlite)"),
        "Touches a well-known credential/secret path.",
    ),
    TripwireRule(
        "pipe-to-interpreter",
        _c(r"(?:curl|wget|fetch|Invoke-WebRequest|iwr)\b[^\n|]*\|\s*"
           r"(?:ba|z|k|d)?sh\b|\|\s*python[23]?\b"),
        "Downloads and immediately executes (curl|sh class).",
    ),
    TripwireRule(
        "outbound-post",
        _c(r"(?:curl|wget)\b[^\n]*(?:-d\b|--data|-F\b|-T\b|--upload-file)|"
           r"requests\.(?:post|put)\s*\(|urllib\.request\.urlopen\s*\([^)]*data="),
        "Sends data outbound — exfiltration shape.",
    ),
    TripwireRule(
        "reverse-shell",
        _c(r"\bnc\b[^\n]*\s-e\b|\bncat\b[^\n]*\s-e\b|/dev/tcp/|"
           r"socket\.socket[^\n]{0,200}(?:connect|SOCK_STREAM)[^\n]{0,200}"
           r"(?:dup2|subprocess|/bin/sh)|pty\.spawn"),
        "Reverse-shell shape.",
    ),

    # ── Escaping the box ────────────────────────────────────────────────────
    TripwireRule(
        "host-escape-path",
        _c(r"/var/run/docker\.sock|/proc/1/(?:root|ns/)|/host(?:fs|/root)?/|"
           r"\bnsenter\b|\bchroot\b|/sys/fs/cgroup/[^\n]*release_agent"),
        "Known container-escape primitive.",
    ),
    TripwireRule(
        "privilege-escalation",
        _c(r"\bsudo\b|\bsu\s+-|\brunas\b|\bsetcap\b|\bchmod\s+[0-7]*[46][0-7]{3}\b"),
        "Attempts privilege escalation.",
    ),

    # ── Wide-blast destruction (log-worthy even though the box contains it) ──
    TripwireRule(
        "recursive-root-delete",
        # Matches unfused/long flags too — not because that makes it a control,
        # but because a detector that misses `rm -r -f` produces a log that lies
        # about what happened.
        _c(r"\brm\b(?:\s+(?:-{1,2}[A-Za-z-]+|\s)*)*\s+(?:/|~|\*|\.\s*$|"
           r"[A-Za-z]:[\\/])|shutil\.rmtree\s*\(\s*['\"]?[/~]|"
           r"\bdel\b[^\n]*\s/[sq]\b|\bformat\b\s+[A-Za-z]:|\bmkfs\b|\bwipefs\b|"
           r"\bdd\b[^\n]*\bof=/dev/"),
        "Wide-blast destructive filesystem operation.",
    ),
    TripwireRule(
        "persistence",
        _c(r"\bcrontab\b|/etc/cron|\bschtasks\b[^\n]*/create|\bsystemctl\b[^\n]*enable|"
           r"\.bashrc\b|\.bash_profile\b|\.zshrc\b|\bauthorized_keys\b|"
           r"HKLM\\|HKCU\\[^\n]*\\Run\b"),
        "Establishes persistence outside the session.",
    ),
    TripwireRule(
        "fork-bomb",
        _c(r":\(\)\s*\{[^}]*\|\s*:\s*&|while\s+true[^\n]{0,40}&\s*done|"
           r"os\.fork\s*\(\s*\)[^\n]{0,80}while"),
        "Fork-bomb shape.",
    ),

    # ── Meta: the model being told to do something by content, not the user ──
    TripwireRule(
        "injection-marker",
        _c(r"(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions|"
           r"^\s*SYSTEM\s*:|<\|im_start\|>|you\s+are\s+now\s+in\s+developer\s+mode)"),
        "Text carries a prompt-injection marker — near-conclusive when it "
        "appears inside a tool ARGUMENT rather than a user message.",
    ),
]


def scan(*subjects: str | Sequence[str] | None) -> list[str]:
    """Return matching rule ids across every subject.

    Accepts strings and argv sequences interchangeably so callers can pass
    `scan(argv, path, content)` without pre-joining. Argv is joined with spaces
    for matching ONLY — the joined form is never executed, and that distinction
    is the whole reason the argv array exists.
    """
    haystacks: list[str] = []
    for subject in subjects:
        if subject is None:
            continue
        if isinstance(subject, str):
            haystacks.append(subject)
        else:
            haystacks.append(" ".join(str(s) for s in subject))
    if not haystacks:
        return []

    blob = "\n".join(haystacks)
    return [rule.rule_id for rule in _RULES if rule.pattern.search(blob)]


def explain(rule_ids: Sequence[str]) -> str:
    """Human-readable notes for the matched ids, for logs and UI notices."""
    by_id = {r.rule_id: r.note for r in _RULES}
    return "; ".join(f"{rid}: {by_id.get(rid, 'unknown rule')}" for rid in rule_ids)


def rule_ids() -> list[str]:
    return [r.rule_id for r in _RULES]
