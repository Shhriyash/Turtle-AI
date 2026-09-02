"""
scripts/testbench/evaluate.py
-----------------------------
Reads GROUND TRUTH from disk (never trusts reply text alone) and writes
scripts/testbench/out/scorecard.md.

Ground-truth sources
  * data/memory/personal/<user_id>/            journal events + rendered topic md
  * data/sessions.sqlite                       per-user messages -> tool-call parts
  * data/scheduler.sqlite                      apscheduler_jobs -> routine jobs
  * data/traces/traces.jsonl                   turtle.turn spans -> latency_ms
  * scripts/testbench/out/session_*.json       harness-captured TTFR + timings

Run:  venv/Scripts/python.exe scripts/testbench/evaluate.py
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from pathlib import Path
from collections import Counter
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TESTBENCH_DIR = Path(__file__).resolve().parent
OUT_DIR = TESTBENCH_DIR / "out"
SESSION_FILE = TESTBENCH_DIR / "persona_session.json"

DATA_DIR = _ROOT / "data"
PERSONAL_DIR = DATA_DIR / "memory" / "personal"
SESSIONS_DB = DATA_DIR / "sessions.sqlite"
SCHEDULER_DB = DATA_DIR / "scheduler.sqlite"
TRACES_FILE = DATA_DIR / "traces" / "traces.jsonl"

from scenario import PERSONA  # noqa: E402


# ---------------------------------------------------------------------------
# user_id resolution
# ---------------------------------------------------------------------------
def resolve_user_id(email: str = PERSONA["email"]) -> str | None:
    if SESSION_FILE.exists():
        try:
            rec = json.loads(SESSION_FILE.read_text(encoding="utf-8")).get(email)
            if rec and rec.get("user_id"):
                return rec["user_id"]
        except Exception:
            pass
    # Fallback: newest personal dir.
    if PERSONAL_DIR.exists():
        dirs = [d for d in PERSONAL_DIR.iterdir() if d.is_dir()]
        if dirs:
            return max(dirs, key=lambda d: d.stat().st_mtime).name
    return None


# ---------------------------------------------------------------------------
# Journal readers
# ---------------------------------------------------------------------------
def iter_journal_events(user_id: str) -> list[dict[str, Any]]:
    """All journal events for a user, in chronological (shard+line) order."""
    jdir = PERSONAL_DIR / user_id / "journal"
    events: list[dict[str, Any]] = []
    if not jdir.exists():
        return events
    for shard in sorted(jdir.glob("*/events.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    # Stable order by observed_at then event_id (ULIDs sort chronologically).
    events.sort(key=lambda e: (e.get("observed_at", ""), e.get("event_id", "")))
    return events


def applied_events(user_id: str) -> list[dict[str, Any]]:
    return [
        e
        for e in iter_journal_events(user_id)
        if e.get("applied") and not e.get("rejected")
    ]


def read_topic_files(user_id: str) -> dict[str, str]:
    base = PERSONAL_DIR / user_id
    out: dict[str, str] = {}
    for fname in (
        "MEMORY.md",
        "identity.md",
        "preferences.md",
        "workflow.md",
        "relations.md",
        "projects.md",
        "contacts.md",
        "communication_style.md",
        "working_style.md",
    ):
        p = base / fname
        if p.exists():
            out[fname] = p.read_text(encoding="utf-8")
    return out


def _event_text(e: dict[str, Any]) -> str:
    """Flatten an event's key + value into a lowercase searchable blob."""
    parts = [str(e.get("key", "")), str(e.get("topic", ""))]
    val = e.get("value")
    if isinstance(val, dict):
        parts.extend(str(v) for v in val.values())
    else:
        parts.append(str(val))
    return " ".join(parts).lower()


def fact_present(user_id: str, needle: str) -> dict[str, Any]:
    """Is `needle` present as an applied journal event OR a rendered md line?"""
    needle_l = needle.lower()
    ev_hits = [e for e in applied_events(user_id) if needle_l in _event_text(e)]
    md = read_topic_files(user_id)
    md_hits = [f for f, text in md.items() if needle_l in text.lower()]
    return {
        "needle": needle,
        "in_journal": bool(ev_hits),
        "in_md": bool(md_hits),
        "present": bool(ev_hits or md_hits),
        "journal_keys": sorted({e.get("key", "") for e in ev_hits}),
        "md_files": md_hits,
    }


# ---------------------------------------------------------------------------
# Session / tool-call readers
# ---------------------------------------------------------------------------
def read_persona_tool_calls(user_id: str) -> list[dict[str, Any]]:
    """Every tool-call part across the persona's sessions, in row order.

    Returns [{session_id, tool_name, updated_at}], parsed straight from the
    persisted pydantic message JSON (part_kind == 'tool-call').
    """
    if not SESSIONS_DB.exists():
        return []
    calls: list[dict[str, Any]] = []
    try:
        con = sqlite3.connect(f"file:{SESSIONS_DB}?mode=ro", uri=True)
    except Exception:
        con = sqlite3.connect(str(SESSIONS_DB))
    try:
        for sid, data, updated in con.execute(
            "select session_id, data, updated_at from sessions order by updated_at"
        ):
            try:
                d = json.loads(data)
            except Exception:
                continue
            if d.get("user_id") != user_id:
                continue
            for m in d.get("messages") or []:
                for p in m.get("parts") or []:
                    if p.get("part_kind") == "tool-call":
                        calls.append(
                            {
                                "session_id": sid,
                                "tool_name": p.get("tool_name"),
                                "updated_at": updated,
                            }
                        )
    finally:
        con.close()
    return calls


def persona_session_ids(user_id: str) -> list[str]:
    if not SESSIONS_DB.exists():
        return []
    ids: list[str] = []
    con = sqlite3.connect(str(SESSIONS_DB))
    try:
        for sid, data in con.execute("select session_id, data from sessions"):
            try:
                if json.loads(data).get("user_id") == user_id:
                    ids.append(sid)
            except Exception:
                continue
    finally:
        con.close()
    return ids


# ---------------------------------------------------------------------------
# Scheduler readers
# ---------------------------------------------------------------------------
def read_routine_jobs(user_id: str) -> list[str]:
    """apscheduler job ids of the form routine::<user_id>::<key>."""
    if not SCHEDULER_DB.exists():
        return []
    jobs: list[str] = []
    con = sqlite3.connect(str(SCHEDULER_DB))
    try:
        try:
            rows = con.execute("select id from apscheduler_jobs").fetchall()
        except Exception:
            return []
        for (jid,) in rows:
            if isinstance(jid, str) and jid.startswith(f"routine::{user_id}::"):
                jobs.append(jid)
    finally:
        con.close()
    return jobs


def workflow_routine_events(user_id: str) -> list[dict[str, Any]]:
    """Applied workflow.* routine events (excludes scheduled_fire.* runtime rows)."""
    out = []
    for e in applied_events(user_id):
        if e.get("topic") != "workflow":
            continue
        key = str(e.get("key", ""))
        if key.startswith("workflow.scheduled_fire"):
            continue
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Trace readers
# ---------------------------------------------------------------------------
def read_turn_latencies(user_id: str | None = None) -> list[float]:
    if not TRACES_FILE.exists():
        return []
    lats: list[float] = []
    for line in TRACES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            span = json.loads(line)
        except Exception:
            continue
        if span.get("name") != "turtle.turn":
            continue
        attrs = span.get("attributes", {})
        if user_id and attrs.get("turtle.user_id") not in (None, user_id):
            continue
        lat = attrs.get("turtle.latency_ms")
        if lat is None:
            st, en = span.get("start_time"), span.get("end_time")
            if isinstance(st, (int, float)) and isinstance(en, (int, float)):
                lat = (en - st) / 1e6
        if isinstance(lat, (int, float)):
            lats.append(float(lat))
    return lats


# ---------------------------------------------------------------------------
# Harness output readers
# ---------------------------------------------------------------------------
def load_session_outputs() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for p in sorted(OUT_DIR.glob("session_*.json")):
        try:
            n = int(p.stem.split("_")[1])
            out[n] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return out


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
def _status(ok: bool | None) -> str:
    if ok is None:
        return "N/A (not run)"
    return "PASS" if ok else "FAIL"


def build_scorecard() -> str:
    user_id = resolve_user_id()
    sessions = load_session_outputs()
    ran = sorted(sessions.keys())

    L: list[str] = []
    L.append("# Turtle Synthetic-Human Test Scorecard")
    L.append("")
    L.append(f"- Persona: **{PERSONA['name']}** <{PERSONA['email']}>")
    L.append(f"- user_id: `{user_id}`")
    L.append(f"- Sessions with captured output: {ran or 'none'}")
    L.append("")

    if user_id is None:
        L.append("> No persona user_id resolved (no persona_session.json and no "
                 "personal memory dir). Run run_session.py first.")
        return "\n".join(L) + "\n"

    # -- Criterion: planted facts persisted -------------------------------
    planted = {
        "vegetarian (diet)": "vegetarian",
        "concise (reply style)": "concise",
        "dog Pixel": "Pixel",
        "project Atlas": "Atlas",
        "best friend Aarav": "Aarav",
    }
    fact_rows = []
    facts_ok = True
    for label, needle in planted.items():
        r = fact_present(user_id, needle)
        # only hold the overall criterion accountable once session 1 was run
        if 1 in ran and not r["present"]:
            facts_ok = False
        fact_rows.append((label, r))

    L.append("## 1. Memory persisted (ground truth = journal events + rendered md)")
    L.append("")
    L.append(f"**{_status(facts_ok if 1 in ran else None)}**")
    L.append("")
    L.append("| Planted fact | Present | In journal | In md | journal keys / md files |")
    L.append("|---|---|---|---|---|")
    for label, r in fact_rows:
        where = ", ".join(r["journal_keys"] + r["md_files"]) or "-"
        L.append(
            f"| {label} | {'yes' if r['present'] else 'NO'} | "
            f"{'yes' if r['in_journal'] else 'no'} | "
            f"{'yes' if r['in_md'] else 'no'} | `{where}` |"
        )
    L.append("")

    # -- Criterion: tool calls --------------------------------------------
    calls = read_persona_tool_calls(user_id)
    tool_counter = Counter(c["tool_name"] for c in calls)
    by_session: dict[str, Counter] = {}
    for c in calls:
        by_session.setdefault(c["session_id"], Counter())[c["tool_name"]] += 1

    web_ok = tool_counter.get("search_web", 0) > 0 if 2 in ran else None

    L.append("## 2. Tool calls (ground truth = persisted tool-call parts)")
    L.append("")
    L.append(f"- search_web fired (session 2 news beat): **{_status(web_ok)}**")
    L.append(f"- Total tool calls across persona sessions: {sum(tool_counter.values())}")
    L.append("")
    if tool_counter:
        L.append("| Tool | Count |")
        L.append("|---|---|")
        for name, n in tool_counter.most_common():
            L.append(f"| `{name}` | {n} |")
    else:
        L.append("_No tool-call parts found in persona sessions yet._")
    L.append("")
    if by_session:
        L.append("Per session_id:")
        L.append("")
        for sid, cnt in by_session.items():
            L.append(f"- `{sid}`: " + ", ".join(f"{k}×{v}" for k, v in cnt.items()))
        L.append("")

    # -- Criterion: flip-back (H1) ----------------------------------------
    style_events = [
        e
        for e in applied_events(user_id)
        if ("concise" in _event_text(e) or "detailed" in _event_text(e))
    ]
    latest_style = style_events[-1] if style_events else None
    if 3 in ran:
        if latest_style is None:
            flip_ok: bool | None = False
        else:
            txt = _event_text(latest_style)
            flip_ok = ("concise" in txt) and ("detailed" not in txt or txt.rfind("concise") > txt.rfind("detailed"))
    else:
        flip_ok = None
    # cross-check rendered md
    md = read_topic_files(user_id)
    md_pref = (md.get("preferences.md", "") + md.get("communication_style.md", "")).lower()

    L.append("## 3. Flip-back / latest-wins (H1)")
    L.append("")
    L.append(f"**{_status(flip_ok)}** - served style must be the LATEST value (concise), not the intervening 'detailed'.")
    L.append("")
    if latest_style is not None:
        L.append(f"- Latest style event: key=`{latest_style.get('key')}` "
                 f"value=`{latest_style.get('value')}` at {latest_style.get('observed_at')}")
    L.append(f"- Rendered preferences/communication md mentions concise: "
             f"{'yes' if 'concise' in md_pref else 'no'}; detailed: "
             f"{'yes' if 'detailed' in md_pref else 'no'}")
    L.append("")

    # -- Criterion: routine -----------------------------------------------
    wf_events = workflow_routine_events(user_id)
    jobs = read_routine_jobs(user_id)
    if 4 in ran:
        routine_ok: bool | None = bool(wf_events) and bool(jobs)
    else:
        routine_ok = None
    L.append("## 4. Routine created (workflow event + scheduler job)")
    L.append("")
    L.append(f"**{_status(routine_ok)}**")
    L.append("")
    L.append(f"- Applied workflow routine events: {len(wf_events)} "
             + (", ".join(f"`{e.get('key')}`" for e in wf_events) if wf_events else ""))
    L.append(f"- APScheduler jobs for persona: {len(jobs)} "
             + (", ".join(f"`{j}`" for j in jobs) if jobs else ""))
    L.append("")

    # -- Criterion: TTFR & latency ----------------------------------------
    ttfrs: list[float] = []
    server_totals: list[float] = []
    server_llms: list[float] = []
    error_frames = 0
    llm_outliers = 0
    for n in ran:
        for turn in sessions[n].get("turns", []):
            if isinstance(turn.get("ttfr_ms"), (int, float)):
                ttfrs.append(float(turn["ttfr_ms"]))
            if isinstance(turn.get("server_total_ms"), (int, float)):
                server_totals.append(float(turn["server_total_ms"]))
            if isinstance(turn.get("server_llm_ms"), (int, float)):
                server_llms.append(float(turn["server_llm_ms"]))
            if turn.get("error"):
                error_frames += 1
    # crude fallback-cascade signal: llm_ms far above the median suggests a
    # failed Gemini rung that fell through to a Groq rung.
    if server_llms:
        med = statistics.median(server_llms)
        llm_outliers = sum(1 for x in server_llms if x > max(2.5 * med, med + 4000))
    trace_lats = read_turn_latencies(user_id)

    L.append("## 5. Latency (TTFR = ws.send -> first `done` frame)")
    L.append("")
    if ttfrs:
        L.append("| Metric | TTFR ms | server total_ms | server llm_ms |")
        L.append("|---|---|---|---|")

        def _row(label: str, fn: Any) -> str:
            def g(vals: list[float]) -> str:
                return f"{fn(vals):.0f}" if vals else "-"
            return f"| {label} | {g(ttfrs)} | {g(server_totals)} | {g(server_llms)} |"

        L.append(_row("min", min))
        L.append(_row("median", statistics.median))
        L.append(_row("p95", lambda v: _pct(v, 0.95)))
        L.append(_row("max", max))
        L.append("")
    else:
        L.append("_No TTFR samples captured yet._")
        L.append("")
    L.append(f"- Turns with error frames: {error_frames}")
    L.append(f"- llm_ms outliers (possible Gemini->Groq fallback rungs): {llm_outliers}")
    if trace_lats:
        L.append(f"- turtle.turn trace spans: {len(trace_lats)}, "
                 f"median latency_ms={statistics.median(trace_lats):.0f}, "
                 f"max={max(trace_lats):.0f}")
    else:
        L.append("- turtle.turn trace spans: none matched persona in traces.jsonl")
    L.append("")

    # -- Summary ----------------------------------------------------------
    L.append("## What works / what's broken")
    L.append("")
    verdicts = [
        ("Memory persisted", facts_ok if 1 in ran else None),
        ("search_web tool", web_ok),
        ("Flip-back latest-wins", flip_ok),
        ("Routine created", routine_ok),
    ]
    works = [name for name, ok in verdicts if ok is True]
    broken = [name for name, ok in verdicts if ok is False]
    pending = [name for name, ok in verdicts if ok is None]
    L.append(f"- **Works:** {', '.join(works) or 'none confirmed yet'}")
    L.append(f"- **Broken:** {', '.join(broken) or 'none detected'}")
    L.append(f"- **Not yet exercised:** {', '.join(pending) or 'none'}")
    if ttfrs:
        L.append(f"- TTFR median {statistics.median(ttfrs):.0f} ms over {len(ttfrs)} turns; "
                 f"{error_frames} error turns.")
    L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    card = build_scorecard()
    out_path = OUT_DIR / "scorecard.md"
    out_path.write_text(card, encoding="utf-8")
    print(card)
    print(f"[evaluate] wrote {out_path}")


if __name__ == "__main__":
    main()
