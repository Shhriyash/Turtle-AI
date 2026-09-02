"""
scripts/trace_replay.py
-----------------------
Phase 3 — trace replay tooling.

Turtle writes one ``turtle.turn`` span per turn to ``data/traces/traces.jsonl``
(see core/observability.py). This CLI reads those spans back offline and, for a
given turn, reconstructs the surrounding context so "why did Turtle answer X"
can be investigated without the running process. It joins three read-only
stores that today share no live linkage:

  1. the trace spans (data/traces/traces.jsonl [+ traces.jsonl.1])
  2. the session message history (data/sessions.sqlite, one JSON blob per
     session under the ``data`` column)
  3. the per-user memory journal
     (data/memory/personal/<uid>/journal/YYYY-MM/events.jsonl)

Nothing is written anywhere; every store is opened read-only.

Usage:
    python -m scripts.trace_replay list [--user U] [--session S] [--last N]
        [--traces data/traces/traces.jsonl]

    python -m scripts.trace_replay show <turn_id>
        [--traces data/traces/traces.jsonl]

    python -m scripts.trace_replay reconstruct <turn_id>
        [--traces data/traces/traces.jsonl]
        [--sessions data/sessions.sqlite]
        [--journal-dir PATH] [--messages K]

Exit codes: 0 on success, 1 when a requested trace/session/row is missing.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Attribute keys mirror core.observability.ATTR_*; import them so the two stay
# in lock-step, but fall back to literals if OTel is unavailable so the replay
# tool never hard-depends on the tracing stack being importable.
try:  # pragma: no cover - exercised implicitly; fallback is the rare path
    from core.observability import (
        ATTR_COST_USD,
        ATTR_ERROR,
        ATTR_INTENT,
        ATTR_LATENCY_MS,
        ATTR_MODEL,
        ATTR_SESSION_ID,
        ATTR_TOKENS_IN,
        ATTR_TOKENS_OUT,
        ATTR_TURN_ID,
        ATTR_USER_ID,
    )
except Exception:  # pragma: no cover
    ATTR_USER_ID = "turtle.user_id"
    ATTR_SESSION_ID = "turtle.session_id"
    ATTR_TURN_ID = "turtle.turn_id"
    ATTR_INTENT = "turtle.intent"
    ATTR_MODEL = "turtle.model"
    ATTR_LATENCY_MS = "turtle.latency_ms"
    ATTR_TOKENS_IN = "turtle.tokens_in"
    ATTR_TOKENS_OUT = "turtle.tokens_out"
    ATTR_COST_USD = "turtle.cost_usd"
    ATTR_ERROR = "turtle.error"

TURN_SPAN_NAME = "turtle.turn"
DEFAULT_TRACES = "data/traces/traces.jsonl"
DEFAULT_SESSIONS = "data/sessions.sqlite"


# ---------------------------------------------------------------------------
# Trace reading
# ---------------------------------------------------------------------------

def _trace_files(traces_path: Path) -> list[Path]:
    """Return existing trace files, oldest first.

    The single-generation rotation in core.observability moves the current
    traces.jsonl to traces.jsonl.1 when it grows too large, so .1 (when it
    exists) holds the OLDER spans and must be read first.
    """
    rotated = traces_path.parent / (traces_path.name + ".1")
    return [p for p in (rotated, traces_path) if p.exists()]


def _iter_span_records(traces_path: Path) -> Iterable[dict[str, Any]]:
    """Yield every span dict from the trace file(s), oldest file first."""
    for path in _trace_files(traces_path):
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record


def _attrs(span: dict[str, Any]) -> dict[str, Any]:
    attrs = span.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _span_attr(span: dict[str, Any], *keys: str) -> Any:
    """First present attribute among *keys* (tolerates prefixed/legacy names)."""
    attrs = _attrs(span)
    for key in keys:
        if key in attrs and attrs[key] not in (None, ""):
            return attrs[key]
    return None


def _iso_from_ns(start_time: Any) -> str:
    """Render a nanosecond epoch (OTel span start/end) as ISO-8601 UTC."""
    if not isinstance(start_time, (int, float)):
        return "-"
    try:
        dt = datetime.fromtimestamp(start_time / 1e9, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return "-"
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _turn_spans(traces_path: Path) -> list[dict[str, Any]]:
    return [s for s in _iter_span_records(traces_path) if s.get("name") == TURN_SPAN_NAME]


def _find_span_by_turn(traces_path: Path, turn_id: str) -> dict[str, Any] | None:
    match: dict[str, Any] | None = None
    for span in _iter_span_records(traces_path):
        if _span_attr(span, ATTR_TURN_ID, "turn_id") == turn_id:
            match = span  # last write wins — newest span for this turn_id
    return match


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> int:
    traces_path = Path(args.traces)
    if not _trace_files(traces_path):
        print(f"error: no trace file found at {traces_path} (or its .1 sibling)")
        return 1

    rows: list[tuple[Any, dict[str, Any]]] = []
    for span in _turn_spans(traces_path):
        if args.user and _span_attr(span, ATTR_USER_ID, "user_id") != args.user:
            continue
        if args.session and _span_attr(span, ATTR_SESSION_ID, "session_id") != args.session:
            continue
        rows.append((span.get("start_time") or 0, span))

    rows.sort(key=lambda r: r[0])
    rows = rows[-args.last:] if args.last and args.last > 0 else rows

    if not rows:
        print("no turtle.turn spans matched.")
        return 0

    header = f"{'start (UTC)':<22} {'turn_id':<34} {'session_id':<34} {'intent':<12} {'ms':>8}  error"
    print(header)
    print("-" * len(header))
    for _, span in rows:
        turn_id = _span_attr(span, ATTR_TURN_ID, "turn_id") or "-"
        session_id = _span_attr(span, ATTR_SESSION_ID, "session_id") or "-"
        intent = _span_attr(span, ATTR_INTENT, "intent") or "-"
        latency = _span_attr(span, ATTR_LATENCY_MS, "latency_ms")
        latency_str = f"{float(latency):.1f}" if isinstance(latency, (int, float)) else "-"
        error = _span_attr(span, ATTR_ERROR, "error") or ""
        print(
            f"{_iso_from_ns(span.get('start_time')):<22} "
            f"{str(turn_id):<34} {str(session_id):<34} "
            f"{str(intent):<12} {latency_str:>8}  {error}"
        )
    print(f"\n{len(rows)} turn(s).")
    return 0


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

def _duration_ms(span: dict[str, Any]) -> float | None:
    start, end = span.get("start_time"), span.get("end_time")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return round((end - start) / 1e6, 2)
    return None


def cmd_show(args: argparse.Namespace) -> int:
    traces_path = Path(args.traces)
    if not _trace_files(traces_path):
        print(f"error: no trace file found at {traces_path} (or its .1 sibling)")
        return 1

    span = _find_span_by_turn(traces_path, args.turn_id)
    if span is None:
        print(f"error: no {TURN_SPAN_NAME} span found for turn_id {args.turn_id!r}")
        return 1

    ctx = span.get("context") or {}
    print(f"name        : {span.get('name')}")
    print(f"trace_id    : {ctx.get('trace_id', '-')}")
    print(f"span_id     : {ctx.get('span_id', '-')}")
    print(f"start (UTC) : {_iso_from_ns(span.get('start_time'))}")
    print(f"end   (UTC) : {_iso_from_ns(span.get('end_time'))}")
    dur = _duration_ms(span)
    print(f"duration_ms : {dur if dur is not None else '-'}")
    print("attributes  :")
    print(json.dumps(_attrs(span), indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0


# ---------------------------------------------------------------------------
# reconstruct
# ---------------------------------------------------------------------------

def _render_part(part: dict[str, Any]) -> tuple[str, str] | None:
    """Map one serialized ModelMessage part to (role, text), or None to skip."""
    part_kind = part.get("part_kind")
    content = part.get("content")

    role_by_kind = {
        "user-prompt": "user",
        "system-prompt": "system",
        "text": "assistant",
        "tool-return": "tool-return",
        "retry-prompt": "retry",
        "thinking": "assistant/thinking",
    }
    if part_kind == "tool-call":
        name = part.get("tool_name", "?")
        args = part.get("args")
        arg_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, default=str)
        return ("assistant/tool-call", f"{name}({arg_str})")

    role = role_by_kind.get(part_kind)
    if role is None:
        return None

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # Multi-modal / structured content: keep only the textual pieces.
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                pieces.append(str(item.get("content") or item.get("text") or item))
            else:
                pieces.append(str(item))
        text = " ".join(pieces)
    else:
        text = "" if content is None else str(content)
    return (role, text)


def _compact(text: str, limit: int = 500) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _read_session_messages(sessions_path: Path, session_id: str) -> tuple[list[Any] | None, str]:
    """Return (messages, note). messages is None when the row is absent.

    Opens the SQLite file strictly read-only (mode=ro URI) so replay never
    mutates production session state.

    Note: completed sessions are compacted to a trailing message tail at
    finalization (SessionStore.mark_finalized keeps only the last
    COMPLETED_SESSION_MESSAGE_TAIL messages), so reconstruct can show at most
    that retained tail for a completed session — earlier turns live only in the
    rolling summary and the personal-memory journal.
    """
    # as_uri() requires an absolute path — the default "data/sessions.sqlite"
    # is relative and would raise ValueError, so resolve first.
    try:
        uri = f"{sessions_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    except (sqlite3.OperationalError, ValueError, OSError) as exc:
        return None, f"could not open sessions DB read-only: {exc}"
    try:
        cur = conn.execute("SELECT data FROM sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
    except sqlite3.OperationalError as exc:
        conn.close()
        return None, f"sessions table query failed: {exc}"
    conn.close()
    if row is None:
        return None, f"no session row for session_id {session_id!r}"
    try:
        data = json.loads(row[0])
    except (json.JSONDecodeError, TypeError) as exc:
        return None, f"session data JSON unparseable: {exc}"
    messages = data.get("messages")
    if not isinstance(messages, list):
        return [], "session row has no message history"
    return messages, ""


def _print_message_history(messages: list[Any], last_k: int) -> None:
    total = len(messages)
    shown = messages if last_k <= 0 or total <= last_k else messages[-last_k:]
    print(
        f"  ({total} message(s) in session; sessions.sqlite stores a flat "
        f"serialized history with no per-turn boundaries, so exact turn "
        f"alignment is not recoverable"
        + (f" — showing the last {len(shown)}" if len(shown) != total else "")
        + ")"
    )
    start_index = total - len(shown)
    for offset, message in enumerate(shown):
        idx = start_index + offset
        if not isinstance(message, dict):
            continue
        kind = message.get("kind", "?")
        for part in message.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            rendered = _render_part(part)
            if rendered is None:
                continue
            role, text = rendered
            print(f"  [{idx:>3}] {kind:<8} {role:<20} {_compact(text)}")


def _journal_dir_for(args: argparse.Namespace, user_id: str | None) -> Path | None:
    if args.journal_dir:
        return Path(args.journal_dir)
    if not user_id:
        return None
    # Mirror core.paths.personal_memory_dir(user_id)/journal WITHOUT its mkdir
    # side effect — replay must not create directories ("no writes anywhere").
    try:
        from core.paths import PERSONAL_MEMORY_DIR
    except Exception:
        return None
    return PERSONAL_MEMORY_DIR / user_id / "journal"


def _iter_journal_events(journal_dir: Path) -> Iterable[dict[str, Any]]:
    if not journal_dir.exists():
        return
    for shard in sorted(journal_dir.glob("*/events.jsonl")):
        with shard.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload


def _print_journal_events(journal_dir: Path, turn_id: str, session_id: str | None) -> None:
    if not journal_dir.exists():
        print(f"  (no journal directory at {journal_dir})")
        return

    exact: list[dict[str, Any]] = []
    same_session: list[dict[str, Any]] = []
    for event in _iter_journal_events(journal_dir):
        ev_turn = str(event.get("turn_id", ""))
        if ev_turn == turn_id or ev_turn.startswith(turn_id + "_"):
            exact.append(event)
        elif session_id and str(event.get("session_id", "")) == session_id:
            same_session.append(event)

    def _render_event(event: dict[str, Any]) -> str:
        statement = event.get("statement") or ""
        if not statement:
            value = event.get("value")
            statement = json.dumps(value, ensure_ascii=False, default=str) if value else ""
        applied = "applied" if event.get("applied") else ("rejected" if event.get("rejected") else "pending")
        return (
            f"  - [{event.get('kind', '?')}/{event.get('topic', '?')}] "
            f"turn={event.get('turn_id', '-')} conf={event.get('confidence', '?')} "
            f"{applied}: {_compact(statement)}"
        )

    if exact:
        print(f"  {len(exact)} journal event(s) with matching turn_id:")
        for event in exact:
            print(_render_event(event))
    else:
        print(
            "  (no journal events carry this exact turn_id — extraction paths "
            "assign their own turn_id conventions, e.g. '<session>_stageb_N', so "
            "trace-turn to journal-turn correlation is best-effort)"
        )
        if same_session:
            print(f"  {len(same_session)} event(s) from the same session_id:")
            for event in same_session:
                print(_render_event(event))


def cmd_reconstruct(args: argparse.Namespace) -> int:
    traces_path = Path(args.traces)
    if not _trace_files(traces_path):
        print(f"error: no trace file found at {traces_path} (or its .1 sibling)")
        return 1

    span = _find_span_by_turn(traces_path, args.turn_id)
    if span is None:
        print(f"error: no {TURN_SPAN_NAME} span found for turn_id {args.turn_id!r}")
        return 1

    exit_code = 0

    print("=" * 72)
    print(f"TURN {args.turn_id}")
    print("=" * 72)
    print("\n[span]")
    print(f"  start (UTC) : {_iso_from_ns(span.get('start_time'))}")
    dur = _duration_ms(span)
    print(f"  duration_ms : {dur if dur is not None else '-'}")
    for label, keys in (
        ("user_id", (ATTR_USER_ID, "user_id")),
        ("session_id", (ATTR_SESSION_ID, "session_id")),
        ("intent", (ATTR_INTENT, "intent")),
        ("model", (ATTR_MODEL, "model")),
        ("tokens_in", (ATTR_TOKENS_IN, "tokens_in")),
        ("tokens_out", (ATTR_TOKENS_OUT, "tokens_out")),
        ("cost_usd", (ATTR_COST_USD, "cost_usd")),
        ("error", (ATTR_ERROR, "error")),
    ):
        value = _span_attr(span, *keys)
        if value is not None:
            print(f"  {label:<12}: {value}")
    print("  all attributes:")
    print("    " + json.dumps(_attrs(span), ensure_ascii=False, sort_keys=True, default=str))

    session_id = _span_attr(span, ATTR_SESSION_ID, "session_id")
    user_id = _span_attr(span, ATTR_USER_ID, "user_id")

    print("\n[session message history]")
    sessions_path = Path(args.sessions)
    if not session_id:
        print("  (span carries no session_id; cannot join sessions DB)")
        exit_code = 1
    elif not sessions_path.exists():
        print(f"  error: sessions DB not found at {sessions_path}")
        exit_code = 1
    else:
        messages, note = _read_session_messages(sessions_path, str(session_id))
        if messages is None:
            print(f"  {note}")
            exit_code = 1
        else:
            if note:
                print(f"  {note}")
            _print_message_history(messages, args.messages)

    print("\n[journal events]")
    journal_dir = _journal_dir_for(args, str(user_id) if user_id else None)
    if journal_dir is None:
        print("  (no --journal-dir given and span carries no user_id to derive one)")
    else:
        _print_journal_events(journal_dir, args.turn_id, str(session_id) if session_id else None)

    return exit_code


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace_replay",
        description="Replay Turtle turn traces and reconstruct their context (read-only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list recent turtle.turn spans")
    p_list.add_argument("--user", default=None, help="filter by turtle.user_id")
    p_list.add_argument("--session", default=None, help="filter by turtle.session_id")
    p_list.add_argument("--last", type=int, default=20, help="show the last N turns (default 20)")
    p_list.add_argument("--traces", default=DEFAULT_TRACES, help=f"traces file (default {DEFAULT_TRACES})")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="pretty-print one span by turn_id")
    p_show.add_argument("turn_id", help="turtle.turn_id to show")
    p_show.add_argument("--traces", default=DEFAULT_TRACES, help=f"traces file (default {DEFAULT_TRACES})")
    p_show.set_defaults(func=cmd_show)

    p_rec = sub.add_parser("reconstruct", help="join span + session history + journal for one turn")
    p_rec.add_argument("turn_id", help="turtle.turn_id to reconstruct")
    p_rec.add_argument("--traces", default=DEFAULT_TRACES, help=f"traces file (default {DEFAULT_TRACES})")
    p_rec.add_argument("--sessions", default=DEFAULT_SESSIONS, help=f"sessions DB (default {DEFAULT_SESSIONS})")
    p_rec.add_argument("--journal-dir", default=None, help="journal dir (default: derived from span user_id)")
    p_rec.add_argument("--messages", type=int, default=20, help="show the last K session messages (default 20)")
    p_rec.set_defaults(func=cmd_reconstruct)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
