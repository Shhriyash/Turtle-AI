"""
Phase 3 — trace replay tooling tests.

Covers:
  (a) core.observability JSONLSpanExporter flush + single-generation rotation,
      and the flush_traces() provider drain hook.
  (b) scripts/trace_replay.py CLI (list / show / reconstruct) against fabricated
      traces.jsonl + a tmp sessions.sqlite whose message blob uses the real
      pydantic-ai serialization shape.
  (c) missing-file / missing-turn paths exit 1 with a message.

Fully offline; every artifact lives under tmp_path (never the repo data/ tree).

Run with:
    pytest test/phase3_trace_replay_test.py -v
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from scripts import trace_replay


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TURN_ID = "turtle_session_20260717_120000_000000_turn_1"
SESSION_ID = "turtle_session_20260717_120000_000000"
USER_ID = "usr_replay"


def _fake_span(**attrs: object) -> SimpleNamespace:
    """A minimal ReadableSpan stand-in that JSONLSpanExporter.export can read."""
    return SimpleNamespace(
        name="turtle.turn",
        context=SimpleNamespace(trace_id=0xABC, span_id=0xDEF),
        start_time=1_752_753_600_000_000_000,
        end_time=1_752_753_600_500_000_000,
        attributes=attrs,
    )


def _write_traces(tmp_path: Path) -> Path:
    traces = tmp_path / "traces" / "traces.jsonl"
    traces.parent.mkdir(parents=True, exist_ok=True)
    span = {
        "name": "turtle.turn",
        "context": {"trace_id": "a" * 32, "span_id": "b" * 16},
        "start_time": 1_752_753_600_000_000_000,
        "end_time": 1_752_753_600_500_000_000,
        "attributes": {
            trace_replay.ATTR_USER_ID: USER_ID,
            trace_replay.ATTR_SESSION_ID: SESSION_ID,
            trace_replay.ATTR_TURN_ID: TURN_ID,
            trace_replay.ATTR_INTENT: "web",
            trace_replay.ATTR_LATENCY_MS: 512.5,
            trace_replay.ATTR_MODEL: "groq:llama-3.1-8b",
            # ad-hoc/legacy unprefixed extras the reader must tolerate
            "memory_context_chars": 42,
            "tools_scoped": "web_search,fetch_url",
        },
    }
    traces.write_text(json.dumps(span) + "\n", encoding="utf-8")
    return traces


def _write_sessions_db(tmp_path: Path) -> Path:
    # Reuse the exact serializer core.session_store._sync_to_backend uses so the
    # fixture's messages blob matches production shape.
    messages = [
        ModelRequest(parts=[UserPromptPart(content="what's the weather in Paris?")]),
        ModelResponse(parts=[TextPart(content="It's sunny and 21°C in Paris.")]),
    ]
    msgs_json = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    data = {
        "status": "active",
        "user_id": USER_ID,
        "messages": msgs_json,
        "summary": [],
        "updated_at": "2026-07-17T12:00:05Z",
    }
    db_path = tmp_path / "sessions.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.execute(
        "INSERT INTO sessions (session_id, data) VALUES (?, ?)",
        (SESSION_ID, json.dumps(data)),
    )
    conn.commit()
    conn.close()
    return db_path


def _write_journal(tmp_path: Path) -> Path:
    journal_dir = tmp_path / "journal"
    shard = journal_dir / "2026-07"
    shard.mkdir(parents=True, exist_ok=True)
    event = {
        "event_id": "EV1",
        "session_id": SESSION_ID,
        "turn_id": TURN_ID,
        "observed_at": "2026-07-17T12:00:03Z",
        "kind": "preference",
        "topic": "preferences",
        "key": "weather_city",
        "value": {"text": "Paris"},
        "confidence": 0.82,
        "source": "inferred",
        "extractor": "llm_turn",
        "statement": "User asks about weather in Paris",
        "applied": True,
    }
    (shard / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    return journal_dir


# ---------------------------------------------------------------------------
# (a) Exporter flush + rotation
# ---------------------------------------------------------------------------

class TestExporterHardening:
    def test_rotation_on_init_when_over_cap(self, tmp_path):
        from core.observability import JSONLSpanExporter, TRACES_ROTATE_BYTES

        log_path = tmp_path / "traces" / "traces.jsonl"
        log_path.parent.mkdir(parents=True)
        # Sparse file just over the cap — no need to write 32 MB of bytes.
        with log_path.open("wb") as f:
            f.seek(TRACES_ROTATE_BYTES + 1)
            f.write(b"\0")
        assert log_path.stat().st_size > TRACES_ROTATE_BYTES

        JSONLSpanExporter(log_path)  # __init__ rotates

        rotated = log_path.parent / (log_path.name + ".1")
        assert rotated.exists(), "oversized traces.jsonl should roll to traces.jsonl.1"
        assert not log_path.exists(), "current traces.jsonl should be moved aside on rotation"
        print("[PASS] exporter rotates oversized traces.jsonl to .1 on init")

    def test_rotation_overwrites_previous_generation(self, tmp_path):
        from core.observability import JSONLSpanExporter, TRACES_ROTATE_BYTES

        log_path = tmp_path / "traces" / "traces.jsonl"
        log_path.parent.mkdir(parents=True)
        rotated = log_path.parent / (log_path.name + ".1")
        rotated.write_text("STALE PREVIOUS GENERATION\n", encoding="utf-8")
        with log_path.open("wb") as f:
            f.seek(TRACES_ROTATE_BYTES + 1)
            f.write(b"\0")

        JSONLSpanExporter(log_path)

        assert rotated.exists()
        assert "STALE" not in rotated.read_text(encoding="utf-8", errors="ignore"), \
            "single-generation rotation must overwrite the old .1"
        print("[PASS] rotation overwrites any previous .1 generation")

    def test_no_rotation_when_small(self, tmp_path):
        from core.observability import JSONLSpanExporter

        log_path = tmp_path / "traces" / "traces.jsonl"
        log_path.parent.mkdir(parents=True)
        log_path.write_text("small\n", encoding="utf-8")

        JSONLSpanExporter(log_path)

        assert log_path.exists(), "a small traces.jsonl must not be rotated"
        assert not (log_path.parent / (log_path.name + ".1")).exists()
        print("[PASS] small traces.jsonl is left untouched")

    def test_export_writes_and_flush_semantics(self, tmp_path):
        from core.observability import JSONLSpanExporter
        from opentelemetry.sdk.trace.export import SpanExportResult

        log_path = tmp_path / "traces" / "traces.jsonl"
        exporter = JSONLSpanExporter(log_path)

        result = exporter.export([_fake_span(**{trace_replay.ATTR_TURN_ID: TURN_ID})])
        assert result == SpanExportResult.SUCCESS

        # force_flush is real (returns True) and shutdown flushes without raising.
        assert exporter.force_flush() is True
        exporter.shutdown()

        # Empty batch is a no-op success (does not create a stray empty file line).
        assert exporter.export([]) == SpanExportResult.SUCCESS

        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["name"] == "turtle.turn"
        assert record["attributes"][trace_replay.ATTR_TURN_ID] == TURN_ID
        assert record["context"]["trace_id"] == format(0xABC, "032x")
        print("[PASS] export writes durably; force_flush True; shutdown flushes; empty batch no-ops")

    def test_flush_traces_callable(self):
        from core.observability import flush_traces

        # In the test process the singleton provider is registered; draining it
        # must never raise regardless of buffer state.
        flush_traces()
        print("[PASS] flush_traces() drains the provider without raising")

    def test_reconstruction_attr_constants_present(self):
        from core.observability import (
            ATTR_CHANNEL,
            ATTR_ERROR,
            ATTR_MEMORY_CONTEXT_CHARS,
            ATTR_OUTPUT_CHARS,
            ATTR_TOOLS_SCOPED,
        )

        for attr in (ATTR_CHANNEL, ATTR_ERROR, ATTR_MEMORY_CONTEXT_CHARS,
                     ATTR_OUTPUT_CHARS, ATTR_TOOLS_SCOPED):
            assert attr.startswith("turtle."), attr
        print("[PASS] reconstruction-grade ATTR constants exported with turtle. prefix")


# ---------------------------------------------------------------------------
# (b) CLI happy paths
# ---------------------------------------------------------------------------

class TestReplayCLI:
    def test_list_finds_the_turn(self, tmp_path, capsys):
        traces = _write_traces(tmp_path)
        rc = trace_replay.main(["list", "--traces", str(traces)])
        out = capsys.readouterr().out
        assert rc == 0
        assert TURN_ID in out
        assert SESSION_ID in out
        assert "web" in out
        assert "512.5" in out
        print("[PASS] list renders the turn row")

    def test_list_filters_by_user_and_session(self, tmp_path, capsys):
        traces = _write_traces(tmp_path)
        rc = trace_replay.main(["list", "--traces", str(traces), "--user", "nobody"])
        out = capsys.readouterr().out
        assert rc == 0
        assert TURN_ID not in out
        assert "no turtle.turn spans matched" in out
        print("[PASS] list --user filter excludes non-matching turns")

    def test_list_reads_rotated_sibling(self, tmp_path, capsys):
        traces = _write_traces(tmp_path)
        # Move the only span into the .1 sibling; list must still find it.
        rotated = traces.parent / (traces.name + ".1")
        rotated.write_text(traces.read_text(encoding="utf-8"), encoding="utf-8")
        traces.unlink()
        rc = trace_replay.main(["list", "--traces", str(traces)])
        out = capsys.readouterr().out
        assert rc == 0
        assert TURN_ID in out
        print("[PASS] list reads traces.jsonl.1 when the current file is gone")

    def test_show_prints_attributes(self, tmp_path, capsys):
        traces = _write_traces(tmp_path)
        rc = trace_replay.main(["show", TURN_ID, "--traces", str(traces)])
        out = capsys.readouterr().out
        assert rc == 0
        assert trace_replay.ATTR_MODEL in out
        assert "groq:llama-3.1-8b" in out
        assert "duration_ms : 500.0" in out
        assert "tools_scoped" in out  # legacy unprefixed extra is shown too
        print("[PASS] show pretty-prints the full span record")

    def test_reconstruct_joins_span_session_journal(self, tmp_path, capsys):
        traces = _write_traces(tmp_path)
        db = _write_sessions_db(tmp_path)
        journal = _write_journal(tmp_path)

        rc = trace_replay.main([
            "reconstruct", TURN_ID,
            "--traces", str(traces),
            "--sessions", str(db),
            "--journal-dir", str(journal),
        ])
        out = capsys.readouterr().out
        assert rc == 0
        # span section
        assert USER_ID in out
        assert "intent" in out
        # session history rendered from the real serialization shape
        assert "what's the weather in Paris?" in out
        assert "It's sunny and 21°C in Paris." in out
        assert "user" in out and "assistant" in out
        # journal event matched by turn_id
        assert "User asks about weather in Paris" in out
        assert "matching turn_id" in out
        print("[PASS] reconstruct joins span + session history + journal event")

    def test_reconstruct_journal_no_exact_match_is_honest(self, tmp_path, capsys):
        traces = _write_traces(tmp_path)
        db = _write_sessions_db(tmp_path)
        # Journal event carries a different (stage-b style) turn_id but same session.
        journal_dir = tmp_path / "journal"
        shard = journal_dir / "2026-07"
        shard.mkdir(parents=True)
        event = {
            "event_id": "EV2", "session_id": SESSION_ID,
            "turn_id": f"{SESSION_ID}_stageb_0", "observed_at": "2026-07-17T12:00:03Z",
            "kind": "fact", "topic": "identity", "key": "name",
            "value": {"text": "Sam"}, "confidence": 0.9, "source": "explicit",
            "extractor": "llm_turn", "statement": "User name is Sam", "applied": True,
        }
        (shard / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

        rc = trace_replay.main([
            "reconstruct", TURN_ID,
            "--traces", str(traces), "--sessions", str(db),
            "--journal-dir", str(journal_dir),
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "no journal events carry this exact turn_id" in out
        assert "same session_id" in out
        assert "User name is Sam" in out
        print("[PASS] reconstruct is honest about imprecise turn correlation, falls back to session")


# ---------------------------------------------------------------------------
# (c) Missing-file / missing-row paths → exit 1
# ---------------------------------------------------------------------------

class TestReplayErrors:
    def test_list_missing_traces_exits_1(self, tmp_path, capsys):
        rc = trace_replay.main(["list", "--traces", str(tmp_path / "nope.jsonl")])
        out = capsys.readouterr().out
        assert rc == 1
        assert "no trace file found" in out
        print("[PASS] list exits 1 on missing traces file")

    def test_show_missing_turn_exits_1(self, tmp_path, capsys):
        traces = _write_traces(tmp_path)
        rc = trace_replay.main(["show", "does_not_exist", "--traces", str(traces)])
        out = capsys.readouterr().out
        assert rc == 1
        assert "no turtle.turn span found" in out
        print("[PASS] show exits 1 on unknown turn_id")

    def test_reconstruct_missing_sessions_db_exits_1(self, tmp_path, capsys):
        traces = _write_traces(tmp_path)
        rc = trace_replay.main([
            "reconstruct", TURN_ID,
            "--traces", str(traces),
            "--sessions", str(tmp_path / "no_such.sqlite"),
            "--journal-dir", str(tmp_path / "journal"),
        ])
        out = capsys.readouterr().out
        assert rc == 1
        assert "sessions DB not found" in out
        print("[PASS] reconstruct exits 1 when sessions DB is missing")

    def test_reconstruct_missing_session_row_exits_1(self, tmp_path, capsys):
        traces = _write_traces(tmp_path)
        # Empty but valid sessions DB → row absent.
        db = tmp_path / "sessions.sqlite"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        conn.commit()
        conn.close()
        rc = trace_replay.main([
            "reconstruct", TURN_ID,
            "--traces", str(traces), "--sessions", str(db),
            "--journal-dir", str(tmp_path / "journal"),
        ])
        out = capsys.readouterr().out
        assert rc == 1
        assert "no session row" in out
        print("[PASS] reconstruct exits 1 when the session row is absent")
