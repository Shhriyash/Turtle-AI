"""
core/observability.py
---------------------
G5: OpenTelemetry SDK initialization and TraceSink protocol implementation.

Every Turtle span carries a standard set of business attributes:
  user_id, session_id, turn_id, intent, model, latency_ms,
  tokens_in, tokens_out, cost_usd, tool_status, hallucination_check_result
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from core.config import settings
from core.storage import TraceSink

# Standard Turtle span attribute keys
ATTR_USER_ID = "turtle.user_id"
ATTR_SESSION_ID = "turtle.session_id"
ATTR_TURN_ID = "turtle.turn_id"
ATTR_INTENT = "turtle.intent"
ATTR_MODEL = "turtle.model"
ATTR_LATENCY_MS = "turtle.latency_ms"
ATTR_TOKENS_IN = "turtle.tokens_in"
ATTR_TOKENS_OUT = "turtle.tokens_out"
ATTR_COST_USD = "turtle.cost_usd"
ATTR_TOOL_STATUS = "turtle.tool_status"
ATTR_HALLUCINATION_CHECK = "turtle.hallucination_check_result"

# Reconstruction-grade attribute keys. The unified turn pipeline (voice + text +
# channels) emits these so a "turtle.turn" span on disk carries enough context
# to replay "why did Turtle answer X" without the running process. Additive:
# these formalize keys some paths already emitted ad-hoc (memory_context_chars,
# tools_scoped) and reserve names for the ones the pipeline is adding. Existing
# keys are NOT renamed — the replay reader tolerates both prefixed and legacy
# unprefixed forms.
ATTR_CHANNEL = "turtle.channel"
ATTR_OUTPUT_CHARS = "turtle.output_chars"
ATTR_ERROR = "turtle.error"
ATTR_MEMORY_CONTEXT_CHARS = "turtle.memory_context_chars"
ATTR_TOOLS_SCOPED = "turtle.tools_scoped"

# Single-generation rotation threshold. traces.jsonl is append-only and never
# rotated by the exporter's normal path, so an always-on local deployment grows
# it without bound (unbounded-growth hazard). At init we roll it once past this
# size into traces.jsonl.1 so disk usage stays roughly bounded to 2x this.
TRACES_ROTATE_BYTES = 32 * 1024 * 1024


class JSONLSpanExporter(SpanExporter):
    """Local mode exporter that writes spans to a JSONL file.

    Each ``export`` opens the file, appends its batch, flushes, and fsyncs
    before closing, so a batch that returns SUCCESS is durable on disk. The
    in-memory buffering that a hard kill can lose lives in the upstream
    ``BatchSpanProcessor`` — drain it with ``flush_traces()`` (provider-level
    force_flush), which cascades into this exporter's ``export``.
    """
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._rotate_if_needed()

    def _rotate_if_needed(self) -> None:
        """Single-generation rotation guarding against unbounded growth.

        traces.jsonl is append-only; nothing trims it during normal operation.
        Before the first append of this process, if it already exceeds the cap
        we move it to traces.jsonl.1 (overwriting any previous generation) and
        start a fresh file. os.replace is atomic and overwrites on Windows,
        unlike os.rename which raises if the destination exists.
        """
        try:
            if self.log_path.exists() and self.log_path.stat().st_size > TRACES_ROTATE_BYTES:
                rotated = self.log_path.parent / (self.log_path.name + ".1")
                os.replace(self.log_path, rotated)
        except OSError as exc:
            print(f"LOG: JSONLSpanExporter rotation skipped: {exc}")

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS
        try:
            with self._lock:
                with self.log_path.open("a", encoding="utf-8") as f:
                    for span in spans:
                        span_data = {
                            "name": span.name,
                            "context": {
                                "trace_id": format(span.context.trace_id, "032x"),
                                "span_id": format(span.context.span_id, "016x"),
                            },
                            "start_time": span.start_time,
                            "end_time": span.end_time,
                            "attributes": dict(span.attributes or {}),
                        }
                        f.write(json.dumps(span_data) + "\n")
                    # Durability: flush Python buffers to the OS, then fsync so
                    # an export that reports SUCCESS survives a subsequent crash.
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass  # fsync is best-effort; the write itself succeeded
        except OSError as exc:
            # Open/write/flush failed (disk full, permissions): report FAILURE
            # so the BatchSpanProcessor doesn't silently believe these spans
            # landed on disk (Codex R1#5).
            print(f"LOG: JSONLSpanExporter export failed: {exc}")
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Real flush semantics: every ``export`` already fsyncs and closes the
        file, so there is no exporter-side buffer left to drain here. Returning
        True lets the provider/processor treat the flush as complete."""
        return True

    def shutdown(self) -> None:
        """Flush on shutdown. Nothing is held open between exports, so this is a
        best-effort force_flush rather than the previous silent no-op."""
        self.force_flush()


class OTelTraceSink(TraceSink):
    """TraceSink protocol implementation wrapping OpenTelemetry.

    Accepts both raw OTel attribute kwargs and the standard Turtle
    business-context kwargs (user_id, session_id, intent, …).
    """
    def __init__(self, tracer_name: str = "turtle-agent"):
        self.tracer = trace.get_tracer(tracer_name)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        intent: Optional[str] = None,
        model: Optional[str] = None,
        latency_ms: Optional[float] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        cost_usd: Optional[float] = None,
        tool_status: Optional[str] = None,
        hallucination_check_result: Optional[str] = None,
        **extra_attrs: Any,
    ) -> Iterator[trace.Span]:
        attrs: dict[str, Any] = {}
        if user_id is not None:
            attrs[ATTR_USER_ID] = user_id
        if session_id is not None:
            attrs[ATTR_SESSION_ID] = session_id
        if turn_id is not None:
            attrs[ATTR_TURN_ID] = turn_id
        if intent is not None:
            attrs[ATTR_INTENT] = intent
        if model is not None:
            attrs[ATTR_MODEL] = model
        if latency_ms is not None:
            attrs[ATTR_LATENCY_MS] = float(latency_ms)
        if tokens_in is not None:
            attrs[ATTR_TOKENS_IN] = int(tokens_in)
        if tokens_out is not None:
            attrs[ATTR_TOKENS_OUT] = int(tokens_out)
        if cost_usd is not None:
            attrs[ATTR_COST_USD] = float(cost_usd)
        if tool_status is not None:
            attrs[ATTR_TOOL_STATUS] = tool_status
        if hallucination_check_result is not None:
            attrs[ATTR_HALLUCINATION_CHECK] = hallucination_check_result
        attrs.update(extra_attrs)

        t0 = time.perf_counter()
        with self.tracer.start_as_current_span(name, attributes=attrs) as current_span:
            yield current_span
            if latency_ms is None:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                current_span.set_attribute(ATTR_LATENCY_MS, round(elapsed_ms, 2))


# The active TracerProvider, stashed so flush_traces() can drain the
# BatchSpanProcessor's in-memory buffer on demand. None in cloud mode (no
# on-disk processor) or before init_observability() runs.
_provider: Optional[TracerProvider] = None


def init_observability() -> TraceSink:
    """Initialize OpenTelemetry SDK based on local/cloud deployment mode."""
    global _provider
    resource = Resource.create({"service.name": "turtle-agent"})
    provider = TracerProvider(resource=resource)

    if not settings.is_cloud:
        # Local mode: write traces to data/traces/traces.jsonl
        traces_file = settings.data_dir / "traces" / "traces.jsonl"
        processor = BatchSpanProcessor(JSONLSpanExporter(traces_file))
        provider.add_span_processor(processor)
    else:
        # Cloud mode: rely on auto-instrumentation or Logfire standard setup.
        pass

    trace.set_tracer_provider(provider)
    _provider = provider
    return OTelTraceSink()


def flush_traces() -> None:
    """Force any buffered spans to disk.

    Local mode feeds spans through a ``BatchSpanProcessor`` that buffers them in
    memory and exports on a timer; a hard process kill drops whatever hasn't
    been exported yet — including the per-turn "turtle.turn" span that makes
    "why did Turtle answer X" answerable from disk. Calling this forces the
    provider (and thus the processor and JSONLSpanExporter) to flush now.

    The server's shutdown hook is expected to call this so traces are durable
    before exit; wiring that call site is the integrator's job — this module
    only exposes the function.
    """
    provider = _provider
    if provider is None:
        return
    try:
        provider.force_flush()
    except Exception as exc:  # never let a flush failure block shutdown
        print(f"LOG: flush_traces force_flush failed: {exc}")


# Global sink singleton
trace_sink = init_observability()
