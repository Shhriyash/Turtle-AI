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


class JSONLSpanExporter(SpanExporter):
    """Local mode exporter that writes spans to a JSONL file."""
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
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
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


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


def init_observability() -> TraceSink:
    """Initialize OpenTelemetry SDK based on local/cloud deployment mode."""
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
    return OTelTraceSink()

# Global sink singleton
trace_sink = init_observability()
