"""
core/observability.py span-attribute + auto-latency tests.

Migrated from test_tier2_verification.py (TestG5Observability) — only the
span-attribute mapping and auto-latency behaviors, plus the business-attribute
constant export. The JSONL exporter's rotation/flush/write path is already
covered by test/phase3_trace_replay_test.py, so it is intentionally NOT
duplicated here.

Spans are emitted through an isolated TracerProvider (no exporter) so nothing
ever reaches the global BatchSpanProcessor that appends to data/traces/.
"""
from __future__ import annotations


class TestG5Observability:
    """OTel TraceSink span attribute mapping + automatic latency measurement."""

    def test_observability_module_importable(self):
        from core.observability import trace_sink, init_observability, OTelTraceSink
        assert trace_sink is not None
        assert callable(init_observability)
        assert OTelTraceSink is not None

    def test_otel_trace_sink_is_trace_sink(self):
        from core.observability import OTelTraceSink
        sink = OTelTraceSink()
        assert hasattr(sink, "span") and callable(sink.span)

    def test_span_accepts_turtle_attributes(self):
        from opentelemetry.sdk.trace import TracerProvider
        from core.observability import OTelTraceSink
        sink = OTelTraceSink()
        # Isolated provider (no exporter) so the span never reaches the global
        # BatchSpanProcessor that appends to the real data/traces/traces.jsonl.
        sink.tracer = TracerProvider().get_tracer("turtle-test")
        with sink.span(
            "test.turn",
            user_id="usr_abc",
            session_id="sess_xyz",
            intent="web",
            model="groq:llama-3.1-8b",
            tokens_in=120,
            tokens_out=80,
            cost_usd=0.0001,
            tool_status="ok",
            hallucination_check_result="pass",
        ) as sp:
            assert sp is not None

    def test_latency_auto_measured_when_not_provided(self):
        """When latency_ms is omitted, the sink measures and sets it automatically."""
        import time
        from opentelemetry.sdk.trace import TracerProvider
        from core.observability import OTelTraceSink, ATTR_LATENCY_MS
        sink = OTelTraceSink()
        sink.tracer = TracerProvider().get_tracer("turtle-test")
        with sink.span("test.latency.auto", user_id="usr_test") as sp:
            time.sleep(0.01)
        attrs = dict(sp.attributes or {})
        assert ATTR_LATENCY_MS in attrs, "latency_ms not auto-set on span"
        assert attrs[ATTR_LATENCY_MS] >= 0

    def test_standard_attribute_constants_exported(self):
        from core.observability import (
            ATTR_USER_ID, ATTR_SESSION_ID, ATTR_INTENT, ATTR_MODEL,
            ATTR_LATENCY_MS, ATTR_TOKENS_IN, ATTR_TOKENS_OUT,
            ATTR_COST_USD, ATTR_TOOL_STATUS, ATTR_HALLUCINATION_CHECK,
        )
        assert all(a.startswith("turtle.") for a in [
            ATTR_USER_ID, ATTR_SESSION_ID, ATTR_INTENT, ATTR_MODEL,
            ATTR_LATENCY_MS, ATTR_TOKENS_IN, ATTR_TOKENS_OUT,
            ATTR_COST_USD, ATTR_TOOL_STATUS, ATTR_HALLUCINATION_CHECK,
        ])
