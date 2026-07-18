"""
core/latency_budgets.py behavior tests.

Migrated from test_tier1_verification.py (TestE4LatencyBudgets) — the real
behavior assertions only (the server-source grep is intentionally dropped).

Covers: per-stage budget constants, the TOOL_S seconds property, the
non-raising sla_exceeded / check_sla logging helpers, and the env override that
rebuilds budgets from TURTLE_TOOL_MAX_MS.
"""
from __future__ import annotations


class TestE4LatencyBudgets:
    """Hard timeouts per stage; SLA breach logging; env-overridable."""

    def test_tool_budget_is_6000ms(self):
        from core.latency_budgets import budgets
        assert budgets.TOOL_MAX_MS == 6000, f"Got {budgets.TOOL_MAX_MS}"

    def test_router_budget_is_400ms(self):
        from core.latency_budgets import budgets
        assert budgets.ROUTER_MAX_MS == 400

    def test_tts_first_byte_budget_is_600ms(self):
        from core.latency_budgets import budgets
        assert budgets.TTS_FIRST_BYTE_MAX_MS == 600

    def test_tool_s_property(self):
        from core.latency_budgets import budgets
        assert budgets.TOOL_S == 6.0

    def test_sla_exceeded_logs_without_raising(self):
        from core.latency_budgets import sla_exceeded
        sla_exceeded("test_stage", 1500.0, 400)  # Must not raise

    def test_check_sla_no_breach_is_silent(self):
        import time
        from core.latency_budgets import check_sla
        start = time.time() - 0.001  # 1 ms elapsed
        check_sla("fast_stage", start, 6000)  # well within budget

    def test_env_override(self):
        import os
        import unittest.mock as mock
        with mock.patch.dict(os.environ, {"TURTLE_TOOL_MAX_MS": "9999"}):
            import core.latency_budgets as lb
            custom = lb.LatencyBudgets()
            assert custom.TOOL_MAX_MS == 9999, f"Got {custom.TOOL_MAX_MS}"
