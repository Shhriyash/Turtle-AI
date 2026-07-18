"""
core/web_search.py routing behavior tests.

Migrated from test_tier1_verification.py (TestB3TavilySearch) — the real
behavior assertions only (the source-grep 'retry-relax hack removed from
server' test is intentionally dropped as test theatre).

Covers: Tavily is primary and DDG is fallback (both async), Tavily result
parsing through search_duckduckgo, the 6.0s timeout constants, query
normalization (site: filter injection), and empty-result formatting.
"""
from __future__ import annotations


class TestB3TavilySearch:
    """Tavily primary; DDG fallback; normalization + formatting helpers."""

    def test_tavily_function_exists(self):
        import asyncio
        from core.web_search import _search_tavily
        assert asyncio.iscoroutinefunction(_search_tavily)

    def test_ddg_fallback_function_exists(self):
        from core.web_search import _search_duckduckgo_fallback
        assert _search_duckduckgo_fallback is not None

    def test_search_duckduckgo_routes_to_tavily_when_key_set(self):
        """With TAVILY_API_KEY set, search_duckduckgo routes through Tavily."""
        import os, asyncio, unittest.mock as mock
        from core.web_search import search_duckduckgo

        fake_results = [{"url": "https://t.co/1", "title": "T1", "content": "snippet1"}]
        fake_resp = mock.AsyncMock()
        fake_resp.raise_for_status = mock.Mock()
        fake_resp.json = mock.Mock(return_value={"results": fake_results})

        with mock.patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            with mock.patch("httpx.AsyncClient.post", return_value=fake_resp):
                client_mock = mock.AsyncMock()
                client_mock.post = mock.AsyncMock(return_value=fake_resp)
                results = asyncio.run(
                    search_duckduckgo(client_mock, "bitcoin price", max_results=5)
                )
        assert len(results) >= 1
        assert results[0].url == "https://t.co/1"

    def test_tavily_timeout_is_6s(self):
        """E4 alignment: Tavily timeout must be exactly 6.0 seconds."""
        from core.web_search import _TAVILY_TIMEOUT
        assert _TAVILY_TIMEOUT == 6.0, f"Expected 6.0, got {_TAVILY_TIMEOUT}"

    def test_ddg_timeout_is_6s(self):
        """E4 alignment: DDG fallback timeout must also be 6.0 seconds."""
        from core.web_search import _DDG_TIMEOUT
        assert _DDG_TIMEOUT == 6.0, f"Expected 6.0, got {_DDG_TIMEOUT}"

    def test_normalize_query_adds_amazon_site_filter(self):
        from core.web_search import _normalize_query
        result = _normalize_query("iphone 15 pro amazon.in")
        assert result.startswith("site:amazon.in"), f"Got: {result!r}"

    def test_format_search_results_empty(self):
        from core.web_search import format_search_results
        out = format_search_results("test query", [])
        assert "No web results" in out
