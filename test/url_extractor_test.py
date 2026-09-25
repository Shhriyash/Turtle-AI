"""
tools/url_tools/extractor.py fetch-routing behavior tests.

Migrated from test_tier2_verification.py (TestB4UrlFetcher). All behavior:
the httpx-first path, SPA detection, the Playwright vs Scrape.do fallback
branch (by token presence), timeout/invalid-URL failure results, direct JSON
return, and graceful degradation when Playwright is not installed.

Fully offline — every network call is mocked; nothing touches data/.

WP1.A note: fetch_url_content_async's step-1 fetch now goes through
tools.url_tools.safe_fetch.safe_get, which uses httpx's streaming API
(`client.stream(...)`) instead of `client.get(...)`, and validates the URL
via safe_fetch.validate_public_url before any request. These tests are
about *routing* behavior (httpx vs Playwright vs Scrape.do), not SSRF
validation (see test/safe_fetch_test.py for that), so an autouse fixture
neutralizes validation and the mock HTTP client fakes `.stream()` as an
async context manager instead of `.get()`.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_ssrf_validation(monkeypatch):
    """Neutralize SSRF validation for these routing tests — the URLs used
    here (example.com, spa-example.com, slow-site.com, ...) are stand-ins
    for "some public URL" and are not meant to be resolved for real."""
    monkeypatch.setattr(
        "tools.url_tools.safe_fetch.validate_public_url", lambda url: None
    )


class _FakeStreamCtx:
    """Async context manager returned by a mocked http_client.stream()."""

    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


def _make_stream_response(text, status_code=200, headers=None):
    """Build a fake httpx streaming response: status/headers plus an
    aiter_bytes() that yields the given text as a single UTF-8 chunk."""
    import unittest.mock as mock

    headers = headers or {"content-type": "text/html"}
    resp = mock.MagicMock()
    resp.status_code = status_code
    resp.headers = headers
    resp.is_redirect = False
    body = text.encode("utf-8")

    async def aiter_bytes():
        yield body

    resp.aiter_bytes = aiter_bytes
    return resp


def _make_stream_client(text, status_code=200, headers=None):
    """Build a fake httpx.AsyncClient whose .stream(...) yields the given
    canned response via an async context manager, matching what
    safe_fetch.safe_get actually calls (not .get())."""
    import unittest.mock as mock

    fake_resp = _make_stream_response(text, status_code, headers)
    client = mock.AsyncMock()
    client.stream = mock.Mock(return_value=_FakeStreamCtx(fake_resp))
    return client


class TestB4UrlFetcher:
    """httpx first; Playwright fallback on SPA detection; Scrape.do in cloud mode."""

    def test_fetch_url_content_async_is_async(self):
        import asyncio
        from tools.url_tools.extractor import fetch_url_content_async
        assert asyncio.iscoroutinefunction(fetch_url_content_async)

    def test_is_spa_content_function_exists(self):
        from tools.url_tools.extractor import _is_spa_content
        assert callable(_is_spa_content)

    def test_is_spa_content_sparse_text(self):
        from tools.url_tools.extractor import _is_spa_content
        assert _is_spa_content("") is True
        assert _is_spa_content("  loading...  ") is True

    def test_is_spa_content_rich_text(self):
        from tools.url_tools.extractor import _is_spa_content
        rich = "The quick brown fox jumps over the lazy dog. " * 10
        assert _is_spa_content(rich) is False

    def test_fetch_with_playwright_function_exists(self):
        import asyncio
        from tools.url_tools.extractor import _fetch_with_playwright
        assert asyncio.iscoroutinefunction(_fetch_with_playwright)

    def test_fetch_with_scraped_do_function_exists(self):
        import asyncio
        from tools.url_tools.extractor import _fetch_with_scraped_do
        assert asyncio.iscoroutinefunction(_fetch_with_scraped_do)

    def test_fetch_static_html_via_httpx(self):
        """httpx path returns content for a normal static HTML page."""
        import asyncio

        STATIC_HTML = """<html><head><title>Test Page</title></head>
        <body>
        <p>This is a rich static page with enough content to pass SPA detection checks.
        The article is about Python programming and web scraping techniques that are
        widely used for data extraction and automation tasks.</p>
        </body></html>"""

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = _make_stream_client(STATIC_HTML)
            return await fetch_url_content_async(mock_client, "https://example.com")

        result = asyncio.run(run())
        assert result.success is True
        assert result.title == "Test Page"
        assert len(result.content) > 0

    def test_spa_detection_triggers_playwright_when_no_token(self):
        """Sparse HTML triggers Playwright fallback when no Scrape.do token."""
        import asyncio, unittest.mock as mock

        SPA_HTML = "<html><body><div id='root'></div></body></html>"
        RENDERED_HTML = """<html><head><title>SPA Rendered</title></head>
        <body><p>Content rendered by JavaScript after hydration. This paragraph is long
        enough to pass the SPA threshold and confirm that Playwright successfully
        rendered the page with full JavaScript execution support.</p></body></html>"""

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = _make_stream_client(SPA_HTML)

            with mock.patch("tools.url_tools.extractor._fetch_with_playwright",
                            new=mock.AsyncMock(return_value=(RENDERED_HTML, 200, "text/html"))) as pw_mock, \
                 mock.patch("tools.url_tools.extractor._fetch_with_scraped_do") as sd_mock, \
                 mock.patch("core.config.settings") as cfg_mock:
                cfg_mock.scraped_do_api_key = None
                result = await fetch_url_content_async(mock_client, "https://spa-example.com")
            pw_mock.assert_called_once()
            sd_mock.assert_not_called()
            return result

        result = asyncio.run(run())
        assert result.success is True

    def test_spa_detection_triggers_scraped_do_when_token_set(self):
        """SPA + Scrape.do token → Scrape.do called, Playwright skipped."""
        import asyncio, unittest.mock as mock

        SPA_HTML = "<html><body><div id='root'></div></body></html>"
        RENDERED_HTML = """<html><head><title>Scraped Page</title></head>
        <body><p>Content fetched through Scrape.do proxy with JS rendering enabled
        for higher success rates and geo bypass across various regions worldwide.</p>
        </body></html>"""

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = _make_stream_client(SPA_HTML)

            fake_secret = mock.MagicMock()
            fake_secret.get_secret_value.return_value = "test-scraped-do-token"

            with mock.patch("tools.url_tools.extractor._fetch_with_playwright") as pw_mock, \
                 mock.patch("tools.url_tools.extractor._fetch_with_scraped_do",
                            new=mock.AsyncMock(return_value=(RENDERED_HTML, 200, "text/html"))) as sd_mock, \
                 mock.patch("core.config.settings") as cfg_mock:
                cfg_mock.scraped_do_api_key = fake_secret
                result = await fetch_url_content_async(mock_client, "https://spa-example.com")
            sd_mock.assert_called_once()
            pw_mock.assert_not_called()
            return result

        result = asyncio.run(run())
        assert result.success is True
        assert "Scraped Page" in result.title

    def test_invalid_url_returns_failure_result(self):
        import asyncio, unittest.mock as mock

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = mock.AsyncMock()
            return await fetch_url_content_async(mock_client, "not-a-url")

        result = asyncio.run(run())
        assert result.success is False
        assert result.error_message is not None

    def test_timeout_returns_failure_result(self):
        import asyncio, unittest.mock as mock
        import httpx

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = mock.AsyncMock()
            mock_client.stream = mock.Mock(side_effect=httpx.TimeoutException("timed out"))
            return await fetch_url_content_async(mock_client, "https://slow-site.com", timeout=5.0)

        result = asyncio.run(run())
        assert result.success is False
        assert "Timeout" in result.error_message or "timeout" in result.error_message.lower()

    def test_json_response_returned_directly(self):
        import asyncio

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = _make_stream_client(
                '{"price": 65000}', headers={"content-type": "application/json"}
            )
            return await fetch_url_content_async(mock_client, "https://api.example.com/data")

        result = asyncio.run(run())
        assert result.success is True
        assert "65000" in result.content

    def test_playwright_import_error_falls_through_gracefully(self):
        """If Playwright is not installed, SPA path returns sparse content without crashing."""
        import asyncio, unittest.mock as mock

        SPA_HTML = "<html><body><div id='root'></div></body></html>"

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async

            async def raise_import(*args, **kwargs):
                raise ImportError("playwright not installed")

            mock_client = _make_stream_client(SPA_HTML)

            with mock.patch("tools.url_tools.extractor._fetch_with_playwright",
                            side_effect=raise_import), \
                 mock.patch("core.config.settings") as cfg_mock:
                cfg_mock.scraped_do_api_key = None
                return await fetch_url_content_async(mock_client, "https://spa.com")

        result = asyncio.run(run())
        assert result.success is True
