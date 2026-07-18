"""
tools/url_tools/extractor.py fetch-routing behavior tests.

Migrated from test_tier2_verification.py (TestB4UrlFetcher). All behavior:
the httpx-first path, SPA detection, the Playwright vs Scrape.do fallback
branch (by token presence), timeout/invalid-URL failure results, direct JSON
return, and graceful degradation when Playwright is not installed.

Fully offline — every network call is mocked; nothing touches data/.
"""
from __future__ import annotations


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
        import asyncio, unittest.mock as mock

        STATIC_HTML = """<html><head><title>Test Page</title></head>
        <body>
        <p>This is a rich static page with enough content to pass SPA detection checks.
        The article is about Python programming and web scraping techniques that are
        widely used for data extraction and automation tasks.</p>
        </body></html>"""

        fake_resp = mock.MagicMock()
        fake_resp.raise_for_status = mock.Mock()
        fake_resp.status_code = 200
        fake_resp.text = STATIC_HTML
        fake_resp.headers = {"content-type": "text/html; charset=utf-8"}

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=fake_resp)
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

        fake_httpx_resp = mock.MagicMock()
        fake_httpx_resp.raise_for_status = mock.Mock()
        fake_httpx_resp.status_code = 200
        fake_httpx_resp.text = SPA_HTML
        fake_httpx_resp.headers = {"content-type": "text/html"}

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=fake_httpx_resp)

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

        fake_httpx_resp = mock.MagicMock()
        fake_httpx_resp.raise_for_status = mock.Mock()
        fake_httpx_resp.status_code = 200
        fake_httpx_resp.text = SPA_HTML
        fake_httpx_resp.headers = {"content-type": "text/html"}

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=fake_httpx_resp)

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
            mock_client.get = mock.AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            return await fetch_url_content_async(mock_client, "https://slow-site.com", timeout=5.0)

        result = asyncio.run(run())
        assert result.success is False
        assert "Timeout" in result.error_message or "timeout" in result.error_message.lower()

    def test_json_response_returned_directly(self):
        import asyncio, unittest.mock as mock

        fake_resp = mock.MagicMock()
        fake_resp.raise_for_status = mock.Mock()
        fake_resp.status_code = 200
        fake_resp.text = '{"price": 65000}'
        fake_resp.json = mock.Mock(return_value={"price": 65000})
        fake_resp.headers = {"content-type": "application/json"}

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=fake_resp)
            return await fetch_url_content_async(mock_client, "https://api.example.com/data")

        result = asyncio.run(run())
        assert result.success is True
        assert "65000" in result.content

    def test_playwright_import_error_falls_through_gracefully(self):
        """If Playwright is not installed, SPA path returns sparse content without crashing."""
        import asyncio, unittest.mock as mock

        SPA_HTML = "<html><body><div id='root'></div></body></html>"

        fake_resp = mock.MagicMock()
        fake_resp.raise_for_status = mock.Mock()
        fake_resp.status_code = 200
        fake_resp.text = SPA_HTML
        fake_resp.headers = {"content-type": "text/html"}

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async

            async def raise_import(*args, **kwargs):
                raise ImportError("playwright not installed")

            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=fake_resp)

            with mock.patch("tools.url_tools.extractor._fetch_with_playwright",
                            side_effect=raise_import), \
                 mock.patch("core.config.settings") as cfg_mock:
                cfg_mock.scraped_do_api_key = None
                return await fetch_url_content_async(mock_client, "https://spa.com")

        result = asyncio.run(run())
        assert result.success is True
