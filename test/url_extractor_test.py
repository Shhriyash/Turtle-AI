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
via safe_fetch.validate_public_url before any request. TestB4UrlFetcher is
about *routing* behavior (httpx vs Playwright vs Scrape.do), not SSRF
validation (see test/safe_fetch_test.py for the exhaustive SSRF-rule
coverage), so it carries a class-scoped autouse fixture that neutralizes
validation — the URLs used there (example.com, spa-example.com, ...) are
stand-ins for "some public URL" and are not meant to be resolved for real
— and its mock HTTP client fakes `.stream()` as an async context manager
instead of `.get()`.

TestSsrfEnforcedAtFetchSites below deliberately does NOT get that fixture:
it proves end-to-end, through fetch_url_content_async, that a refusal
happens *before* the outbound call on all three fetch sites (httpx,
Scrape.do, Playwright) — the thing a grep can't prove. It is the
regression test for "someone deleted the validate_public_url(...) call
from _fetch_with_scraped_do / _fetch_with_playwright".
"""
from __future__ import annotations

import pytest

from tools.url_tools import safe_fetch


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

    @pytest.fixture(autouse=True)
    def _no_ssrf_validation(self, monkeypatch):
        """Neutralize SSRF validation for these routing tests — the URLs used
        here (example.com, spa-example.com, slow-site.com, ...) are stand-ins
        for "some public URL" and are not meant to be resolved for real."""
        monkeypatch.setattr(
            "tools.url_tools.safe_fetch.validate_public_url", lambda url: None
        )

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


def _reject_from_second_call(monkeypatch):
    """Patch safe_fetch.validate_public_url so the 1st call (always the
    step-1 httpx validation) passes, and every call from the 2nd onward
    raises UnsafeUrlError.

    This is how we get end-to-end coverage, through fetch_url_content_async,
    of the *site-specific* validate_public_url call inside
    _fetch_with_scraped_do / _fetch_with_playwright — without it, deleting
    that call would leave all of TestB4UrlFetcher green (see module
    docstring). A single fixed bad literal-IP URL can't do this, because
    the exact same `url` string is reused at every fetch site, and a real
    validate_public_url call is deterministic on that string: if step 1
    already accepted it, the later site-specific call would accept it too.
    Counting calls lets us prove "this call site still runs" independent
    of step 1's own check.
    """
    calls = {"n": 0}

    def fake_validate(url):
        calls["n"] += 1
        if calls["n"] > 1:
            raise safe_fetch.UnsafeUrlError("blocked for regression test")

    monkeypatch.setattr("tools.url_tools.safe_fetch.validate_public_url", fake_validate)
    return calls


class _FakeNavRoute:
    """Fake playwright.async_api.Route for the fake-Playwright harness below."""

    def __init__(self, request):
        self.request = request
        self.aborted = False

    async def abort(self, error_code=None):
        self.aborted = True

    async def continue_(self, **kwargs):
        pass


class _FakeNavRequest:
    def __init__(self, url, is_navigation=True):
        self.url = url
        self._is_navigation = is_navigation

    def is_navigation_request(self):
        return self._is_navigation


class _FakePlaywrightPage:
    """Simulates a Chromium page whose goto() walks through `nav_urls`
    (the initial URL plus each server redirect) invoking whatever handler
    was registered via page.route(), exactly like real Playwright would
    invoke it for the initial navigation and every internal redirect."""

    def __init__(self, nav_urls, rendered_html="<html><body><p>" + ("ok " * 20) + "</p></body></html>"):
        self._handler = None
        self._nav_urls = list(nav_urls)
        self._rendered_html = rendered_html

    async def set_extra_http_headers(self, headers):
        pass

    async def route(self, pattern, handler):
        self._handler = handler

    async def goto(self, url, wait_until=None, timeout=None):
        assert self._handler is not None, "route() must be called before goto()"
        for hop_url in self._nav_urls:
            request = _FakeNavRequest(hop_url)
            route = _FakeNavRoute(request)
            await self._handler(route, request)
            if route.aborted:
                raise RuntimeError(f"net::ERR_FAILED navigating to {hop_url}")
        import unittest.mock as mock
        resp = mock.MagicMock()
        resp.status = 200
        resp.headers = {"content-type": "text/html"}
        return resp

    async def content(self):
        return self._rendered_html


class _FakePlaywrightBrowser:
    def __init__(self, page):
        self._page = page
        self.closed = False

    async def new_page(self):
        return self._page

    async def close(self):
        self.closed = True


class _FakeChromiumLauncher:
    def __init__(self, browser):
        self._browser = browser
        self.launch_called = False

    async def launch(self, headless=True):
        self.launch_called = True
        return self._browser


class _FakePlaywrightCtx:
    def __init__(self, chromium):
        self.chromium = chromium

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class TestSsrfEnforcedAtFetchSites:
    """End-to-end (through fetch_url_content_async) proof that a refusal
    happens *before* the outbound call at every fetch site — no
    _no_ssrf_validation fixture here, so real safe_fetch validation runs.
    """

    def test_httpx_path_refuses_metadata_url_before_any_request(self):
        import asyncio, unittest.mock as mock
        from tools.url_tools.extractor import fetch_url_content_async
        from tools.url_tools import safe_fetch

        async def run():
            mock_client = mock.AsyncMock()  # .stream/.get left unconfigured
            return await fetch_url_content_async(
                mock_client, "http://169.254.169.254/latest/meta-data/"
            )

        result = asyncio.run(run())
        assert result.success is False
        assert result.error_message == safe_fetch.REFUSAL_MESSAGE
        # 169.254.169.254 nowhere in the message — nothing leaked.
        assert "169.254" not in result.error_message

    def test_scrape_do_path_refuses_before_request_is_built(self, monkeypatch):
        """SPA content + Scrape.do token: the Scrape.do request must never
        be issued if the (re-)validation inside _fetch_with_scraped_do
        refuses the URL."""
        import asyncio, unittest.mock as mock
        from tools.url_tools.extractor import fetch_url_content_async
        from tools.url_tools import safe_fetch

        SPA_HTML = "<html><body><div id='root'></div></body></html>"
        calls = _reject_from_second_call(monkeypatch)

        async def run():
            mock_client = _make_stream_client(SPA_HTML)
            mock_client.get = mock.AsyncMock()  # would be the Scrape.do call

            fake_secret = mock.MagicMock()
            fake_secret.get_secret_value.return_value = "test-scraped-do-token"

            with mock.patch("core.config.settings") as cfg_mock:
                cfg_mock.scraped_do_api_key = fake_secret
                result = await fetch_url_content_async(mock_client, "https://example.com")
            return mock_client, result

        mock_client, result = asyncio.run(run())
        assert calls["n"] >= 2, "site-specific validate_public_url call did not run"
        mock_client.get.assert_not_called()
        assert result.success is False
        assert result.error_message == safe_fetch.REFUSAL_MESSAGE

    def test_playwright_path_refuses_before_browser_launch(self, monkeypatch):
        """SPA content, no Scrape.do token: the browser must never be
        launched if the (re-)validation inside _fetch_with_playwright
        refuses the URL."""
        import asyncio, unittest.mock as mock
        from tools.url_tools.extractor import fetch_url_content_async
        from tools.url_tools import safe_fetch

        SPA_HTML = "<html><body><div id='root'></div></body></html>"
        calls = _reject_from_second_call(monkeypatch)

        page = _FakePlaywrightPage(nav_urls=["https://example.com"])
        browser = _FakePlaywrightBrowser(page)
        chromium = _FakeChromiumLauncher(browser)
        playwright_ctx = _FakePlaywrightCtx(chromium)

        async def run():
            mock_client = _make_stream_client(SPA_HTML)

            with mock.patch("core.config.settings") as cfg_mock, \
                 mock.patch("playwright.async_api.async_playwright",
                             return_value=playwright_ctx):
                cfg_mock.scraped_do_api_key = None
                result = await fetch_url_content_async(mock_client, "https://example.com")
            return result

        result = asyncio.run(run())
        assert calls["n"] >= 2, "site-specific validate_public_url call did not run"
        assert chromium.launch_called is False
        assert result.success is False
        assert result.error_message == safe_fetch.REFUSAL_MESSAGE

    def test_playwright_redirect_to_private_address_is_refused(self, monkeypatch):
        """The Playwright page itself navigates to a public URL, but the
        server responds with a redirect to a private/loopback address.
        Request interception must catch that second navigation, and the
        failure must surface as the exact fixed REFUSAL_MESSAGE — not a
        raw Playwright exception string that could contain the target."""
        import asyncio, unittest.mock as mock
        from tools.url_tools.extractor import fetch_url_content_async
        from tools.url_tools import safe_fetch

        SPA_HTML = "<html><body><div id='root'></div></body></html>"

        # Public initial URL (literal IP — no DNS involved), then a
        # same-site-looking redirect straight to loopback.
        page = _FakePlaywrightPage(
            nav_urls=["http://93.184.216.34/start", "http://127.0.0.1/admin"]
        )
        browser = _FakePlaywrightBrowser(page)
        chromium = _FakeChromiumLauncher(browser)
        playwright_ctx = _FakePlaywrightCtx(chromium)

        async def run():
            mock_client = _make_stream_client(SPA_HTML)

            with mock.patch("core.config.settings") as cfg_mock, \
                 mock.patch("playwright.async_api.async_playwright",
                             return_value=playwright_ctx):
                cfg_mock.scraped_do_api_key = None
                result = await fetch_url_content_async(
                    mock_client, "http://93.184.216.34/start"
                )
            return result

        result = asyncio.run(run())
        assert chromium.launch_called is True  # browser *did* launch this time
        assert result.success is False
        assert result.error_message == safe_fetch.REFUSAL_MESSAGE
        # Never leak the blocked target back to the model.
        assert "127.0.0.1" not in result.error_message
