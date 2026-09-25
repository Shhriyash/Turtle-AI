"""
Core URL content extraction functionality.

B4: httpx first → SPA detection → Playwright (local) or Scrape.do+render (cloud) fallback.
"""

import re
import json
import urllib.parse
from urllib.parse import urlparse
from typing import Optional
import httpx
from bs4 import BeautifulSoup

from . import safe_fetch
from .models import UrlAnalysisResult

_SPA_THRESHOLD = 200  # chars of visible text below which we consider the page a SPA


def _is_spa_content(html_text: str) -> bool:
    """Return True when visible text is too sparse to be a static page."""
    return len(html_text.strip()) < _SPA_THRESHOLD


async def _fetch_with_scraped_do(
    http_client: httpx.AsyncClient,
    url: str,
    token: str,
    headers: dict,
    timeout: float,
) -> tuple[str, int, str]:
    """Fetch via Scrape.do API (render=true) — cloud JS-rendering + geo bypass."""
    # The outer request always goes to the fixed api.scrape.do host (plain
    # http, port 80 — permitted by our own rules), so SSRF validation of
    # *that* URL buys nothing. What matters is the model-supplied target
    # embedded in the query string: we refuse to ask a third party to fetch
    # an internal address on our behalf, which also stops this tool being
    # used as an abuse proxy against arbitrary internal hosts.
    try:
        safe_fetch.validate_public_url(url)
    except safe_fetch.UnsafeUrlError as exc:
        raise safe_fetch.UnsafeUrlError(safe_fetch.REFUSAL_MESSAGE) from exc
    target = (
        f"http://api.scrape.do"
        f"?token={token}&url={urllib.parse.quote(url, safe='')}&render=true"
    )
    resp = await http_client.get(
        target, headers=headers, timeout=timeout, follow_redirects=True
    )
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "text/html").lower()
    return resp.text, resp.status_code, content_type


async def _fetch_with_playwright(
    url: str, headers: dict, timeout: float
) -> tuple[str, int, str]:
    """Fetch via local Playwright headless browser for JS-rendered pages."""
    from playwright.async_api import async_playwright

    # A local Chromium instance navigating straight to the target — the
    # sharpest SSRF edge on this path, since the browser reaches the host's
    # own network stack. Validate before launching the browser at all, so an
    # obviously-bad URL fails fast without paying browser-startup cost.
    try:
        safe_fetch.validate_public_url(url)
    except safe_fetch.UnsafeUrlError as exc:
        raise safe_fetch.UnsafeUrlError(safe_fetch.REFUSAL_MESSAGE) from exc

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_extra_http_headers(headers)

            # Re-validate every *navigation* Chromium performs (the initial
            # goto plus every server redirect it follows internally) via
            # request interception — mirroring safe_get's manual redirect
            # loop for the httpx path, and enforcing the same MAX_REDIRECTS
            # hop limit. This must be registered before page.goto(), or the
            # first navigation escapes it entirely.
            #
            # Deliberately scoped to navigation requests
            # (request.is_navigation_request()) rather than every request:
            # blocking/validating every subresource load (JS, CSS, images,
            # XHR/fetch) would routinely break legitimate JS-rendered pages,
            # which is the entire reason this fallback exists. This is a
            # real, named residual gap, not an oversight: a page's own
            # script can still issue an XHR/fetch to an internal address
            # (e.g. the cloud metadata endpoint) after the page loads, and
            # that traffic is NOT validated by this handler.
            nav_state = {"count": 0, "blocked": False}

            async def _validate_navigation(route, request):
                if not request.is_navigation_request():
                    await route.continue_()
                    return
                nav_state["count"] += 1
                if nav_state["count"] > safe_fetch.MAX_REDIRECTS + 1:
                    nav_state["blocked"] = True
                    await route.abort()
                    return
                try:
                    safe_fetch.validate_public_url(request.url)
                except safe_fetch.UnsafeUrlError:
                    nav_state["blocked"] = True
                    await route.abort()
                    return
                await route.continue_()

            await page.route("**/*", _validate_navigation)

            try:
                resp = await page.goto(url, wait_until="networkidle", timeout=int(timeout * 1000))
            except Exception as exc:
                if nav_state["blocked"]:
                    # Translate Playwright's own navigation-failure exception
                    # (e.g. net::ERR_FAILED for an aborted request) into our
                    # fixed refusal — never leak the raw Playwright error
                    # string, which could contain the blocked target URL.
                    raise safe_fetch.UnsafeUrlError(safe_fetch.REFUSAL_MESSAGE) from exc
                raise

            if nav_state["blocked"]:
                raise safe_fetch.UnsafeUrlError(safe_fetch.REFUSAL_MESSAGE)

            status = resp.status if resp else 200
            content_type = (
                resp.headers.get("content-type", "text/html").lower() if resp else "text/html"
            )
            text = await page.content()
        finally:
            await browser.close()

    return text, status, content_type


def _build_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }


def _parse_html(
    html_text: str,
    parsed_url,
    url: str,
    max_content_length: int,
    status_code: int,
    content_type: str,
) -> UrlAnalysisResult:
    """Parse an HTML string into a UrlAnalysisResult."""
    content_bytes = html_text.encode("utf-8", errors="replace")
    soup = BeautifulSoup(content_bytes, "lxml")

    visible_text = soup.get_text().strip()
    if not visible_text:
        return UrlAnalysisResult(
            title="Dynamic Content Page",
            description="Page loads content via JavaScript",
            keywords=None,
            headings=[],
            content=(
                "This page appears to use JavaScript for content loading and "
                "requires browser rendering to display its full content."
            ),
            links=[],
            url=url,
            success=True,
        )

    # Metadata
    title_tag = soup.find("title")
    title_text = title_tag.get_text().strip() if title_tag else "No title"

    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = meta_desc.get("content", "").strip() if meta_desc else None

    meta_kw = soup.find("meta", attrs={"name": "keywords"})
    keywords = meta_kw.get("content", "").strip() if meta_kw else None

    # Strip noise
    for el in soup(["script", "style", "nav", "footer", "header", "aside",
                    "noscript", "iframe", "embed", "object", "form", "button"]):
        el.decompose()

    # Prefer semantic content containers
    main_content = None
    for selector in ["main", "article", '[role="main"]', ".content", ".main-content",
                     ".post-content", ".entry-content", ".article-content", "#content"]:
        main_content = soup.select_one(selector)
        if main_content:
            break
    content_source = main_content or soup.find("body") or soup

    headings = [
        f"{h.name.upper()}: {h.get_text().strip()}"
        for h in content_source.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        if h.get_text().strip() and len(h.get_text().strip()) > 3
    ]

    text_elements = [
        el.get_text().strip()
        for el in content_source.find_all(["p", "li", "div"])
        if el.get_text().strip() and len(el.get_text().strip()) > 20
    ]

    links = []
    for a in content_source.find_all("a", href=True):
        link_text = a.get_text().strip()
        link_url = a["href"]
        if link_text and link_url and len(link_text) > 3:
            if link_url.startswith("/"):
                link_url = f"{parsed_url.scheme}://{parsed_url.netloc}{link_url}"
            links.append(f"- {link_text}: {link_url}")

    full_text = " ".join(text_elements)
    cleaned = re.sub(r"\s+", " ", full_text).strip()

    if not cleaned or len(cleaned) < 50:
        cleaned = (
            f"Limited content extracted. "
            f"Raw HTML: {len(content_bytes)} bytes | "
            f"Status: {status_code} | Content-Type: {content_type}"
        )

    if len(cleaned) > max_content_length:
        cleaned = cleaned[:max_content_length] + "..."

    return UrlAnalysisResult(
        title=title_text,
        description=description,
        keywords=keywords,
        headings=headings,
        content=cleaned,
        links=links,
        url=url,
        success=True,
    )


async def fetch_url_content_async(
    http_client: httpx.AsyncClient,
    url: str,
    max_content_length: int = 8000,
    timeout: float = 20.0,
) -> UrlAnalysisResult:
    """
    Fetch and extract content from a URL.

    B4 strategy:
      1. httpx plain fetch (fast path)
      2. SPA detected (sparse body) → Scrape.do render=true (when API key set)
                                    → Playwright headless (local fallback)
    """
    # Validate URL
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return UrlAnalysisResult(
                title="", description=None, keywords=None, headings=[],
                content="", links=[], url=url, success=False,
                error_message=f"Invalid URL format: {url}",
            )
    except Exception as exc:
        return UrlAnalysisResult(
            title="", description=None, keywords=None, headings=[],
            content="", links=[], url=url, success=False,
            error_message=f"URL validation failed: {exc}",
        )

    headers = _build_headers()

    try:
        # ── Step 1: httpx (SSRF-validated, redirect-safe, body-capped) ──────
        response = await safe_fetch.safe_get(
            http_client, url, headers=headers, timeout=timeout
        )
        if response.status_code >= 400:
            return UrlAnalysisResult(
                title="", description=None, keywords=None, headings=[],
                content="", links=[], url=url, success=False,
                error_message=f"HTTP {response.status_code}",
            )
        content_type = response.headers.get("content-type", "").lower()
        html_text = response.text
        status_code = response.status_code

        # JSON — return as-is
        if "json" in content_type:
            try:
                formatted = json.dumps(json.loads(html_text), indent=2)
            except Exception:
                formatted = html_text
            return UrlAnalysisResult(
                title="JSON Content", description="API Response", keywords=None,
                headings=[], content=formatted, links=[], url=url, success=True,
            )

        # Non-HTML — return raw snippet
        if "html" not in content_type:
            return UrlAnalysisResult(
                title=f"Content ({content_type})", description="Non-HTML content",
                keywords=None, headings=[],
                content=html_text[:max_content_length],
                links=[], url=url, success=True,
            )

        # ── Step 2: SPA detection ──────────────────────────────────────────
        quick_soup = BeautifulSoup(html_text.encode("utf-8", errors="replace"), "lxml")
        visible = quick_soup.get_text()
        if _is_spa_content(visible):
            # Try JS-rendering fallback
            try:
                from core.config import settings
                scraped_do_token = (
                    settings.scraped_do_api_key.get_secret_value()
                    if settings.scraped_do_api_key
                    else None
                )
            except Exception:
                scraped_do_token = None

            if scraped_do_token:
                html_text, status_code, content_type = await _fetch_with_scraped_do(
                    http_client, url, scraped_do_token, headers, timeout
                )
            else:
                try:
                    html_text, status_code, content_type = await _fetch_with_playwright(
                        url, headers, timeout
                    )
                except ImportError:
                    pass  # Playwright not installed; proceed with sparse httpx content

        # ── Step 3: parse HTML ─────────────────────────────────────────────
        return _parse_html(html_text, parsed, url, max_content_length, status_code, content_type)

    except safe_fetch.UnsafeUrlError:
        # Fixed, non-leaky refusal text — never echo the resolved IP or the
        # internal validation reason back to the model.
        return UrlAnalysisResult(
            title="", description=None, keywords=None, headings=[],
            content="", links=[], url=url, success=False,
            error_message=safe_fetch.REFUSAL_MESSAGE,
        )
    except httpx.TimeoutException:
        return UrlAnalysisResult(
            title="", description=None, keywords=None, headings=[],
            content="", links=[], url=url, success=False,
            error_message=f"Timeout fetching {url} ({timeout}s exceeded)",
        )
    except httpx.HTTPStatusError as exc:
        return UrlAnalysisResult(
            title="", description=None, keywords=None, headings=[],
            content="", links=[], url=url, success=False,
            error_message=f"HTTP {exc.response.status_code}",
        )
    except Exception as exc:
        return UrlAnalysisResult(
            title="", description=None, keywords=None, headings=[],
            content="", links=[], url=url, success=False,
            error_message=str(exc),
        )


def fetch_url_content_sync(
    url: str,
    max_content_length: int = 8000,
    timeout: float = 20.0,
) -> UrlAnalysisResult:
    """Synchronous wrapper around fetch_url_content_async."""
    import asyncio
    import concurrent.futures

    async def _fetch():
        async with httpx.AsyncClient() as client:
            return await fetch_url_content_async(client, url, max_content_length, timeout)

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(asyncio.run, _fetch()).result()
    except RuntimeError:
        return asyncio.run(_fetch())
