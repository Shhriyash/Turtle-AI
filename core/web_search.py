"""
core/web_search.py
------------------
B3: Replaced DDG-HTML scraping with Tavily Search API.
Tavily is a dedicated search API with structured results and real-time data.
DDG is retained as a fallback when TAVILY_API_KEY is absent.

Timeout: 6 s with one retry (E4 latency budget).
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from urllib.parse import quote_plus

import httpx


# ---------------------------------------------------------------------------
# Shared result type (unchanged API surface)
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


# ---------------------------------------------------------------------------
# Tavily (primary)
# ---------------------------------------------------------------------------

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_TAVILY_TIMEOUT = 6.0   # E4 hard cap
_TAVILY_RETRIES = 1


async def _search_tavily(
    http_client: httpx.AsyncClient,
    query: str,
    max_results: int = 10,
) -> list[SearchResult]:
    """Call Tavily Search API and return normalised SearchResult list."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }

    last_exc: Exception | None = None
    for attempt in range(_TAVILY_RETRIES + 1):
        try:
            resp = await http_client.post(
                TAVILY_SEARCH_URL,
                json=payload,
                timeout=_TAVILY_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            results: list[SearchResult] = []
            seen_urls: set[str] = set()
            for item in data.get("results", []):
                url = str(item.get("url", "")).strip()
                title = str(item.get("title", "")).strip()
                snippet = str(item.get("content", "") or item.get("snippet", "")).strip()
                if not url or url in seen_urls or not title:
                    continue
                seen_urls.add(url)
                results.append(SearchResult(title=title, url=url, snippet=snippet))
                if len(results) >= max_results:
                    break
            return results
        except Exception as exc:
            last_exc = exc
            if attempt < _TAVILY_RETRIES:
                await asyncio.sleep(0.3)

    raise RuntimeError(f"Tavily search failed after {_TAVILY_RETRIES + 1} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# DuckDuckGo (fallback when Tavily key is absent)
# ---------------------------------------------------------------------------

DDG_HTML_SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_DDG_TIMEOUT = 6.0   # E4 alignment


async def _search_duckduckgo_fallback(
    http_client: httpx.AsyncClient,
    query: str,
    max_results: int = 10,
) -> list[SearchResult]:
    """Fallback DDG-HTML scraping. Used only when TAVILY_API_KEY is absent."""
    from bs4 import BeautifulSoup

    response = await http_client.get(
        DDG_HTML_SEARCH_URL.format(query=quote_plus(query)),
        headers={"User-Agent": DEFAULT_USER_AGENT},
        follow_redirects=True,
        timeout=_DDG_TIMEOUT,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    results: list[SearchResult] = []
    seen_urls: set[str] = set()

    for node in soup.select(".result"):
        link = node.select_one(".result__title a")
        if not link:
            continue
        url = (link.get("href") or "").strip()
        title = link.get_text(" ", strip=True)
        snippet_node = node.select_one(".result__snippet")
        snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
        if not url or url in seen_urls or not title:
            continue
        seen_urls.add(url)
        results.append(SearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= max_results:
            break

    return results


# ---------------------------------------------------------------------------
# Public search entry point
# ---------------------------------------------------------------------------

def _normalize_query(query: str) -> str:
    normalized = " ".join((query or "").split())
    if "amazon.in" in normalized.lower() and "site:amazon.in" not in normalized.lower():
        normalized = f"site:amazon.in {normalized}"
    return normalized


async def search_duckduckgo(
    http_client: httpx.AsyncClient,
    query: str,
    max_results: int = 5,
) -> list[SearchResult]:
    """
    Primary search interface (kept for backward compatibility with tool registrations).

    Routing logic:
      1. If TAVILY_API_KEY is set  → use Tavily (structured, reliable, 6 s cap).
      2. Otherwise                 → fall back to DDG HTML scraping.

    NOTE: The DDG retry-relax hack previously in turtle_server.py is DELETED
    (B3). Tavily handles site: filters natively; for DDG fallback we accept
    best-effort results without a secondary retry.
    """
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return []

    if os.getenv("TAVILY_API_KEY", "").strip():
        try:
            return await _search_tavily(http_client, normalized_query, max_results=max_results)
        except Exception as exc:
            print(f"LOG: Tavily search failed ({exc}), falling back to DDG")

    return await _search_duckduckgo_fallback(http_client, normalized_query, max_results=max_results)


# ---------------------------------------------------------------------------
# Formatter (unchanged)
# ---------------------------------------------------------------------------

def format_search_results(query: str, results: list[SearchResult]) -> str:
    if not results:
        return (
            f"No web results found for query: {query}\n"
            "If this is a shopping request, try a more specific product, budget, or brand."
        )

    lines = [f"Web results for query: {query}"]
    for idx, result in enumerate(results, start=1):
        lines.append(f"{idx}. {result.title}")
        if result.snippet:
            lines.append(f"Snippet: {result.snippet}")
        lines.append(f"URL: {result.url}")
        lines.append("")

    return "\n".join(lines).strip()