from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus

from bs4 import BeautifulSoup
import httpx


DDG_HTML_SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


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
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return []

    response = await http_client.get(
        DDG_HTML_SEARCH_URL.format(query=quote_plus(normalized_query)),
        headers={"User-Agent": DEFAULT_USER_AGENT},
        follow_redirects=True,
        timeout=20.0,
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
