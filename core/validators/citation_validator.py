"""
core/validators/citation_validator.py
--------------------------------------
H2: Citation-grounded answer validator.

When a tool (web search, URL fetch) returned URLs, the post-response check
confirms the response cites at least one of those URLs when making factual
claims. If not, a re-prompt message is produced so the caller can ask the
model to cite or retract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_URL_RE = re.compile(r"https?://[^\s\)\]\"'>]+")

_REPROMPT_TEMPLATE = (
    "Your answer used facts from web sources but did not cite any of the "
    "returned URLs. Please revise your answer to include at least one source "
    "URL, or retract any claims you cannot support."
)


@dataclass
class CitationCheckResult:
    passed: bool
    urls_returned: list[str]
    urls_cited: list[str]
    reprompt: str | None = field(default=None)


def _extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text)


def check_citation(response: str, tool_urls: list[str]) -> CitationCheckResult:
    """Check whether *response* cites at least one URL from *tool_urls*.

    Rules:
    - If tool_urls is empty there is nothing to cite → always passes.
    - A URL is considered "cited" when it appears verbatim in the response
      (scheme + host prefix match is sufficient for redirected/shortened URLs).
    - Only the host+path portion must match; query strings are ignored for the
      containment check so shortened redirect chains still count.
    """
    if not tool_urls:
        return CitationCheckResult(passed=True, urls_returned=[], urls_cited=[])

    cited: list[str] = []
    response_urls = set(_extract_urls(response))

    for tool_url in tool_urls:
        tool_bare = _strip_query(tool_url)
        for r_url in response_urls:
            if _strip_query(r_url).startswith(tool_bare) or tool_bare.startswith(_strip_query(r_url)):
                cited.append(tool_url)
                break
        else:
            # Also accept substring match for shortened URLs
            if tool_url in response or _host(tool_url) in response:
                cited.append(tool_url)

    passed = len(cited) > 0
    return CitationCheckResult(
        passed=passed,
        urls_returned=list(tool_urls),
        urls_cited=cited,
        reprompt=None if passed else _REPROMPT_TEMPLATE,
    )


def _strip_query(url: str) -> str:
    return url.split("?")[0].rstrip("/")


def _host(url: str) -> str:
    """Extract scheme+host from a URL (e.g. 'https://example.com')."""
    m = re.match(r"https?://[^/]+", url)
    return m.group(0) if m else url
