# Tool: search_url

## Purpose
Fetch and extract the full readable content of a specific web page. Handles both static HTML and (with Playwright fallback) JavaScript-rendered single-page apps.

## When to USE
- User supplies a specific URL they want you to read or summarise
- A previous search_web call returned URLs and the user wants you to "open" or "read" one of them
- You need to verify a claim by reading the source article, not just the snippet
- User says "go to that link", "read that page", "open this URL", "what does [URL] say"

## When NOT to USE
- No concrete URL is present in the conversation — use search_web to find one first
- The URL is a file download link (PDF, ZIP) — flag this to the user instead

## Parameters
- `url` (required, string): Fully-qualified URL including scheme (https://).
  - MUST be a real URL from the conversation or from a tool result — never invent URLs.
  - GOOD: `"https://www.bbc.com/news/article-12345"`
  - BAD: Made-up or hallucinated domains

## Return shape
Plain text of the page content, truncated to ~3500 chars. Includes page title and main body text.

## Citation requirement (B6)
When you summarise or quote content from a URL, always mention the source URL in your response.

## Common failure modes
- **Paywalled page**: Returns only the intro paragraph. Inform the user the full article is behind a paywall.
- **JS-heavy SPA**: If httpx returns empty body, Playwright fallback is triggered automatically.
- **404 / unreachable**: Returns an error message — tell the user the page is unavailable.

## Examples

**Example 1 — reading a news article**
User: "Can you read this article for me? https://www.reuters.com/article/..."
→ call `search_url(url="https://www.reuters.com/article/...")`

**Example 2 — follow-up from search results**
User: "Open the first result" (after a search_web call returned URLs)
→ call `search_url(url="<url from first search result>")`
