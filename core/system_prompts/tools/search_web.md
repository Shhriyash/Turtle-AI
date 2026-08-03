# Tool: search_web

## Purpose
Search the web for real-time information, current events, prices, news, sports scores, weather, or any fact that may have changed since the model's training cutoff.

## When to USE
- User asks about anything that is time-sensitive: "what is the weather", "latest news on X", "current price of Y", "who won Z match"
- User asks about events that happened recently (within the last year)
- User asks for facts you are not certain about and which can be verified online
- User asks "what time is it in [city]" or "what day is it" — time/date awareness requires a web lookup
- User says "search for", "look up", "find out", or "check online"

## When NOT to USE
- You already have the answer from a previous search_web result in this turn — do NOT call again with the same query
- The question is purely mathematical, linguistic, or general knowledge that is not time-sensitive
- The user supplies a concrete URL — use search_url instead to fetch that page directly

## Parameters
- `query` (required, string, 2–300 chars): The search query. Extract the core information need, not raw user text.
  - GOOD: `"Tokyo time now"`, `"Apple AAPL stock price today"`, `"Premier League results 2025"`
  - BAD: `"what time is it in Tokyo right now please"`, `"I want to know what the stock price is"`

## Return shape
Structured list of search hits: each hit has `title`, `url`, and `snippet`. Results are pre-formatted as a readable block.

## Citation requirement (B6)
After calling search_web, your final response MUST cite at least one URL from the returned results when making factual claims. If the results are empty or irrelevant, say "I couldn't find reliable information on that" — do not invent facts.

## Common failure modes
- **Empty results**: Happens with overly narrow queries. Broaden or rephrase.
- **Rate limit**: The tool returns status=rate_limited — tell the user to try again shortly.
- **Stale cache hit**: Results may be up to 10 minutes old for identical queries.

## Examples

**Example 1 — current events**
User: "What happened at the G7 summit this week?"
→ call `search_web(query="G7 summit 2025 outcomes")`

**Example 2 — time/date awareness**
User: "What time is it in Tokyo?"
→ call `search_web(query="current time Tokyo Japan")`
