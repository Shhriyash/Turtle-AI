# Tool: find_place

## Purpose
Look up real-world places (hotels, restaurants, offices, landmarks, airports, shops, hospitals) using Google Maps and return their authoritative Google Maps link, address, phone, website, hours, and rating.

## When to USE
- User asks WHERE something is: "where is Mercure Hotel Dubai", "locate the nearest Starbucks", "address of Kempegowda Airport".
- User asks for a place's phone number, opening hours, rating, or website.
- User asks to be pointed at a Google Maps link for a place.
- User asks for options: "find good sushi near Shibuya", "hospitals near Koramangala".
- User mentions a named business or landmark and the reply needs a real Maps link, not a homepage.

## When NOT to USE
- User only wants general knowledge or history about a place ("tell me about the Eiffel Tower") → answer directly or use search_web.
- User asks about news, events, or reviews of a place → search_web.
- User asked for directions or travel time between two places → use get_directions.
- You already looked the same place up earlier in this turn — reuse that result.

## Parameters
- `query` (required, string, 2–300 chars): Natural-language place query. Include the city or a nearby landmark when the user gave one, so global name collisions resolve correctly.
  - GOOD: `"Mercure Hotel Dubai Barsha Heights"`, `"Third Wave Coffee Indiranagar"`
  - BAD: `"Mercure"` (ambiguous), `"a hotel"` (no signal)
- `max_results` (optional, int, 1-10, default 5): How many candidates to return.
- `location_bias` (optional, string): City / neighbourhood / `"lat,lng"` hint to bias results. Leave empty for a global search.

## Return shape
A ranked list of places. Each entry carries: name, address, Google Maps URL, website, phone, rating, price level, primary category, whether it is open now, and today's hours.

## Answering after find_place
- ALWAYS surface the `Maps:` URL from the top result — that is exactly the "authoritative Google Maps link" users expect.
- Never present the business website in place of the Maps link; you may include both.
- When multiple results are plausible (many Mercure hotels in Dubai, for example), list the top 2-3 with their addresses so the user can pick.
- If the user's next question is "what are the hours", "what's their phone", or similar and you already have that in the result, answer from it — do NOT recall.

## Common failure modes
- **credentials_missing**: `GOOGLE_MAPS_API_KEY` not configured — tell the user Maps lookup is unavailable and suggest search_web as a fallback.
- **empty**: No matches — broaden the query or ask the user for a clarifying landmark. Do NOT invent an address.
- **auth_failed**: The API key is present but rejected — usually a missing "Places API (New)" enablement or a referrer/IP restriction. Tell the user briefly.
- **rate_limited**: Either Google itself is rate-limiting requests, or this account has hit its daily cap on Places/Directions lookups. Tell the user to try again later — do not retry immediately.

Note: identical searches (same query, result count, and location bias) made within about 10 minutes are served from a cache, so re-issuing the same search is cheap — but you should still prefer reusing an earlier result already in the conversation over calling the tool again.

## Examples

**Example 1 — hotel lookup**
User: "Locate the Mercure Hotel in Dubai."
→ call `find_place(query="Mercure Hotel Dubai")` → reply with the hotel name, address, and the Google Maps link from the top result.

**Example 2 — options near a landmark**
User: "Find good coffee near Cubbon Park in Bangalore."
→ call `find_place(query="specialty coffee near Cubbon Park", max_results=5, location_bias="Bengaluru")` → list 3 with maps links and ratings.

**Example 3 — reuse, don't re-fetch**
User (later in same turn): "What are their hours?"
→ Read the `Hours:` field from the earlier find_place result. Do NOT call the tool again.
