# Tool: place_details

## Purpose
Fetch extended details for a specific Google Places `place_id` — full weekly opening hours, an editorial summary, and any fields not surfaced by find_place's summary.

## When to USE
- User asks for the full weekly hours of a place you already looked up.
- User asks for a description, "what is this place", or a summary blurb of a specific place returned by find_place.
- User asks for information about ONE specific place they've already selected.

## When NOT to USE
- The user hasn't chosen a specific place yet → call find_place first.
- You already have the field the user is asking about in an existing find_place result — read it from there, don't refetch.
- You do not have a real `place_id` from a prior tool result. NEVER invent a `place_id`.

## Parameters
- `place_id` (required, string): The exact `place_id` from an earlier find_place result. Never invent or guess this.

## Return shape
A place record + the full 7-day opening hours + an editorial summary if Google has one.

## Common failure modes
- **credentials_missing**: `GOOGLE_MAPS_API_KEY` not configured.
- **bad_request**: The `place_id` was malformed — this usually means you invented one. Call find_place instead.
- **empty**: The `place_id` no longer resolves — the place may have closed.

## Example
User: "What are the full opening hours for Mercure Hotel Dubai Barsha?"
→ You already called find_place earlier and captured its `place_id`.
→ call `place_details(place_id="<id>")` and answer with the weekly hours.
