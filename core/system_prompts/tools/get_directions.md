# Tool: get_directions

## Purpose
Compute a real-world route between two locations using Google's Routes API. Returns the distance, ETA (traffic-aware for driving), and a Google Maps directions URL the user can open.

## When to USE
- User asks how to GET from A to B: "how do I get from the airport to Marina", "route to the office", "walking directions to the mall".
- User asks how FAR two places are: "distance from Bangalore to Mysore", "how far is my hotel from the beach".
- User asks HOW LONG it takes to travel: "drive time from JFK to Manhattan", "how long does it take to walk to the metro".
- User asks about a specific travel mode: driving, walking, cycling, transit, two-wheeler.

## When NOT to USE
- User only wants to LOCATE a single place → use find_place.
- User wants generic navigation advice, tips, or road rules → answer directly or use search_web.
- The user's two endpoints are ambiguous — resolve them with find_place first if unsure.

## Parameters
- `origin` (required, string): Starting address, place name, or `"lat,lng"` pair.
- `destination` (required, string): Ending address, place name, or `"lat,lng"` pair.
- `travel_mode` (optional, string, default `"DRIVE"`): One of `DRIVE`, `WALK`, `BICYCLE`, `TRANSIT`, `TWO_WHEELER`.

Map user language to travel_mode:
- "drive", "car", "driving" → DRIVE
- "walk", "on foot", "walking" → WALK
- "cycle", "bicycle", "bike" (unless it's a motorbike) → BICYCLE
- "bus", "train", "metro", "transit", "public transport" → TRANSIT
- "motorbike", "scooter", "two wheeler" → TWO_WHEELER

## Return shape
Distance (text + metres), duration (text + seconds), the travel mode, and a shareable Google Maps directions URL. Present the Maps URL in the reply so the user can tap through and start navigation.

## Common failure modes
- **credentials_missing**: `GOOGLE_MAPS_API_KEY` not configured — say Maps routing is unavailable.
- **empty**: No route was found — usually a typo in the endpoint names. Ask the user to clarify or try a nearby landmark.
- **invalid**: `travel_mode` was outside the supported set — retry with a valid mode.

## Examples

**Example 1 — driving between cities**
User: "How long to drive from Bangalore to Mysore?"
→ call `get_directions(origin="Bangalore", destination="Mysore", travel_mode="DRIVE")`
→ reply with distance, drive time, and the Maps directions link.

**Example 2 — walk from a hotel**
User: "How far is the beach from Mercure Hotel Al Barsha on foot?"
→ call `get_directions(origin="Mercure Hotel Al Barsha, Dubai", destination="Jumeirah Beach", travel_mode="WALK")`

**Example 3 — public transport**
User: "What's the fastest way to get to the airport by metro?"
→ call `get_directions(origin="<user's known location>", destination="<airport>", travel_mode="TRANSIT")`
