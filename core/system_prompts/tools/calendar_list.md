# Tool: calendar_list

## Purpose
List upcoming events from the user's primary Google Calendar.

## When to USE
- User asks "what's on my calendar", "do I have anything this week", "what meetings do I have today", "show me my schedule"

## When NOT to USE
- User wants to *create* an event — use calendar_create instead

## Parameters
- `max_results` (optional, int 1–20, default 5): Number of upcoming events to return.
- `time_min_iso` (optional, string): Only return events after this ISO 8601 datetime. Leave empty to default to now.

## Return shape
List of upcoming events, each with title, start/end datetime, Google Calendar URL, and Meet link if present.
On failure: error message with code `credentials_missing` or `upstream_error`.

## Common failure modes
- **credentials_missing**: Google Calendar is not configured — tell the user to set `GOOGLE_CALENDAR_CREDENTIALS_JSON`.
- **Empty list**: No upcoming events in the requested window.
