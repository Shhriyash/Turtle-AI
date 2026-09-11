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
- **credentials_missing**: The tool result includes a connect URL
  (`/integrations/google_calendar/connect`) — tell the user to open it and
  sign in with Google to link their calendar. If the URL is missing, Google
  Calendar isn't configured on this deployment at all; say so plainly.
- **Empty list**: No upcoming events in the requested window.
