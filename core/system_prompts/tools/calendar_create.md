# Tool: calendar_create

## Purpose
Create a Google Calendar event on behalf of the user, optionally with attendees and a Google Meet link.

## When to USE
- User says "schedule a meeting", "create an event", "book a call", "set up a reminder", "add to my calendar"
- User mentions a specific date, time, and activity to schedule

## When NOT to USE
- User just wants to *view* their calendar — use calendar_list instead
- Date or time is ambiguous — ask for clarification before calling; NEVER invent dates

## Parameters
- `title` (required, string): Event summary / name.
- `start_iso` (required, string): Start datetime in ISO 8601 with timezone, e.g. `"2026-05-10T14:00:00+05:30"`. Derive only from what the user explicitly stated.
- `end_iso` (required, string): End datetime ISO 8601. Must be after start_iso.
- `attendee_emails` (optional, list of strings): Only include email addresses the user explicitly provided. NEVER guess or fabricate emails.
- `description` (optional, string): Event description or agenda.
- `add_google_meet` (optional, bool, default true): Attach a Google Meet link.

## Return shape
On success: event title, start/end, Google Calendar URL, and Meet link (if requested).
On failure: error message with code `credentials_missing` (not configured) or `upstream_error`.

## Common failure modes
- **credentials_missing**: Google Calendar is not configured — tell the user to set `GOOGLE_CALENDAR_CREDENTIALS_JSON`.
- **Invalid ISO date**: The date string was malformed — recheck the format and retry.
