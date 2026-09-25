# Tool: calendar_confirm

## Purpose
Create the Google Calendar event that was staged by a prior `calendar_create`
call. This is the ONLY tool that actually writes to the user's calendar.

## When to USE
- The user was just shown a calendar draft (from `calendar_create`) and replies affirmatively — "confirm", "yes", "go ahead", "looks good", "book it"

## When NOT to USE
- No draft has been shown yet — call `calendar_create` first
- The user wants to change something about the draft — call `calendar_create` again with the corrected fields instead; do not call `calendar_confirm` on a draft the user just asked to change
- User wants to view their calendar — use `calendar_list`

## Parameters
None. It acts on the pending draft already stored from `calendar_create`.

## Return shape
On success: event title, start/end, Google Calendar URL, and Meet link (if requested).
On failure: an error message; the draft is kept so the user can retry with `calendar_confirm` again.
If there is no pending draft (never created, or the draft expired after an hour of inactivity), it says so and asks you to call `calendar_create` again.
Confirming the same draft twice within a few minutes creates only one event — the duplicate call returns the same result instead of creating a second event.

## Common failure modes
- **No pending draft**: Tell the user their draft expired or was never created, then re-run `calendar_create` with the details.
- **credentials_missing**: The tool result includes a connect URL
  (`/integrations/google_calendar/connect`) — tell the user to open it and
  sign in with Google to link their calendar. If the URL is missing, Google
  Calendar isn't configured on this deployment at all; say so plainly.
