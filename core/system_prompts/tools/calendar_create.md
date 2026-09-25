# Tool: calendar_create

## Purpose
Stage a Google Calendar event as a DRAFT for the user to review. This tool does
**NOT** create the event — it only stores the proposed event and shows it back
to the user. `calendar_confirm` is the tool that actually creates it, once the
user confirms.

## When to USE
- User says "schedule a meeting", "create an event", "book a call", "set up a reminder", "add to my calendar"
- User mentions a specific date, time, and activity to schedule

## When NOT to USE
- User just wants to *view* their calendar — use calendar_list instead
- Date or time is ambiguous — ask for clarification before calling; NEVER invent dates
- User is confirming a draft you already showed them ("yes", "confirm", "go ahead") — call `calendar_confirm` instead, not `calendar_create` again

## Parameters
- `title` (required, string): Event summary / name.
- `start_iso` (required, string): Start datetime in ISO 8601 with timezone, e.g. `"2026-05-10T14:00:00+05:30"`. Derive only from what the user explicitly stated.
- `end_iso` (required, string): End datetime ISO 8601. Must be after start_iso.
- `attendee_emails` (optional, list of strings): Only include email addresses the user explicitly provided. NEVER guess or fabricate emails.
- `description` (optional, string): Event description or agenda.
- `add_google_meet` (optional, bool, default true): Attach a Google Meet link.
- `notify_attendees` (optional, bool, default false): Send calendar invite
  emails to attendees. Set this to true **only** when the user explicitly
  asked to notify/invite/email the attendees ("invite them", "let them
  know", "send them the invite"). Leave it false — even when
  `attendee_emails` is non-empty — for anything else, e.g. "add Alice to
  this" without an explicit notify request.

## Return shape
Always a rendered draft (title, start, end, attendees, description, Meet
yes/no, notify yes/no) ending with a prompt to say "confirm" or describe a
change. There is no success/failure event result here — nothing has been
created yet.

## Next step
After this tool returns, wait for the user's confirmation, then call
`calendar_confirm` (no arguments) to actually create the event. If the user
asks for a change instead, call `calendar_create` again with the corrected
fields — the new draft replaces the old one.

## Common failure modes
- **Invalid ISO date**: The date string was malformed — recheck the format and retry.
