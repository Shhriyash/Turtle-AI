# Tool: send_email_assistant

## Purpose
Send an email on behalf of the user. This tool handles extraction of recipients, subject, and body from natural language, validates the fields, and actually sends the email.

## When to USE
- User explicitly says "send an email", "email [person/address]", "write an email to", "shoot a mail to", "draft and send"
- User confirms they want to send a previously drafted message
- ANY time the user's intent is to actually deliver an email message

## When NOT to USE
- User is asking about an email they received — use search_web or history_tool instead
- User just wants help writing text without sending — offer to compose the text directly without calling this tool
- The request is about calendar invites — use a calendar tool instead

## Parameters
- `query` (required, string): The user's email request in their own words. Include ALL relevant detail:
  - Recipient name/address
  - CC/BCC mentions
  - Subject line
  - Body content or intent
  - Do NOT paraphrase — pass through the relevant raw user text so the email agent can extract fields correctly.

## Return shape
On success: "Email sent successfully!" followed by the To/Cc/Bcc/Subject header and send confirmation.
On missing fields: A prompt asking for the missing information (recipients / subject / body).
On invalid email format: A message identifying the invalid addresses.

## Citation requirement
Not applicable — this is a side-effecting action tool, not an information retrieval tool.

## Common failure modes
- **Missing recipient**: Tool will ask the user for the email address.
- **Missing subject or body**: Tool will ask the user to provide the missing field.
- **Invalid email format**: Tool reports which address is malformed.
- **Email config missing**: TURTLE_EMAIL_ADDRESS or TURTLE_EMAIL_PASSKEY not set in environment.

## Examples

**Example 1 — full send**
User: "Email john@example.com, subject 'Meeting tomorrow', body 'Hey John, are you free at 3pm?'"
→ call `send_email_assistant(query="Email john@example.com, subject 'Meeting tomorrow', body 'Hey John, are you free at 3pm?'")`

**Example 2 — partial info, multi-turn**
User: "Send an email to Sarah about the project update"
→ call `send_email_assistant(query="Send an email to Sarah about the project update")`
→ Tool returns: "Please provide Sarah's email address"
→ User gives address → call again with full details
