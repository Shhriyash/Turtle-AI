# Email Agent Architecture

## Purpose

`send_email_assistant` is the main-agent tool entrypoint for email sending workflows.

Main agent responsibilities:
- detect email-send intent
- delegate to `send_email_assistant`

Email flow responsibilities:
- extract recipients, subject, and content
- keep partial state across turns
- validate required fields locally
- send immediately when complete
- ask only for missing fields when incomplete

## Components

### 1. Main delegator tool

File:
- `G:\dem\Turtle Voice\turtle\apps\turtle_voice.py`

Function:
- `send_email_assistant(ctx, query)`

Behavior:
- receives user request from `main_assistant`
- loads pending email state from `SessionStore`
- runs deterministic extraction hints
- calls `email_agent` for plain-text JSON extraction
- parses and merges extraction with pending state
- validates recipients and required fields
- sends email directly or asks for missing info

### 2. Email extraction agent

File:
- `G:\dem\Turtle Voice\turtle\apps\turtle_voice.py`

Object:
- `email_agent = Agent(..., output_type=str, ...)`

Prompt file:
- `G:\dem\Turtle Voice\turtle\core\system_prompts\email_agent.txt`

Role:
- extraction only
- returns JSON text only
- no direct SMTP side effects
- no final send ownership

### 3. Extraction parser

File:
- `G:\dem\Turtle Voice\turtle\core\email_flow.py`

Functions:
- `parse_email_extraction_response(response_text)`
- `EmailExtractionOutput`

Behavior:
- parses plain JSON output from the email extractor
- tolerates fenced JSON or surrounding text
- falls back to empty fields on malformed output

### 4. Session-backed pending state

File:
- `G:\dem\Turtle Voice\turtle\core\session_store.py`

State key:
- `pending_email`

Fields:
- `recipients: list[str]`
- `subject: str`
- `content: str`

Methods:
- `get_pending_email()`
- `set_pending_email(...)`
- `clear_pending_email()`

Persistence:
- stored in active session manifest
- survives app restart while session is active

### 5. Direct send path

Files:
- `G:\dem\Turtle Voice\turtle\apps\turtle_voice.py`
- `G:\dem\Turtle Voice\turtle\core\email_flow.py`
- `G:\dem\Turtle Voice\turtle\tools\email_tools\email_toolkit.py`

Functions:
- `validate_send_email_args(recipients, subject, content)`
- `send_email_now(details)`

Role:
- validate exact extracted fields in Python
- call SMTP send directly without a second LLM or tool-calling hop

Behavior:
- reject missing recipient, subject, or body before send
- reject invalid email formats before send
- load SMTP config via `create_email_tool_from_env()`
- send through `EmailTool.send_email(...)`
- return success or failure text to the main agent

## Input and output formats

### Input into `send_email_assistant`

Type:
- `query: str`

Example:
- `"Send an email to shriyashbeohar1 at the rate gmail.com subject hello body world"`

Context available:
- full message history from main-agent run
- pending email state from session store

### Email agent output

Type:
- `str`

Expected JSON shape:
- `{"recipients": ["user@example.com"], "subject": "hello", "content": "world", "send_intent": true}`

Notes:
- prompt requires JSON only
- parser still tolerates fenced JSON or minor output noise

### Output from `send_email_assistant`

Type:
- `str`

Forms:
- success:
  - includes recipient list and subject
- missing fields:
  - explicit prompt listing only missing items
- invalid recipient format:
  - asks user to provide recipient again
- config or send failure:
  - explicit failure message

## Runtime flow

1. Main agent calls `send_email_assistant`.
2. Tool reads pending state.
3. Deterministic parser extracts:
- spoken email normalization (`at the rate`, `dot`)
- email regex recipients
- subject and content markers
4. Email agent extracts JSON fields from latest request plus context.
5. Tool parses JSON into `EmailExtractionOutput`.
6. Tool merges:
- pending state
- deterministic extraction
- LLM extraction
7. Tool validates recipients.
8. If missing required fields:
- persist pending state
- ask only for missing parts
9. If complete:
- run local validation
- call `send_email_now(...)` directly
- clear pending state on success

## History access model

Main agent history:
- maintained in `SessionStore.message_history`

Delegated email history:
- `send_email_assistant` passes current run context when calling `email_agent`

Implication:
- email extraction can resolve short references using current conversation context
- extraction is still constrained to avoid inventing missing fields

## Reliability guards

- deterministic recipient normalization before validation
- no structured-output tool path in the extractor
- plain JSON parsing under local control
- pending state persisted per active session
- missing-field gating before SMTP call
- invalid-recipient gating before SMTP call
- no second LLM hop for final send
