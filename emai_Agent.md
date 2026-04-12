# Email Agent Architecture

## Purpose

`send_email_assistant` is the main-agent tool entrypoint for email sending workflows.

Main agent responsibilities:
- detect email-send intent
- delegate to `send_email_assistant`

Email flow responsibilities:
- extract recipients/cc/bcc/subject/content
- keep partial state across turns
- validate required fields
- send immediately when complete
- ask only for missing fields when incomplete

## Components

### 1. Main delegator tool

File:
- [apps/turtle_voice.py](G:\dem\Turtle Voice\turtle\apps\turtle_voice.py)

Function:
- `send_email_assistant(ctx, query)`

Behavior:
- receives user request from `main_assistant`
- loads pending email state from `SessionStore`
- runs deterministic extraction hints
- calls `email_agent` for structured extraction
- merges extraction with pending state
- validates recipients and required fields
- sends email or asks for missing info

### 2. Email extraction agent

File:
- [apps/turtle_voice.py](G:\dem\Turtle Voice\turtle\apps\turtle_voice.py)

Object:
- `email_agent = Agent(..., output_type=EmailExtractionOutput, ...)`

Prompt file:
- [core/system_prompts/email_agent.txt](G:\dem\Turtle Voice\turtle\core\system_prompts\email_agent.txt)

Role:
- extraction only
- no direct SMTP side effects

### 3. Session-backed pending state

File:
- [core/session_store.py](G:\dem\Turtle Voice\turtle\core\session_store.py)

State key:
- `pending_email`

Fields:
- `recipients: list[str]`
- `cc_recipients: list[str]`
- `bcc_recipients: list[str]`
- `subject: str`
- `content: str`

Methods:
- `get_pending_email()`
- `set_pending_email(...)`
- `clear_pending_email()`

Persistence:
- stored in active session manifest
- survives app restart while session is active

### 4. Email send agent

File:
- [apps/turtle_voice.py](G:\dem\Turtle Voice\turtle\apps\turtle_voice.py)

Object:
- `email_send_agent = Agent(..., output_type=str, ...)`

Prompt file:
- [core/system_prompts/email_send_agent.txt](G:\dem\Turtle Voice\turtle\core\system_prompts\email_send_agent.txt)

Role:
- call the validated send tool with exact values
- retry argument formatting if validation rejects the call

### 5. Validated send path

Files:
- [apps/turtle_voice.py](G:\dem\Turtle Voice\turtle\apps\turtle_voice.py)
- [tools/email_tools/email_toolkit.py](G:\dem\Turtle Voice\turtle\tools\email_tools\email_toolkit.py)

Functions:
- `send_email(ctx, recipients, subject, content)`
- `_send_email_now(details)`

Behavior:
- `send_email` is guarded by `args_validator`
- validator raises `ModelRetry` for missing or invalid fields
- loads SMTP config via `create_email_tool_from_env()`
- sends through `EmailTool.send_email(...)`
- returns success/failure message

## Input and output formats

## Input into `send_email_assistant`

Type:
- `query: str`

Example:
- `"Send an email to shriyashbeohar1 at the rate gmail.com subject hello body world"`

Context available:
- full message history from main-agent run (`ctx.messages`)
- pending email state from session store

## Email agent structured output

Model:
- `EmailExtractionOutput`

Schema:
- `recipients: list[str]`
- `cc_recipients: list[str]`
- `bcc_recipients: list[str]`
- `subject: str`
- `content: str`
- `send_intent: bool`

Notes:
- extraction prompt requires empty strings for missing fields
- no free-form explanation in extraction output

## Output from `send_email_assistant`

Type:
- `str`

Forms:
- success:
  - includes to/cc/bcc (when present) and subject
- missing fields:
  - explicit prompt listing only missing required items (`to`, `subject`, `content`)
- invalid email format:
  - asks user to provide corrected address (to/cc/bcc)
- config/send failure:
  - explicit failure message

## Runtime flow

1. Main agent calls `send_email_assistant`.
2. Tool reads pending state.
3. Deterministic parser extracts:
   - spoken email normalization (`at the rate`, `dot`)
   - email regex recipients
   - labeled `cc` and `bcc` recipients
   - subject/content markers
4. Email agent extracts structured fields from latest request + context.
5. Tool merges:
   - pending state
   - deterministic extraction
   - LLM extraction
6. Tool validates recipients.
   - validates `to`, `cc`, and `bcc` formats
7. If missing required fields:
   - persist pending state
   - ask only for missing parts
8. If complete:
   - call `email_send_agent`
   - `email_send_agent` calls validated `send_email` tool
   - clear pending state on success

## History access model

Main agent history:
- maintained in `SessionStore.message_history`

Delegated email history:
- `send_email_assistant` passes `message_history=ctx.messages` when calling `email_agent`

Implication:
- email extraction can resolve short references using current conversation context
- extraction is still constrained to avoid inventing missing fields

## Reliability guards

- deterministic recipient normalization before validation
- deterministic cc/bcc normalization before validation
- structured extraction output (`EmailExtractionOutput`)
- pending state persisted per active session
- missing-field gating before SMTP call
- invalid-address gating (to/cc/bcc) before SMTP call
- `args_validator + ModelRetry` on the send tool path
