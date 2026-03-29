# Mitigation Plan

## Goal

Build a voice-first assistant that:

- keeps reliable short-term context during a live run
- persists cross-session memory without corrupting retrieval quality
- uses a strong main agent without forcing every task through weak multi-hop delegation
- executes tools safely, predictably, and fast

## Target architecture

### 1. Memory split

- `Working memory`: native PydanticAI `message_history` for the active run.
- `Session store`: persisted native `ModelMessage` history for crash recovery and session resume.
- `Fact store`: structured long-term user facts such as names, emails, preferences, aliases, accounts.
- `Episodic memory`: post-session RAG built from completed sessions only.

Rule:
- active session recall must never depend on RAG
- RAG is for past-session recall, not live slot filling

### 2. Agent split

- `Main agent`: intent routing, conversation, tool selection, final response synthesis.
- `Deterministic tools`: shell, file explorer, email send, URL fetch, web search wrapper, memory read/write.
- `Specialist subagents`: only for genuinely open-ended tasks such as writing, summarizing, research synthesis, code reasoning.

Rule:
- do not use agent-to-agent delegation for tasks that can be represented as validated tool arguments

### 3. Execution split

- `Immediate tools`: safe reads, search, URL fetch, memory lookup.
- `Validated side-effect tools`: email send, shell command run, file write, delete, external actions.
- `Deferred/approval tools`: high-risk actions if needed later.
- `Durable execution`: only for workflows that must survive crashes or await external confirmation.

## Phase plan

## Phase 0: Stabilize current runtime

### Changes

- add persistent per-session `message_history` in voice mode
- stop running voice turns with only the latest transcription
- pass the same message history to all sub-runs that depend on local context
- remove prompt claims that say history exists unless it is actually provided

### Implementation

- maintain `message_history: list[ModelMessage]` in `voice_chat()`
- after each run, update it from `result.all_messages()`
- on startup, try to restore session history from disk using `ModelMessagesTypeAdapter`
- use `history_processors` to keep token growth bounded

### Acceptance

- follow-ups like `to me`, `correct the email`, `send it now`, `same subject` resolve correctly in the same run
- active-session recall works without calling RAG

## Phase 1: Replace transcript memory with native session persistence

### Changes

- store active sessions as serialized PydanticAI messages, not flattened transcript pairs
- add crash recovery for unfinished sessions
- keep a session manifest with session id, timestamps, mode, and summary metadata

### Implementation

- write `message_history` to `data/session/<session_id>/messages.json`
- serialize using `to_jsonable_python(...)` and restore with `ModelMessagesTypeAdapter.validate_python(...)`
- atomically write session files
- on startup:
  - if an unfinished session exists, restore it
  - if session is stale, offer or automatically finalize it

### Acceptance

- killing the process mid-session does not lose active history
- restarting the app can continue or finalize the previous session

## Phase 2: Rebuild long-term memory properly

### Changes

- keep your design decision: build long-term RAG only after session end
- stop chunking by raw character windows
- stop storing only transcript prose
- add structured metadata and retrievable session summaries

### Implementation

- session finalization pipeline:
  1. load completed `ModelMessage` history
  2. derive structured turns and tool events
  3. extract durable facts
  4. create a session summary
  5. create turn-group chunks for RAG
  6. embed and upsert to vector DB

- chunking strategy:
  - chunk by turn groups, not characters
  - preserve boundaries: user turn, assistant reply, tool call, tool result
  - preserve timestamps and session id
  - no mid-word or mid-sentence splits
  - chunk size target by token estimate, with overlap only at turn boundaries

- vector schema:
  - `memory_id`
  - `session_id`
  - `turn_range`
  - `kind` = `summary|fact_context|turn_chunk|tool_trace`
  - `content`
  - `timestamp_start`
  - `timestamp_end`
  - `tags`

- storage behavior:
  - append-only writes
  - rebuildable index
  - no fake deletion that leaves index drift behind

### Acceptance

- no stored chunk starts mid-word
- session retrieval returns coherent conversation units
- exact past facts can be found with higher precision than generic semantic search

## Phase 3: Add a structured fact store

### Changes

- move stable user data out of RAG
- facts should be first-class records, not buried in transcript chunks

### Fact types

- identity: name, nicknames, emails, phone numbers
- preferences: tone, defaults, favorite tools, common recipients
- operational defaults: default sender, output paths, browser choice
- confirmed corrections: `my email is ...`, `use this account`, `correct recipient is ...`

### Implementation

- create `data/memory/facts.json` or SQLite
- store facts with:
  - `key`
  - `value`
  - `source_session_id`
  - `confidence`
  - `last_confirmed_at`
- add tools:
  - `remember_fact`
  - `lookup_fact`
  - `update_fact`
  - `forget_fact` if needed

Rule:
- exact facts are resolved from fact store first
- RAG is fallback for fuzzy memory questions

### Acceptance

- `what is my email` resolves without semantic search if previously confirmed
- corrected facts override older ones cleanly

## Phase 4: Rework tool architecture

### Changes

- make tools the primary execution path
- use subagents only where open-ended generation is required

### Tool categories

- `read tools`: web search, URL read, file read, directory inspect, memory lookup
- `write tools`: email send, file write, note save
- `system tools`: shell command runner, process status, environment checks
- `specialist generators`: email drafting, long-form writing, research synthesis

### Routing rules

- main agent decides:
  - answer directly
  - call deterministic tool
  - call specialist generator, then deterministic tool

- example:
  - `send an email to X saying Y`
    - direct call to validated `send_email`
  - `draft a professional email to X about Y`
    - call writing subagent or draft tool
    - then optionally call `send_email`

### Acceptance

- simple actions do not require agent-to-agent delegation
- specialist agents are only used when they add real value

## Phase 5: Harden tool contracts

### Changes

- use strict schemas, validators, retries, and dynamic tool availability

### Implementation

- use `args_validator` for:
  - email recipient validity
  - shell command safety policy
  - file path policy
  - required field combinations

- use `ModelRetry` for:
  - ambiguous recipient
  - missing subject/body
  - unsafe shell command
  - malformed path or URL

- use `prepare` / `prepare_tools` to expose only relevant tools per turn
  - example: hide shell and file-write tools during casual chat
  - example: enable email tools only when intent is messaging or drafting

- use `ToolReturn` where the model needs rich context but the app needs structured return values

### Acceptance

- invalid tool arguments trigger correction, not silent failure
- tool list is smaller and more relevant on each turn

## Phase 6: Build a proper task state layer

### Changes

- add lightweight session task state for multi-turn actions
- do not rely on the model to remember partial tool arguments

### Example state objects

- `pending_email = {recipients, subject, body, draft_status}`
- `pending_shell = {command, cwd, approval_status}`
- `pending_file_action = {path, operation, content}`

### Rules

- when a user gives partial info, update task state deterministically
- when all required fields are present, execute
- task state expires after completion, cancellation, or timeout

### Acceptance

- multi-turn tasks complete reliably even if each utterance is partial
- STT noise affects fewer fields because state is explicit

## Phase 7: Safety and durability

### Changes

- add approval and durable workflows only where they matter

### Immediate

- email send can remain immediate if validated
- shell command runner should support policy-based approval
- destructive file ops should require approval

### Later

- use deferred tools for approval gates
- use DBOS only for:
  - long-running jobs
  - workflows waiting for approval or external response
  - crash-safe finalization of session-to-RAG indexing

Rule:
- do not introduce DBOS into the whole app first
- use it only around specific workflows with clear recovery value

## Phase 8: Performance and observability

### Changes

- reduce token waste and measure actual failure modes

### Implementation

- use `history_processors`:
  - keep recent turns verbatim
  - summarize older active-session turns when needed

- cache:
  - web search results for short TTL
  - URL fetch results for current session
  - embeddings for finalized chunks

- log metrics:
  - tool call rate
  - tool validation failures
  - fallback-key rate
  - retries by tool
  - STT correction rate
  - RAG hit quality
  - average response latency

### Acceptance

- lower token usage per turn
- stable latency under longer conversations
- measurable improvement in tool success rate

## Recommended implementation order

1. Phase 0
2. Phase 1
3. Phase 6
4. Phase 5
5. Phase 4
6. Phase 2
7. Phase 3
8. Phase 8
9. Phase 7

Reason:
- fix live reliability first
- then recoverability
- then explicit task state
- then tool correctness
- then long-term memory quality
- durable execution comes last

## Non-goals

- do not use RAG for active-session follow-ups
- do not let the main agent see every tool on every turn
- do not persist only flattened transcript pairs
- do not chunk by character windows
- do not add more subagents unless the task is genuinely open-ended

## First implementation slice

Ship this first:

- voice-mode `message_history`
- native session persistence with restore
- explicit `pending_email` state
- validated `send_email` tool
- corrected prompts
- turn-based chunker skeleton for finalized sessions

This slice will remove the current biggest failures without requiring a full memory platform rewrite.
