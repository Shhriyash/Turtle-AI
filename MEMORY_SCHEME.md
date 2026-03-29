# Persistent Memory Scheme

## Goal

Build persistent memory that:

- survives process restarts and new sessions
- keeps active conversations coherent
- personalizes Turtle over time
- does not flood prompts or grow uncontrollably

This scheme separates storage memory from prompt memory.

- `storage memory`: what is kept on disk long term
- `prompt memory`: the small subset injected into a model run

Prompt memory must stay bounded. Storage memory can grow, but in structured form.

## Memory layers

### 1. Working memory

Purpose:
- keep the current conversation coherent

Source:
- native PydanticAI `message_history`

Lifetime:
- current session only, but persisted for crash recovery

Storage:
- `data/sessions/active/messages.json`
- archived after session end

Used for:
- follow-ups like `to me`, `send it now`, `same subject`

Rule:
- active-session recall must come from working memory first, not RAG

### 2. Fact memory

Purpose:
- store exact durable user data

Examples:
- user name
- email addresses
- timezone
- common recipients
- preferred sender identity
- project paths
- model defaults

Characteristics:
- structured
- low volume
- exact-match retrievable
- updated in place when corrected

Rule:
- exact facts must never depend on semantic chunk retrieval

### 3. Preference memory

Purpose:
- store how the user wants Turtle to behave

Examples:
- concise responses
- low humor
- formal email tone
- ask before sending email
- default model preferences
- preferred output style

Characteristics:
- structured
- moderate volume
- one current value per preference key

Rule:
- preferences are compact state, not transcript excerpts

### 4. Pattern memory

Purpose:
- capture repeated behavior and working habits

Examples:
- often asks for short responses
- commonly sends emails after drafting
- often uses a particular project folder
- frequently asks for web lookups before decisions

Characteristics:
- aggregated, not raw logs
- evidence-based
- lower authority than explicit facts/preferences

Rule:
- patterns should influence suggestions, not override explicit instructions

### 5. Episodic memory

Purpose:
- retrieve useful information from past sessions

Examples:
- previous discussions
- earlier decisions
- prior troubleshooting context

Storage:
- post-session turn-based chunks in vector DB

Characteristics:
- fuzzy retrieval
- session-scoped
- not injected by default

Rule:
- episodic memory is for past-session recall, not exact personalization

## Storage model

Use SQLite for structured persistent memory.

Suggested database:
- `data/memory/turtle_memory.db`

Suggested tables:

### `user_facts`

Columns:
- `id`
- `key`
- `value_json`
- `canonical_value`
- `category`
- `confidence`
- `source_session_id`
- `source_turn_id`
- `created_at`
- `updated_at`
- `last_confirmed_at`
- `active`

Examples:
- `user.primary_email`
- `user.name`
- `user.timezone`
- `email.default_sender_address`

### `user_preferences`

Columns:
- `id`
- `key`
- `value_json`
- `category`
- `confidence`
- `source_session_id`
- `created_at`
- `updated_at`
- `last_confirmed_at`
- `active`

Examples:
- `assistant.response_style = concise`
- `assistant.humor_level = low`
- `email.default_tone = formal`

### `user_patterns`

Columns:
- `id`
- `pattern_key`
- `value_json`
- `evidence_count`
- `confidence`
- `first_seen_at`
- `last_seen_at`
- `active`

Examples:
- `workflow.prefers_draft_before_send`
- `workflow.common_project_path`

### `memory_events`

Purpose:
- audit trail for every add, update, merge, confirm, deactivate

Columns:
- `id`
- `memory_type`
- `memory_id`
- `event_type`
- `payload_json`
- `source_session_id`
- `created_at`

### `session_summaries`

Purpose:
- compact persistent summaries of past sessions

Columns:
- `session_id`
- `summary_text`
- `topics_json`
- `created_at`

## Retrieval model

Never inject all memory into every prompt.

Use task-scoped retrieval.

### Retrieval priority

1. working memory
2. fact memory
3. preference memory
4. pattern memory
5. episodic memory / RAG

### Retrieval by task

#### General conversation

Retrieve:
- name
- response style
- humor level
- any high-priority assistant behavior preferences

Prompt cap:
- 3 to 6 short memory lines

#### Email task

Retrieve:
- primary email
- preferred sender name
- common recipients or aliases
- default email tone
- send-confirmation preference

Prompt cap:
- 5 to 8 short lines

#### Web search task

Retrieve:
- response style preference
- domain/task preferences only if relevant

Prompt cap:
- 1 to 4 short lines

#### Memory question

Retrieve:
- facts first
- then episodic memory
- then session summaries

Rule:
- exact question `what is my email` should hit fact memory first

## Prompt injection policy

Create a small generated memory block per run.

Example:

```text
Relevant user memory:
- User name: Shriyash
- Preferred response style: concise
- Humor level: low
- Primary email: shriyashbeohar1@gmail.com
- Default email tone: formal
```

Rules:
- max 8 lines
- only include relevant items
- only include active, high-confidence memory
- never dump raw database rows
- never inject raw session transcripts by default

## Write policy

Not every observation should become memory.

### Write classes

#### A. Auto-save exact facts

Save immediately if explicit and unambiguous:
- `my email is ...`
- `my name is ...`
- `use this folder ...`
- `set default model to ...`

Action:
- upsert into fact memory

#### B. Confirmed preferences

Save if explicit:
- `keep responses concise`
- `be formal in emails`
- `do not use much humor`

Action:
- upsert into preference memory

#### C. Inferred patterns

Do not store after one observation.

Examples:
- user often asks for summaries
- user usually drafts before sending emails

Action:
- add/merge into pattern memory only after repeated evidence

### Confidence policy

- `1.0`
  Explicit exact statement from user

- `0.8`
  Explicit preference with clear wording

- `0.5`
  Inferred pattern with repeated evidence

- `< 0.5`
  Keep as event only, do not inject into prompts

## Update policy

Memory must be mutable.

Rules:
- exact fact correction replaces previous active value
- preference changes overwrite current preference
- patterns aggregate evidence counts
- deactivated values remain in audit history but are not injected

Example:
- old: `user.primary_email = a@gmail.com`
- new: `user.primary_email = b@gmail.com`
- result:
  - old row inactive
  - new row active
  - memory event recorded

## Compaction policy

Persistent memory should grow slowly, not indefinitely.

### Facts

- one active value per key unless multi-value is intentional
- old superseded values become inactive

### Preferences

- one active value per preference key
- overwrite rather than append duplicates

### Patterns

- aggregate evidence count
- merge duplicate patterns
- decay stale low-confidence patterns over time

### Episodic memory

- store finalized session chunks and summaries
- no prompt injection by default
- retrieval only on demand

## Session finalization flow

At session end:

1. archive native `message_history`
2. extract durable facts from the session
3. extract explicit preferences from the session
4. update pattern statistics from the session
5. create a short session summary
6. create turn-based episodic chunks
7. embed and store chunks in vector DB

Rule:
- long-term memory writes happen after session end unless the fact is operationally needed immediately

Immediate-write exceptions:
- corrected email
- sender identity
- critical path or config preference

## Memory extraction pipeline

Use a background extractor after session end.

Responsibilities:
- identify exact facts
- identify explicit preferences
- identify candidate patterns
- reject transient one-off statements

Output:
- structured memory write operations, not prose

Rule:
- the extractor proposes structured updates
- the memory layer applies merge/update rules

## RAG scheme

RAG remains persistent, but only for episodic memory.

### What goes into RAG

- finalized session summaries
- finalized turn-group chunks
- tool-related conversation context if useful

### What should not go into RAG as primary memory

- primary email
- user name
- stable preferences
- config defaults

### Chunking rules

- chunk by turn groups, not character windows
- preserve user/assistant/tool boundaries
- preserve timestamps
- preserve session id
- preserve topic tags if available
- never split mid-word
- avoid splitting mid-turn

## Personalization strategy

Do not use a separate always-on personalization agent in the main loop.

Better design:

- main agent remains the orchestrator
- main agent uses memory tools
- optional post-session memory extractor can be a specialist component

Why:
- lower token cost
- lower routing complexity
- more reliable behavior
- easier audit and correction

## Tools to add

The main agent should get a small memory toolset:

- `lookup_memory(query, category=None)`
- `remember_fact(key, value, category, confidence=1.0)`
- `remember_preference(key, value, category, confidence=1.0)`
- `update_memory(key, value, category)`
- `list_relevant_memory(task_type)`

Internal or background-only tools:

- `extract_session_memory(session_id)`
- `merge_pattern_evidence(...)`
- `finalize_session_summary(session_id)`

## Token control rules

This is the key anti-bloat policy:

- memory database can grow
- prompt memory must stay bounded

Enforcement:

- inject max 8 memory lines
- inject only relevant categories
- inject only active high-confidence items
- inject no raw transcript chunks unless user asks about history
- keep old sessions in archive and vector DB, not in normal prompts

## Practical examples

### Example 1: explicit fact

User says:
- `My email is shriyashbeohar1@gmail.com`

Action:
- write `user.primary_email`
- confidence `1.0`
- available in future sessions

### Example 2: explicit preference

User says:
- `Keep your responses concise`

Action:
- write `assistant.response_style = concise`
- confidence `0.8` or `1.0`

### Example 3: inferred pattern

Observed across many sessions:
- user usually asks for an email draft before sending

Action:
- increase evidence for `workflow.prefers_draft_before_send`
- only promote after threshold

### Example 4: memory query

User asks:
- `What is my email?`

Resolution:
- check fact memory first
- answer directly
- do not query RAG unless fact memory misses

## Recommended rollout

### Phase 1

- persistent session history
- fact memory store
- preference memory store
- task-scoped retrieval

### Phase 2

- post-session extractor
- pattern memory
- session summaries

### Phase 3

- rebuilt episodic RAG
- better ranking and retrieval controls

## Non-goals

- no full memory dump into prompts
- no Word/PDF as primary memory source
- no transcript chunks as primary personalization store
- no separate always-on personalization agent

## Final rule

Personalization should be:

- structured
- persistent
- selective
- auditable
- cheap at prompt time

That is how Turtle becomes more personal without becoming bloated or unreliable.
