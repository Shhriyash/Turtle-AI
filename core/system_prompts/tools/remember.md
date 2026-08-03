# remember

## Purpose

Permanently store a fact the user explicitly asked Turtle to remember, or a clearly-stated durable personal fact the user just shared.

## When to USE

Use this tool when the user says things like:

- "remember that..."
- "don't forget..."
- "note that my X is Y"

Use it for durable personal facts about the user, their preferences, workflow, contacts, relations, projects, corrections, working style, communication style, tool preferences, or decision style.

## When NOT to USE

Do not use this tool for:

- Temporary states ("I'm tired today").
- Hypotheticals.
- Facts about third parties that are not related to the user.
- Anything the user did not state themselves.

## Parameters

- `topic`: one of `identity`, `preferences`, `workflow`, `contacts`, `relations`, `projects`, `corrections`, `working_style`, `communication_style`, `tool_preferences`, `decision_style`.
- `key`: short snake_case identifier (e.g. `favourite_editor`).
- `value`: verbatim-faithful fact as the user stated it.

## Rules

- Store each distinct fact **exactly once**, under a single canonical snake_case key. Do **not** call `remember` more than once for the same fact.
- Prefer the distilled value over echoing the whole sentence. For "I'm working on a project codenamed Atlas", store `topic: projects`, `key: codename`, `value: Atlas` — a single entry, not both a `codename_atlas` sentence and a separate `project_codename` entry.

## Return

Returns a confirmation string.

Only claim "I'll remember" after this tool returns ok.
