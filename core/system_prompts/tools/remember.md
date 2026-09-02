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
- `key`: short snake_case identifier for the KIND of fact (e.g. `email`, `phone`, `favourite_editor`). Use the SAME key for values of the same kind — the `mode` parameter decides whether a new value replaces or joins the old one.
- `value`: verbatim-faithful fact as the user stated it.
- `mode`: `replace` (default) or `add`.

## Rules

- Store each distinct fact **once** — don't call `remember` twice for the same fact, and prefer the distilled value over echoing the whole sentence (for "I'm working on a project codenamed Atlas", use `key: codename`, `value: Atlas`).
- **`mode: replace`** (default) for a single-valued fact or a correction — the new value supersedes the previous one under this key (e.g. the user's name, their primary city, "actually my editor is nvim now").
- **`mode: add`** when the user gives ANOTHER value for something they can have several of — a second email, an extra phone number, another address or project. Reuse the SAME `key` (e.g. `key: email`) with `mode: add`, and each value is kept alongside the others instead of overwriting. Use `add` whenever the user says "also", "another", "add", or lists more than one. This is a judgement call about the fact, not a keyword rule — if the user clearly has multiple of something, accumulate.

## Return

Returns a confirmation string.

Only claim "I'll remember" after this tool returns ok.
