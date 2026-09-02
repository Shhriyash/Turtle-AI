# Tool: sandbox_list_dir

## Purpose
List the files and directories in your sandbox workspace.

## When to USE
- Orienting yourself before reading or running anything — do this FIRST rather
  than guessing at filenames
- The user refers to "my file" or "the script" without naming it
- Confirming a file you wrote actually landed where you expected

## When NOT to USE
- You already know the exact path and just want the contents — use
  `sandbox_read_file`
- You need a recursive tree — use `sandbox_run(argv=["find", ".", "-type", "f"])`

## Parameters
- `path` (optional, string, default `"."`): Workspace-relative directory.
  Absolute paths and `..` are refused.

## Return shape
The directory path, then one line per entry. Directories are marked with a
trailing `/`, symlinks with `@`, and files show their size in bytes. A long
listing ends with `… (listing truncated)`.

## Common failure modes
- **`[sandbox denied] directory not found`** — list the parent directory instead
  of guessing further.
- **`[sandbox denied] path is not a directory`** — you passed a file path; use
  `sandbox_read_file`.

## Examples

**Example 1 — orient before acting**
User: "Run my analysis script"
→ `sandbox_list_dir(path=".")` to find its real name, then `sandbox_run`

**Example 2 — check a subdirectory**
User: "What data files do I have?"
→ `sandbox_list_dir(path="data")`
