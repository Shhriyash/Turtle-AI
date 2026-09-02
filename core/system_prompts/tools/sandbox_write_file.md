# Tool: sandbox_write_file

## Purpose
Write a file into your sandbox workspace.

## When to USE
- Saving a script you wrote so you can run it with `sandbox_run`
- Saving output, results, or a report the user asked for
- Editing an existing file (read it first, then write the full new contents)

## When NOT to USE
- Writing outside the workspace — impossible, and you should say so
- Writing somewhere on the user's real machine — the workspace is a separate,
  isolated directory

## Parameters
- `path` (required, string): Workspace-relative path, e.g. `"src/main.py"`.
  Parent directories are created automatically. Absolute paths and `..` are
  refused.
- `content` (required, string): The **complete** new file contents. There is no
  append and no partial edit — whatever you pass replaces the file entirely.

## Return shape
`Wrote N bytes to <path>`.

## Important behaviour
- **This overwrites.** If the file exists, its previous contents are gone from
  the workspace. A snapshot is taken automatically beforehand, but do not treat
  that as an undo you can offer the user casually — read the file first if you
  are modifying rather than creating.
- **Full contents only.** To change one line, read the file, apply the change in
  your own reasoning, and write the whole file back.
- There is a size limit; very large content is refused with a clear message.

## Common failure modes
- **`[sandbox denied] content is N bytes; the limit is M`** — split the work or
  generate the file with a script instead of inlining it.
- **`[sandbox denied] path escapes the workspace`** — use a relative path.

## Examples

**Example 1 — create a script**
User: "Write a script that sums a CSV column"
→ `sandbox_write_file(path="sum.py", content="import csv\n...")`
→ then `sandbox_run(argv=["python3", "sum.py"])`

**Example 2 — edit an existing file**
User: "Rename the function in utils.py"
→ `sandbox_read_file(path="utils.py")`
→ `sandbox_write_file(path="utils.py", content="<full updated file>")`
