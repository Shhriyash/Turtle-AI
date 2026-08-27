#!/usr/bin/env python3
"""
core/sandbox/guest/fsops.py
---------------------------
Runs INSIDE the sandbox container, bind-mounted read-only at /opt/turtle/fsops.py.
Emits exactly one JSON object on stdout.

This file must stay stdlib-only and Python-3.8-compatible: it executes against
whatever interpreter the sandbox image ships, which is not necessarily the one
Turtle runs on. Nothing here may import from `core.*`.

Why file I/O happens in here instead of host-side, which would be simpler and
faster: a host-side open() on the bind-mounted workspace FOLLOWS SYMLINKS INTO
THE REAL HOST FILESYSTEM. The model can create a symlink inside /workspace (it
has a shell), point it at ~/.ssh/id_rsa, and a host-side "jailed" read hands it
over — the exact escape a Path.resolve() string check cannot stop, because
resolve() is doing the following. Run the read inside the mount namespace and
the same symlink resolves to the CONTAINER's /root/.ssh, which is empty. The
namespace is the jail; this script just gives it a structured interface.

The containment checks below are belt-and-braces for clean error messages. They
are NOT the boundary. Do not let anyone talk you into moving this host-side
because "we validate the path anyway".
"""
import json
import os
import sys

WORKSPACE = "/workspace"
MAX_LIST_ENTRIES = 500


def fail(message, code="error"):
    json.dump({"ok": False, "error": message, "code": code}, sys.stdout)
    sys.stdout.write("\n")
    sys.exit(0)  # exit 0: the JSON body carries the outcome, not the exit code


def jailed(rel):
    """Resolve a caller-supplied relative path under WORKSPACE."""
    if rel is None:
        rel = "."
    rel = rel.strip().replace("\\", "/")
    if not rel:
        rel = "."
    if rel.startswith("/"):
        fail("path must be relative to the workspace, not absolute", "bad_path")
    if any(part == ".." for part in rel.split("/")):
        fail("path must not contain '..' segments", "bad_path")
    target = os.path.realpath(os.path.join(WORKSPACE, rel))
    root = os.path.realpath(WORKSPACE)
    if target != root and not target.startswith(root + os.sep):
        fail("path escapes the workspace", "bad_path")
    return target


def entry_type(path):
    if os.path.islink(path):
        return "symlink"
    if os.path.isdir(path):
        return "dir"
    if os.path.isfile(path):
        return "file"
    return "other"


def op_read(rel, max_bytes):
    target = jailed(rel)
    if not os.path.exists(target):
        fail("file not found: " + rel, "not_found")
    if os.path.isdir(target):
        fail("path is a directory; use list", "is_dir")
    size = os.path.getsize(target)
    with open(target, "rb") as fh:
        raw = fh.read(max_bytes)
    truncated = size > max_bytes
    return {
        "ok": True,
        "path": os.path.relpath(target, WORKSPACE),
        "size_bytes": size,
        "truncated": truncated,
        # errors="replace": a byte-capped read routinely lands mid-UTF-8, and a
        # decode error would turn "your file is long" into "the tool crashed".
        "content": raw.decode("utf-8", "replace"),
    }


def op_write(rel, max_bytes):
    target = jailed(rel)
    if os.path.isdir(target):
        fail("path is a directory", "is_dir")
    payload = sys.stdin.buffer.read(max_bytes + 1)
    if len(payload) > max_bytes:
        fail("content exceeds max_write_bytes (%d)" % max_bytes, "too_large")
    parent = os.path.dirname(target)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    existed = os.path.exists(target)
    # Write to a temp file in the same dir then rename: a half-written file after
    # a timeout/OOM-kill is worse than no write, especially since the model will
    # read it back and reason about the truncated content as if it were real.
    tmp = target + ".turtle-tmp"
    with open(tmp, "wb") as fh:
        fh.write(payload)
    # os.replace, not os.rename: rename REFUSES to clobber an existing target on
    # Windows, so every overwrite would fail there. The container is Linux, but
    # this helper is also driven directly by the test suite on the dev machine,
    # and a helper that only works on one OS is a helper nobody can debug.
    os.replace(tmp, target)
    return {
        "ok": True,
        "path": os.path.relpath(target, WORKSPACE),
        "bytes_written": len(payload),
        "overwrote": existed,
    }


def op_list(rel):
    target = jailed(rel)
    if not os.path.exists(target):
        fail("directory not found: " + rel, "not_found")
    if not os.path.isdir(target):
        fail("path is not a directory", "not_dir")
    names = sorted(os.listdir(target))
    truncated = len(names) > MAX_LIST_ENTRIES
    entries = []
    for name in names[:MAX_LIST_ENTRIES]:
        full = os.path.join(target, name)
        etype = entry_type(full)
        size = None
        mtime = None
        try:
            # follow_symlinks=False: stat'ing through a symlink would report the
            # TARGET's size, and a symlink pointing outside the workspace would
            # leak host-side metadata into the listing.
            st = os.stat(full, follow_symlinks=False)
            size = st.st_size if etype == "file" else None
            mtime = st.st_mtime
        except OSError:
            pass
        entries.append({
            "name": name,
            "type": etype,
            "size_bytes": size,
            "mtime": mtime,
        })
    return {
        "ok": True,
        "path": os.path.relpath(target, WORKSPACE),
        "entries": entries,
        "truncated": truncated,
    }


def op_mkdir(rel):
    target = jailed(rel)
    if not os.path.isdir(target):
        os.makedirs(target)
    return {"ok": True, "path": os.path.relpath(target, WORKSPACE)}


def main(argv):
    if len(argv) < 2:
        fail("usage: fsops.py <read|write|list|mkdir> <path> [max_bytes]", "usage")
    op = argv[1]
    rel = argv[2] if len(argv) > 2 else "."
    try:
        limit = int(argv[3]) if len(argv) > 3 else 262144
    except ValueError:
        limit = 262144

    try:
        if op == "read":
            result = op_read(rel, limit)
        elif op == "write":
            result = op_write(rel, limit)
        elif op == "list":
            result = op_list(rel)
        elif op == "mkdir":
            result = op_mkdir(rel)
        else:
            fail("unknown op: " + op, "usage")
    except OSError as exc:
        fail("%s: %s" % (type(exc).__name__, exc), "os_error")
    except Exception as exc:  # noqa: BLE001 - must always emit parseable JSON
        fail("%s: %s" % (type(exc).__name__, exc), "internal")

    json.dump(result, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv)
