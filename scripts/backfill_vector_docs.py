from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.identity import identity_manager
from core.personal_memory_store import PersonalMemoryStore


def _line_to_text(raw_line: str) -> str:
    text = raw_line.strip()
    if text.startswith("- "):
        text = text[2:].strip()
    return text


def _docs_for_user(user_id: str) -> list[tuple[str, str, dict]]:
    """Re-derive the (doc_id, text, metadata) triples embed_personal_memory
    would have produced for every line currently on record for this user,
    across every topic. doc_id scheme (`topic_{topic_name}_{idx}`) mirrors
    core/background_tasks.py::embed_personal_memory exactly, so re-running
    the real per-write embed job later for the same line lands on the same
    doc_id and safely overwrites this backfill's row rather than duplicating it.
    """
    store = PersonalMemoryStore(user_id)
    docs: list[tuple[str, str, dict]] = []
    for topic_name in store.topic_paths:
        doc = store.load_topic(topic_name)
        for idx, raw_line in enumerate(doc.lines or []):
            text = _line_to_text(str(raw_line))
            if not text:
                continue
            docs.append((f"topic_{topic_name}_{idx}", text, {"topic": topic_name, "line_index": idx}))
    return docs


async def _resolve_user_ids(user_arg: str) -> list[str]:
    if user_arg != "all":
        return [user_arg]
    users = await identity_manager.list_users()
    return [u["user_id"] for u in users if u.get("user_id")]


async def _run(user_arg: str, dry_run: bool) -> int:
    user_ids = await _resolve_user_ids(user_arg)
    if not user_ids:
        print("No users found.")
        return 0

    total_docs = 0
    vector_store = None
    if not dry_run:
        # Constructed lazily, and only on the real-write path: PgVectorStore's
        # embedder is Cohere, whose __init__ raises without COHERE_API_KEY, so
        # building it here would crash every dry run for anyone without a key
        # set -- exactly the case a dry run exists to be safe for.
        from core.storage.factory import get_vector_store

        vector_store = get_vector_store()

    for user_id in user_ids:
        docs = _docs_for_user(user_id)
        print(f"user={user_id}: {len(docs)} fact line(s) to re-embed")
        for doc_id, text, metadata in docs:
            if dry_run:
                print(f"  [dry-run] would upsert doc_id={doc_id!r} text={text!r} metadata={metadata!r}")
                continue
            await vector_store.upsert(user_id, doc_id, text, metadata)
            print(f"  upserted doc_id={doc_id!r}")
        total_docs += len(docs)

    print(f"\n{'Would re-embed' if dry_run else 'Re-embedded'} {total_docs} fact line(s) "
          f"across {len(user_ids)} user(s).")
    if dry_run:
        print("Dry run only. Re-run with --confirm to write to vector_docs.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-embed every user's current personal-memory topics into vector_docs (ledger 2.2)."
    )
    parser.add_argument("--user", required=True, help="'all', or a single user_id (e.g. usr_abc123).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be re-embedded without writing or calling the embedder. Implied unless --confirm is passed.",
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Actually perform the re-embed. Without this flag the script always runs as a dry run.",
    )
    args = parser.parse_args()

    # Dry-run-by-default posture, matching scripts/wipe_data.py: writing to
    # every user's vector_docs is exactly the kind of one-shot admin action
    # that must require an explicit opt-in, not an accidentally-bare default.
    dry_run = not args.confirm

    return asyncio.run(_run(args.user, dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
