"""End-to-end smoke test: does _build_personal_tier find 'Best Friend: Aarav'
when queried with the same paraphrases the model emits?
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.personal_memory_store import PersonalMemoryStore
from core.retrieval_broker import RetrievalBroker
from core.task_history import TaskHistoryStore


async def main() -> None:
    user_id = "usr_7ef119f3ab8b"
    base = Path("data") / "memory" / "personal" / user_id
    store = PersonalMemoryStore(user_id=user_id, base_dir=base)
    task_store = TaskHistoryStore(history_path=base / "task_history.jsonl")
    broker = RetrievalBroker(store=store, task_store=task_store)
    print(f"relations registered? {'relations' in store.topic_paths}, exists? {store.topic_paths.get('relations', Path('?')).exists()}")

    queries = [
        "user's best friend",
        "do you remember my best friend",
        "best friend",
        "user's friend",
        "what is the user's timezone",
        "my morning routine",
    ]

    for q in queries:
        result = await broker._build_personal_tier(q)
        print(f"\n=== query: {q!r} ===")
        print(result if result else "(empty)")


if __name__ == "__main__":
    asyncio.run(main())
