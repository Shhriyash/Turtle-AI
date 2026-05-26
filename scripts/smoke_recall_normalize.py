"""Quick smoke test for _normalize_recall_query + _topics_for_keywords."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.retrieval_broker import _normalize_recall_query, _topics_for_keywords


CASES: list[str] = [
    "user's best friend",
    "user's friend",
    "do you remember my best friend",
    "do you remember what I told you about my job",
    "my morning routine",
    "best friend",
    "your wife's name",
    "who is my brother",
    "Tell me my preferred email tone",
    "what is the user's timezone",
]


def main() -> None:
    for case in CASES:
        normalized = _normalize_recall_query(case)
        topics = _topics_for_keywords(normalized)
        print(f"{case!r:60s} -> {normalized!r:40s} topics={topics}")


if __name__ == "__main__":
    main()
