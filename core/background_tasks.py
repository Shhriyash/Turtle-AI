"""
core/background_tasks.py
------------------------
G3/D5: Background tasks registry and implementations.
"""
from __future__ import annotations

import logging
from typing import Any

from core.worker import task
from core.storage.local.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)


@task("embed_personal_memory")
async def embed_personal_memory(user_id: str, topic_name: str, lines: list[str]) -> None:
    """Embeds all lines of a personal memory topic into the vector store."""
    vs = FAISSVectorStore()
    
    for idx, line in enumerate(lines):
        text = line.strip()
        if text.startswith("- "): 
            text = text[2:].strip()
        if not text:
            continue
            
        doc_id = f"topic_{topic_name}_{idx}"
        await vs.upsert(
            user_id=user_id,
            doc_id=doc_id,
            text=text,
            metadata={"topic": topic_name, "line_index": idx}
        )
