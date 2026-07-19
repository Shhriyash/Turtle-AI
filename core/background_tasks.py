"""
core/background_tasks.py
------------------------
G3/D5: Background tasks registry and implementations.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from core.worker import task
from core.storage.local.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)

# Tenants that must never receive a real vector embed. "default" is the
# PersonalMemoryStore constructor default used pervasively by tests and legacy
# single-tenant stores; "" is an un-scoped store. A real tenant id is usr_*.
# Embedding for either would land in the SHARED
# data/memory/personal/default/vector index — cross-tenant collapse — so the
# job hard-skips it.
_SKIP_TENANTS = {"", "default"}

# Module-level lazy singleton. Sharing ONE FAISSVectorStore across jobs means
# its instance-scoped per-user locks (faiss_store.py) actually serialize
# concurrent embeds for a single user; a fresh store per job defeated them,
# letting concurrent embeds interleave _load_tenant -> add -> _save_tenant and
# clobber index.bin.
_vs_singleton: FAISSVectorStore | None = None
_vs_lock = threading.Lock()


def _vector_store() -> FAISSVectorStore:
    """Return the process-wide shared FAISSVectorStore, constructing it once.

    Construction is deferred to first use (never import time) because
    CohereEmbedding.__init__ requires COHERE_API_KEY — building it at import
    would break keyless environments and the offline test suite. The
    threading.Lock guards the one-time init against concurrent first callers.
    """
    global _vs_singleton
    if _vs_singleton is None:
        with _vs_lock:
            if _vs_singleton is None:
                _vs_singleton = FAISSVectorStore()
    return _vs_singleton


@task("embed_personal_memory")
async def embed_personal_memory(user_id: str, topic_name: str, lines: list[str]) -> None:
    """Embeds all lines of a personal memory topic into the vector store."""
    # Offline kill-switch: skip the whole job (and the live Cohere call it
    # makes) when embedding is disabled. The test suite sets this to "0" so the
    # detached embed never fires a network call or writes under data/.
    enabled = os.getenv("TURTLE_PERSONAL_EMBED_ENABLED", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        logger.warning(
            "LOG: embed_personal_memory skipped — TURTLE_PERSONAL_EMBED_ENABLED disabled"
        )
        return

    # Tenant guard: an un-scoped / default write must never land in the shared
    # default/vector dir (cross-tenant collapse). Return BEFORE constructing the
    # store so keyless/offline callers are never forced to build a Cohere client.
    # Normalized comparison so "Default" / " default " can't slip past the
    # guard while the canonical id is still used for storage (Codex P5 #7).
    normalized_uid = (user_id or "").strip()
    if not normalized_uid or normalized_uid.lower() in _SKIP_TENANTS:
        logger.warning(
            "LOG: embed_personal_memory skipped for un-scoped tenant %r (topic=%s)",
            user_id,
            topic_name,
        )
        return

    # Defensive: a bare str payload would iterate CHARACTERS, embedding one doc
    # per character. Normalize to lines.
    if isinstance(lines, str):
        lines = lines.splitlines()

    vs = _vector_store()

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
