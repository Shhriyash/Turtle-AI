"""
core/storage/__init__.py
------------------------
G1: Storage abstraction layer.
Defines protocols for FactStore, VectorStore, SessionStore, Queue, BlobStore, and TraceSink.
"""
from __future__ import annotations

from typing import Protocol, Any, Optional
from pydantic import BaseModel

class Session(BaseModel):
    session_id: str
    data: dict[str, Any]

class Fact(BaseModel):
    id: str
    topic: str
    key: str
    value: str
    metadata: dict[str, Any] = {}

class Hit(BaseModel):
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = {}

class FactStore(Protocol):
    async def upsert_fact(self, user_id: str, topic: str, key: str, value: str) -> None: ...
    async def get_facts(self, user_id: str, topic: str) -> list[Fact]: ...

class VectorStore(Protocol):
    async def upsert(self, user_id: str, doc_id: str, text: str, metadata: dict[str, Any]) -> None: ...
    async def search(self, user_id: str, query: str, k: int) -> list[Hit]: ...

class SessionStoreProtocol(Protocol):
    """Minimal session persistence surface: only ``get``/``put`` are required so
    lightweight/custom backends stay trivial to implement.

    The SessionStore wrapper additionally *duck-types* an optional extended
    surface — ``init_db()``, ``list_sessions(status_filter, user_id)`` and
    ``delete(session_id)`` — via ``hasattr``/``inspect`` and degrades gracefully
    when a backend omits it (e.g. it passes ``user_id`` only when the signature
    accepts it, and re-filters tenancy in Python regardless). The bundled
    ``SQLiteSessionStore`` implements the full surface with a real, indexed
    ``user_id`` column.
    """
    async def get(self, session_id: str) -> Optional[Session]: ...
    async def put(self, session: Session) -> None: ...

class Queue(Protocol):
    async def enqueue(self, job_name: str, **kwargs: Any) -> str: ...

class BlobStore(Protocol):
    async def put(self, key: str, data: bytes) -> str: ...

class AbstractContextManager(Protocol):
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

class TraceSink(Protocol):
    def span(self, name: str, **attrs: Any) -> AbstractContextManager: ...
