"""
core/storage/factory.py
------------------------
Single seam that selects local vs. cloud storage backends off
settings.is_cloud. Callers that used to import a concrete local class
directly (SQLiteSessionStore, get_faiss_vector_store) should import the
matching get_*() here instead, so TURTLE_DEPLOY=cloud repoints them without
touching call sites again in a future backend swap.
"""
from __future__ import annotations

from typing import Any, Dict

from core.config import settings
from core.storage import SessionStoreProtocol, VectorStore

# Process-wide singleton cache for the cloud vector store, mirroring
# core.storage.local.faiss_store's own singleton registry — avoids
# reconstructing the embedder client and re-issuing CREATE TABLE IF NOT
# EXISTS on every one of the 3 call sites' invocations.
_CLOUD_VECTOR_STORE_SINGLETONS: Dict[int, Any] = {}


def get_session_store_backend() -> SessionStoreProtocol:
    """SQLite locally, Postgres in cloud mode. See core/session_store.py's
    SessionStore, which takes this as its `backend`."""
    if settings.is_cloud:
        from core.storage.cloud.postgres_store import PostgresSessionStore

        return PostgresSessionStore()
    from core.storage.local.sqlite_store import SQLiteSessionStore

    return SQLiteSessionStore()


def get_vector_store(embedding_dimension: int = 1024) -> VectorStore:
    """FAISS (process-singleton) locally, pgvector (shared Postgres) in cloud
    mode. Replaces direct calls to
    core.storage.local.faiss_store.get_faiss_vector_store at all 3 call sites
    (apps/turtle_server.py x2, core/background_tasks.py) — same signature and
    return-type Protocol, so the migration to cloud storage doesn't add code
    at those call sites, just swaps the import.
    """
    if settings.is_cloud:
        store = _CLOUD_VECTOR_STORE_SINGLETONS.get(embedding_dimension)
        if store is None:
            from core.storage.cloud.pgvector_store import PgVectorStore

            store = PgVectorStore(embedding_dimension=embedding_dimension)
            _CLOUD_VECTOR_STORE_SINGLETONS[embedding_dimension] = store
        return store
    from core.storage.local.faiss_store import get_faiss_vector_store

    return get_faiss_vector_store(embedding_dimension=embedding_dimension)


# Process-wide singletons for the rate-limiter/gate classes (BOTH local and
# cloud paths cache one instance — note_prompt/try_consume_answer and
# check_and_record only work as a pair because the same object's state is
# read back across separate calls; a fresh instance per call would silently
# lose every outstanding prompt / rate-limit counter between calls).
_rate_limiter: Any = None
_channel_gate_buffer: Any = None


def get_ws_rate_limiter() -> Any:
    """core.guardrails.ws_rate_limiter locally, Redis-backed in cloud mode.
    Deliberately NOT branched inside core/guardrails.py itself: the Redis
    implementation needs WebSocketRateLimitExceeded from that same module, so
    branching there would be a circular import — this factory sits above
    both and only one of the two ever gets imported at runtime.
    """
    global _rate_limiter
    if _rate_limiter is not None:
        return _rate_limiter
    if settings.is_cloud:
        from core.storage.cloud.redis_backends import RedisWebSocketRateLimiter

        _rate_limiter = RedisWebSocketRateLimiter()
    else:
        from core.guardrails import ws_rate_limiter

        _rate_limiter = ws_rate_limiter
    return _rate_limiter


def get_link_code_store() -> Any:
    """core.account_linking.LinkCodeStore(identity_manager.db_path) locally,
    PostgresLinkCodeStore in cloud mode.

    Not just a Postgres-vs-SQLite swap: identity_manager.db_path (what the
    local class's constructor needs) only exists on the LOCAL
    core.identity.IdentityManager — in cloud mode identity_manager is a
    PostgresIdentityManager with no db_path attribute at all, so the two
    call sites in apps/turtle_server.py that used to build
    LinkCodeStore(identity_manager.db_path) directly would raise
    AttributeError outright. This factory is what lets both call sites stay
    identical regardless of mode — both store types expose the same
    issue/peek/reserve/release_reservation/consume/purge_expired surface.
    """
    if settings.is_cloud:
        from core.storage.cloud.account_linking_store import PostgresLinkCodeStore

        return PostgresLinkCodeStore()
    from core.account_linking import LinkCodeStore
    from core.identity import identity_manager

    return LinkCodeStore(identity_manager.db_path)


def get_confirmation_state_backend(user_id: str) -> Any:
    """PostgresConfirmationState in cloud mode, None locally (ConfirmationGate
    falls back to its own local-JSON-file backend when no explicit backend is
    passed — see core/confirmation_gate.py's __init__). Not cached as a
    singleton like the rate limiter/gate buffer above: this is a per-user
    object with no shared cross-call state of its own (each load/save is a
    fresh Postgres round trip), so a new instance per call is free.
    """
    if settings.is_cloud:
        from core.storage.cloud.confirmation_state_store import PostgresConfirmationState

        return PostgresConfirmationState(user_id)
    return None


def get_channel_gate_buffer() -> Any:
    """core.channel_gate.ChannelGateBuffer locally, Redis-backed in cloud
    mode. Same circular-import rationale as get_ws_rate_limiter(): the Redis
    class needs parse_gate_answer/DEFAULT_TTL_SECONDS from core.channel_gate,
    so the branch lives here rather than in that module.
    """
    global _channel_gate_buffer
    if _channel_gate_buffer is not None:
        return _channel_gate_buffer
    if settings.is_cloud:
        from core.storage.cloud.redis_backends import RedisChannelGateBuffer

        _channel_gate_buffer = RedisChannelGateBuffer()
    else:
        from core.channel_gate import ChannelGateBuffer

        _channel_gate_buffer = ChannelGateBuffer()
    return _channel_gate_buffer
