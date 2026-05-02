"""
core/worker.py
--------------
G3: Worker queue facade and job registry.
Local mode: uses asyncio.create_task. Cloud mode: will use Arq + Redis.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Awaitable, Callable

from core.config import settings
from core.storage import Queue

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Callable[..., Awaitable[Any]]] = {}


def task(name: str) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorator to register a background task function."""
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        _REGISTRY[name] = func
        return func
    return decorator


class LocalWorkerQueue(Queue):
    """Local mode queue using asyncio.create_task."""
    async def enqueue(self, job_name: str, **kwargs: Any) -> str:
        func = _REGISTRY.get(job_name)
        if not func:
            raise ValueError(f"Job {job_name} not registered")

        job_id = f"job_{uuid.uuid4().hex[:12]}"

        async def _wrapper() -> None:
            try:
                await func(**kwargs)
            except Exception as e:
                logger.error(f"Background job '{job_name}' failed: {e}", exc_info=True)

        asyncio.create_task(_wrapper(), name=job_id)
        return job_id


# Instantiate globally
if settings.is_cloud:
    # In cloud mode, this would instantiate an Arq wrapper.
    # Fallback to local for now until Redis integration is active.
    queue_service: Queue = LocalWorkerQueue()
else:
    queue_service: Queue = LocalWorkerQueue()
