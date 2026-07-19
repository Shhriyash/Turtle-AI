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

# Strong references to in-flight fire-and-forget tasks. CPython may garbage
# collect a task whose only reference is the event loop's weak set, cancelling
# it mid-flight. Retaining the task here until it completes closes that hazard.
_TASKS: set[asyncio.Task[Any]] = set()


def track_task(task_obj: asyncio.Task[Any]) -> None:
    """Retain a strong reference to a detached task and observe its failures.

    Adds ``task_obj`` to the module-level ``_TASKS`` set so it cannot be GC'd
    while running, and attaches a done-callback that discards it on completion
    and logs any exception it raised. This is the safety net for tasks whose
    coroutine does NOT funnel through ``_wrapper`` (e.g. the outer
    ``create_task(queue_service.enqueue(...))`` in personal_memory_store) — the
    wrapper's own catch never sees those, so failures would otherwise vanish.
    """
    _TASKS.add(task_obj)

    def _on_done(t: asyncio.Task[Any]) -> None:
        _TASKS.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.error(
                f"Tracked background task '{t.get_name()}' raised: {exc}",
                exc_info=exc,
            )

    task_obj.add_done_callback(_on_done)


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

        task_obj = asyncio.create_task(_wrapper(), name=job_id)
        track_task(task_obj)
        return job_id


# Instantiate globally
if settings.is_cloud:
    # In cloud mode, this would instantiate an Arq wrapper.
    # Fallback to local for now until Redis integration is active.
    queue_service: Queue = LocalWorkerQueue()
else:
    queue_service: Queue = LocalWorkerQueue()
