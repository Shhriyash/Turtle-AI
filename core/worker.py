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

# user_id -> set of live tasks operating on that user's data. Populated only by
# callers who pass user_id= to track_task. Enables drain_user_tasks() so account
# linking can wait for the source's in-flight writes to finish BEFORE snapshot,
# closing the "detached writer escapes the source lock" race Codex flagged.
_TASKS_BY_USER: dict[str, set[asyncio.Task[Any]]] = {}


def track_task(task_obj: asyncio.Task[Any], user_id: str | None = None) -> None:
    """Retain a strong reference to a detached task and observe its failures.

    Adds ``task_obj`` to the module-level ``_TASKS`` set so it cannot be GC'd
    while running, and attaches a done-callback that discards it on completion
    and logs any exception it raised. This is the safety net for tasks whose
    coroutine does NOT funnel through ``_wrapper`` (e.g. the outer
    ``create_task(queue_service.enqueue(...))`` in personal_memory_store) — the
    wrapper's own catch never sees those, so failures would otherwise vanish.

    When ``user_id`` is supplied, the task is ALSO indexed under that user so
    ``drain_user_tasks(user_id)`` can wait for it before a destructive
    operation on that user's data (e.g. account-link merge).
    """
    _TASKS.add(task_obj)
    if user_id:
        _TASKS_BY_USER.setdefault(user_id, set()).add(task_obj)

    def _on_done(t: asyncio.Task[Any]) -> None:
        _TASKS.discard(t)
        if user_id:
            bucket = _TASKS_BY_USER.get(user_id)
            if bucket is not None:
                bucket.discard(t)
                if not bucket:
                    _TASKS_BY_USER.pop(user_id, None)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.error(
                f"Tracked background task '{t.get_name()}' raised: {exc}",
                exc_info=exc,
            )

    task_obj.add_done_callback(_on_done)


async def drain_user_tasks(user_id: str, timeout: float = 5.0) -> int:
    """Wait for every tracked task operating on ``user_id`` to finish.

    Returns the number of tasks awaited. Used before account-link merge so an
    in-flight extraction or reflection for the SOURCE account can't append into
    a now-unreachable journal after the mapping is re-pointed. Bounded by
    ``timeout`` so a hung task can't stall linking indefinitely.

    Snapshots the task set once — new tasks scheduled AFTER the drain begins
    are not awaited (the caller is expected to also block new turns for the
    same user before calling this).
    """
    tasks = list(_TASKS_BY_USER.get(user_id, set()))
    if not tasks:
        return 0
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            f"drain_user_tasks({user_id!r}): timed out waiting for "
            f"{len(tasks)} task(s) after {timeout}s"
        )
    return len(tasks)


def task(name: str) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorator to register a background task function."""
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        _REGISTRY[name] = func
        return func
    return decorator


# Cap on background jobs executing at once. Without it, enqueue() was a bare
# create_task: a burst (e.g. an embed job per journal event) launched unbounded
# coroutines, each doing blocking-ish work, and starved the turn pipeline.
# Excess jobs queue on the semaphore instead of all running at once.
MAX_CONCURRENT_JOBS = 8
_job_semaphore: asyncio.Semaphore | None = None


def _get_job_semaphore() -> asyncio.Semaphore:
    # Created lazily: a Semaphore binds to the running loop, and this module is
    # imported long before the app loop exists (and re-used across test loops).
    global _job_semaphore
    if _job_semaphore is None:
        _job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    return _job_semaphore


class LocalWorkerQueue(Queue):
    """Local mode queue using asyncio.create_task, bounded by a semaphore."""
    async def enqueue(self, job_name: str, **kwargs: Any) -> str:
        func = _REGISTRY.get(job_name)
        if not func:
            raise ValueError(f"Job {job_name} not registered")

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        semaphore = _get_job_semaphore()

        # Any per-user job MUST be drainable, or account-link merge will snapshot
        # a source journal while its embed/etc is still writing. Job payloads for
        # per-tenant work already carry user_id in kwargs (embed_personal_memory,
        # etc.); pick it up so track_task can index this job under that user.
        job_user_id = str(kwargs.get("user_id", "") or "") or None

        async def _wrapper() -> None:
            try:
                async with semaphore:
                    await func(**kwargs)
            except Exception as e:
                logger.error(f"Background job '{job_name}' failed: {e}", exc_info=True)

        task_obj = asyncio.create_task(_wrapper(), name=job_id)
        track_task(task_obj, user_id=job_user_id)
        return job_id


# Instantiate globally
if settings.is_cloud:
    # In cloud mode, this would instantiate an Arq wrapper.
    # Fallback to local for now until Redis integration is active.
    queue_service: Queue = LocalWorkerQueue()
else:
    queue_service: Queue = LocalWorkerQueue()
