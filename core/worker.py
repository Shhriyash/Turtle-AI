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


def dispatch_embed_personal_memory_job(user_id: str, topic_name: str, lines: list[str]) -> None:
    """Fire the embed_personal_memory job for a topic write, without making
    core.personal_memory_store.PersonalMemoryStore.write_topic() (a plain
    sync method, called from ~5 places including inside a journal replay
    that is itself called synchronously) become async just to await it.

    Local mode: an in-process detached asyncio task running the job to
    completion — a genuinely long-lived process, proven fine there.

    Cloud mode: found in a post-migration audit to carry the SAME risk
    already fixed for Discord's deferred interaction processing — a
    detached asyncio.create_task has no confirmed guarantee of surviving
    past this invocation's response on Vercel's Python runtime (whose
    documented post-response background-work API, waitUntil()/after(), is
    JS-only). The embed job itself does a live Cohere HTTP call plus a
    pgvector upsert — exactly the kind of multi-step work that could get cut
    off mid-flight. So cloud mode's detached task does NOT run the job
    directly; it self-invokes POST /internal/embed-personal-memory
    (apps/cron_tick_routes.py) as an independent request instead, the same
    pattern apps/channels/discord.py's _kick_off_deferred_processing uses —
    the detached task's own job shrinks to "reliably kick off a second,
    independently-completing invocation", bounded to how long it takes to
    send one HTTP request rather than an embed+upsert round trip, and the
    second invocation runs to completion with its own normal timeout budget
    with no ambiguity.

    Best-effort either way: no running event loop (e.g. an offline script)
    is a silent no-op, matching the original inline behavior in write_topic.

    Residual trade-off worth naming: track_task's user_id tag (below) exists
    so drain_user_tasks(user_id) can wait for in-flight embeds before an
    account-link merge snapshots a source journal. In cloud mode, what gets
    tracked here is only the SELF-INVOKE task (milliseconds — send the
    request and return), not the actual embed+upsert running on the separate
    invocation it triggered, which this process has no handle on. A merge
    could therefore still race a just-triggered cloud embed. Narrower than
    the bug this fixes, though: the journal event (the source of truth) is
    already durably in Postgres either way, and the vector store is a
    rebuildable search INDEX over it — the same posture already given to the
    SQLite FTS5 read-model cache elsewhere in this migration — so the worst
    case is a fact temporarily missing from RAG search until the next write
    to that topic re-triggers its embed, not a fact lost.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    if settings.is_cloud:
        task_obj = loop.create_task(
            _self_invoke_embed_job(user_id=user_id, topic_name=topic_name, lines=lines)
        )
    else:
        task_obj = loop.create_task(
            queue_service.enqueue(
                "embed_personal_memory", user_id=user_id, topic_name=topic_name, lines=lines
            )
        )
    track_task(task_obj, user_id=user_id)


async def _self_invoke_embed_job(user_id: str, topic_name: str, lines: list[str]) -> None:
    """Cloud-mode body of dispatch_embed_personal_memory_job — see its
    docstring. Mirrors apps/channels/discord.py's
    _kick_off_deferred_processing: a short read timeout paired with a
    generous connect/write timeout guarantees the request was fully SENT
    (so the target invocation is genuinely dispatched) without waiting for
    it to finish.

    WP 1.B / S-7.3: authenticates with INTERNAL_JOB_SECRET (not the retired
    CRON_SHARED_SECRET), and sends only an opaque job id — the real payload
    (including user_id) is stashed in Redis first
    (core.internal_auth.store_job_payload) so it's never trusted from the
    wire by apps/cron_tick_routes.py's endpoint.
    """
    import json

    import httpx

    from core.internal_auth import sign_request, store_job_payload
    from core.storage.cloud import CloudBackendUnavailable

    secret = settings.internal_job_secret.get_secret_value() if settings.internal_job_secret else ""
    if not secret:
        # No internal-automation secret configured — run the job in this
        # same detached task rather than silently dropping the embed. Less
        # robust on serverless, but strictly no worse than before this fix,
        # and one env var away from the safe path.
        logger.warning(
            "INTERNAL_JOB_SECRET unset — embedding %s/%s in-process (detached task)",
            user_id, topic_name,
        )
        func = _REGISTRY.get("embed_personal_memory")
        if func is not None:
            await func(user_id=user_id, topic_name=topic_name, lines=lines)
        return

    payload = {"user_id": user_id, "topic_name": topic_name, "lines": lines}
    try:
        job_id = await store_job_payload(payload)
        body = json.dumps({"job_id": job_id}).encode("utf-8")
        envelope = sign_request(secret, body)
    except CloudBackendUnavailable as exc:
        logger.warning(
            "job store unavailable, embedding %s/%s in-process: %s",
            user_id, topic_name, exc,
        )
        func = _REGISTRY.get("embed_personal_memory")
        if func is not None:
            await func(user_id=user_id, topic_name=topic_name, lines=lines)
        return

    url = f"{settings.public_base_url.rstrip('/')}/internal/embed-personal-memory"
    headers = {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
        **envelope.headers(),
    }
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                url,
                content=body,
                headers=headers,
                timeout=httpx.Timeout(connect=5.0, read=0.1, write=5.0, pool=5.0),
            )
    except httpx.ReadTimeout:
        pass  # Expected: the request was sent; we deliberately don't await its reply.
    except Exception as exc:
        logger.error(
            "self-invoke for embed_personal_memory failed user=%s topic=%s: %s",
            user_id, topic_name, exc,
        )
        func = _REGISTRY.get("embed_personal_memory")
        if func is not None:
            await func(user_id=user_id, topic_name=topic_name, lines=lines)
