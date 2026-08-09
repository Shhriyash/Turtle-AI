"""
Phase 9 — production concurrency hardening.

1. FAISSVectorStore.search/upsert were `async def` bodies that acquired a
   threading.Lock and then did synchronous work including a BLOCKING Cohere HTTP
   embed. Run inline on the event loop this froze every user's turn for the
   duration of the embed; worse, a second coroutine on the same tenant would
   block-acquire the lock on the sole loop thread, deadlocking until the HTTP
   call returned. Both are now offloaded via asyncio.to_thread.

2. Background jobs were unbounded bare create_task — a burst launched unlimited
   concurrent coroutines and starved the turn pipeline. Now semaphore-bounded.

3. The per-turn extraction and reflector tasks were untracked, so asyncio's weak
   reference let them be GC'd mid-flight, silently dropping memory writes.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest


# ── 1. FAISS must not block the event loop ───────────────────────────────────

def test_faiss_search_and_upsert_are_offloaded():
    """The async entry points must hand off to a thread, not run the blocking
    body inline. Assert the sync bodies exist and the async wrappers use them."""
    from core.storage.local.faiss_store import FAISSVectorStore

    assert not inspect.iscoroutinefunction(FAISSVectorStore._search_sync)
    assert not inspect.iscoroutinefunction(FAISSVectorStore._upsert_sync)
    assert inspect.iscoroutinefunction(FAISSVectorStore.search)
    assert inspect.iscoroutinefunction(FAISSVectorStore.upsert)

    for fn in (FAISSVectorStore.search, FAISSVectorStore.upsert):
        src = inspect.getsource(fn)
        assert "to_thread" in src, f"{fn.__name__} must offload to a worker thread"


def test_faiss_search_does_not_stall_the_loop():
    """A slow embed inside FAISS must not prevent other coroutines running.

    Simulates the real failure: the body sleeps synchronously (as a blocking
    HTTP embed would). If it ran on the loop, the concurrent ticker would be
    frozen for the whole duration and record ~0 ticks.
    """
    import time

    from core.storage.local.faiss_store import FAISSVectorStore

    store = FAISSVectorStore.__new__(FAISSVectorStore)

    def _slow_search(user_id, query, k):
        time.sleep(0.30)          # stands in for the blocking Cohere call
        return []

    store._search_sync = _slow_search  # type: ignore[method-assign]

    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(30):
            await asyncio.sleep(0.01)
            ticks += 1

    async def main():
        await asyncio.gather(
            FAISSVectorStore.search(store, "u", "q", 3),
            ticker(),
        )

    asyncio.run(main())
    assert ticks >= 20, (
        f"event loop was stalled during the FAISS call (only {ticks} ticks) — "
        "the blocking body is still running inline"
    )


# ── 2. background jobs are bounded ───────────────────────────────────────────

def test_worker_queue_bounds_concurrency():
    from core import worker

    assert worker.MAX_CONCURRENT_JOBS >= 1

    peak = 0
    live = 0

    async def _job(**kwargs):
        nonlocal peak, live
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1

    async def main():
        worker._job_semaphore = None          # rebind to this loop
        worker._REGISTRY["_test_job"] = _job
        try:
            await asyncio.gather(
                *(worker.queue_service.enqueue("_test_job") for _ in range(40))
            )
            await asyncio.sleep(0.4)
        finally:
            worker._REGISTRY.pop("_test_job", None)

    asyncio.run(main())
    assert peak <= worker.MAX_CONCURRENT_JOBS, (
        f"{peak} jobs ran at once, cap is {worker.MAX_CONCURRENT_JOBS}"
    )
    assert peak > 0, "no jobs ran at all"


# ── 3. critical background tasks are retained ────────────────────────────────

def test_per_turn_extraction_task_is_tracked():
    import apps.turtle_server as ts

    src = inspect.getsource(ts._queue_confirmation_candidates_from_turn)
    assert "track_task" in src, (
        "per-turn memory extraction must be retained — a bare create_task can be "
        "GC'd mid-flight, silently dropping what the user just disclosed"
    )


def test_reflector_task_is_tracked():
    import core.periodic_reflector as pr

    src = inspect.getsource(pr.PeriodicReflector.on_turn)
    assert "track_task" in src, (
        "reflection (Stage B + rolling summary) must be retained"
    )
