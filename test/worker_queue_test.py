"""
core/worker.py local queue + @task registry tests.

Migrated from test_tier2_verification.py (TestG3WorkerQueue) — the four pure
core.worker behaviors. The embed_personal_memory registration test is
intentionally NOT migrated here: it imports core.background_tasks to trigger
registration, and that coverage lives with the memory subsystem, not the queue
primitives.

Covers: module import, @task registration into _REGISTRY, enqueue returning a
job_ id, and unknown-job ValueError.
"""
from __future__ import annotations


class TestG3WorkerQueue:
    """LocalWorkerQueue uses asyncio.create_task; task decorator registers jobs."""

    def test_worker_queue_importable(self):
        from core.worker import LocalWorkerQueue, queue_service, task
        assert LocalWorkerQueue is not None
        assert queue_service is not None
        assert callable(task)

    def test_task_decorator_registers_job(self):
        from core.worker import _REGISTRY, task

        @task("_test_job_xyz")
        async def _test_job_xyz():
            pass

        assert "_test_job_xyz" in _REGISTRY

    def test_local_queue_enqueue_returns_job_id(self):
        import asyncio
        from core.worker import LocalWorkerQueue, task

        @task("_test_noop")
        async def _test_noop(**_):
            pass

        async def run():
            q = LocalWorkerQueue()
            job_id = await q.enqueue("_test_noop")
            assert job_id.startswith("job_")
            return job_id

        asyncio.run(run())

    def test_unknown_job_raises(self):
        import asyncio, pytest
        from core.worker import LocalWorkerQueue

        async def run():
            q = LocalWorkerQueue()
            with pytest.raises(ValueError, match="not registered"):
                await q.enqueue("nonexistent_job_abc")

        asyncio.run(run())
