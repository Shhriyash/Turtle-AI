"""Phase 5 W1 — background-task layer hardening.

These pin the production fixes for the detached personal-memory embed job and
the worker's fire-and-forget task handling:

  (a) the embed job refuses to run for the un-scoped default/empty tenant, and
      does NOT even construct the vector store to do so (the store would embed
      into the SHARED default/vector dir — cross-tenant collapse);
  (b) the offline kill-switch env var short-circuits the whole job;
  (c) write_topic skips the enqueue for the default tenant but enqueues for a
      real usr_* tenant;
  (d) a bare str content is normalized to a list of lines before the enqueue
      (a str payload would embed one document per character);
  (e) the worker retains detached tasks and its done-callback drains _TASKS and
      logs failures (both the wrapper path and the bypassed-wrapper path);
  (f) the shared vector-store singleton is constructed once and reused.

All tests are fully offline: the embed job never reaches Cohere, stores are
built under tmp dirs, and core.paths.PERSONAL_MEMORY_DIR is monkeypatched so
nothing is written under the repo's data/ tree.
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

import core.background_tasks as bg
import core.paths as core_paths
import core.worker as worker
from core.personal_memory_store import PersonalMemoryStore
from core.worker import LocalWorkerQueue, task, track_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(base, user_id: str = "default") -> PersonalMemoryStore:
    return PersonalMemoryStore(
        user_id=user_id,
        base_dir=base,
        index_path=base / "MEMORY.md",
        logs_dir=base / "logs",
        topic_paths={
            "identity": base / "identity.md",
            "preferences": base / "preferences.md",
        },
    )


class _BoomStore:
    """Stand-in for FAISSVectorStore that fails loudly if ever constructed."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("FAISSVectorStore must not be constructed here")


# ---------------------------------------------------------------------------
# (a) embed job skips default/empty tenant WITHOUT constructing the store
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("uid", ["", "default"])
def test_embed_skips_unscoped_tenant_without_constructing_store(monkeypatch, uid):
    # Kill-switch ON (embedding enabled) so ONLY the tenant guard can stop it.
    monkeypatch.setenv("TURTLE_PERSONAL_EMBED_ENABLED", "1")
    monkeypatch.setattr(bg, "FAISSVectorStore", _BoomStore)
    monkeypatch.setattr(bg, "_vs_singleton", None)

    # If the guard fails, _BoomStore raises and the test errors.
    asyncio.run(
        bg.embed_personal_memory(user_id=uid, topic_name="identity", lines=["- Name: X"])
    )


# ---------------------------------------------------------------------------
# (b) offline kill-switch env respected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["0", "false", "False", "no", "off"])
def test_kill_switch_skips_embed(monkeypatch, value):
    monkeypatch.setenv("TURTLE_PERSONAL_EMBED_ENABLED", value)
    monkeypatch.setattr(bg, "FAISSVectorStore", _BoomStore)
    monkeypatch.setattr(bg, "_vs_singleton", None)

    # Even a real tenant is skipped when disabled — store never constructed.
    asyncio.run(
        bg.embed_personal_memory(
            user_id="usr_killswitch", topic_name="identity", lines=["- Name: X"]
        )
    )


def test_kill_switch_enabled_reaches_store(monkeypatch):
    """Sanity: with the switch ON and a real tenant, the guard falls through to
    the store (proving the earlier skips are the env/tenant guards, not a
    blanket no-op)."""
    monkeypatch.setenv("TURTLE_PERSONAL_EMBED_ENABLED", "1")

    constructed = {"n": 0}
    upserts: list[dict] = []

    class _FakeVS:
        def __init__(self, *a, **k):
            constructed["n"] += 1

        async def upsert(self, **kwargs):
            upserts.append(kwargs)

    monkeypatch.setattr(bg, "FAISSVectorStore", _FakeVS)
    monkeypatch.setattr(bg, "_vs_singleton", None)

    asyncio.run(
        bg.embed_personal_memory(
            user_id="usr_real", topic_name="identity", lines=["- Name: X", "  ", "- City: Y"]
        )
    )

    assert constructed["n"] == 1
    assert [u["text"] for u in upserts] == ["Name: X", "City: Y"]
    assert all(u["user_id"] == "usr_real" for u in upserts)


# ---------------------------------------------------------------------------
# (c) write_topic gates the enqueue on the tenant
# ---------------------------------------------------------------------------

def test_write_topic_enqueue_gated_by_tenant(monkeypatch, tmp_path):
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", tmp_path / "pm")
    enq = AsyncMock(return_value="job_x")
    monkeypatch.setattr("core.personal_memory_store.queue_service.enqueue", enq)

    async def run():
        # default tenant -> enqueue skipped
        d_base = tmp_path / "d"
        d_base.mkdir()
        default_store = _make_store(d_base, user_id="default")
        default_store.write_topic(
            "preferences", ["Prefers concise responses"], {"confidence": "confirmed"}
        )
        await asyncio.sleep(0)
        assert enq.call_count == 0

        # real tenant -> enqueue fires exactly once, scoped to that tenant
        u_base = tmp_path / "u"
        u_base.mkdir()
        usr_store = _make_store(u_base, user_id="usr_x")
        usr_store.write_topic(
            "preferences", ["Prefers concise responses"], {"confidence": "confirmed"}
        )
        await asyncio.sleep(0)
        assert enq.call_count == 1
        _, kwargs = enq.call_args
        assert kwargs["user_id"] == "usr_x"
        assert kwargs["topic_name"] == "preferences"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# (d) str content normalized to lines before the enqueue
# ---------------------------------------------------------------------------

def test_str_content_normalized_to_lines_before_enqueue(monkeypatch, tmp_path):
    monkeypatch.setattr(core_paths, "PERSONAL_MEMORY_DIR", tmp_path / "pm")
    enq = AsyncMock(return_value="job_x")
    monkeypatch.setattr("core.personal_memory_store.queue_service.enqueue", enq)

    async def run():
        base = tmp_path / "u"
        base.mkdir()
        store = _make_store(base, user_id="usr_y")
        store.write_topic("preferences", "line one\nline two", {"confidence": "confirmed"})
        await asyncio.sleep(0)

        assert enq.call_count == 1
        _, kwargs = enq.call_args
        assert not isinstance(kwargs["lines"], str)
        assert kwargs["lines"] == ["line one", "line two"]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# (e) worker retention + done-callback drain/logging
# ---------------------------------------------------------------------------

def test_worker_task_retention_and_failure_logging(caplog):
    async def run():
        @task("_p5_raises_job")
        async def _p5_raises_job(**_):
            raise RuntimeError("boom-enqueue")

        q = LocalWorkerQueue()

        # --- wrapper path: enqueue a job that raises -------------------------
        with caplog.at_level(logging.ERROR, logger="core.worker"):
            job_id = await q.enqueue("_p5_raises_job")
            ours = [t for t in worker._TASKS if t.get_name() == job_id]
            assert len(ours) == 1  # task retained while in flight
            await asyncio.gather(*ours, return_exceptions=True)
            for _ in range(3):
                await asyncio.sleep(0)  # let the done-callback run

        # done-callback drained the retained task ...
        assert all(t.get_name() != job_id for t in worker._TASKS)
        # ... and the wrapper logged the failure.
        assert any("failed" in r.getMessage() for r in caplog.records)

        # --- bypassed-wrapper path: track_task on a raising coroutine --------
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="core.worker"):
            async def _direct_raise():
                raise ValueError("boom-direct")

            t = asyncio.create_task(_direct_raise(), name="_p5_direct")
            track_task(t)
            assert t in worker._TASKS
            await asyncio.gather(t, return_exceptions=True)
            for _ in range(3):
                await asyncio.sleep(0)

        assert t not in worker._TASKS
        # the done-callback itself logged the otherwise-unobserved exception.
        assert any("raised" in r.getMessage() for r in caplog.records)

    asyncio.run(run())


# ---------------------------------------------------------------------------
# (f) shared singleton is constructed once and reused
# ---------------------------------------------------------------------------

def test_shared_vector_store_singleton_reused(monkeypatch):
    constructed: list[object] = []

    class _FakeVS:
        def __init__(self, *a, **k):
            constructed.append(self)

    monkeypatch.setattr(bg, "FAISSVectorStore", _FakeVS)
    monkeypatch.setattr(bg, "_vs_singleton", None)

    first = bg._vector_store()
    second = bg._vector_store()

    assert first is second
    assert len(constructed) == 1
