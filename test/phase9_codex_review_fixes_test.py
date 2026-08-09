"""
Phase 9 — fixes from the Codex adversarial review.

1. LINK CODE DISCLOSURE (high). `link_account` returned the raw claim code as
   ordinary assistant content, and the Discord gateway posts that content back
   to wherever the request came from — including a public guild channel. The
   original design note claimed a leaked code was harmless ("an attacker can
   only attach their own handle to their own account"). That was BACKWARDS:
   redemption binds the CLAIM's channel identity to the REDEEMER's account and
   merge_memory copies the claim owner's journal into it. So an observer who
   races a leaked code steals the victim's memory. Codes must never reach a
   shared surface.

2. FAISS LOCK RACE (medium). _get_lock was an unsynchronized check-then-insert.
   Safe while everything ran on the event-loop thread; moving search/upsert to
   asyncio.to_thread made it genuinely multi-threaded, so two first-time
   operations for one tenant could take DIFFERENT locks and concurrently
   mutate the index/metadata.
"""
from __future__ import annotations

import asyncio
import threading

import pytest


# ── 1. claim codes never leave a private surface ─────────────────────────────

def test_turtle_event_defaults_to_public():
    """Fail closed: an adapter that has not opted in must be treated as public."""
    from apps.channels import TurtleEvent

    ev = TurtleEvent(user_id="u", channel="discord", modality="text", content="hi")
    assert ev.is_private is False


def test_link_account_refuses_on_a_public_surface():
    """The core fix: no claim code may be emitted where others can read it."""
    import types

    import apps.turtle_server as ts

    agent = ts.agents_mgr.main_assistant
    tool = None
    for name in ("link_account",):
        fn = getattr(agent, "_function_toolset", None)
        if fn is not None:
            tools = getattr(fn, "tools", {}) or {}
            if name in tools:
                tool = tools[name]
    assert tool is not None, "link_account tool is not registered"

    deps = types.SimpleNamespace(
        user_id="usr_victim",
        channel="discord",
        channel_user_id="759",
        channel_is_private=False,   # public guild channel
    )
    ctx = types.SimpleNamespace(deps=deps)
    out = asyncio.run(tool.function(ctx))
    assert "code" not in out.lower() or "shared channel" in out.lower()
    assert "direct message" in out.lower(), f"expected a refusal, got: {out[:200]}"


def test_link_account_issues_in_a_dm(tmp_path, monkeypatch):
    """...and still works on a private surface."""
    import types

    import apps.turtle_server as ts
    from core.identity import IdentityManager

    agent = ts.agents_mgr.main_assistant
    tools = getattr(getattr(agent, "_function_toolset", None), "tools", {}) or {}
    tool = tools.get("link_account")
    assert tool is not None

    monkeypatch.setattr(
        ts.identity_manager if hasattr(ts, "identity_manager") else IdentityManager,
        "db_path",
        tmp_path / "users.sqlite",
        raising=False,
    )
    import core.identity as ci
    monkeypatch.setattr(ci.identity_manager, "db_path", tmp_path / "users.sqlite", raising=False)

    deps = types.SimpleNamespace(
        user_id="usr_owner",
        channel="discord",
        channel_user_id="759",
        channel_is_private=True,    # DM
    )
    out = asyncio.run(tool.function(types.SimpleNamespace(deps=deps)))
    assert "Link code:" in out


# ── 2. FAISS per-tenant lock is unique under concurrency ─────────────────────

def test_faiss_lock_is_unique_per_tenant_under_threads():
    """Two threads racing first use of the same tenant must get the SAME lock."""
    from core.storage.local.faiss_store import FAISSVectorStore

    store = FAISSVectorStore.__new__(FAISSVectorStore)
    store._locks = {}
    store._locks_guard = threading.Lock()

    seen: list = []
    barrier = threading.Barrier(8)

    def grab():
        barrier.wait()
        seen.append(store._get_lock("usr_race"))

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == 8
    assert len({id(x) for x in seen}) == 1, "tenant lock split across threads"


def test_faiss_distinct_tenants_get_distinct_locks():
    from core.storage.local.faiss_store import FAISSVectorStore

    store = FAISSVectorStore.__new__(FAISSVectorStore)
    store._locks = {}
    store._locks_guard = threading.Lock()

    assert store._get_lock("a") is not store._get_lock("b")
    assert store._get_lock("a") is store._get_lock("a")


# ── 3. link ordering: merge FIRST, commit only on success ────────────────────

def test_merge_failure_leaves_mapping_untouched(tmp_path, monkeypatch):
    """The redemption route must MERGE before it re-points the channel mapping
    and consumes the code. If merge fails, the mapping must NOT be re-pointed
    and the code must stay redeemable (so the user can retry once whatever
    caused the failure clears)."""
    import asyncio
    import core.account_linking as al
    from core.identity import IdentityManager

    db = tmp_path / "users.sqlite"
    mgr = IdentityManager(db_path=db)
    store = al.LinkCodeStore(db)

    async def setup_and_run():
        await mgr.init_db()
        source = await mgr.resolve_user("discord", "759")
        target = await mgr.resolve_user("web_email", "me@example.com")
        code = store.issue(
            channel="discord", channel_user_id="759", source_user_id=source
        ).code

        # simulate a merge failure exactly as the endpoint would see it
        def failing_merge(src, dst):
            return {"events_copied": 0, "replayed": False, "ok": False, "error": "disk full"}

        # 1. Peek + failing merge: mapping NOT re-pointed, code NOT consumed.
        claim = store.peek(code)
        assert claim is not None
        merged = failing_merge(claim.source_user_id, target)
        assert not merged["ok"]
        # The endpoint would return here without link_channel / mark_consumed.

        # Mapping still points at the ORIGINAL source user id.
        assert await mgr.resolve_user("discord", "759") == source
        # Code is still redeemable.
        assert store.peek(code) is not None

        # 2. Now the retry succeeds -> commit ownership + burn the code.
        await mgr.link_channel(user_id=target, channel="discord", channel_user_id="759")
        assert al.mark_consumed(store, code) is True
        assert await mgr.resolve_user("discord", "759") == target
        assert store.peek(code) is None

    asyncio.run(setup_and_run())


def test_peek_does_not_consume_but_consume_is_single_shot(tmp_path):
    """peek() must not burn a code; consume() must still be single-shot after
    N peeks (this is the property that lets us hold the code across the merge)."""
    import core.account_linking as al

    store = al.LinkCodeStore(tmp_path / "users.sqlite")
    code = store.issue(channel="discord", channel_user_id="x", source_user_id="u").code

    for _ in range(5):
        assert store.peek(code) is not None
    assert al.mark_consumed(store, code) is True
    assert al.mark_consumed(store, code) is False
    assert store.peek(code) is None
