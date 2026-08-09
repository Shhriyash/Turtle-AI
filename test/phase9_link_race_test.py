"""
Phase 9 — the two race findings from the Codex verification pass.

  [medium] Two-target race: two authenticated redeemers with different target
    user_ids both passed peek, both merged the source into their own account,
    then last-writer-wins the mapping — memory disclosed into two accounts,
    "single-use" was a lie because the loser observed mark_consumed=False and
    the endpoint only logged it and still returned 200.

  [medium] Detached writers escape the source lock: per-turn extraction and
    reflector run as background tasks that continue writing through the old
    source SharedState after the handler releases the lock; a redemption can
    snapshot / merge / repoint / release and the delayed task then appends to
    the now-unreachable source journal, stranding those writes with no retry.
"""
from __future__ import annotations

import asyncio

import pytest

import core.account_linking as al


# ── two-target reservation is atomic ─────────────────────────────────────────

def test_reserve_blocks_a_different_target(tmp_path):
    store = al.LinkCodeStore(tmp_path / "users.sqlite")
    code = store.issue(channel="discord", channel_user_id="759", source_user_id="usr_src").code

    status_a, claim_a = store.reserve(code, "usr_target_A")
    assert status_a == "ok" and claim_a is not None

    # A different target hitting the same live code MUST be rejected — this is
    # what closed the race Codex found (two 200s, memory into two accounts).
    status_b, claim_b = store.reserve(code, "usr_target_B")
    assert status_b == "locked"
    assert claim_b is None


def test_reserve_same_target_is_idempotent(tmp_path):
    """The same target retrying inside the reservation window MUST get through
    (transient failure retry), not be locked out by its own prior reservation."""
    store = al.LinkCodeStore(tmp_path / "users.sqlite")
    code = store.issue(channel="discord", channel_user_id="759", source_user_id="usr_src").code

    assert store.reserve(code, "usr_A")[0] == "ok"
    assert store.reserve(code, "usr_A")[0] == "ok"
    assert store.reserve(code, "usr_A")[0] == "ok"


def test_release_reservation_unblocks_retry(tmp_path):
    """Merge failure releases the reservation eagerly so a different target
    could redeem after a real failure (this MOSTLY exists for the same target's
    fast retry; blocking a different target here would strand a legitimate
    re-issue). Retention still holds until TTL if we don't release."""
    store = al.LinkCodeStore(tmp_path / "users.sqlite")
    code = store.issue(channel="discord", channel_user_id="759", source_user_id="usr_src").code

    store.reserve(code, "usr_A")
    store.release_reservation(code, "usr_A")
    # After release, a different target CAN reserve — the merge for target A
    # already failed and rolled back, so this is a normal race window not a leak.
    assert store.reserve(code, "usr_B")[0] == "ok"


def test_only_the_holder_can_release(tmp_path):
    """release_reservation is target-scoped; a stranger can't drop someone
    else's lock. Otherwise an attacker could park a release call and race in."""
    store = al.LinkCodeStore(tmp_path / "users.sqlite")
    code = store.issue(channel="discord", channel_user_id="759", source_user_id="usr_src").code
    store.reserve(code, "usr_A")

    store.release_reservation(code, "usr_evil")  # no-op — doesn't hold it
    assert store.reserve(code, "usr_B")[0] == "locked"


def test_reserve_expires_after_ttl(tmp_path, monkeypatch):
    """A crashed redeemer's stale reservation ages out so a different target
    can eventually redeem (e.g. after the source user reissues a code)."""
    from datetime import UTC, datetime, timedelta

    store = al.LinkCodeStore(tmp_path / "users.sqlite")
    code = store.issue(channel="discord", channel_user_id="759", source_user_id="usr_src").code
    store.reserve(code, "usr_A")

    later = datetime.now(UTC) + timedelta(seconds=al.RESERVATION_TTL_SECONDS + 2)
    monkeypatch.setattr(al, "_utc_now", lambda: later)
    assert store.reserve(code, "usr_B")[0] == "ok"


def test_reserve_rejects_invalid_and_consumed(tmp_path):
    store = al.LinkCodeStore(tmp_path / "users.sqlite")
    assert store.reserve("NOPE", "usr_A")[0] == "invalid"
    assert store.reserve("", "usr_A")[0] == "invalid"

    code = store.issue(channel="discord", channel_user_id="759", source_user_id="usr_src").code
    assert store.reserve(code, "usr_A")[0] == "ok"
    al.mark_consumed(store, code)
    assert store.reserve(code, "usr_A")[0] == "invalid"
    assert store.reserve(code, "usr_B")[0] == "invalid"


# ── drain-user-tasks awaits in-flight writers ────────────────────────────────

def test_drain_user_tasks_waits_for_tagged_tasks():
    """Tasks tagged for a user must finish before drain returns."""
    from core.worker import drain_user_tasks, track_task

    finished: list[str] = []

    async def slow_writer(marker: str, delay: float = 0.05):
        await asyncio.sleep(delay)
        finished.append(marker)

    async def run():
        t1 = asyncio.create_task(slow_writer("A1"))
        t2 = asyncio.create_task(slow_writer("A2"))
        t_other = asyncio.create_task(slow_writer("B1"))
        track_task(t1, user_id="usr_A")
        track_task(t2, user_id="usr_A")
        track_task(t_other, user_id="usr_B")

        awaited = await drain_user_tasks("usr_A", timeout=2.0)
        # both A tasks must have completed before drain returned
        assert "A1" in finished and "A2" in finished
        assert awaited == 2
        # B was not awaited — let it finish so the loop exits cleanly
        await t_other

    asyncio.run(run())


def test_drain_user_tasks_is_bounded():
    """A hung task must not stall linking indefinitely — timeout is honoured."""
    from core.worker import drain_user_tasks, track_task
    import time

    async def hung():
        await asyncio.sleep(3.0)

    async def run():
        t = asyncio.create_task(hung())
        track_task(t, user_id="usr_hung")
        t0 = time.perf_counter()
        awaited = await drain_user_tasks("usr_hung", timeout=0.2)
        elapsed = time.perf_counter() - t0
        # returned early via TimeoutError inside gather; still reported the count
        assert awaited == 1
        assert elapsed < 1.0, f"drain waited {elapsed:.2f}s past its timeout"
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


def test_drain_untagged_users_is_a_noop():
    """A user with zero tracked tasks returns 0 immediately."""
    from core.worker import drain_user_tasks
    assert asyncio.run(drain_user_tasks("usr_never_seen")) == 0
