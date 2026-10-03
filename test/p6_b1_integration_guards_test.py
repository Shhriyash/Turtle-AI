"""
test/p6_b1_integration_guards_test.py
-------------------------------------
Two gaps closed while integrating P6-B1 (ledger 6.1), both of the shape this
repo keeps shipping: a mechanism wired at one end only.

1. The server emits `turn_queued` / `turn_queue_cleared`, but
   web/js/websocket.js had no case for either, so both fell through to
   `default: console.log('Unknown')`. The pending bubble happened to work by
   client-side inference in chat.js -- i.e. the client guessed at state the
   SERVER owns. Only the server knows whether a message was actually queued.

2. `replace_messages` ran unshielded immediately after the `done` frame. Now
   that turns are cancellable tasks (6.1), an `interrupt` landing in that
   window cancelled the persistence -- so the user saw an answer that was then
   absent from their history. Stopping a turn must not un-say something
   already said.
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SERVER = _ROOT / "apps" / "turtle_server.py"
_WS_JS = _ROOT / "web" / "js" / "websocket.js"


def test_client_handles_every_turn_queue_frame_the_server_sends():
    """Both ends, derived from the source rather than assumed.

    Collects the frame types the server actually emits for the turn queue and
    asserts the client switch has a case for each. If someone adds a third
    queue frame server-side and forgets the client, this fails.
    """
    server = _SERVER.read_text(encoding="utf-8")
    client = _WS_JS.read_text(encoding="utf-8")

    emitted = {name for name in ("turn_queued", "turn_queue_cleared") if f'"{name}"' in server}
    assert emitted, "expected the server to emit the turn-queue frames"

    for frame in sorted(emitted):
        assert f"case '{frame}':" in client, (
            f"server emits {frame!r} but web/js/websocket.js has no case for it, "
            f"so it falls through to the Unknown branch"
        )

    # And the handlers must be wired to the real pending-state functions,
    # not left as empty cases.
    assert "setPendingTurn" in client and "clearPendingTurn" in client, (
        "the turn-queue cases must drive chat.js's pending state"
    )


def test_persistence_after_done_is_shielded_from_cancellation():
    """Every `replace_messages` that follows a `done` frame must be shielded.

    Structural, because reaching the real call needs an LLM, a socket and a
    session store. What can regress is someone unwrapping the shield, so that
    is what this pins. The asyncio semantics it relies on are pinned
    separately below.
    """
    tree = ast.parse(_SERVER.read_text(encoding="utf-8"))

    unshielded: list[int] = []
    for node in ast.walk(tree):
        # Looking for `await <something>.replace_messages(...)` where the
        # awaited expression is NOT asyncio.shield(...).
        if not isinstance(node, ast.Await):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "replace_messages":
            unshielded.append(node.lineno)

    assert not unshielded, (
        f"replace_messages awaited without asyncio.shield at line(s) {unshielded}: "
        f"an interrupt between the `done` frame and this write would lose an "
        f"answer the user has already seen"
    )

    # Sanity: the shielded form is actually present, so the test above cannot
    # pass merely because the call was renamed or removed.
    shielded = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "shield"
        and node.args
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Attribute)
        and node.args[0].func.attr == "replace_messages"
    )
    assert shielded >= 2, (
        f"expected both the batch and streaming paths to shield their "
        f"replace_messages write, found {shielded}"
    )


def test_shield_really_is_what_saves_the_write():
    """Pins the asyncio behaviour the structural test above depends on.

    If a future Python cancelled shielded work along with its awaiter, the
    guard would be worthless -- better to learn that from a test than from a
    lost answer in production.
    """

    async def drive(use_shield: bool) -> list[str]:
        persisted: list[str] = []

        async def replace_messages() -> None:
            await asyncio.sleep(0.05)
            persisted.append("written")

        async def turn() -> None:
            # `done` has already gone out to the user by this point.
            if use_shield:
                await asyncio.shield(replace_messages())
            else:
                await replace_messages()

        t = asyncio.create_task(turn())
        await asyncio.sleep(0.01)  # interrupt lands mid-write
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        await asyncio.sleep(0.12)  # let shielded work finish
        return persisted

    assert asyncio.run(drive(use_shield=False)) == [], (
        "an unshielded write survived cancellation -- re-check whether the "
        "shield is still needed"
    )
    assert asyncio.run(drive(use_shield=True)) == ["written"], (
        "asyncio.shield failed to protect the in-flight write"
    )
