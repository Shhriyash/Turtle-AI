"""
Quick parallel tool-call smoke test.
Each search_web call sleeps 1s (monkeypatched). If calls run in parallel,
total time ≈ 1s. If sequential, total time ≈ 2s.

Usage:  python test_parallel.py
"""
import asyncio
import time
import unittest.mock as mock

CALL_LOG: list[tuple[str, float]] = []


async def fake_search(query: str, *_, **__):
    t = time.perf_counter()
    CALL_LOG.append((query, t))
    await asyncio.sleep(1)
    return [{"title": f"Result for {query}", "url": "https://example.com", "content": "fake"}]


async def main():
    with mock.patch("core.web_search.search_tavily", new=fake_search), \
         mock.patch("core.web_search.search_duckduckgo", new=fake_search):

        from apps.turtle_server import TurtleAgentManager
        mgr = TurtleAgentManager()

        start = time.perf_counter()
        response = await mgr.handle_text_message(
            "Search for 'weather in Tokyo' and 'weather in London' and tell me both results.",
            session_id="parallel-test",
        )
        elapsed = time.perf_counter() - start

    print("\n--- Response ---")
    print(response[:400])
    print("\n--- Timing ---")
    for q, t in CALL_LOG:
        print(f"  search started: {q!r:30s}  t+{t - start:.3f}s")

    print(f"\n  Total elapsed: {elapsed:.2f}s")
    if elapsed < 1.8:
        print("  ✓ PARALLEL  (both calls fired simultaneously)")
    else:
        print("  ✗ SEQUENTIAL (calls ran one after the other)")


asyncio.run(main())
