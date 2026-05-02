"""
evals/runner.py
---------------
H1: Eval harness runner.

Hits the live Turtle server WebSocket for each prompt in the frozen set and
records: tool-call accuracy, response text, latency p50/p95, and cost estimate.

Usage::

    python evals/runner.py --prompts evals/prompts/tier1_baseline.json
                           --output evals/results/run_YYYYMMDD.json
                           --url ws://127.0.0.1:8765/ws
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_single_prompt(
    ws_url: str,
    prompt_obj: dict[str, Any],
    *,
    timeout_s: float = 45.0,
) -> dict[str, Any]:
    """Send one prompt to the Turtle server and collect the full response."""
    if websockets is None:
        raise RuntimeError("websockets package not installed. Run: pip install websockets")

    prompt_text = prompt_obj["prompt"]
    start = time.time()
    tool_calls_observed: list[str] = []
    response_text = ""
    timings: dict[str, Any] = {}
    error: str | None = None

    try:
        async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
            # Send text message
            await ws.send(json.dumps({"type": "text", "content": prompt_text}))

            deadline = time.time() + timeout_s
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue

                if isinstance(raw, bytes):
                    # TTS audio chunk — not relevant for text eval
                    continue

                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "done":
                    response_text = msg.get("content", "")
                    break
                elif msg_type == "timing":
                    timings = {k: v for k, v in msg.items() if k != "type"}
                elif msg_type == "status":
                    status = msg.get("status", "")
                    # Infer tool calls from server-side status messages
                    if "searching" in status.lower():
                        tool_calls_observed.append("search_web")
                    elif "analyzing" in status.lower():
                        tool_calls_observed.append("search_url")
                    elif "email" in status.lower():
                        tool_calls_observed.append("send_email_assistant")
                    elif "history" in status.lower():
                        tool_calls_observed.append("history_tool")
                elif msg_type == "error":
                    error = msg.get("message", "unknown error")
                    break

    except Exception as exc:
        error = str(exc)

    elapsed_ms = round((time.time() - start) * 1000)
    return {
        "id": prompt_obj["id"],
        "category": prompt_obj["category"],
        "prompt": prompt_text,
        "response": response_text,
        "tool_calls_observed": list(set(tool_calls_observed)),
        "expected_tool_calls": prompt_obj.get("expected_tool_calls", []),
        "latency_ms": elapsed_ms,
        "timings": timings,
        "error": error,
    }


def score_result(result: dict[str, Any]) -> dict[str, Any]:
    """Score a single eval result."""
    expected = set(result["expected_tool_calls"])
    observed = set(result["tool_calls_observed"])

    if not expected:
        # No tools expected — check no tools were called
        tool_accuracy = 1.0 if not observed else 0.0
    else:
        # At least the expected tools should be called (precision/recall)
        tp = len(expected & observed)
        precision = tp / len(observed) if observed else 0.0
        recall = tp / len(expected)
        tool_accuracy = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    has_citation = (
        "http://" in result["response"] or
        "https://" in result["response"] or
        "source:" in result["response"].lower() or
        "according to" in result["response"].lower()
    ) if result["response"] else False

    hallucination_risk = (
        not has_citation and
        bool(expected & {"search_web", "search_url"}) and
        bool(result["response"])
    )

    return {
        **result,
        "tool_accuracy": round(tool_accuracy, 3),
        "has_citation": has_citation,
        "hallucination_risk": hallucination_risk,
        "pass": tool_accuracy >= 0.8 and not hallucination_risk,
    }


async def run_eval(
    prompts_path: str,
    output_path: str,
    ws_url: str = "ws://127.0.0.1:8765/ws",
    max_concurrent: int = 1,  # Sequential by default to avoid WS connection issues
    timeout_s: float = 45.0,
) -> None:
    prompts = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
    print(f"Running {len(prompts)} eval prompts against {ws_url}")

    results: list[dict[str, Any]] = []
    latencies: list[float] = []

    for i, prompt_obj in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] {prompt_obj['id']}: {prompt_obj['prompt'][:60]!r}")
        result = await run_single_prompt(ws_url, prompt_obj, timeout_s=timeout_s)
        scored = score_result(result)
        results.append(scored)
        latencies.append(scored["latency_ms"])

        status = "PASS" if scored["pass"] else "FAIL"
        citation = "cited" if scored["has_citation"] else "no-cite"
        risk = " HALLUCINATION-RISK" if scored["hallucination_risk"] else ""
        print(f"     {status} tool_acc={scored['tool_accuracy']:.2f} {citation} {scored['latency_ms']}ms{risk}")

    # Summary statistics
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    p50 = sorted_latencies[int(n * 0.50)] if n else 0
    p95 = sorted_latencies[int(n * 0.95)] if n else 0
    pass_rate = sum(1 for r in results if r["pass"]) / len(results) if results else 0
    mean_acc = sum(r["tool_accuracy"] for r in results) / len(results) if results else 0
    hallucination_count = sum(1 for r in results if r["hallucination_risk"])

    summary = {
        "total_prompts": len(prompts),
        "pass_rate": round(pass_rate, 3),
        "mean_tool_accuracy": round(mean_acc, 3),
        "hallucination_risk_count": hallucination_count,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "results": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== Eval Summary ===")
    print(f"Pass rate:           {pass_rate:.1%} ({sum(1 for r in results if r['pass'])}/{len(results)})")
    print(f"Mean tool accuracy:  {mean_acc:.2f}")
    print(f"Hallucination risks: {hallucination_count}")
    print(f"Latency p50/p95:     {p50}ms / {p95}ms")
    print(f"Results written to:  {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Turtle eval harness")
    parser.add_argument("--prompts", default="evals/prompts/tier1_baseline.json")
    parser.add_argument("--output", default=f"evals/results/run_{int(time.time())}.json")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    asyncio.run(run_eval(
        prompts_path=args.prompts,
        output_path=args.output,
        ws_url=args.url,
        timeout_s=args.timeout,
    ))


if __name__ == "__main__":
    main()
