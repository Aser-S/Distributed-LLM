"""Async load generator for stress testing and scalability validation.

Generates concurrent requests with configurable concurrency levels,
patterns (sustained, burst, wave), and failure injection.

Usage:
  from client.async_load_generator import run_async_load_test
  import asyncio

  async def main():
    await run_async_load_test(scheduler, num_users=100, pattern="sustained")

  asyncio.run(main())
"""

import asyncio
import time
from typing import Callable, Optional
from common.models import Request


async def run_async_load_test(
    scheduler,
    num_users: int = 100,
    pattern: str = "sustained",
    duration_s: float | None = None,
    target_rps: float | None = None,
    print_progress: bool = True,
) -> dict:
    """Run async load test with specified concurrency and pattern.

    Args:
        scheduler: Scheduler instance
        num_users: concurrent request slots (rate limiting)
        pattern: "sustained" (steady), "burst" (spike), "wave" (ramp)
        duration_s: test duration (None = run until naturally complete)
        target_rps: target requests/sec (None = unlimited)
        print_progress: print periodic progress updates

    Returns:
        dict: summary stats (total_requests, success, failures, avg_latency, min/max_latency)
    """
    start_time = time.time()
    semaphore = asyncio.Semaphore(num_users)
    results = []
    request_id_counter = 0
    lock = asyncio.Lock()

    async def bounded_request(query_text: str) -> dict | None:
        """Submit a request respecting concurrency limit."""
        nonlocal request_id_counter
        async with semaphore:
            async with lock:
                request_id_counter += 1
                req_id = request_id_counter

            try:
                req = Request(id=req_id, query=query_text)
                result = await scheduler.handle_request_async(req)
                return {"success": True, "latency": result.get("latency", 0.0), "id": req_id}
            except Exception as e:
                return {"success": False, "latency": 0.0, "id": req_id, "error": str(e)[:100]}

    async def generate_requests() -> None:
        """Generate requests according to pattern."""
        req_counter = 0
        wave_cycle = 0

        while True:
            # Check duration
            elapsed = time.time() - start_time
            if duration_s is not None and elapsed > duration_s:
                break

            # Pattern: sustained, burst, or wave
            if pattern == "sustained":
                # Steady stream
                delay = 0.01 if target_rps is None else (num_users / target_rps) / num_users
                await asyncio.sleep(delay)
            elif pattern == "burst":
                # Fire all at once, then pause
                if req_counter % (num_users * 2) == 0:
                    await asyncio.sleep(5.0)  # pause between bursts
                else:
                    await asyncio.sleep(0.001)
            elif pattern == "wave":
                # Ramp up and down
                cycle_time = time.time() - start_time
                wave_pos = (cycle_time % 10.0) / 10.0  # 10-second cycle
                if wave_pos < 0.5:
                    # Ramp up
                    load_factor = wave_pos * 2  # 0 -> 1
                else:
                    # Ramp down
                    load_factor = (1.0 - wave_pos) * 2  # 1 -> 0
                delay = (1.0 - load_factor) * 0.05
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0.01)

            query = f"Query {req_counter}"
            task = asyncio.create_task(bounded_request(query))
            results.append(task)
            req_counter += 1

            if req_counter > num_users * 100:  # safety cap
                break

    async def progress_printer() -> None:
        """Periodically print progress."""
        while True:
            await asyncio.sleep(5.0)
            completed = sum(1 for t in results if t.done())
            elapsed = time.time() - start_time
            rate = completed / max(elapsed, 0.1)
            if print_progress:
                print(f"[LoadTest] elapsed={elapsed:.1f}s, submitted={len(results)}, completed={completed}, "
                      f"rate={rate:.1f} req/s")

    # Run generator and progress tasks
    gen_task = asyncio.create_task(generate_requests())
    progress_task = asyncio.create_task(progress_printer()) if print_progress else None

    # Wait for generator to finish or timeout
    try:
        await asyncio.wait_for(gen_task, timeout=duration_s + 5.0 if duration_s else None)
    except asyncio.TimeoutError:
        gen_task.cancel()

    # Wait for all submitted requests to complete
    if results:
        await asyncio.gather(*results, return_exceptions=True)

    if progress_task:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    # Summarize results
    end_time = time.time()
    elapsed = end_time - start_time
    successful = []
    failed = []

    for task in results:
        try:
            result = task.result() if task.done() else None
            if result:
                if result.get("success"):
                    successful.append(result["latency"])
                else:
                    failed.append(result)
        except Exception:
            failed.append({"success": False, "latency": 0.0})

    latencies = sorted(successful)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    p50_latency = latencies[len(latencies) // 2] if latencies else 0.0
    p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    p99_latency = latencies[int(len(latencies) * 0.99)] if latencies else 0.0

    summary = {
        "pattern": pattern,
        "num_users": num_users,
        "duration_s": elapsed,
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / max(1, len(results)),
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
        "p99_latency": p99_latency,
        "throughput_rps": len(results) / elapsed,
    }

    return summary


async def run_scalability_benchmark(scheduler, concurrency_levels: list[int] | None = None) -> None:
    """Run a series of load tests at different concurrency levels.

    Prints a summary table comparing throughput, latency, and success rates.
    """
    if concurrency_levels is None:
        concurrency_levels = [100, 500, 1000]

    print("\n=== SCALABILITY BENCHMARK ===")
    print(f"{'Users':<10} {'Pattern':<12} {'Throughput':<12} {'Avg Latency':<12} {'P99 Latency':<12} {'Success%':<10}")
    print("-" * 80)

    for num_users in concurrency_levels:
        for pattern in ["sustained", "burst"]:
            summary = await run_async_load_test(
                scheduler,
                num_users=num_users,
                pattern=pattern,
                duration_s=30.0,
                print_progress=False,
            )
            print(
                f"{num_users:<10} {pattern:<12} "
                f"{summary['throughput_rps']:<12.2f} "
                f"{summary['avg_latency']:<12.3f}s "
                f"{summary['p99_latency']:<12.3f}s "
                f"{summary['success_rate']*100:<10.1f}%"
            )

    print()
