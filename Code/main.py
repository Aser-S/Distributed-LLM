"""End-to-end entry point: client -> scheduler -> LB -> workers."""

import sys
import asyncio
from pathlib import Path

# Make sure Code/ is on sys.path so absolute imports resolve when run from anywhere.
sys.path.insert(0, str(Path(__file__).parent))

from workers.gpu_worker import GPUWorker
from master.scheduler import Scheduler
from client.load_generator import run_load_test
from client.async_load_generator import run_async_load_test, run_scalability_benchmark
from common.hardening import run_full_hardening_suite


def main_sync():
    """Synchronous smoke test with 4 users (existing baseline)."""
    print("=== SYNC BASELINE TEST (4 users) ===\n")
    workers = [GPUWorker(i) for i in range(4)]
    scheduler = Scheduler(workers, strategy="load_aware")

    run_load_test(scheduler, num_users=4)

    print("\n--- CLUSTER STATUS ---")
    status = scheduler.cluster_status()
    print(f"strategy={status['strategy']}, workers={status['worker_count']}, "
          f"total_processed={status['total_processed']}, "
          f"cluster_avg_latency={status['cluster_avg_latency']}s, "
          f"cluster_throughput={status['cluster_throughput_rps']} rps")
    print(f"GPU metrics: avg_util={status['cluster_gpu_util_avg_pct']}%, "
          f"overload_count={status['cluster_gpu_overload_count']}")
    print(f"Reliability: retry_rate={status['retry_rate']}, "
          f"timeout_rate={status['timeout_rate']}")
    for w in status["workers"]:
        print(f"  worker {w['worker_id']}: processed={w['processed']}, "
              f"avg={w['avg_latency']}s, p50={w['p50_latency']}s, "
              f"p95={w['p95_latency']}s, p99={w['p99_latency']}s, "
              f"tput={w['throughput_rps']} rps, "
              f"gpu_util={w['gpu_util_pct']}%, "
              f"gpu_sat={w['gpu_saturation_score']}, "
              f"gpu_ovld={w['gpu_overload_score']}")


async def main_async():
    """Async stress test with configurable concurrency."""
    print("\n=== ASYNC STRESS TEST ===\n")
    workers = [GPUWorker(i) for i in range(4)]
    scheduler = Scheduler(workers, strategy="gpu_aware")
    scheduler.start_workers()

    try:
        # Run a moderate async load test (100 users, 30 seconds)
        summary = await run_async_load_test(
            scheduler,
            num_users=100,
            pattern="sustained",
            duration_s=30.0,
            print_progress=True,
        )

        print("\n--- ASYNC LOAD TEST SUMMARY ---")
        print(f"Pattern: {summary['pattern']}")
        print(f"Duration: {summary['duration_s']:.1f}s")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Successful: {summary['successful']} ({summary['success_rate']*100:.1f}%)")
        print(f"Failed: {summary['failed']}")
        print(f"Throughput: {summary['throughput_rps']:.2f} req/s")
        print(f"Latency - Avg: {summary['avg_latency']:.3f}s, "
              f"Min: {summary['min_latency']:.3f}s, Max: {summary['max_latency']:.3f}s")
        print(f"Latency - P50: {summary['p50_latency']:.3f}s, "
              f"P95: {summary['p95_latency']:.3f}s, P99: {summary['p99_latency']:.3f}s")

        status = scheduler.cluster_status()
        print(f"\nFinal cluster state:")
        print(f"  Total processed: {status['total_processed']}")
        print(f"  Avg latency: {status['cluster_avg_latency']}s")
        print(f"  Throughput: {status['cluster_throughput_rps']} rps")
        print(f"  GPU util avg: {status['cluster_gpu_util_avg_pct']}%")

    finally:
        await scheduler.stop_workers()


async def main_hardening():
    """Run system hardening validation suite."""
    print("\n=== HARDENING VALIDATION ===\n")
    workers = [GPUWorker(i) for i in range(2)]
    scheduler = Scheduler(workers, strategy="load_aware")
    scheduler.start_workers()

    try:
        results = await run_full_hardening_suite(scheduler, with_load_injection=False)
    finally:
        await scheduler.stop_workers()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distributed LLM cluster demo")
    parser.add_argument("--mode", choices=["sync", "async", "hardening"], default="sync",
                        help="Test mode: sync (baseline), async (stress), or hardening")
    args = parser.parse_args()

    if args.mode == "sync":
        main_sync()
    elif args.mode == "async":
        asyncio.run(main_async())
    elif args.mode == "hardening":
        asyncio.run(main_hardening())
