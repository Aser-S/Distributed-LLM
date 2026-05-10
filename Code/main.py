"""End-to-end entry point: client -> scheduler -> LB -> workers."""

import sys
from pathlib import Path

# Make sure Code/ is on sys.path so absolute imports resolve when run from anywhere.
sys.path.insert(0, str(Path(__file__).parent))

from workers.gpu_worker import GPUWorker
from master.scheduler import Scheduler
from client.load_generator import run_load_test


def main():
    # Phase 3 default: 4 GPU workers, 4-user sync smoke test.
    # Each request now hits real Ollama (~3-5 s); 4 users via threading + RR
    # gives one request per worker. For the async pipeline + bigger loads,
    # use the Step 13 async demo instead.
    workers = [GPUWorker(i) for i in range(4)]
    scheduler = Scheduler(workers)

    run_load_test(scheduler, num_users=4)

    print("\n--- CLUSTER STATUS ---")
    status = scheduler.cluster_status()
    print(f"workers={status['worker_count']}, "
          f"total_processed={status['total_processed']}, "
          f"cluster_avg_latency={status['cluster_avg_latency']}s")
    for w in status["workers"]:
        print(f"  worker {w['worker_id']}: processed={w['processed']}, "
              f"avg_latency={w['avg_latency']}s")


if __name__ == "__main__":
    main()
