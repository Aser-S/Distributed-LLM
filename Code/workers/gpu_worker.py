"""GPU worker node - Phase 1 / Step 3: split lifecycle from work; variable latency."""

import random
import time
import threading


class GPUWorker:
    """Simulates a single GPU node that processes one request at a time.

    Step 3 splits process() (lifecycle: lock, timing, counters) from _do_work()
    (the actual computation). The seam lets Phase 3 swap in the real LLM call
    without touching lifecycle code.
    """

    # Simulated GPU work range, in seconds. Replaced by real LLM latency in Phase 3.
    SIM_LATENCY_MIN = 0.05
    SIM_LATENCY_MAX = 0.20

    def __init__(self, worker_id: int):
        self.id = worker_id
        self.busy = False
        self.processed_count = 0
        self.total_latency = 0.0
        self._lock = threading.Lock()

    def process(self, request) -> dict:
        """Lifecycle: mark busy, run _do_work, record latency, release."""
        with self._lock:
            self.busy = True
        start = time.time()
        try:
            print(f"[Worker {self.id}] received request {request.id}")
            result = self._do_work(request)
            latency = time.time() - start
            return {
                "id": request.id,
                "result": result,
                "latency": latency,
                "worker_id": self.id,
            }
        finally:
            elapsed = time.time() - start
            with self._lock:
                self.busy = False
                self.processed_count += 1
                self.total_latency += elapsed

    def _do_work(self, request) -> str:
        """Stubbed GPU computation. Phase 3 replaces this with the real LLM call."""
        delay = random.uniform(self.SIM_LATENCY_MIN, self.SIM_LATENCY_MAX)
        time.sleep(delay)
        return f"[stub] worker={self.id} answered query={request.query!r}"

    def __repr__(self) -> str:
        avg = (self.total_latency / self.processed_count) if self.processed_count else 0.0
        return (
            f"GPUWorker(id={self.id}, processed={self.processed_count}, "
            f"avg_latency={avg:.3f}s)"
        )
