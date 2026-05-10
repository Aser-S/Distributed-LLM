"""GPU worker node - Phase 1 / Step 1: skeleton with identity and a stub process()."""

import time
import threading


class GPUWorker:
    """Simulates a single GPU node that processes one request at a time.

    Phase 1 keeps process() as a stub (sleeps briefly, returns a fake result).
    Real LLM + RAG integration arrives in Phase 3.
    """

    def __init__(self, worker_id: int):
        self.id = worker_id
        self.busy = False                 # used by least-connections in Phase 2
        self.processed_count = 0          # used by metrics in Phase 5
        self._lock = threading.Lock()     # protects busy + counter under threaded clients

    def process(self, request) -> dict:
        """Stub processing. Real LLM/RAG wiring is added in Phase 3."""
        with self._lock:
            self.busy = True
        start = time.time()
        try:
            print(f"[Worker {self.id}] received request {request.id}")
            time.sleep(0.05)  # placeholder for GPU work
            result = f"[stub] worker={self.id} answered query={request.query!r}"
            latency = time.time() - start
            return {"id": request.id, "result": result, "latency": latency}
        finally:
            with self._lock:
                self.busy = False
                self.processed_count += 1

    def __repr__(self) -> str:
        return f"GPUWorker(id={self.id}, processed={self.processed_count})"
