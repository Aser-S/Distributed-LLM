"""Master scheduler - Phase 1 / Step 2: worker registration.

The Scheduler is the master controller. It owns the registry of GPU workers
(add / remove / list). Step 2 only handles the registry; dispatching via the
load balancer is wired in Step 4.
"""


class Scheduler:
    """Master controller. Source of truth for which workers are in the cluster."""

    def __init__(self, workers=None):
        self.workers = {}  # worker_id -> GPUWorker
        if workers:
            self.register_workers(workers)

    def register_worker(self, worker):
        """Add a single worker to the registry."""
        if worker.id in self.workers:
            print(f"[Scheduler] worker {worker.id} already registered, skipping")
            return
        self.workers[worker.id] = worker
        print(f"[Scheduler] registered worker {worker.id} (total={len(self.workers)})")

    def register_workers(self, workers):
        """Bulk-register a list of workers."""
        for w in workers:
            self.register_worker(w)

    def unregister_worker(self, worker_id):
        """Remove a worker from the registry. Used by fault tolerance in Phase 4."""
        if worker_id not in self.workers:
            print(f"[Scheduler] worker {worker_id} not registered, nothing to remove")
            return
        del self.workers[worker_id]
        print(f"[Scheduler] unregistered worker {worker_id} (total={len(self.workers)})")

    def get_workers(self):
        """Return current workers as a list."""
        return list(self.workers.values())

    def handle_request(self, request):
        """Step 2 stub: confirms registry is populated; dispatch is wired in Step 4."""
        if not self.workers:
            raise RuntimeError("[Scheduler] no workers registered")
        print(
            f"[Scheduler] received request {request.id} "
            f"(workers available={len(self.workers)})"
        )
        return None  # not dispatched yet — Step 4 wires the LB
