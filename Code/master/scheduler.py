"""Master scheduler - Phase 2 / Step 10: pluggable balancing strategy.

The Scheduler owns the worker registry AND a load balancer. Strategies:
  - "round_robin"        -> existing lb.load_balancer.LoadBalancer (cycles workers)
  - "least_connections"  -> LeastConnectionsBalancer below (picks lowest inflight)

Whenever the registry changes the balancer is rebuilt with the current workers.
"""

from lb.load_balancer import LoadBalancer
from workers.gpu_worker import GPUWorker


class LeastConnectionsBalancer:
    """Picks the worker with the fewest in-flight requests; ties broken by id.

    Drop-in replacement for `LoadBalancer`: same `get_next_worker` / `dispatch`
    contract. Relies on `worker.inflight` (added in Step 7).
    """

    def __init__(self, workers: list[GPUWorker]):
        self.workers = workers

    def get_next_worker(self) -> GPUWorker:
        return min(self.workers, key=lambda w: (w.inflight, w.id))

    def dispatch(self, request):
        return self.get_next_worker().process(request)


class Scheduler:
    """Master controller. Source of truth for the cluster; dispatches via LB."""

    SUPPORTED_STRATEGIES = ("round_robin", "least_connections")

    def __init__(self, workers=None, strategy: str = "round_robin"):
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"unknown strategy {strategy!r}; expected one of {self.SUPPORTED_STRATEGIES}"
            )
        self.strategy = strategy
        self.workers = {}  # worker_id -> GPUWorker
        self.lb = None
        if workers:
            self.register_workers(workers)

    def register_worker(self, worker):
        if worker.id in self.workers:
            print(f"[Scheduler] worker {worker.id} already registered, skipping")
            return
        self.workers[worker.id] = worker
        self._rebuild_lb()
        print(f"[Scheduler] registered worker {worker.id} (total={len(self.workers)})")

    def register_workers(self, workers):
        for w in workers:
            self.register_worker(w)

    def unregister_worker(self, worker_id):
        if worker_id not in self.workers:
            print(f"[Scheduler] worker {worker_id} not registered, nothing to remove")
            return
        del self.workers[worker_id]
        self._rebuild_lb()
        print(f"[Scheduler] unregistered worker {worker_id} (total={len(self.workers)})")

    def get_workers(self):
        return list(self.workers.values())

    def _rebuild_lb(self):
        """Sync the LB with the current registry, honouring the chosen strategy."""
        if not self.workers:
            self.lb = None
            return
        workers = self.get_workers()
        if self.strategy == "round_robin":
            self.lb = LoadBalancer(workers)
        elif self.strategy == "least_connections":
            self.lb = LeastConnectionsBalancer(workers)
        else:
            raise ValueError(f"unknown strategy {self.strategy!r}")

    def handle_request(self, request):
        """Dispatch a request to a worker via the LoadBalancer (round robin)."""
        if not self.lb:
            raise RuntimeError("[Scheduler] no workers registered")
        print(f"[Scheduler] dispatching request {request.id} (workers={len(self.workers)})")
        return self.lb.dispatch(request)

    def cluster_status(self) -> dict:
        """Snapshot of the cluster: per-worker stats + aggregates.

        Used to verify load distribution and to feed monitoring in Phase 5.
        """
        workers = self.get_workers()
        per_worker = []
        total_processed = 0
        total_latency = 0.0
        for w in workers:
            avg = (w.total_latency / w.processed_count) if w.processed_count else 0.0
            per_worker.append({
                "worker_id": w.id,
                "processed": w.processed_count,
                "busy": w.busy,
                "avg_latency": round(avg, 4),
            })
            total_processed += w.processed_count
            total_latency += w.total_latency
        cluster_avg = (total_latency / total_processed) if total_processed else 0.0
        return {
            "strategy": self.strategy,
            "worker_count": len(workers),
            "total_processed": total_processed,
            "cluster_avg_latency": round(cluster_avg, 4),
            "workers": per_worker,
        }
