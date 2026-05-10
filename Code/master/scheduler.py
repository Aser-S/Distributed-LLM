"""Master scheduler - Phase 1 / Step 4: round-robin dispatch via LoadBalancer.

The Scheduler owns the worker registry AND a LoadBalancer. Whenever the
registry changes (register / unregister), the LB is rebuilt so it always
sees the current cluster.
"""

from lb.load_balancer import LoadBalancer


class Scheduler:
    """Master controller. Source of truth for the cluster; dispatches via LB."""

    def __init__(self, workers=None):
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
        """Sync the LB with the current registry. RR index resets - acceptable here."""
        self.lb = LoadBalancer(self.get_workers()) if self.workers else None

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
            "worker_count": len(workers),
            "total_processed": total_processed,
            "cluster_avg_latency": round(cluster_avg, 4),
            "workers": per_worker,
        }
