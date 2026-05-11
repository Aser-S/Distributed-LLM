"""Master scheduler - Phase 4 / Step 15: scheduler-side health checks.

Strategies for `get_next_worker`:
  - "round_robin"        -> lb.load_balancer.LoadBalancer (cycles workers)
  - "least_connections"  -> LeastConnectionsBalancer below (picks lowest inflight)

Execution surfaces:
  - handle_request(request)        -> sync, calls worker.process(request)
  - handle_request_async(request)  -> async, calls await worker.submit(request)

Step 14: `cluster_status()` now reports `last_heartbeat_age_s` per worker.
Step 15: `_health_check_loop()` runs every HEALTH_CHECK_INTERVAL seconds and
         evicts any worker whose heartbeat age exceeds STALE_THRESHOLD.
"""

import asyncio
import time

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

    # Health-check tuning knobs.
    HEALTH_CHECK_INTERVAL = 2.0   # seconds between health sweeps
    STALE_THRESHOLD = 3.0         # seconds of silence before a worker is evicted

    def __init__(self, workers=None, strategy: str = "round_robin"):
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"unknown strategy {strategy!r}; expected one of {self.SUPPORTED_STRATEGIES}"
            )
        self.strategy = strategy
        self.workers = {}  # worker_id -> GPUWorker
        self.lb = None
        self._health_monitor_task: asyncio.Task | None = None
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
        """Sync dispatch via the LoadBalancer. Calls worker.process() (blocking)."""
        if not self.lb:
            raise RuntimeError("[Scheduler] no workers registered")
        print(f"[Scheduler] dispatching request {request.id} (workers={len(self.workers)})")
        return self.lb.dispatch(request)

    # --- Step 13: async dispatch via worker queues ---

    def start_workers(self) -> None:
        """Spawn drain loops on every registered worker, then start health monitor.

        Must be called from inside a running event loop. Idempotent: re-calling
        is a no-op (each worker's start() guards itself; monitor checked here too).
        """
        for w in self.workers.values():
            w.start()
        self.start_health_monitor()

    async def stop_workers(self) -> None:
        """Stop health monitor, then drain all worker queues."""
        await self.stop_health_monitor()
        if not self.workers:
            return
        await asyncio.gather(*(w.stop() for w in self.workers.values()))

    # --- Step 15: scheduler-side health checks ---

    def start_health_monitor(self) -> None:
        """Spawn the health-check background task if not already running."""
        if self._health_monitor_task is not None:
            return
        self._health_monitor_task = asyncio.create_task(
            self._health_check_loop(), name="scheduler-health-monitor"
        )

    async def stop_health_monitor(self) -> None:
        """Cancel and await the health-check task."""
        if self._health_monitor_task is None:
            return
        self._health_monitor_task.cancel()
        try:
            await self._health_monitor_task
        except asyncio.CancelledError:
            pass
        self._health_monitor_task = None

    async def _health_check_loop(self) -> None:
        """Periodically evict workers whose heartbeat has gone stale.

        A worker is considered dead when `time.time() - w.last_heartbeat`
        exceeds STALE_THRESHOLD. Eviction calls unregister_worker(), which
        also rebuilds the LB so no future requests land on the dead worker.
        """
        try:
            while True:
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
                now = time.time()
                stale = [
                    wid for wid, w in list(self.workers.items())
                    if now - w.last_heartbeat > self.STALE_THRESHOLD
                ]
                for wid in stale:
                    print(
                        f"[Scheduler] worker {wid} heartbeat stale "
                        f"({now - self.workers[wid].last_heartbeat:.1f}s > "
                        f"{self.STALE_THRESHOLD}s) — evicting"
                    )
                    self.unregister_worker(wid)
        except asyncio.CancelledError:
            return

    async def handle_request_async(self, request) -> dict:
        """Async dispatch: LB picks the worker, the worker's queue runs the request.

        The LB's selection logic (RR or least-connections) is reused unchanged;
        only the execution surface differs (await submit vs sync process).
        """
        if not self.lb:
            raise RuntimeError("[Scheduler] no workers registered")
        worker = self.lb.get_next_worker()
        print(f"[Scheduler] dispatching request {request.id} -> worker {worker.id} (async)")
        return await worker.submit(request)

    def cluster_status(self) -> dict:
        """Snapshot of the cluster: per-worker stats + aggregates.

        Used to verify load distribution and to feed monitoring in Phase 5.
        """
        workers = self.get_workers()
        per_worker = []
        total_processed = 0
        total_latency = 0.0
        now = time.time()
        for w in workers:
            avg = (w.total_latency / w.processed_count) if w.processed_count else 0.0
            per_worker.append({
                "worker_id": w.id,
                "processed": w.processed_count,
                "busy": w.busy,
                "avg_latency": round(avg, 4),
                "last_heartbeat_age_s": round(now - w.last_heartbeat, 2),
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
