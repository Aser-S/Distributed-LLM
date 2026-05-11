"""Master scheduler - Phase 4 (Steps 14-18) + Phase 5 Steps 19-21.

Strategies for `get_next_worker`:
  - "round_robin"        -> lb.load_balancer.LoadBalancer (cycles workers)
  - "least_connections"  -> LeastConnectionsBalancer (picks lowest inflight)
  - "load_aware"         -> LoadAwareBalancer (picks lowest pending; Step 19)

Execution surfaces:
  - handle_request(request)        -> sync, calls worker.process(request)
  - handle_request_async(request)  -> async, calls await worker.submit(request)
                                      with per-task timeout + reassignment

Step 14: `cluster_status()` reports `last_heartbeat_age_s` per worker.
Step 15: `_health_check_loop()` evicts workers whose heartbeat goes stale.
Step 16: `handle_request_async` wraps submit() in asyncio.wait_for(timeout).
Step 17: On timeout / exception, the request is reassigned to another worker
         up to MAX_ATTEMPTS times. The failing worker is skipped on retry.
Step 19: `worker.pending` is incremented at dispatch time and decremented on
         completion, so the balancer can distribute a burst evenly even
         before any request has reached its consumer.
Step 20: per-worker p50/p95 latency + recent throughput, derived from the
         sliding window `worker.recent` and surfaced via cluster_status().
Step 21: optional dashboard loop (`start_dashboard()` / `stop_dashboard()`)
         that prints a single rolling status line every DASHBOARD_INTERVAL
         seconds — opt-in so existing tests/scripts stay quiet by default.
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from collections import deque

from lb.load_balancer import LoadBalancer
from workers.gpu_worker import GPUWorker
from common.metrics import percentiles as _calc_percentiles
from common.structured_logging import get_logger


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


class LoadAwareBalancer:
    """Step 19: picks the worker with the lowest `pending` count.

    `pending` is incremented by the scheduler the moment a worker is chosen
    (NOT when the request actually starts executing on a consumer), so a
    burst of N async requests gets distributed across all workers instead of
    piling onto worker 0 like least-connections did.

    Ties are broken by id for determinism.
    """

    def __init__(self, workers: list[GPUWorker]):
        self.workers = workers

    def get_next_worker(self) -> GPUWorker:
        return min(self.workers, key=lambda w: (w.pending, w.id))

    def dispatch(self, request):
        return self.get_next_worker().process(request)


class GPUAwareBalancer:
    """Phase 23: picks worker with lowest saturation score (simulated GPU pressure).

    Considers saturation_score from GPU simulator: combines GPU util + queue pressure.
    Falls back to pending count for ties.
    """

    def __init__(self, workers: list[GPUWorker]):
        self.workers = workers

    def get_next_worker(self) -> GPUWorker:
        # Primary key: saturation score; secondary: pending count; tertiary: id.
        return min(
            self.workers,
            key=lambda w: (w.gpu_sim.util_pct / 100.0, w.pending, w.id),
        )

    def dispatch(self, request):
        return self.get_next_worker().process(request)


class Scheduler:
    """Master controller. Source of truth for the cluster; dispatches via LB."""

    SUPPORTED_STRATEGIES = ("round_robin", "least_connections", "load_aware", "gpu_aware")

    # Health-check tuning knobs.
    HEALTH_CHECK_INTERVAL = 2.0   # seconds between health sweeps
    STALE_THRESHOLD = 3.0         # seconds of silence before a worker is evicted

    # Per-request tuning knobs (Steps 16-17).
    REQUEST_TIMEOUT = 30.0        # seconds before a single submit() is abandoned
    MAX_ATTEMPTS = 3              # total tries (1 initial + 2 reassignments)

    # Metrics window (Step 20).
    THROUGHPUT_WINDOW = 10.0      # seconds over which req/s is computed

    # Dashboard (Step 21).
    DASHBOARD_INTERVAL = 5.0      # seconds between dashboard status prints

    def __init__(self, workers=None, strategy: str = "round_robin"):
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"unknown strategy {strategy!r}; expected one of {self.SUPPORTED_STRATEGIES}"
            )
        self.strategy = strategy
        self.workers = {}  # worker_id -> GPUWorker
        self.lb = None
        self._health_monitor_task: asyncio.Task | None = None
        self._dashboard_task: asyncio.Task | None = None
        self._metrics_lock = threading.Lock()
        self.metrics = SchedulerMetrics()
        if workers:
            self.register_workers(workers)

    def _record_completion(self) -> None:
        with self._metrics_lock:
            self.metrics.completed.append(time.time())

    def _record_retry(self) -> None:
        with self._metrics_lock:
            self.metrics.retries += 1

    def _record_timeout(self) -> None:
        with self._metrics_lock:
            self.metrics.timeouts += 1

    def _record_failure(self) -> None:
        with self._metrics_lock:
            self.metrics.failures += 1

    def _record_overload(self) -> None:
        with self._metrics_lock:
            self.metrics.overload_events += 1

    def _scheduler_metrics_snapshot(self, now: float) -> tuple[int, float, int, int, int, int]:
        with self._metrics_lock:
            recent = sum(
                1 for ts in self.metrics.completed if now - ts <= self.THROUGHPUT_WINDOW
            )
            tput = recent / self.THROUGHPUT_WINDOW
            return (
                len(self.metrics.completed),
                tput,
                self.metrics.retries,
                self.metrics.timeouts,
                self.metrics.failures,
                self.metrics.overload_events,
            )

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
        elif self.strategy == "load_aware":
            self.lb = LoadAwareBalancer(workers)
        elif self.strategy == "gpu_aware":
            self.lb = GPUAwareBalancer(workers)
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
        """Stop dashboard + health monitor, then drain all worker queues."""
        await self.stop_dashboard()
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

    # --- Step 21: aggregate monitoring / dashboard line ---

    def start_dashboard(self, interval: float | None = None) -> None:
        """Start an opt-in background task that prints a rolling status line.

        Disabled by default — call this explicitly from main / your harness
        when you want continuous visibility into a long-running run.
        `interval` overrides DASHBOARD_INTERVAL for this run only.
        """
        if self._dashboard_task is not None:
            return
        period = interval if interval is not None else self.DASHBOARD_INTERVAL
        self._dashboard_task = asyncio.create_task(
            self._dashboard_loop(period), name="scheduler-dashboard"
        )

    async def stop_dashboard(self) -> None:
        """Cancel and await the dashboard task."""
        if self._dashboard_task is None:
            return
        self._dashboard_task.cancel()
        try:
            await self._dashboard_task
        except asyncio.CancelledError:
            pass
        self._dashboard_task = None

    async def _dashboard_loop(self, interval: float) -> None:
        """Print one summary line per `interval` seconds.

        Format keeps everything on one line so a tail -f / scrolling terminal
        stays readable. Per-worker block uses `id:processed@p95s/tput` triples.

        Phase 21: also emit a cluster snapshot event for structured logging.
        """
        logger = get_logger()
        try:
            while True:
                await asyncio.sleep(interval)
                status = self.cluster_status()
                workers = "  ".join(
                    f"w{w['worker_id']}:{w['processed']}@p95={w['p95_latency']}s/"
                    f"{w['throughput_rps']}rps/pend={w['pending']}"
                    for w in status["workers"]
                )
                print(
                    f"[Dashboard] strategy={status['strategy']} "
                    f"workers={status['worker_count']} "
                    f"total={status['total_processed']} "
                    f"avg={status['cluster_avg_latency']}s "
                    f"tput={status['cluster_throughput_rps']}rps | {workers}"
                )
                # Phase 21: emit cluster snapshot for structured logging
                logger.cluster_snapshot(
                    status["strategy"],
                    status["worker_count"],
                    status["total_processed"],
                    status["cluster_avg_latency"],
                    status["cluster_throughput_rps"],
                    status["cluster_gpu_util_avg_pct"],
                    status["retry_rate"],
                    status["timeout_rate"],
                )
        except asyncio.CancelledError:
            return

    async def _health_check_loop(self) -> None:
        """Periodically evict workers whose heartbeat has gone stale.

        A worker is considered dead when `time.time() - w.last_heartbeat`
        exceeds STALE_THRESHOLD. Eviction calls unregister_worker(), which
        also rebuilds the LB so no future requests land on the dead worker.
        """
        logger = get_logger()
        try:
            while True:
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
                now = time.time()
                stale = [
                    wid for wid, w in list(self.workers.items())
                    if now - w.last_heartbeat > self.STALE_THRESHOLD
                ]
                for wid in stale:
                    age = now - self.workers[wid].last_heartbeat
                    logger.evict_worker(wid, "stale_heartbeat")
                    print(
                        f"[Scheduler] worker {wid} heartbeat stale "
                        f"({age:.1f}s > "
                        f"{self.STALE_THRESHOLD}s) — evicting"
                    )
                    self.unregister_worker(wid)
        except asyncio.CancelledError:
            return

    async def handle_request_async(self, request) -> dict:
        """Async dispatch with per-task timeout (Step 16) + reassignment (Step 17).

        Picks a worker via the LB, awaits `worker.submit(request)` with a
        REQUEST_TIMEOUT bound. On timeout or any exception the worker is
        added to a skip-set and another worker is chosen, up to MAX_ATTEMPTS.
        If every attempt fails, the last exception is re-raised so the caller
        sees the actual failure mode.
        """
        if not self.lb:
            raise RuntimeError("[Scheduler] no workers registered")

        logger = get_logger()
        logger.request_received(request.id, self.strategy)

        tried: set[int] = set()
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            worker = self._pick_worker(skip=tried)
            if worker is None:
                # No healthy worker remains that we haven't already tried.
                break
            tried.add(worker.id)
            queue_depth = worker._queue.qsize() if worker._queue is not None else 0
            logger.dispatch(request.id, worker.id, attempt, worker.pending, queue_depth)
            print(
                f"[Scheduler] dispatching request {request.id} -> worker {worker.id} "
                f"(attempt {attempt}/{self.MAX_ATTEMPTS}, pending={worker.pending})"
            )
            # Step 19: account for this dispatch BEFORE awaiting so the next
            # LB pick in a burst sees an up-to-date load reading.
            worker.pending += 1
            try:
                result = await asyncio.wait_for(
                    worker.submit(request), timeout=self.REQUEST_TIMEOUT
                )
                # Phase 21: log completion
                latency_ms = result.get("latency", 0.0) * 1000
                logger.completed(request.id, worker.id, latency_ms, 0.0)
                self._record_completion()
                return result
            except asyncio.TimeoutError as e:
                last_error = e
                logger.timeout(request.id, worker.id, attempt, self.REQUEST_TIMEOUT)
                self._record_timeout()
                if attempt < self.MAX_ATTEMPTS:
                    logger.reassign(request.id, worker.id, -1, attempt)
                    self._record_retry()
                print(
                    f"[Scheduler] request {request.id} TIMED OUT on worker {worker.id} "
                    f"after {self.REQUEST_TIMEOUT}s — reassigning"
                )
            except Exception as e:
                last_error = e
                self._record_failure()
                if attempt < self.MAX_ATTEMPTS:
                    logger.reassign(request.id, worker.id, -1, attempt)
                    self._record_retry()
                print(
                    f"[Scheduler] request {request.id} FAILED on worker {worker.id}: "
                    f"{type(e).__name__}: {e} — reassigning"
                )
            finally:
                # Always release the slot regardless of outcome.
                worker.pending = max(0, worker.pending - 1)

        # Out of attempts (or workers). Surface the failure to the caller.
        if last_error is None:
            self._record_overload()
            raise RuntimeError(
                f"[Scheduler] no available workers for request {request.id}"
            )
        raise RuntimeError(
            f"[Scheduler] request {request.id} failed after {len(tried)} attempt(s); "
            f"last error: {type(last_error).__name__}: {last_error}"
        ) from last_error

    def _pick_worker(self, skip: set[int]):
        """Pick a registered worker not in `skip`, preferring the LB's choice.

        Uses the LB's get_next_worker() first (so RR rotation / least-conn
        logic is preserved); if that worker is in `skip`, scans the rest.
        Returns None if every registered worker has been tried.
        """
        if not self.lb:
            return None
        # Fast path: LB's first pick is fine if it hasn't been tried.
        primary = self.lb.get_next_worker()
        if primary.id not in skip:
            return primary
        # Fallback: linear scan of the registry for any untried worker.
        for w in self.workers.values():
            if w.id not in skip:
                return w
        return None

    def cluster_status(self) -> dict:
        """Snapshot of the cluster: per-worker stats + aggregates.

        Step 20: each worker entry now also carries p50_latency, p95_latency,
        and throughput_rps (requests per second over the last THROUGHPUT_WINDOW
        seconds). The cluster aggregate sums per-worker throughput.

        Phase 23: adds GPU simulation metrics (util %, VRAM, queue pressure, saturation).
        """
        workers = self.get_workers()
        per_worker = []
        total_processed = 0
        total_latency = 0.0
        cluster_throughput = 0.0
        cluster_util = 0.0
        cluster_overload_events = 0
        now = time.time()
        for w in workers:
            avg = (w.total_latency / w.processed_count) if w.processed_count else 0.0
            latency_samples = w.end_to_end if w.end_to_end else w.recent
            p50, p95, p99 = _percentiles([lat for _, lat in latency_samples], (50, 95, 99))
            qw_p50, qw_p95, qw_p99 = _percentiles([lat for _, lat in w.queue_wait], (50, 95, 99))
            recent_count = sum(
                1 for ts, _ in w.recent if now - ts <= self.THROUGHPUT_WINDOW
            )
            tput = recent_count / self.THROUGHPUT_WINDOW

            # Phase 23: GPU metrics snapshot
            gpu_state = w.gpu_sim.update(
                w.inflight, w._queue.qsize() if w._queue is not None else 0
            )

            per_worker.append({
                "worker_id": w.id,
                "processed": w.processed_count,
                "busy": w.busy,
                "inflight": w.inflight,
                "pending": w.pending,
                "queue_depth": w._queue.qsize() if w._queue is not None else 0,
                "avg_latency": round(avg, 4),
                "p50_latency": round(p50, 4),
                "p95_latency": round(p95, 4),
                "p99_latency": round(p99, 4),
                "queue_wait_p50": round(qw_p50, 4),
                "queue_wait_p95": round(qw_p95, 4),
                "queue_wait_p99": round(qw_p99, 4),
                "throughput_rps": round(tput, 3),
                "last_heartbeat_age_s": round(now - w.last_heartbeat, 2),
                # Phase 23: GPU simulation metrics
                "gpu_util_pct": gpu_state.util_pct,
                "gpu_vram_used_mb": gpu_state.vram_used_mb,
                "gpu_vram_total_mb": w.gpu_sim.total_vram_mb,
                "gpu_queue_pressure": gpu_state.queue_pressure,
                "gpu_saturation_score": gpu_state.saturation_score,
                "gpu_overload_score": gpu_state.overload_score,
            })
            total_processed += w.processed_count
            total_latency += w.total_latency
            cluster_throughput += tput
            cluster_util += gpu_state.util_pct
            if gpu_state.overload_score > 0.5:
                cluster_overload_events += 1

        cluster_avg = (total_latency / total_processed) if total_processed else 0.0
        avg_util = cluster_util / max(1, len(workers))

        sched_total, sched_tput, retry_count, timeout_count, failure_count, overload_events = (
            self._scheduler_metrics_snapshot(now)
        )
        retry_rate = retry_count / max(1, sched_total + retry_count)
        timeout_rate = timeout_count / max(1, sched_total + timeout_count)
        return {
            "strategy": self.strategy,
            "worker_count": len(workers),
            "total_processed": total_processed,
            "cluster_avg_latency": round(cluster_avg, 4),
            "cluster_throughput_rps": round(cluster_throughput, 3),
            "scheduler_processed": sched_total,
            "scheduler_throughput_rps": round(sched_tput, 3),
            "retry_count": retry_count,
            "timeout_count": timeout_count,
            "failure_count": failure_count,
            "overload_events": overload_events,
            "retry_rate": round(retry_rate, 4),
            "timeout_rate": round(timeout_rate, 4),
            # Phase 23: cluster-level GPU metrics
            "cluster_gpu_util_avg_pct": round(avg_util, 2),
            "cluster_gpu_overload_count": cluster_overload_events,
            "workers": per_worker,
        }


def _percentiles(values: list[float], percentiles: tuple[int, ...]) -> tuple[float, ...]:
    """Compatibility shim for tests; delegates to common.metrics.percentiles()."""
    return _calc_percentiles(values, percentiles)


@dataclass
class SchedulerMetrics:
    completed: deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    retries: int = 0
    timeouts: int = 0
    failures: int = 0
    overload_events: int = 0


    def record_completion(self, ts: float) -> None:
        self.completed.append(ts)
