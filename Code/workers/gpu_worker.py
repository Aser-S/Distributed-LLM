"""GPU worker node - Phase 4 / Steps 14 + 18.

`process()` / `process_async()` are the sync / async execution surfaces.
`_do_work` calls real LLM via `infer()` (Step 12).
`start()` spawns a heartbeat task that updates `last_heartbeat` every
HEARTBEAT_INTERVAL seconds. The Scheduler polls this timestamp (Step 15)
to detect dead / hung workers.

Step 18: `simulate_failure()` flips a kill switch — stops the heartbeat
(so the scheduler's health check evicts the worker) and makes any new
`_do_work` call raise `WorkerDeadError` (so Step 17 reassignment fires).
"""


class WorkerDeadError(RuntimeError):
    """Raised by a worker that has been killed via simulate_failure()."""

import asyncio
import time
import threading

from llm.inference import infer


class GPUWorker:
    DEFAULT_CONCURRENCY = 4
    HEARTBEAT_INTERVAL = 1.0  # seconds between heartbeat timestamp updates

    def __init__(self, worker_id: int, concurrency: int = DEFAULT_CONCURRENCY):
        self.id = worker_id
        self.concurrency = max(1, concurrency)        # max parallel requests on this worker
        self.inflight = 0                              # number of requests currently being processed
        self.processed_count = 0
        self.total_latency = 0.0
        self.last_heartbeat = time.time()              # updated by _heartbeat_loop; read by Scheduler health checks
        self._lock = threading.Lock()
        self._queue: asyncio.Queue | None = None       # created in start()
        self._consumers: list[asyncio.Task] = []       # N drain coroutines
        self._heartbeat_task: asyncio.Task | None = None
        self._dead = False                             # Step 18: kill switch
        self.pending = 0                               # Step 19: dispatched-but-not-yet-completed

    @property
    def busy(self) -> bool:
        """True iff any request is currently in flight. Used by least-connections in Step 10."""
        return self.inflight > 0

    def process(self, request) -> dict:
        """Synchronous lifecycle. Used by the existing round-robin LB and Phase 1 tests."""
        self._claim()
        start = time.time()
        try:
            print(f"[Worker {self.id}] received request {request.id}")
            result = self._do_work(request)
            return self._build_response(request, result, time.time() - start)
        finally:
            self._release(time.time() - start)

    async def process_async(self, request) -> dict:
        """Async lifecycle. Offloads `_do_work` to a worker thread.

        `_do_work` is intentionally still sync: in Phase 3 it becomes a
        `requests.post` to Ollama, which is also blocking. `asyncio.to_thread`
        is the right wrapper for both the sleep stub and the real call.
        """
        self._claim()
        start = time.time()
        try:
            print(f"[Worker {self.id}] received request {request.id} (async)")
            result = await asyncio.to_thread(self._do_work, request)
            return self._build_response(request, result, time.time() - start)
        finally:
            self._release(time.time() - start)

    def _do_work(self, request) -> str:
        """Step 12: full RAG + LLM pipeline via `infer()`.

        `infer()` is the coworkers' module-level entry point: it retrieves
        context from rag/retriever.py and forwards the augmented prompt to
        Ollama on llama3.2:1b. Returns the LLM's response text directly, or
        a tagged error string if Ollama is unreachable / errors out.

        Step 18: if `simulate_failure()` has been called, raise immediately
        so the scheduler's reassignment path takes over.
        """
        if self._dead:
            raise WorkerDeadError(f"worker {self.id} has been killed")
        result = infer(request.query)
        if result.get("status") != "success":
            return f"[infer-error] {result.get('error', 'unknown')}: {result.get('response', '')[:200]}"
        return result["response"]

    def _claim(self) -> None:
        with self._lock:
            self.inflight += 1

    def _release(self, elapsed: float) -> None:
        with self._lock:
            self.inflight -= 1
            self.processed_count += 1
            self.total_latency += elapsed

    def _build_response(self, request, result: str, latency: float) -> dict:
        return {
            "id": request.id,
            "result": result,
            "latency": latency,
            "worker_id": self.id,
        }

    # --- Step 8: per-worker async queue ---

    def start(self) -> None:
        """Spawn the consumer pool + the heartbeat task.

        Must be called from inside a running event loop.
        """
        if self._consumers:
            return
        self._queue = asyncio.Queue()
        self._consumers = [
            asyncio.create_task(self._consumer(), name=f"worker-{self.id}-c{i}")
            for i in range(self.concurrency)
        ]
        self.last_heartbeat = time.time()
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name=f"worker-{self.id}-hb"
        )

    async def stop(self) -> None:
        """Cancel heartbeat, send shutdown sentinels, wait for consumers to exit."""
        if not self._consumers or self._queue is None:
            return
        # Stop heartbeat first so it doesn't tick during shutdown.
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        # Drain consumers.
        for _ in self._consumers:
            await self._queue.put(None)
        await asyncio.gather(*self._consumers)
        self._consumers = []
        self._queue = None

    def simulate_failure(self) -> None:
        """Step 18: pretend this worker just crashed.

        Effects:
          * `_dead = True` -> any new `_do_work` call raises WorkerDeadError,
            triggering Step 17 reassignment on the scheduler side.
          * Heartbeat task is cancelled -> after STALE_THRESHOLD seconds,
            the scheduler's health monitor (Step 15) evicts this worker.
          * Consumers are NOT cancelled — in-flight requests fail naturally
            via WorkerDeadError on their next _do_work call.

        Idempotent: re-calling on an already-dead worker is a no-op.
        """
        if self._dead:
            return
        self._dead = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        print(f"[Worker {self.id}] SIMULATED FAILURE — heartbeat stopped, "
              f"new work will raise WorkerDeadError")

    async def _heartbeat_loop(self) -> None:
        """Tick `last_heartbeat` every HEARTBEAT_INTERVAL seconds until cancelled."""
        try:
            while True:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                self.last_heartbeat = time.time()
        except asyncio.CancelledError:
            return

    async def submit(self, request) -> dict:
        """Enqueue a request and await its response."""
        if self._queue is None:
            raise RuntimeError(f"Worker {self.id}: call start() before submit()")
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((request, future))
        return await future

    async def _consumer(self) -> None:
        """One of N consumers. Pulls items off the shared queue until the sentinel."""
        if self._queue is None:
            return
        while True:
            item = await self._queue.get()
            if item is None:                # shutdown sentinel
                self._queue.task_done()
                return
            request, future = item
            try:
                result = await self.process_async(request)
                # Guard: the future may already be cancelled if the caller
                # gave up on us (Step 16 wait_for timeout) — setting a result
                # on a settled future raises InvalidStateError.
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
            finally:
                self._queue.task_done()

    def __repr__(self) -> str:
        avg = (self.total_latency / self.processed_count) if self.processed_count else 0.0
        qsize = self._queue.qsize() if self._queue is not None else 0
        return (
            f"GPUWorker(id={self.id}, processed={self.processed_count}, "
            f"avg_latency={avg:.3f}s, inflight={self.inflight}, qsize={qsize})"
        )
