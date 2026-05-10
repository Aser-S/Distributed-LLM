"""GPU worker node - Phase 3 / Step 12: real LLM pipeline in `_do_work`.

`process()` is the sync lifecycle (used by round-robin LB).
`process_async()` adds an async surface that offloads `_do_work` to a thread.
`_do_work` calls `llm.inference.infer(query)`, which internally retrieves RAG
context and forwards the full prompt to Ollama. The TA's "use AI model, no
simulation" requirement is satisfied here.
"""

import asyncio
import time
import threading

from llm.inference import infer


class GPUWorker:
    DEFAULT_CONCURRENCY = 4

    def __init__(self, worker_id: int, concurrency: int = DEFAULT_CONCURRENCY):
        self.id = worker_id
        self.concurrency = max(1, concurrency)        # max parallel requests on this worker
        self.inflight = 0                              # number of requests currently being processed
        self.processed_count = 0
        self.total_latency = 0.0
        self._lock = threading.Lock()
        self._queue: asyncio.Queue | None = None       # created in start()
        self._consumers: list[asyncio.Task] = []       # N drain coroutines

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
        """
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
        """Spawn `concurrency` consumer coroutines sharing one queue.

        Must be called from inside a running event loop.
        """
        if self._consumers:
            return
        self._queue = asyncio.Queue()
        self._consumers = [
            asyncio.create_task(self._consumer(), name=f"worker-{self.id}-c{i}")
            for i in range(self.concurrency)
        ]

    async def stop(self) -> None:
        """Send one shutdown sentinel per consumer and wait for all to exit."""
        if not self._consumers or self._queue is None:
            return
        for _ in self._consumers:
            await self._queue.put(None)
        await asyncio.gather(*self._consumers)
        self._consumers = []
        self._queue = None

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
                future.set_result(result)
            except Exception as e:
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
