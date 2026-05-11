# Distributed LLM — GPU Cluster Task Distribution

**CSE354 — Distributed Computing**
Efficient load balancing and GPU cluster task distribution for handling 1000+ concurrent LLM+RAG requests.

This project runs **real Ollama inference** (model `llama3.2:1b`) with a lightweight RAG retriever, while **simulating GPU resource behavior** (utilization, VRAM pressure, saturation) to support 1000+ concurrent-user stress testing on commodity hardware. A master scheduler routes incoming requests across a pool of asynchronous GPU worker nodes, with active health checks, per-request timeouts, automatic task reassignment, and structured observability.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │              Client (load gen)           │
                    │   async_load_generator (asyncio)         │
                    │   load_generator (threaded)              │
                    └────────────────────┬─────────────────────┘
                                         │ Request(id, query)
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │                Scheduler                 │
                    │  • registry: {worker_id -> GPUWorker}    │
                    │  • strategy: round_robin | least_conn    │
                    │             | load_aware | gpu_aware     │
                    │  • health monitor task (Step 15)         │
                    │  • per-request timeout + reassignment    │
                    │    (Steps 16-17)                         │
                    │  • metrics + structured logging          │
                    └────────────────────┬─────────────────────┘
                                         │ via LB.get_next_worker()
                                         ▼
                ┌────────────────────────────────────────────────┐
                │            Load Balancer (pluggable)           │
                │   round_robin  (lb/load_balancer.py)           │
                │   least_connections | load_aware | gpu_aware   │
                │   (master/scheduler.py)                        │
                └────────────────────┬───────────────────────────┘
                                     │ submit(request)
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
       ┌────────────┐         ┌────────────┐         ┌────────────┐
       │ GPUWorker 0│         │ GPUWorker 1│   ...   │ GPUWorker N│
       │            │         │            │         │            │
        │ asyncio.Q  │         │ asyncio.Q  │         │ asyncio.Q  │
        │ N consumers│         │ N consumers│         │ N consumers│
        │ heartbeat  │         │ heartbeat  │         │ heartbeat  │
        │ GPU sim    │         │ GPU sim    │         │ GPU sim    │
       └─────┬──────┘         └─────┬──────┘         └─────┬──────┘
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    ▼
                    ┌──────────────────────────────────┐
                    │   llm.inference.infer(prompt)    │
                    │   1. rag.retriever.retrieve_…    │
                    │   2. requests.post Ollama HTTP   │
                    │      llama3.2:1b @ :11434        │
                    └──────────────────────────────────┘
```

**Design pillars**

- **Three coexisting execution surfaces** on each worker — `process()` (sync), `process_async()` (async), and queue-backed `submit()` (actor-style). The first two are kept for the synchronous round-robin LB; production traffic goes through `submit()`.
- **`_do_work()` is a seam.** Started as a `time.sleep` stub in Phase 1 and was swapped for the real `infer()` call in Phase 3 without touching the surrounding lifecycle.
- **Hybrid realism.** The LLM and RAG are real; only GPU utilization/VRAM/queue pressure are simulated to enable large-scale stress testing without enterprise GPUs.
- **Scheduler owns liveness, LB owns ordering.** The load balancer is a pure selector — no state about who is healthy. The scheduler evicts dead workers, and the LB sees a clean registry on rebuild.
- **Absolute imports only**, matching `main.py`'s entry contract.

---

## Repository layout

```
Code/
├── main.py                  # entry point: sync | async | hardening modes
├── client/
│   ├── async_load_generator.py  # asyncio load/stress generator
│   └── load_generator.py        # threaded load generator (coworker-owned)
├── common/
│   ├── metrics.py           # rolling windows + p50/p95/p99
│   ├── gpu_sim.py           # GPU util/VRAM/saturation simulation
│   ├── structured_logging.py# JSON event logging
│   ├── hardening.py         # shutdown/recovery validation suite
│   └── models.py            # Request / Response dataclasses (coworker-owned)
├── lb/
│   └── load_balancer.py     # round-robin LB (coworker-owned)
├── llm/
│   └── inference.py         # infer() -> Ollama HTTP (coworker-owned)
├── rag/
│   └── retriever.py         # keyword-overlap retrieval over in-memory KB
├── master/
│   └── scheduler.py         # ★ Scheduler — owned by this work
├── workers/
│   └── gpu_worker.py        # ★ GPUWorker — owned by this work
├── test_step15.py           # health-check eviction
├── test_step16_17.py        # per-task timeout + reassignment
├── test_step18.py           # simulate_failure end-to-end
├── test_step19.py           # load-aware burst distribution
└── test_step20.py           # latency percentiles + throughput math
```

★ = files this project owns. All other modules are coworker-owned and treated as stable contracts.

---

## Implementation progress

| Phase | Step | Title | Status |
|-------|------|-------|--------|
| **1 — Basic Core** | 1 | GPUWorker skeleton | ✅ |
|  | 2 | Scheduler worker registry | ✅ |
|  | 3 | Worker lifecycle + `_do_work` seam | ✅ |
|  | 4 | Round-robin dispatch via LB | ✅ |
|  | 5 | `cluster_status()` snapshot | ✅ |
|  | 6 | End-to-end client → scheduler → worker | ✅ |
| **2 — Concurrency** | 7 | `process_async()` via `asyncio.to_thread` | ✅ |
|  | 8 | Per-worker `asyncio.Queue` | ✅ |
|  | 9 | N consumer coroutines sharing one queue | ✅ |
|  | 10 | Least-connections strategy | ✅ |
| **3 — Integration** | 11 | Wire `retrieve_context` into worker | ✅ |
|  | 12 | Wire `infer()` end-to-end | ✅ |
|  | 13 | Async pipeline through `handle_request_async` | ✅ |
| **4 — Fault Tolerance** | 14 | Worker heartbeats | ✅ |
|  | 15 | Scheduler-side health checks + eviction | ✅ |
|  | 16 | Per-task timeout (`asyncio.wait_for`) | ✅ |
|  | 17 | Task reassignment on timeout / exception | ✅ |
|  | 18 | Failure simulation (`simulate_failure()`) | ✅ |
| **5 — Optimization** | 19 | Load-aware scheduling (`pending` counter) | ✅ |
| **6 — Metrics & Observability** | 20 | p50/p95/p99 latency + throughput | ✅ |
|  | 21 | Structured observability (JSON logs + snapshots) | ✅ |
|  | 22 | Async stress tests (100/500/1000 users) | ✅ |
| **7 — GPU Simulation & Hardening** | 23 | GPU util/VRAM simulation + gpu_aware routing | ✅ |
|  | 24 | Hardening suite (shutdown/recovery/consistency) | ✅ |

---

## Public contracts

### `Scheduler`

```python
Scheduler(workers: list[GPUWorker] | None = None,
          strategy: str = "round_robin")
# strategies: "round_robin" | "least_connections" | "load_aware" | "gpu_aware"

# Registry
.register_worker(w) / .register_workers(ws) / .unregister_worker(id)
.get_workers() -> list[GPUWorker]

# Sync dispatch (Phase 1 LB)
.handle_request(request) -> dict

# Async dispatch (Phase 2+) — includes timeout + reassignment
async .handle_request_async(request) -> dict

# Lifecycle (Phase 2-4)
.start_workers()              # spawns consumers + heartbeats + health monitor
async .stop_workers()         # graceful drain + cancel monitors

# Observability
.cluster_status() -> dict     # per-worker stats + heartbeat ages + GPU/queue metrics

# Tunables (class attributes)
HEALTH_CHECK_INTERVAL = 2.0   # health sweep period (s)
STALE_THRESHOLD       = 3.0   # heartbeat age before eviction (s)
REQUEST_TIMEOUT       = 30.0  # single-attempt deadline (s)
MAX_ATTEMPTS          = 3     # 1 initial + up to 2 reassignments
THROUGHPUT_WINDOW     = 10.0  # window for req/s in cluster_status (s)
```

### `GPUWorker`

```python
GPUWorker(worker_id: int, concurrency: int = 4)

# Execution
.process(request) -> dict
async .process_async(request) -> dict
async .submit(request) -> dict        # enqueue + await

# Lifecycle
.start()           # spawns N consumer tasks + heartbeat task
async .stop()      # sentinel-drains the queue, cancels heartbeat

# Failure simulation (Step 18)
.simulate_failure()                   # kill switch, idempotent

# State
.inflight: int                        # consumer-side (Step 7)
.pending: int                         # scheduler-side dispatch counter (Step 19)
.busy: bool                           # @property: inflight > 0
.processed_count: int
.total_latency: float
.last_heartbeat: float
.recent: deque[(completed_at, latency)]   # Step 20 metrics window
```

---

## Running it

### Prerequisites

- Python 3.12
- `requests` (`pip install requests`)
- [Ollama](https://ollama.com) running locally with `llama3.2:1b` pulled:
  ```powershell
  ollama pull llama3.2:1b
  curl http://localhost:11434/api/tags    # sanity check
  ```

### End-to-end demo (sync baseline)

```powershell
cd Code
python main.py --mode sync
```

`python main.py` also works (defaults to sync mode).

Expected output (per-request latency depends on your CPU/GPU):

```
[Scheduler] registered worker 0 (total=1)
...
[Worker 0] received request 0
[Client] Response 0 | Latency: 2.49s
...
--- CLUSTER STATUS ---
workers=4, total_processed=4, cluster_avg_latency=2.71s
  worker 0: processed=1, avg_latency=2.49s
  worker 1: processed=1, avg_latency=2.56s
  worker 2: processed=1, avg_latency=2.79s
  worker 3: processed=1, avg_latency=2.99s
```

### Async stress testing (100/500/1000 users)

```powershell
cd Code
python main.py --mode async
```

To run a full benchmark sweep:

```python
from client.async_load_generator import run_scalability_benchmark
# benchmark = await run_scalability_benchmark(scheduler, [100, 500, 1000], duration_s=30)
```

### Hardening suite (shutdown/recovery/consistency)

```powershell
cd Code
python main.py --mode hardening
```

### Fault-tolerance test suite

These tests don't touch Ollama — they stub `_do_work` so they finish in seconds:

```powershell
python test_step15.py         # health-check eviction
python test_step16_17.py      # timeout + reassignment
python test_step18.py         # simulate_failure end-to-end
python test_step19.py         # load-aware burst distribution
python test_step20.py         # p50/p95 + throughput math
```

Each should print `[test] PASS` lines and exit cleanly.

---

## Design decisions worth knowing

1. **`infer()` already calls `retrieve_context` internally.** The worker does *not* call the retriever explicitly — that would double-retrieve. Step 11 was a transitional checkpoint and was rolled into Step 12.

2. **`inflight: int` replaced `busy: bool`** in Step 7 so the least-connections balancer can rank workers by depth rather than just busy/idle. `busy` survives as a `@property` for compatibility with the LB-side `dispatch()` path.

3. **Health checks don't drive request retries; timeouts do.** A slow worker is not necessarily dead. Step 15 (heartbeat-based eviction) and Steps 16-17 (per-request timeout + reassignment) are independent signals.

4. **`asyncio.wait_for` cancels the awaiter, not the worker.** When a request times out, the consumer keeps running and eventually finishes — but the future is already settled, so the consumer's `set_result()` is guarded by `if not future.done()`. This wastes capacity but never corrupts state. A cleaner cancellation propagation is a Phase 5 candidate.

5. **`simulate_failure()` kills the heartbeat AND flips `_dead`.** That way a single hook exercises both fault-tolerance paths: Step 17 reassignment fires on the next request, and Step 15 eviction fires after `STALE_THRESHOLD`.

6. **The load balancer is treated as a stable, coworker-owned black box.** Least-connections and load-aware live in `master/scheduler.py`, not `lb/load_balancer.py`, so we never modify the coworker file.

7. **`inflight` vs `pending` are two different signals (Step 19).**
   - `inflight` is incremented inside the worker, *after* a consumer dequeues the request — it counts what's actually executing.
   - `pending` is incremented by the scheduler, *before* `submit()` is awaited — it counts what's been committed but not yet completed.
   The load-aware balancer ranks by `pending` because that is the only signal that updates fast enough to spread an async burst across workers.

---

## Advanced phases (20–24) implemented

- **Metrics**: per-worker p50/p95/p99 + throughput, scheduler retry/timeout/overload rates.
- **Observability**: JSON structured logging + periodic cluster snapshots.
- **Stress testing**: async load generator with 100/500/1000-user benchmarks.
- **GPU simulation**: util/VRAM/saturation/overload scoring + gpu_aware routing.
- **Hardening**: graceful shutdown, recovery validation, consistency checks.

### Resolved: least-connections burst bias

Step 19 closed this. The scheduler now increments `worker.pending` the moment a worker is chosen — before `submit()` is awaited — so a burst of N async requests gets distributed evenly across workers instead of all landing on worker 0. See [test_step19.py](Code/test_step19.py) for a before/after comparison between strategies.

---

## Authors

- **Scheduler + GPU workers:** Omar ALashker (this work).
- **RAG retriever, LLM inference, load balancer, client load generator, common models:** coworkers.
