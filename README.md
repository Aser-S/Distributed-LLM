# Distributed LLM — GPU Cluster Task Distribution

<p align="center">
  <strong>CSE354: Distributed Computing — Ain Shams University, Faculty of Engineering</strong><br>
  Efficient load balancing and GPU cluster task distribution for handling 1000+ concurrent LLM+RAG requests.
</p>

<p align="center">
  <a href="#architecture">Architecture</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#implementation-phases">Phases</a> •
  <a href="#api-reference">API</a> •
  <a href="#testing">Testing</a> •
  <a href="#team">Team</a>
</p>

---

## Overview

This project implements a production-oriented distributed system for serving Large Language Model (LLM) inference requests at scale. It combines **real LLM inference** via Ollama (Llama 3.2 1B) with a **Retrieval-Augmented Generation (RAG)** pipeline powered by ChromaDB vector search, while simulating GPU resource behavior to enable stress testing on commodity hardware without enterprise GPUs.

The architecture follows a **master-worker pattern** with a pluggable load balancing layer, async request queues, heartbeat-based health monitoring, automatic task reassignment, and structured observability.

### Key Capabilities

- **1,000+ concurrent users** via async load generation with bounded concurrency
- **Real LLM inference** through Ollama HTTP API (not simulated)
- **RAG enrichment** with ChromaDB + Ollama embeddings + keyword fallback
- **4 load balancing strategies**: Round Robin, Least Connections, Load-Aware, GPU-Aware
- **Fault tolerance**: heartbeats, health check eviction, per-request timeout, automatic reassignment
- **GPU simulation**: utilization, VRAM, saturation scoring for scheduling decisions
- **Structured observability**: JSON event logging, p50/p95/p99 latency, throughput metrics
- **System hardening**: graceful shutdown, consistency validation, recovery verification

---

## Architecture

```
                    +------------------------------------------+
                    |           Client (Load Generator)        |
                    |    async_load_generator (asyncio)        |
                    |    load_generator (threaded)             |
                    +-------------------+----------------------+
                                        | Request(id, query)
                                        v
                    +------------------------------------------+
                    |                Scheduler                 |
                    |  + Registry: {worker_id -> GPUWorker}    |
                    |  + Strategy: round_robin | least_conn    |
                    |              load_aware | gpu_aware      |
                    |  + Health monitor task (heartbeat)       |
                    |  + Per-request timeout + reassignment    |
                    |  + Metrics + structured logging          |
                    +-------------------+----------------------+
                                        | via LB.get_next_worker()
                                        v
                +-----------------------------------------------+
                |           Load Balancer (Pluggable)           |
                |    round_robin  (lb/load_balancer.py)          |
                |    least_connections | load_aware | gpu_aware  |
                |    (master/scheduler.py)                       |
                +-------------------+---------------------------+
                                    | submit(request)
              +---------------------+---------------------+
              v                     v                     v
       +-------------+      +-------------+      +-------------+
       | GPUWorker 0 |      | GPUWorker 1 | ...  | GPUWorker N |
       |             |      |             |      |             |
       | asyncio.Q   |      | asyncio.Q   |      | asyncio.Q   |
       | N consumers |      | N consumers |      | N consumers |
       | heartbeat   |      | heartbeat   |      | heartbeat   |
       | GPU sim     |      | GPU sim     |      | GPU sim     |
       +------+------+      +------+------+      +------+------+
              |                     |                     |
              +---------------------+---------------------+
                                    v
                    +-----------------------------------+
                    |    llm.inference.infer(prompt)    |
                    |    1. rag.retriever.retrieve_...  |
                    |    2. requests.post Ollama HTTP   |
                    |       llama3.2:1b @ :11434        |
                    +-----------------------------------+
```

### Design Principles

1. **Three Execution Surfaces**: Each worker supports `process()` (sync), `process_async()` (async), and queue-backed `submit()` (actor-style). Production traffic uses `submit()`; sync methods remain for backward compatibility and testing.

2. **`_do_work()` as a Seam**: The worker's processing method was intentionally separated from the lifecycle wrapper. Started as a `time.sleep` stub and swapped for real `infer()` calls without touching surrounding code.

3. **Hybrid Realism**: The LLM and RAG are real production services. Only GPU utilization/VRAM/queue pressure are simulated to enable large-scale testing without enterprise GPU hardware.

4. **Scheduler Owns Liveness, LB Owns Ordering**: The load balancer is a pure selector with no health state. The scheduler evicts dead workers; the LB sees a clean registry on every rebuild.

5. **Absolute Imports Only**: All modules use absolute imports matching `main.py`'s entry contract for consistent module resolution.

---

## Features

### Load Balancing Strategies

| Strategy | Selection Criteria | Best For |
|----------|-------------------|----------|
| `round_robin` | Cyclical rotation | Even distribution, low overhead |
| `least_connections` | Fewest in-flight requests | Variable processing times |
| `load_aware` | Lowest pending count | Async burst distribution |
| `gpu_aware` | Lowest GPU saturation | Resource-constrained environments |

### Fault Tolerance

| Mechanism | Detection | Response |
|-----------|-----------|----------|
| Heartbeat Monitoring | `last_heartbeat > 3s` | Worker evicted from registry |
| Per-Request Timeout | `asyncio.wait_for > 30s` | Reassignment to different worker |
| Task Reassignment | Timeout or `WorkerDeadError` | Retry on untried worker (up to 3 attempts) |
| Failure Simulation | `simulate_failure()` call | Worker marked dead for testing |

### GPU Simulation Metrics

- **GPU Utilization** (0-100%): EMA-smoothed based on inflight requests and queue depth
- **VRAM Usage**: Base allocation + per-request overhead + queue memory
- **Queue Pressure** (0-1): Queue depth relative to 2x concurrency capacity
- **Saturation Score** (0-1): Composite of 60% normalized GPU util + 40% queue pressure
- **Overload Score** (0-1): Threshold-triggered indicator for load shedding

### Observability

Structured JSON logging emits the following event types:
- `request_received` — scheduler receives request
- `dispatch` — worker selected for request
- `completed` — request finished successfully
- `timeout` — request exceeded deadline
- `reassign` — request moved to different worker
- `evict_worker` — worker removed by health monitor
- `cluster_snapshot` — periodic cluster state summary

---

## Quick Start

### Prerequisites

- Python 3.12+
- `requests` library (`pip install requests`)
- `chromadb` library (`pip install chromadb`)
- [Ollama](https://ollama.com) installed and running locally

### Install Ollama Models

```bash
# Pull the LLM
ollama pull llama3.2:1b

# Pull the embeddings model for RAG
ollama pull nomic-embed-text

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### Run the System

```bash
cd Code

# Sync baseline test (4 users)
python main.py --mode sync

# Async stress test (100 users, 30 seconds)
python main.py --mode async

# Hardening validation suite
python main.py --mode hardening
```

### Run Fault Tolerance Tests

These tests don't require Ollama — they use stubs and complete in seconds:

```bash
cd Code
python test_step15.py         # Health-check eviction
python test_step16_17.py      # Timeout + reassignment
python test_step18.py         # simulate_failure end-to-end
python test_step19.py         # Load-aware burst distribution
python test_step20.py         # Latency percentiles + throughput
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server endpoint |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embeddings model for RAG |
| `CHROMA_PERSIST_DIR` | `Code/data/chroma` | ChromaDB storage path |

---

## Project Structure

```
Code/
├── main.py                          # Entry point: sync | async | hardening
├── client/
│   ├── async_load_generator.py      # Async stress testing (100/500/1000 users)
│   └── load_generator.py            # Threaded load generator
├── common/
│   ├── models.py                    # Request / Response dataclasses
│   ├── metrics.py                   # Rolling windows + p50/p95/p99
│   ├── gpu_sim.py                   # GPU util/VRAM/saturation simulation
│   ├── structured_logging.py        # JSON event logging
│   └── hardening.py                 # Shutdown/recovery validation suite
├── lb/
│   └── load_balancer.py             # Round-robin load balancer
├── llm/
│   └── inference.py                 # infer() -> Ollama HTTP API
├── rag/
│   └── retriever.py                 # ChromaDB + keyword retrieval
├── master/
│   └── scheduler.py                 # Master scheduler with registry, health, metrics
├── workers/
│   └── gpu_worker.py                # GPU worker with queues, heartbeats, GPU sim
├── test_step15.py                   # Health-check eviction test
├── test_step16_17.py                # Timeout + reassignment test
├── test_step18.py                   # Failure simulation test
├── test_step19.py                   # Load-aware burst distribution test
└── test_step20.py                   # Latency percentile + throughput test
```

---

## Implementation Phases

| Phase | Steps | Focus Area |
|-------|-------|------------|
| **1 — Basic Core** | 1-6 | Worker skeleton, registry, round-robin dispatch, end-to-end flow |
| **2 — Concurrency** | 7-10 | Async processing, per-worker queues, N consumers, least-connections |
| **3 — Integration** | 11-13 | RAG + LLM wiring, async pipeline through scheduler |
| **4 — Fault Tolerance** | 14-18 | Heartbeats, health checks, timeout, reassignment, failure simulation |
| **5 — Optimization** | 19 | Load-aware scheduling with pending counter |
| **6 — Metrics & Observability** | 20-22 | p50/p95/p99, structured logging, 1000-user stress tests |
| **7 — GPU Simulation & Hardening** | 23-24 | GPU metrics, graceful shutdown, recovery validation |

---

## API Reference

### Scheduler

```python
from master.scheduler import Scheduler
from workers.gpu_worker import GPUWorker

# Create workers and scheduler
workers = [GPUWorker(i) for i in range(4)]
scheduler = Scheduler(workers, strategy="load_aware")

# Start workers (spawns consumers + heartbeats + health monitor)
scheduler.start_workers()

# Sync dispatch
result = scheduler.handle_request(Request(id=1, query="Hello"))

# Async dispatch (with timeout and reassignment)
result = await scheduler.handle_request_async(Request(id=2, query="World"))

# Get cluster status
status = scheduler.cluster_status()
# Returns: worker_count, total_processed, cluster_avg_latency,
#          cluster_throughput_rps, per-worker stats, GPU metrics

# Graceful shutdown
await scheduler.stop_workers()
```

### GPUWorker

```python
from workers.gpu_worker import GPUWorker

worker = GPUWorker(worker_id=0, concurrency=4)

# Lifecycle
worker.start()              # Spawn consumers + heartbeat
await worker.stop()         # Graceful drain + cancel

# Execution
result = worker.process(request)           # Sync
result = await worker.process_async(request)  # Async
result = await worker.submit(request)      # Queue-based

# Failure simulation (testing)
worker.simulate_failure()   # Kill switch for fault tolerance tests

# State inspection
print(worker.busy)              # True if inflight > 0
print(worker.processed_count)   # Completed requests
print(worker.pending)           # Dispatched but not completed
```

### Load Test Generator

```python
from client.async_load_generator import run_async_load_test, run_scalability_benchmark

# Single load test
summary = await run_async_load_test(
    scheduler,
    num_users=100,          # Concurrent request slots
    pattern="sustained",    # "sustained" | "burst" | "wave"
    duration_s=30.0,        # Test duration
    print_progress=True
)
# summary: total_requests, successful, failed, throughput_rps,
#          avg_latency, p50_latency, p95_latency, p99_latency

# Full benchmark sweep
await run_scalability_benchmark(scheduler, [100, 500, 1000])
```

### RAG Retrieval

```python
from rag.retriever import retrieve_context

# Retrieve relevant context for a query
context = retrieve_context("What is distributed computing?", top_k=2)
# Returns: concatenated relevant documents as a string

# Used internally by llm.inference.infer()
```

---

## Testing

### Component Tests

Each fault-tolerance mechanism has a dedicated test file:

```bash
python test_step15.py    # Verify health monitor evicts stale workers
python test_step16_17.py # Verify timeout triggers reassignment
python test_step18.py    # Verify simulate_failure triggers eviction
python test_step19.py    # Verify load-aware distributes bursts evenly
python test_step20.py    # Verify latency percentile calculations
```

### Integration Tests

Run the full system with different modes:

```bash
# Baseline (small, fast)
python main.py --mode sync

# Stress test (moderate load)
python main.py --mode async

# Hardening (shutdown/recovery checks)
python main.py --mode hardening
```

### Benchmarking

For formal performance evaluation:

```python
import asyncio
from client.async_load_generator import run_scalability_benchmark

async def benchmark():
    workers = [GPUWorker(i) for i in range(4)]
    scheduler = Scheduler(workers, strategy="gpu_aware")
    scheduler.start_workers()
    
    try:
        await run_scalability_benchmark(
            scheduler,
            concurrency_levels=[100, 500, 1000],
        )
    finally:
        await scheduler.stop_workers()

asyncio.run(benchmark())
```

---

## Design Decisions

1. **No Double Retrieval**: `infer()` already calls `retrieve_context()` internally. The worker calls `infer()` directly — never the retriever — to avoid redundant context retrieval.

2. **`inflight` vs `pending`**: `inflight` counts requests currently executing (consumer-side); `pending` counts requests dispatched but not yet completed (scheduler-side). The load-aware balancer uses `pending` because it updates faster for burst distribution.

3. **Health Checks Don't Drive Retries**: Slow workers are not immediately evicted. Per-request timeout handles transient slowdowns; heartbeat eviction handles permanent failures. This separation prevents flapping.

4. **Timeout Cancels Awaiter, Not Worker**: When a request times out, the scheduler's await is cancelled but the worker consumer continues. The future guard (`if not future.done()`) prevents state corruption. This trades some wasted capacity for correctness.

5. **Load Balancer as Black Box**: New strategies (least-connections, load-aware, gpu-aware) were added to `master/scheduler.py`, not `lb/load_balancer.py`, preserving the coworker-owned file as a stable contract.

---

## Performance Notes

- **Single Ollama Instance**: Throughput is ultimately limited by the single Ollama process. For higher throughput, run multiple Ollama instances behind the scheduler with more workers.
- **Model Size**: Llama 3.2 1B provides fast responses for demonstration. Larger models will increase latency proportionally.
- **Concurrent Users**: The async design handles 1000+ concurrent users efficiently, but Ollama's processing queue determines actual throughput.
- **GPU Simulation**: GPU metrics are simulated for scheduling decisions. They don't reflect real GPU state but provide realistic scheduling signals.

---

## Team

| Member | Responsibility |
|--------|---------------|
| **Omar ALashker** | Master Scheduler, GPU Workers, Fault Tolerance, System Integration |
| **Aser Sherif** | Project Coordination, RAG and LLM Integration |
| **Muhammed Yassin** | Load Balancer, Client Components |
| **Ziad Tamer** | Client Load Generation |
| **Mohamed Beder** | RAG Implementation, LLM Integration |

---

## Course Information

- **Course**: CSE354 — Distributed Computing
- **Institution**: Ain Shams University, Faculty of Engineering
- **Semester**: 2nd Semester 2025/2026
- **Project Type**: Group Project (2-5 students)

---

## License

This project was developed for academic purposes as part of the CSE354 coursework at Ain Shams University.
