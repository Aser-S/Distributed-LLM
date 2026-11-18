# ⚡ Distributed-LLM

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Type](https://img.shields.io/badge/Type-Simulation-blueviolet?style=flat-square)

A simulation of a **distributed LLM serving architecture**: a scheduler hands incoming user requests to a load balancer, which spreads them across a pool of simulated GPU workers, with a RAG retriever in the pipeline.

## 🧩 Architecture

```
 Load generator (1000 users)
          │
          ▼
     Scheduler  ──►  Load Balancer  ──►  GPU Worker 0..3
   (master/)            (lb/)             (workers/)
                                              │
                                  RAG retriever + LLM inference
                                       (rag/, llm/)
```

| Module | Role |
|---|---|
| `client/load_generator.py` | Generates concurrent user load for the test |
| `master/scheduler.py` | Receives requests and dispatches them |
| `lb/load_balancer.py` | Chooses which worker handles each request |
| `workers/gpu_worker.py` | Simulated GPU worker |
| `rag/retriever.py` | Retrieval step for RAG |
| `llm/inference.py` | Inference logic |
| `common/models.py` | Shared data models |

## 🚀 Run it

```bash
git clone https://github.com/Aser-S/Distributed-LLM.git
cd Distributed-LLM/Code
python main.py
```

`main.py` spins up 4 simulated GPU workers and runs a load test with 1000 users.

## 🔭 Ideas for next steps

- Compare load-balancing strategies (round-robin vs. least-loaded)
- Report latency and throughput metrics
- Swap the simulated workers for real model servers
