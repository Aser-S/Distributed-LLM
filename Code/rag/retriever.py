"""RAG retriever - simple keyword-overlap retrieval over an in-memory KB.

NOTE TO TEAMMATES:
    This file was restored after rag/retriever.py was accidentally overwritten
    with the inference engine code. The implementation below is intentionally
    minimal (no new dependencies) but does *real* retrieval, not a hardcoded
    stub. Replace `_KNOWLEDGE_BASE` and `_score` with your intended vector-DB
    backend (FAISS / Chroma / Qdrant / etc.) when ready - the public contract
    `retrieve_context(query: str) -> str` must stay the same so
    `llm/inference.py` keeps working.

Public API (consumed by llm/inference.py):
    retrieve_context(query: str, top_k: int = 2) -> str
        Returns the top-k most relevant documents from the knowledge base,
        joined into a single context string. Empty string if nothing matches.
"""

from __future__ import annotations

import re


# Tiny in-memory knowledge base. Each entry is one document. Replace with a
# real corpus / vector store in production.
_KNOWLEDGE_BASE: list[str] = [
    "Distributed computing splits a workload across multiple machines that "
    "communicate over a network, solving problems faster than any single machine.",
    "Load balancing distributes incoming requests across multiple servers using "
    "strategies such as round robin, least connections, and load-aware routing.",
    "Round robin scheduling sends each request to the next worker in a fixed "
    "rotation, regardless of current load on each worker.",
    "Least connections scheduling sends each new request to the worker with "
    "the fewest active in-flight requests at the time of dispatch.",
    "GPU clusters parallelize deep-learning inference by spreading model "
    "computation across multiple GPUs to maximize throughput.",
    "Fault tolerance in distributed systems is the ability to keep operating "
    "correctly when individual nodes fail; common techniques include "
    "heartbeats, task reassignment, and replication.",
    "A large language model (LLM) generates text by predicting one token at a "
    "time conditioned on a preceding context window of tokens.",
    "Retrieval-Augmented Generation (RAG) enriches an LLM prompt with relevant "
    "passages retrieved from a knowledge base, improving factual grounding.",
    "An asyncio event loop in Python schedules coroutines cooperatively, "
    "allowing many I/O-bound tasks to make progress on a single thread.",
    "Ollama is a local runtime for running open-weights LLMs such as "
    "llama3.2 and serving them through an HTTP API on port 11434.",
    "Heartbeats are periodic 'I am alive' messages sent from a worker to its "
    "scheduler so the scheduler can detect failed nodes and reassign work.",
    "A request queue decouples producers and consumers: producers enqueue "
    "work without waiting, and consumers drain the queue at their own pace.",
]


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Lowercase + split on non-alphanumeric. Cheap and dependency-free."""
    return set(_TOKEN_RE.findall(text.lower()))


def _score(query_tokens: set[str], doc_tokens: set[str]) -> int:
    """Number of shared tokens between query and document.

    A simple Jaccard-numerator-style score is enough for a coursework demo;
    swap this for cosine similarity over embeddings when wiring a real vector
    store.
    """
    return len(query_tokens & doc_tokens)


def retrieve_context(query: str, top_k: int = 2) -> str:
    """Return the top-k most relevant documents, joined as a single string.

    Empty query or no token overlap returns "" - the LLM can still answer
    without context in that case.
    """
    if not query or not query.strip():
        return ""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return ""

    scored: list[tuple[int, str]] = []
    for doc in _KNOWLEDGE_BASE:
        s = _score(q_tokens, _tokenize(doc))
        if s > 0:
            scored.append((s, doc))

    if not scored:
        return ""

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return " ".join(doc for _, doc in scored[:top_k])
