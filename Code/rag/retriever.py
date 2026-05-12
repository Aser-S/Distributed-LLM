"""RAG retriever backed by ChromaDB + Ollama embeddings.

Public API (consumed by llm/inference.py):
    retrieve_context(query: str, top_k: int = 2) -> str

This keeps the original contract while swapping the in-memory keyword
retrieval for a real vector DB. If ChromaDB is not installed or Ollama
embeddings are unavailable, it falls back to the keyword retriever with
an explicit warning.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import requests


# Ollama config (embeddings)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Chroma config (persistent local store)
CHROMA_COLLECTION = "rag_kb"
CHROMA_PERSIST_DIR = Path(
    os.getenv(
        "CHROMA_PERSIST_DIR",
        Path(__file__).resolve().parents[1] / "data" / "chroma",
    )
)


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
_collection_lock = threading.Lock()
_collection = None


def _tokenize(text: str) -> set[str]:
    """Lowercase + split on non-alphanumeric. Cheap and dependency-free."""
    return set(_TOKEN_RE.findall(text.lower()))


def _score(query_tokens: set[str], doc_tokens: set[str]) -> int:
    """Number of shared tokens between query and document."""
    return len(query_tokens & doc_tokens)


def _keyword_retrieve(query: str, top_k: int) -> str:
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


def _ollama_embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Ollama's /api/embeddings endpoint."""
    embeddings: list[list[float]] = []
    for text in texts:
        payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("invalid embedding response from Ollama")
        embeddings.append(embedding)
    return embeddings


def _get_collection():
    """Get or initialize the Chroma collection, seeding docs on first use."""
    global _collection
    if _collection is not None:
        return _collection
    with _collection_lock:
        if _collection is not None:
            return _collection
        try:
            import chromadb  # type: ignore
        except ImportError:
            print("[RAG] chromadb not installed; falling back to keyword retrieval.")
            return None

        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
        if collection.count() == 0:
            try:
                embeddings = _ollama_embed(_KNOWLEDGE_BASE)
            except requests.exceptions.RequestException as exc:
                print(f"[RAG] Ollama embeddings failed ({exc}); falling back to keyword retrieval.")
                return None
            collection.add(
                ids=[f"doc-{i}" for i in range(len(_KNOWLEDGE_BASE))],
                documents=_KNOWLEDGE_BASE,
                embeddings=embeddings,
            )
        _collection = collection
        return _collection


def retrieve_context(query: str, top_k: int = 2) -> str:
    """Return the top-k most relevant documents, joined as a single string."""
    if not query or not query.strip():
        return ""
    top_k = max(1, int(top_k))

    collection = _get_collection()
    if collection is None:
        return _keyword_retrieve(query, top_k)

    try:
        query_embedding = _ollama_embed([query])[0]
    except requests.exceptions.RequestException as exc:
        print(f"[RAG] Ollama embeddings failed ({exc}); falling back to keyword retrieval.")
        return _keyword_retrieve(query, top_k)

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    if not isinstance(results, dict):
        return ""
    documents = results.get("documents")
    if not documents or not isinstance(documents, list):
        return ""
    first = documents[0] if documents else []
    if not isinstance(first, list):
        return ""
    docs = [d for d in first if isinstance(d, str)]
    return " ".join(docs) if docs else ""
