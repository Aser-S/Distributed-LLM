"""
Simple Ollama-only inference engine.

This file provides a minimal, easy-to-read API that only uses a local
Ollama model running at `http://localhost:11434`.

Usage (quick):
  1. Install and run Ollama (see README below).
  2. In Python: `from llm.inference_engine import infer; print(infer('Hello'))`

API:
  - `infer(prompt, max_tokens=256, context='')` -> single response dict
  - `stream_infer(prompt, max_tokens=256)` -> simple chunked response
  - `health_check()` -> checks Ollama + returns basic stats
  - `set_model(name)` -> change the Ollama model name used for requests
  - `get_stats()` / `reset_stats()` -> basic engine statistics

This is intentionally simple: no simulators, no remote providers.
"""

import time
import threading
import requests
import uuid
from typing import Dict, Any
from rag.retriever import retrieve_context 

# Configuration: update the model name if you want a different local model
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"


class InferenceEngine:
    """Lightweight stats collector for Ollama calls."""

    def __init__(self):
        self.model = OLLAMA_MODEL
        self.total_requests = 0
        self.total_tokens = 0
        self.total_latency = 0.0
        self.lock = threading.Lock()

    def _update_stats(self, latency_s: float, tokens: int) -> None:
        with self.lock:
            self.total_requests += 1
            self.total_latency += latency_s
            self.total_tokens += tokens

    def _get_stats(self) -> Dict[str, Any]:
        with self.lock:
            avg = (self.total_latency / self.total_requests) if self.total_requests else 0.0
            return {
                "model": self.model,
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "avg_latency_ms": round(avg * 1000, 2),
            }


_engine = InferenceEngine()

# could just call ollama_infer instead of infer()
# Example Usage: infer(prompt, max_tokens)
def infer(prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
    """Send a single synchronous request to the local Ollama server.

    Returns a dictionary with keys: request_id, model, prompt, response, tokens,
    performance, mode, status (success/error), and optionally error.
    """

    context = retrieve_context(prompt)

    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    content = f"Context: {context}\n\nUse this context strictly and override any other data you might have, the context is the most relevant\n\nQuestion: {prompt}"
    payload = {
        "model": _engine.model,
        "prompt": content,
        "stream": False,
        "options": {"num_predict": min(max_tokens, 512)},
    }
    try:
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60)   # setting timeout
        data = resp.json()
        if resp.status_code != 200:
            raise Exception(f"Ollama {resp.status_code}: {data}")
        # Ollama returns a `response` string and may include `eval_count` tokens
        text = data.get("response", "")
        latency = time.time() - start
        out_tok = data.get("eval_count", len(text.split()))
        in_tok = data.get("prompt_eval_count", len(content.split()))
        _engine._update_stats(latency, out_tok)
        return {
            "request_id": request_id,
            "prompt": prompt[:200],
            "response": text,
            "tokens": {"input": in_tok, "output": out_tok, "total": in_tok + out_tok},
            "performance": {"latency_ms": round(latency * 1000, 2)},
            "status": "success",
        }
    except requests.exceptions.ConnectionError:
        latency = time.time() - start
        _engine._update_stats(latency, 0)
        return {
            "request_id": request_id,
            "prompt": prompt[:200],
            "response": "[Ollama not running — start with: ollama serve]",
            "tokens": {"input": 0, "output": 0, "total": 0},
            "performance": {"latency_ms": round(latency * 1000, 2)},
            "status": "error",
            "error": "Ollama not reachable at localhost:11434",
        }
    except Exception as e:
        latency = time.time() - start
        _engine._update_stats(latency, 0)
        return {
            "request_id": request_id,
            "prompt": prompt[:200],
            "response": f"[Ollama Error: {str(e)[:200]}]",
            "tokens": {"input": 0, "output": 0, "total": 0},
            "performance": {"latency_ms": round(latency * 1000, 2)},
            "status": "error",
            "error": str(e),
        }

# might not need at all
def health_check() -> Dict[str, Any]:
    """Simple check for Ollama availability using the base URL."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/", timeout=10)
        if resp.status_code == 200:
            return {"status": "running"}
        else:
            return {"status": "error"}
    except requests.exceptions.RequestException:
        return {"status": "offline"}


def get_stats() -> Dict[str, Any]:
    return _engine._get_stats()


def reset_stats() -> None:
    with _engine.lock:
        _engine.total_requests = 0
        _engine.total_tokens = 0
        _engine.total_latency = 0.0