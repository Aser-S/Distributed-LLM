"""Structured logging for request lifecycle and cluster observability.

Emits JSON-formatted events to track:
  - request_received: scheduler receives a request
  - dispatch: scheduler picks a worker
  - enqueue: request added to worker queue
  - start_processing: worker begins processing (after queue wait)
  - completed: request finished successfully
  - timeout: request timed out
  - reassign: request reassigned to another worker
  - heartbeat: worker heartbeat update
  - evict_worker: worker evicted by health monitor
  - cluster_snapshot: periodic snapshot of cluster state

All events are timestamped and include relevant context (request_id, worker_id, metrics, etc.).
"""

import json
import time
from typing import Any


class StructuredLogger:
    """Lightweight JSON event logger for distributed system observability."""

    def __init__(self, enabled: bool = True, write_fn=None):
        self.enabled = enabled
        self.write_fn = write_fn or (lambda line: print(line))

    def emit(self, event: str, **context) -> None:
        """Emit a JSON event with timestamp and context."""
        if not self.enabled:
            return
        data = {
            "ts": time.time(),
            "event": event,
            **context,
        }
        self.write_fn(json.dumps(data))

    def request_received(self, request_id: int, strategy: str) -> None:
        self.emit("request_received", request_id=request_id, strategy=strategy)

    def dispatch(self, request_id: int, worker_id: int, attempt: int, pending: int, queue_depth: int) -> None:
        self.emit(
            "dispatch",
            request_id=request_id,
            worker_id=worker_id,
            attempt=attempt,
            pending=pending,
            queue_depth=queue_depth,
        )

    def enqueue(self, request_id: int, worker_id: int, queue_depth: int) -> None:
        self.emit("enqueue", request_id=request_id, worker_id=worker_id, queue_depth=queue_depth)

    def start_processing(self, request_id: int, worker_id: int, queue_wait_ms: float) -> None:
        self.emit(
            "start_processing",
            request_id=request_id,
            worker_id=worker_id,
            queue_wait_ms=round(queue_wait_ms, 2),
        )

    def completed(self, request_id: int, worker_id: int, latency_ms: float, queue_wait_ms: float) -> None:
        self.emit(
            "completed",
            request_id=request_id,
            worker_id=worker_id,
            latency_ms=round(latency_ms, 2),
            queue_wait_ms=round(queue_wait_ms, 2),
        )

    def timeout(self, request_id: int, worker_id: int, attempt: int, timeout_s: float) -> None:
        self.emit(
            "timeout",
            request_id=request_id,
            worker_id=worker_id,
            attempt=attempt,
            timeout_s=timeout_s,
        )

    def reassign(self, request_id: int, from_worker: int, to_worker: int, attempt: int) -> None:
        self.emit(
            "reassign",
            request_id=request_id,
            from_worker=from_worker,
            to_worker=to_worker,
            attempt=attempt,
        )

    def heartbeat(self, worker_id: int, age_s: float) -> None:
        self.emit("heartbeat", worker_id=worker_id, age_s=round(age_s, 2))

    def evict_worker(self, worker_id: int, reason: str) -> None:
        self.emit("evict_worker", worker_id=worker_id, reason=reason)

    def cluster_snapshot(self, strategy: str, worker_count: int, total_processed: int,
                        avg_latency_s: float, throughput_rps: float, avg_util_pct: float,
                        retry_rate: float, timeout_rate: float) -> None:
        self.emit(
            "cluster_snapshot",
            strategy=strategy,
            worker_count=worker_count,
            total_processed=total_processed,
            avg_latency_s=round(avg_latency_s, 4),
            throughput_rps=round(throughput_rps, 3),
            avg_util_pct=round(avg_util_pct, 2),
            retry_rate=round(retry_rate, 4),
            timeout_rate=round(timeout_rate, 4),
        )

    def overload_event(self, worker_id: int, queue_pressure: float, saturation_score: float, queue_depth: int) -> None:
        self.emit(
            "overload_event",
            worker_id=worker_id,
            queue_pressure=round(queue_pressure, 3),
            saturation_score=round(saturation_score, 3),
            queue_depth=queue_depth,
        )


# Global logger instance
_logger: StructuredLogger | None = None


def init_logger(enabled: bool = True) -> StructuredLogger:
    """Initialize and return the global structured logger."""
    global _logger
    _logger = StructuredLogger(enabled=enabled)
    return _logger


def get_logger() -> StructuredLogger:
    """Get the global logger instance, initializing if needed."""
    global _logger
    if _logger is None:
        _logger = StructuredLogger(enabled=True)
    return _logger


def disable_logging() -> None:
    """Disable all structured logging."""
    global _logger
    _logger = StructuredLogger(enabled=False)
