"""Simulated GPU resource state for realistic distributed scheduling without enterprise GPUs.

Each worker tracks simulated GPU metrics:
  - util_pct: simulated GPU utilization (0-100%)
  - vram_used_mb: simulated VRAM in use
  - queue_pressure: score based on queue depth vs concurrency
  - saturation_score: composite metric (0-1) from util + queue pressure
  - overload_score: threshold trigger for overload events (0-1)

Formulas use lightweight heuristics and exponential moving averages (EMA) to smooth spikes.
No external dependencies; update per heartbeat or per-request.
"""

import time
import math
from dataclasses import dataclass


# Default GPU specs (simulating a modest GPU cluster node)
DEFAULT_GPU_VRAM_MB = 8192  # 8 GB
DEFAULT_BASE_VRAM_MB = 2048  # OS + model overhead


@dataclass
class SimulatedGPUState:
    """Point-in-time snapshot of simulated GPU state."""
    util_pct: float  # 0-100%
    vram_used_mb: int
    queue_pressure: float  # 0-1
    saturation_score: float  # 0-1
    overload_score: float  # 0-1 (>0.8 = overloaded)


class GPUSimulator:
    """Lightweight stateful GPU simulator for one worker.

    Updates EMA-smoothed utilization and VRAM based on inflight count,
    queue depth, and request history. Designed for scheduling decisions,
    not for accurate GPU emulation.
    """

    def __init__(
        self,
        worker_id: int,
        concurrency: int = 4,
        total_vram_mb: int = DEFAULT_GPU_VRAM_MB,
        base_vram_mb: int = DEFAULT_BASE_VRAM_MB,
        request_vram_mb: int = 500,
        queue_vram_per_item_mb: int = 128,
    ):
        self.worker_id = worker_id
        self.concurrency = max(1, concurrency)
        self.total_vram_mb = total_vram_mb
        self.base_vram_mb = base_vram_mb
        self.request_vram_mb = request_vram_mb
        self.queue_vram_per_item_mb = queue_vram_per_item_mb

        # EMA smoothing factor for util (0=no smoothing, 1=ignore new data)
        self.ema_alpha = 0.3  # blend 30% new, 70% old

        # Current state
        self.util_pct = 0.0
        self.vram_used_mb = base_vram_mb
        self.last_update = time.time()

    def update(self, inflight: int, queue_depth: int) -> SimulatedGPUState:
        """Update simulated state based on current load; return snapshot."""
        now = time.time()
        elapsed = max(0.01, now - self.last_update)
        self.last_update = now

        # --- GPU Utilization ---
        # Base: inflight requests push util up; idle time decays it.
        # Each inflight request: ~20% util per slot. Queue adds ~4% per item.
        target_util = 20.0 * inflight + 4.0 * queue_depth + 5.0  # +5 for baseline noise
        target_util = min(100.0, max(0.0, target_util))

        # EMA: blend new target with old value
        self.util_pct = (
            self.ema_alpha * target_util + (1.0 - self.ema_alpha) * self.util_pct
        )

        # Add small random jitter (±5%) for realism
        jitter = (hash((self.worker_id, int(now))) % 11 - 5) * 0.5  # -5% to +5%
        self.util_pct = max(0.0, min(100.0, self.util_pct + jitter))

        # --- VRAM Usage ---
        vram_inflight = self.base_vram_mb + inflight * self.request_vram_mb
        vram_queued = queue_depth * self.queue_vram_per_item_mb
        self.vram_used_mb = int(vram_inflight + vram_queued)
        self.vram_used_mb = min(self.total_vram_mb, self.vram_used_mb)

        # --- Queue Pressure ---
        # 0.0 = empty; 1.0 = queue >= 2*concurrency
        queue_capacity = 2.0 * self.concurrency
        queue_pressure = min(1.0, queue_depth / max(1.0, queue_capacity))

        # --- Saturation Score ---
        # Composite: 60% GPU util + 40% queue pressure
        util_norm = self.util_pct / 100.0
        saturation_score = 0.6 * util_norm + 0.4 * queue_pressure
        saturation_score = min(1.0, max(0.0, saturation_score))

        # --- Overload Score ---
        # Trigger when saturation > 0.8; scale 0-1 above that threshold.
        overload_threshold = 0.8
        if saturation_score >= overload_threshold:
            overload_score = min(1.0, (saturation_score - overload_threshold) / 0.2)
        else:
            overload_score = 0.0

        return SimulatedGPUState(
            util_pct=round(self.util_pct, 2),
            vram_used_mb=self.vram_used_mb,
            queue_pressure=round(queue_pressure, 3),
            saturation_score=round(saturation_score, 3),
            overload_score=round(overload_score, 3),
        )

    def vram_available_mb(self) -> int:
        """Available VRAM for new requests (may be negative if oversubscribed)."""
        return self.total_vram_mb - self.vram_used_mb

    def vram_utilization_pct(self) -> float:
        """VRAM utilization as a percentage."""
        if self.total_vram_mb <= 0:
            return 0.0
        return 100.0 * self.vram_used_mb / self.total_vram_mb
