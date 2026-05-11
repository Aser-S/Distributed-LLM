from __future__ import annotations

from collections import deque
import math
import time


class RollingWindow:
    """Timestamped rolling window for metric samples."""

    def __init__(self, maxlen: int = 200, window_s: float | None = None):
        self.samples: deque[tuple[float, float]] = deque(maxlen=maxlen)
        self.window_s = window_s

    def add(self, value: float, ts: float | None = None) -> None:
        ts = time.time() if ts is None else ts
        self.samples.append((ts, value))
        self._trim(ts)

    def _trim(self, now: float) -> None:
        if self.window_s is None:
            return
        cutoff = now - self.window_s
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def values(self, now: float | None = None) -> list[float]:
        now = time.time() if now is None else now
        self._trim(now)
        return [value for _, value in self.samples]

    def count_recent(self, window_s: float, now: float | None = None) -> int:
        now = time.time() if now is None else now
        cutoff = now - window_s
        return sum(1 for ts, _ in self.samples if ts >= cutoff)


def percentiles(values: list[float], percentiles: tuple[int, ...]) -> tuple[float, ...]:
    """Return nearest-rank percentiles (ceil-based) over a sorted copy."""
    if not values:
        return tuple(0.0 for _ in percentiles)
    s = sorted(values)
    n = len(s)
    out = []
    for p in percentiles:
        idx = min(n - 1, max(0, math.ceil(p / 100.0 * n)))
        out.append(s[idx])
    return tuple(out)
