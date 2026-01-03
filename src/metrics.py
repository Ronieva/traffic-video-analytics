from __future__ import annotations
import time
from collections import deque

class PerfMeter:
    """Tracks FPS and latency with a rolling window."""
    def __init__(self, window: int = 60) -> None:
        self.window = window
        self.lat_ms = deque(maxlen=window)
        self._t0 = None

    def tic(self) -> None:
        self._t0 = time.perf_counter()

    def toc(self) -> float:
        if self._t0 is None:
            return 0.0
        dt = (time.perf_counter() - self._t0) * 1000.0
        self.lat_ms.append(dt)
        return dt

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.lat_ms) / len(self.lat_ms) if self.lat_ms else 0.0

    @property
    def fps(self) -> float:
        # approximate FPS from avg latency
        lat = self.avg_latency_ms
        return 1000.0 / lat if lat > 1e-9 else 0.0

