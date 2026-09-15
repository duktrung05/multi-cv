from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from threading import Lock
from typing import Dict, Iterator


class MetricsRegistry:
    def __init__(self) -> None:
        self._count = defaultdict(int)
        self._sum_ms = defaultdict(float)
        self._lock = Lock()

    @contextmanager
    def timer(self, metric: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            with self._lock:
                self._count[metric] += 1
                self._sum_ms[metric] += elapsed_ms

    def incr(self, metric: str, value: int = 1) -> None:
        with self._lock:
            self._count[metric] += value

    def snapshot(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        with self._lock:
            for k, cnt in self._count.items():
                out[f"{k}.count"] = float(cnt)
                if self._sum_ms.get(k):
                    out[f"{k}.avg_ms"] = self._sum_ms[k] / max(cnt, 1)
        return out
