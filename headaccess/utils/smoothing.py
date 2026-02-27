"""Moving average smoothing utilities."""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple


class MovingAverageFilter:
    """Applies moving average to 2D coordinates."""

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.window_size = window_size
        self._buffer: Deque[Tuple[float, float]] = deque(maxlen=window_size)

    def add(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Add a point and return the smoothed output."""
        self._buffer.append(point)
        avg_x = sum(p[0] for p in self._buffer) / len(self._buffer)
        avg_y = sum(p[1] for p in self._buffer) / len(self._buffer)
        return avg_x, avg_y

    def reset(self) -> None:
        """Clear internal state."""
        self._buffer.clear()
