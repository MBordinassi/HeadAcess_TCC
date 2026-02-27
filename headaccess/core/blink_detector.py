"""Blink detection for triggering mouse clicks."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

from config import BlinkConfig


class BlinkAction(str, Enum):
    """Supported mouse actions triggered by eye blink."""

    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"


class BlinkDetector:
    """Detects unilateral blinks via Eye Aspect Ratio (EAR)."""

    def __init__(self, config: BlinkConfig) -> None:
        self._config = config
        self._left_closed_frames = 0
        self._right_closed_frames = 0
        self._cooldown = 0

    @staticmethod
    def _distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx * dx + dy * dy) ** 0.5

    def _ear(
        self,
        eye_points: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    ) -> float:
        left_corner, right_corner, top, bottom = eye_points
        horizontal = self._distance(left_corner, right_corner)
        vertical = self._distance(top, bottom)
        if horizontal <= 1e-6:
            return 1.0
        return vertical / horizontal

    def update(
        self,
        eye_data: Dict[str, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]],
    ) -> Optional[BlinkAction]:
        """Process eye landmarks and return click action when a blink is confirmed."""
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        left_ear = self._ear(eye_data["left"])
        right_ear = self._ear(eye_data["right"])

        left_closed = left_ear < self._config.ear_threshold
        right_closed = right_ear < self._config.ear_threshold

        self._left_closed_frames = self._left_closed_frames + 1 if left_closed else 0
        self._right_closed_frames = self._right_closed_frames + 1 if right_closed else 0

        if (
            self._left_closed_frames >= self._config.min_consecutive_frames
            and self._right_closed_frames == 0
        ):
            self._left_closed_frames = 0
            self._cooldown = self._config.cooldown_frames
            return BlinkAction.LEFT_CLICK

        if (
            self._right_closed_frames >= self._config.min_consecutive_frames
            and self._left_closed_frames == 0
        ):
            self._right_closed_frames = 0
            self._cooldown = self._config.cooldown_frames
            return BlinkAction.RIGHT_CLICK

        return None
