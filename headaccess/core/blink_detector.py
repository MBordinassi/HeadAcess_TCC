"""Blink detection for hold/release mouse control."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple

from config import BlinkConfig


class BlinkAction(str, Enum):
    """Supported mouse button state transitions triggered by eye closure."""

    LEFT_DOWN = "left_down"
    LEFT_UP = "left_up"
    RIGHT_DOWN = "right_down"
    RIGHT_UP = "right_up"


class BlinkDetector:
    """Detects sustained unilateral eye closure and emits hold/release events."""

    def __init__(self, config: BlinkConfig) -> None:
        self._config = config
        self._left_closed_frames = 0
        self._right_closed_frames = 0
        self._left_open_frames = 0
        self._right_open_frames = 0
        self._left_is_held = False
        self._right_is_held = False

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
    ) -> list[BlinkAction]:
        """Process eye landmarks and return button transitions for this frame."""
        events: list[BlinkAction] = []

        left_ear = self._ear(eye_data["left"])
        right_ear = self._ear(eye_data["right"])

        # Hysteresis avoids rapid toggling near threshold:
        # close when EAR is low, release only when EAR is clearly open.
        left_close = left_ear < self._config.ear_close_threshold
        right_close = right_ear < self._config.ear_close_threshold
        left_open = left_ear > self._config.ear_open_threshold
        right_open = right_ear > self._config.ear_open_threshold

        self._left_closed_frames = self._left_closed_frames + 1 if left_close else 0
        self._right_closed_frames = self._right_closed_frames + 1 if right_close else 0
        self._left_open_frames = self._left_open_frames + 1 if left_open else 0
        self._right_open_frames = self._right_open_frames + 1 if right_open else 0

        if (
            left_close
            and self._left_closed_frames >= self._config.min_closed_frames_for_hold
            and not self._left_is_held
        ):
            self._left_is_held = True
            events.append(BlinkAction.LEFT_DOWN)

        if (
            right_close
            and self._right_closed_frames >= self._config.min_closed_frames_for_hold
            and not self._right_is_held
        ):
            self._right_is_held = True
            events.append(BlinkAction.RIGHT_DOWN)

        if (
            self._left_is_held
            and self._left_open_frames >= self._config.min_open_frames_for_release
        ):
            self._left_is_held = False
            self._left_closed_frames = 0
            events.append(BlinkAction.LEFT_UP)

        if (
            self._right_is_held
            and self._right_open_frames >= self._config.min_open_frames_for_release
        ):
            self._right_is_held = False
            self._right_closed_frames = 0
            events.append(BlinkAction.RIGHT_UP)

        return events
