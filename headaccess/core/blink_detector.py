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
        self._left_ear_smooth: float | None = None
        self._right_ear_smooth: float | None = None
        self._left_open_baseline: float | None = None
        self._right_open_baseline: float | None = None
        self._debug_state: dict[str, float | bool] = {}

        # Smoothing/adaptation constants tuned for webcam noise.
        self._ear_smoothing_alpha = 0.35
        self._baseline_alpha = 0.03
        self._unilateral_margin = 0.025

    @staticmethod
    def _distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _ema(previous: float | None, value: float, alpha: float) -> float:
        """Exponential moving average helper."""
        if previous is None:
            return value
        return previous + alpha * (value - previous)

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

        left_ear_raw = self._ear(eye_data["left"])
        right_ear_raw = self._ear(eye_data["right"])

        self._left_ear_smooth = self._ema(
            self._left_ear_smooth, left_ear_raw, self._ear_smoothing_alpha
        )
        self._right_ear_smooth = self._ema(
            self._right_ear_smooth, right_ear_raw, self._ear_smoothing_alpha
        )
        left_ear = self._left_ear_smooth
        right_ear = self._right_ear_smooth

        if left_ear >= self._config.ear_open_threshold:
            self._left_open_baseline = self._ema(
                self._left_open_baseline, left_ear, self._baseline_alpha
            )
        elif self._left_open_baseline is None:
            self._left_open_baseline = left_ear

        if right_ear >= self._config.ear_open_threshold:
            self._right_open_baseline = self._ema(
                self._right_open_baseline, right_ear, self._baseline_alpha
            )
        elif self._right_open_baseline is None:
            self._right_open_baseline = right_ear

        left_close_threshold = max(
            self._config.ear_close_threshold,
            (self._left_open_baseline or self._config.ear_open_threshold) * 0.78,
        )
        right_close_threshold = max(
            self._config.ear_close_threshold,
            (self._right_open_baseline or self._config.ear_open_threshold) * 0.78,
        )
        left_open_threshold = max(
            self._config.ear_open_threshold,
            left_close_threshold + 0.03,
        )
        right_open_threshold = max(
            self._config.ear_open_threshold,
            right_close_threshold + 0.03,
        )

        # Hysteresis avoids rapid toggling near threshold:
        # close when EAR is low, release only when EAR is clearly open.
        left_close = left_ear < left_close_threshold
        right_close = right_ear < right_close_threshold
        left_open = left_ear > left_open_threshold
        right_open = right_ear > right_open_threshold

        # Avoid triggering hold on normal bilateral blinks.
        left_dominant_close = left_close and (left_ear + self._unilateral_margin < right_ear)
        right_dominant_close = right_close and (
            right_ear + self._unilateral_margin < left_ear
        )

        self._left_closed_frames = self._left_closed_frames + 1 if left_dominant_close else 0
        self._right_closed_frames = self._right_closed_frames + 1 if right_dominant_close else 0
        self._left_open_frames = self._left_open_frames + 1 if left_open else 0
        self._right_open_frames = self._right_open_frames + 1 if right_open else 0

        if (
            left_dominant_close
            and self._left_closed_frames >= self._config.min_closed_frames_for_hold
            and not self._left_is_held
        ):
            self._left_is_held = True
            events.append(BlinkAction.LEFT_DOWN)

        if (
            right_dominant_close
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

        self._debug_state = {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "left_close_threshold": left_close_threshold,
            "right_close_threshold": right_close_threshold,
            "left_open_threshold": left_open_threshold,
            "right_open_threshold": right_open_threshold,
            "left_held": self._left_is_held,
            "right_held": self._right_is_held,
        }

        return events

    def get_debug_state(self) -> dict[str, float | bool]:
        """Expose current EAR telemetry for debug overlay/logs."""
        return self._debug_state.copy()
