"""Blink detection for hold/release mouse control."""

from __future__ import annotations

import time
from enum import Enum
from typing import Dict, Tuple

from config import BlinkConfig

EyePoints = Tuple[Tuple[int, int], ...]


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
        self._left_closed_since: float | None = None
        self._right_closed_since: float | None = None
        self._left_open_since: float | None = None
        self._right_open_since: float | None = None
        self._last_action_time = 0.0
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
        eye_points: EyePoints,
    ) -> float:
        left_corner, upper_outer, upper_inner, right_corner, lower_inner, lower_outer = (
            eye_points
        )
        horizontal = self._distance(left_corner, right_corner)
        vertical_outer = self._distance(upper_outer, lower_outer)
        vertical_inner = self._distance(upper_inner, lower_inner)
        if horizontal <= 1e-6:
            return 1.0
        return (vertical_outer + vertical_inner) / (2.0 * horizontal)

    def update(
        self,
        eye_data: Dict[str, EyePoints],
    ) -> list[BlinkAction]:
        """Process eye landmarks and return button transitions for this frame."""
        events: list[BlinkAction] = []
        now = time.monotonic()

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
            (self._left_open_baseline or self._config.ear_open_threshold) * 0.90,
        )
        right_close_threshold = max(
            self._config.ear_close_threshold,
            (self._right_open_baseline or self._config.ear_open_threshold) * 0.90,
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

        both_closed = left_close and right_close

        # A bilateral blink is treated as a left click because it is the easiest
        # gesture for most users. One-eye winks still select left/right.
        left_dominant_close = both_closed or (
            left_close
            and (
                not right_close
                or left_ear + self._config.unilateral_margin < right_ear
            )
        )
        right_dominant_close = (not both_closed) and (
            right_close
            and (
                not left_close
                or right_ear + self._config.unilateral_margin < left_ear
            )
        )

        self._left_closed_since = (
            self._left_closed_since if left_dominant_close else None
        )
        if left_dominant_close and self._left_closed_since is None:
            self._left_closed_since = now

        self._right_closed_since = (
            self._right_closed_since if right_dominant_close else None
        )
        if right_dominant_close and self._right_closed_since is None:
            self._right_closed_since = now

        self._left_open_since = self._left_open_since if left_open else None
        if left_open and self._left_open_since is None:
            self._left_open_since = now

        self._right_open_since = self._right_open_since if right_open else None
        if right_open and self._right_open_since is None:
            self._right_open_since = now

        left_closed_duration = (
            now - self._left_closed_since if self._left_closed_since is not None else 0.0
        )
        right_closed_duration = (
            now - self._right_closed_since if self._right_closed_since is not None else 0.0
        )
        left_open_duration = (
            now - self._left_open_since if self._left_open_since is not None else 0.0
        )
        right_open_duration = (
            now - self._right_open_since if self._right_open_since is not None else 0.0
        )
        cooldown_ready = (
            now - self._last_action_time >= self._config.click_cooldown_seconds
        )

        if (
            left_dominant_close
            and left_closed_duration >= self._config.min_closed_seconds_for_hold
            and not self._left_is_held
            and cooldown_ready
        ):
            self._left_is_held = True
            self._last_action_time = now
            events.append(BlinkAction.LEFT_DOWN)

        if (
            right_dominant_close
            and right_closed_duration >= self._config.min_closed_seconds_for_hold
            and not self._right_is_held
            and cooldown_ready
        ):
            self._right_is_held = True
            self._last_action_time = now
            events.append(BlinkAction.RIGHT_DOWN)

        if (
            self._left_is_held
            and left_open_duration >= self._config.min_open_seconds_for_release
        ):
            self._left_is_held = False
            self._left_closed_since = None
            events.append(BlinkAction.LEFT_UP)

        if (
            self._right_is_held
            and right_open_duration >= self._config.min_open_seconds_for_release
        ):
            self._right_is_held = False
            self._right_closed_since = None
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
            "left_close": left_close,
            "right_close": right_close,
            "both_closed": both_closed,
            "left_closed_duration": left_closed_duration,
            "right_closed_duration": right_closed_duration,
        }

        return events

    def get_debug_state(self) -> dict[str, float | bool]:
        """Expose current EAR telemetry for debug overlay/logs."""
        return self._debug_state.copy()
