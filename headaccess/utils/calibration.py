"""Calibration helpers for neutral head pose."""

from __future__ import annotations

from typing import Optional, Tuple


class CalibrationManager:
    """Stores and serves neutral pose coordinates."""

    def __init__(self) -> None:
        self._neutral_nose: Optional[Tuple[float, float]] = None

    @property
    def is_calibrated(self) -> bool:
        """Whether a neutral point is available."""
        return self._neutral_nose is not None

    def calibrate(self, nose_point: Tuple[float, float]) -> None:
        """Define the current nose point as neutral."""
        self._neutral_nose = nose_point

    def get_neutral(self) -> Optional[Tuple[float, float]]:
        """Return the neutral nose point if calibrated."""
        return self._neutral_nose

    def reset(self) -> None:
        """Reset calibration state."""
        self._neutral_nose = None
