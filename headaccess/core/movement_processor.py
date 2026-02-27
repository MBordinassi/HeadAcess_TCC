"""Head movement processing pipeline."""

from __future__ import annotations

from typing import Optional, Tuple

from config import MovementConfig
from utils.calibration import CalibrationManager
from utils.screen_mapping import ScreenMapper
from utils.smoothing import MovingAverageFilter


class MovementProcessor:
    """Converts tracked head pose into smooth cursor coordinates."""

    def __init__(self, config: MovementConfig, screen_size: Tuple[int, int]) -> None:
        self._config = config
        self._screen_size = screen_size
        self._screen_center = (screen_size[0] / 2, screen_size[1] / 2)
        self._calibration = CalibrationManager()
        self._smoother = MovingAverageFilter(config.smoothing_window)

    @property
    def is_calibrated(self) -> bool:
        """Whether movement processing has a neutral pose."""
        return self._calibration.is_calibrated

    def calibrate(self, nose_point: Tuple[float, float]) -> None:
        """Set neutral position and reset smoothing history."""
        self._calibration.calibrate(nose_point)
        self._smoother.reset()

    def process(
        self,
        nose_point: Tuple[float, float],
        frame_size: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """Transform nose position to a stable target screen coordinate."""
        neutral = self._calibration.get_neutral()
        if neutral is None:
            return None

        mapped = ScreenMapper.map_nose_to_screen(
            nose_point=nose_point,
            neutral_point=neutral,
            frame_size=frame_size,
            screen_size=self._screen_size,
            sensitivity_x=self._config.sensitivity_x,
            sensitivity_y=self._config.sensitivity_y,
        )

        dead_zone_applied = ScreenMapper.apply_dead_zone(
            current_point=mapped,
            center_point=self._screen_center,
            dead_zone_px=self._config.dead_zone_px,
        )

        smooth_x, smooth_y = self._smoother.add(dead_zone_applied)
        out_x = int(max(0, min(smooth_x, self._screen_size[0] - 1)))
        out_y = int(max(0, min(smooth_y, self._screen_size[1] - 1)))
        return out_x, out_y

    def get_neutral(self) -> Optional[Tuple[float, float]]:
        """Expose neutral pose for debug overlay."""
        return self._calibration.get_neutral()
