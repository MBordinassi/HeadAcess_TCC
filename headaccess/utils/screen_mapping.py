"""Screen mapping utilities."""

from __future__ import annotations

from typing import Tuple


class ScreenMapper:
    """Maps camera-space movement to screen-space coordinates."""

    @staticmethod
    def map_nose_to_screen(
        nose_point: Tuple[float, float],
        neutral_point: Tuple[float, float],
        frame_size: Tuple[int, int],
        screen_size: Tuple[int, int],
        sensitivity_x: float,
        sensitivity_y: float,
    ) -> Tuple[float, float]:
        """Convert relative nose offset to absolute screen position."""
        frame_w, frame_h = frame_size
        screen_w, screen_h = screen_size

        dx = (nose_point[0] - neutral_point[0]) / max(frame_w, 1)
        dy = (nose_point[1] - neutral_point[1]) / max(frame_h, 1)

        center_x = screen_w / 2
        center_y = screen_h / 2

        target_x = center_x + dx * screen_w * sensitivity_x
        target_y = center_y + dy * screen_h * sensitivity_y
        return target_x, target_y

    @staticmethod
    def apply_dead_zone(
        current_point: Tuple[float, float],
        center_point: Tuple[float, float],
        dead_zone_px: int,
    ) -> Tuple[float, float]:
        """Ignore tiny movements around center to reduce jitter."""
        out_x, out_y = current_point

        if abs(current_point[0] - center_point[0]) < dead_zone_px:
            out_x = center_point[0]
        if abs(current_point[1] - center_point[1]) < dead_zone_px:
            out_y = center_point[1]

        return out_x, out_y
