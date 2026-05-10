"""Debug rendering utilities for HeadAccess."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

EyePoints = Tuple[Tuple[int, int], ...]


class DebugOverlay:
    """Draws visual telemetry over camera frames."""

    @staticmethod
    def _draw_eye(
        output: np.ndarray,
        points: EyePoints,
        label: str,
        is_closed: bool,
        is_held: bool,
    ) -> None:
        color = (0, 0, 255) if is_closed else (0, 220, 0)
        if is_held:
            color = (0, 165, 255)

        contour = np.array(points, dtype=np.int32)
        cv2.polylines(output, [contour], isClosed=True, color=color, thickness=2)
        for point in points:
            cv2.circle(output, point, 3, color, -1)

        left = min(point[0] for point in points)
        top = min(point[1] for point in points)
        status = "HELD" if is_held else ("CLOSED" if is_closed else "OPEN")
        cv2.putText(
            output,
            f"{label}: {status}",
            (left, max(15, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    @staticmethod
    def draw(
        frame: np.ndarray,
        nose_point: Optional[Tuple[int, int]],
        face_center: Optional[Tuple[int, int]],
        target_point: Optional[Tuple[int, int]],
        fps: float,
        calibrated: bool,
        neutral_point: Optional[Tuple[float, float]],
        landmarks: Optional[list[Tuple[int, int]]] = None,
        blink_status: Optional[str] = None,
        eye_data: Optional[dict[str, EyePoints]] = None,
        blink_state: Optional[dict[str, float | bool]] = None,
    ) -> np.ndarray:
        """Render landmarks, vectors and status text."""
        output = frame.copy()

        if landmarks:
            for x, y in landmarks:
                cv2.circle(output, (x, y), 1, (255, 0, 0), -1)

        if face_center is not None:
            cv2.circle(output, face_center, 5, (0, 255, 255), -1)
            cv2.putText(
                output,
                "Face Center",
                (face_center[0] + 8, face_center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        if nose_point is not None:
            cv2.circle(output, nose_point, 6, (0, 255, 0), -1)
            cv2.putText(
                output,
                "Nose",
                (nose_point[0] + 8, nose_point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        if neutral_point is not None and nose_point is not None:
            neutral_int = (int(neutral_point[0]), int(neutral_point[1]))
            cv2.circle(output, neutral_int, 6, (255, 255, 0), 2)
            cv2.arrowedLine(output, neutral_int, nose_point, (0, 0, 255), 2)

        if eye_data is not None:
            blink_state = blink_state or {}
            if "left" in eye_data:
                DebugOverlay._draw_eye(
                    output,
                    eye_data["left"],
                    "L",
                    bool(blink_state.get("left_close")),
                    bool(blink_state.get("left_held")),
                )
            if "right" in eye_data:
                DebugOverlay._draw_eye(
                    output,
                    eye_data["right"],
                    "R",
                    bool(blink_state.get("right_close")),
                    bool(blink_state.get("right_held")),
                )

        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            output,
            f"Calibrated: {'YES' if calibrated else 'NO'}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if calibrated else (0, 0, 255),
            2,
        )

        if target_point is not None:
            cv2.putText(
                output,
                f"Cursor: {target_point[0]}, {target_point[1]}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

        if blink_status:
            cv2.putText(
                output,
                blink_status,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 255, 200),
                2,
            )

        cv2.putText(
            output,
            "C: calibrate | ESC: quit",
            (10, output.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return output
