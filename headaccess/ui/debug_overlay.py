"""Debug rendering utilities for HeadAccess."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class DebugOverlay:
    """Draws visual telemetry over camera frames."""

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
