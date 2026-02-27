"""HeadAccess application entry point."""

from __future__ import annotations

import logging
import time

import cv2

from config import APP, BLINK, CAMERA, FACEMESH, MOVEMENT
from core.blink_detector import BlinkAction, BlinkDetector
from core.camera import CameraStream
from core.face_tracker import FaceTracker
from core.mouse_controller import MouseController
from core.movement_processor import MovementProcessor
from ui.debug_overlay import DebugOverlay


def setup_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, APP.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run() -> None:
    """Start HeadAccess control loop."""
    setup_logging()
    logger = logging.getLogger("HeadAccess")

    camera = CameraStream(CAMERA)
    tracker = FaceTracker(FACEMESH)
    mouse = MouseController()
    movement = MovementProcessor(MOVEMENT, mouse.screen_size)
    blink_detector = BlinkDetector(BLINK)

    last_time = time.time()

    try:
        camera.start()
        logger.info("app_started")

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.001)
                continue

            face_data = tracker.process(frame)
            target_cursor = None
            fps = 1.0 / max(time.time() - last_time, 1e-6)
            last_time = time.time()

            if face_data is not None:
                frame_h, frame_w = frame.shape[:2]
                nose_point = (float(face_data.nose_px[0]), float(face_data.nose_px[1]))

                # Auto-calibration happens once on first valid face frame.
                if not movement.is_calibrated:
                    movement.calibrate(nose_point)
                    logger.info("calibration_auto_initial x=%s y=%s", *nose_point)

                target_cursor = movement.process(nose_point, (frame_w, frame_h))
                if target_cursor is not None:
                    mouse.move_to(*target_cursor)

                blink_action = blink_detector.update(face_data.eye_data)
                if blink_action == BlinkAction.LEFT_CLICK:
                    mouse.left_click()
                elif blink_action == BlinkAction.RIGHT_CLICK:
                    mouse.right_click()

                if APP.debug_mode:
                    frame = DebugOverlay.draw(
                        frame=frame,
                        nose_point=face_data.nose_px,
                        face_center=face_data.face_center_px,
                        target_point=target_cursor,
                        fps=fps,
                        calibrated=movement.is_calibrated,
                        neutral_point=movement.get_neutral(),
                        landmarks=face_data.landmarks_px,
                    )
            else:
                if APP.debug_mode:
                    frame = DebugOverlay.draw(
                        frame=frame,
                        nose_point=None,
                        face_center=None,
                        target_point=target_cursor,
                        fps=fps,
                        calibrated=movement.is_calibrated,
                        neutral_point=movement.get_neutral(),
                        landmarks=None,
                    )

            if APP.debug_mode:
                cv2.imshow("HeadAccess Debug", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") or key == ord("C"):
                if face_data is not None:
                    movement.calibrate(
                        (float(face_data.nose_px[0]), float(face_data.nose_px[1]))
                    )
                    logger.info(
                        "calibration_manual x=%s y=%s",
                        face_data.nose_px[0],
                        face_data.nose_px[1],
                    )
            if key == 27:
                logger.info("app_stop_requested source=keyboard")
                break

    finally:
        camera.stop()
        tracker.close()
        cv2.destroyAllWindows()
        logger.info("app_stopped")


if __name__ == "__main__":
    run()
