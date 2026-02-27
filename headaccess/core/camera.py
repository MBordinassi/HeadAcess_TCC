"""Threaded camera capture module."""

from __future__ import annotations

import logging
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from config import CameraConfig


class CameraStream:
    """Continuously captures frames from a webcam in a background thread."""

    def __init__(self, config: CameraConfig) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None

    def start(self) -> None:
        """Start webcam capture and frame acquisition thread."""
        self._capture = cv2.VideoCapture(self._config.device_index)
        if not self._capture.isOpened():
            raise RuntimeError("Unable to open webcam")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._capture.set(cv2.CAP_PROP_FPS, self._config.fps)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._logger.info(
            "camera_started device=%s width=%s height=%s fps=%s",
            self._config.device_index,
            self._config.width,
            self._config.height,
            self._config.fps,
        )

    def _capture_loop(self) -> None:
        """Read frames until stream is stopped."""
        assert self._capture is not None
        while self._running:
            success, frame = self._capture.read()
            if not success:
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the latest available frame."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self) -> None:
        """Stop capture and release resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self._logger.info("camera_stopped")
