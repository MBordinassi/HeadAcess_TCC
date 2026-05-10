"""Facial landmark tracking using MediaPipe.

Supports:
- Legacy Solution API (`mp.solutions.face_mesh`) when available.
- Tasks API (`FaceLandmarker`) for newer builds where `solutions` is absent.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

from config import FaceMeshConfig

EyePoints = Tuple[Tuple[int, int], ...]


@dataclass
class FaceData:
    """Normalized and pixel-space face information from one frame."""

    landmarks_px: Optional[List[Tuple[int, int]]]
    nose_px: Tuple[int, int]
    face_center_px: Tuple[int, int]
    eye_data: Dict[str, EyePoints]


class FaceTracker:
    """Tracks facial landmarks and exposes key points for control logic."""

    NOSE_INDEX = 1

    LEFT_EYE = {
        "left_corner": 33,
        "upper_outer": 160,
        "upper_inner": 158,
        "right_corner": 133,
        "lower_inner": 153,
        "lower_outer": 144,
    }

    RIGHT_EYE = {
        "left_corner": 362,
        "upper_inner": 385,
        "upper_outer": 387,
        "right_corner": 263,
        "lower_outer": 373,
        "lower_inner": 380,
    }

    KEY_INDICES = frozenset(
        [
            NOSE_INDEX,
            *LEFT_EYE.values(),
            *RIGHT_EYE.values(),
        ]
    )

    TASK_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

    def __init__(self, config: FaceMeshConfig) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._use_tasks_api = not hasattr(mp, "solutions")
        self._face_mesh = None
        self._task_landmarker = None
        self._video_timestamp_ms = 0
        self._processing_size = (config.processing_width, config.processing_height)
        self._include_all_landmarks = config.include_all_landmarks

        if self._use_tasks_api:
            self._init_tasks_api(config)
        else:
            self._init_solutions_api(config)

    def _init_solutions_api(self, config: FaceMeshConfig) -> None:
        """Initialize classic FaceMesh API."""
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config.max_num_faces,
            refine_landmarks=config.refine_landmarks,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._logger.info("face_tracker_backend=solutions")

    def _init_tasks_api(self, config: FaceMeshConfig) -> None:
        """Initialize FaceLandmarker Tasks API (new MediaPipe builds)."""
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        model_path = self._ensure_task_model()
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=config.max_num_faces,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._task_landmarker = vision.FaceLandmarker.create_from_options(options)
        self._logger.info("face_tracker_backend=tasks model=%s", model_path)

    def _ensure_task_model(self) -> Path:
        """Ensure FaceLandmarker model exists locally, downloading if needed."""
        model_dir = self._resolve_model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "face_landmarker.task"
        if model_path.exists():
            return model_path

        self._logger.info("downloading_face_model url=%s", self.TASK_MODEL_URL)
        try:
            urlretrieve(self.TASK_MODEL_URL, model_path)
        except Exception as exc:  # pragma: no cover - external I/O
            raise RuntimeError(
                "Failed to download FaceLandmarker model. "
                f"Download manually from {self.TASK_MODEL_URL} "
                f"to {model_path}"
            ) from exc
        return model_path

    def _resolve_model_dir(self) -> Path:
        """Resolve model directory for source and frozen (PyInstaller) runs."""
        if getattr(sys, "frozen", False):
            bundled_base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
            bundled_model_dir = bundled_base / "models"
            if bundled_model_dir.exists():
                return bundled_model_dir

            local_app_data = os.getenv("LOCALAPPDATA")
            if local_app_data:
                return Path(local_app_data) / "HeadAccess" / "models"
            return Path.home() / ".headaccess" / "models"

        return Path(__file__).resolve().parents[1] / "models"

    def process(self, frame_bgr: np.ndarray) -> Optional[FaceData]:
        """Process one frame and return face data if detected."""
        frame_h, frame_w = frame_bgr.shape[:2]
        landmarks = self._extract_landmarks(frame_bgr)
        if landmarks is None:
            return None

        key_points_px = {
            index: (int(landmarks[index].x * frame_w), int(landmarks[index].y * frame_h))
            for index in self.KEY_INDICES
        }
        landmarks_px = None
        if self._include_all_landmarks:
            landmarks_px = [
                (int(landmark.x * frame_w), int(landmark.y * frame_h))
                for landmark in landmarks
            ]

        nose_px = key_points_px[self.NOSE_INDEX]
        face_center_px = (
            int(sum(landmark.x for landmark in landmarks) / len(landmarks) * frame_w),
            int(sum(landmark.y for landmark in landmarks) / len(landmarks) * frame_h),
        )

        eye_data = {
            "left": (
                key_points_px[self.LEFT_EYE["left_corner"]],
                key_points_px[self.LEFT_EYE["upper_outer"]],
                key_points_px[self.LEFT_EYE["upper_inner"]],
                key_points_px[self.LEFT_EYE["right_corner"]],
                key_points_px[self.LEFT_EYE["lower_inner"]],
                key_points_px[self.LEFT_EYE["lower_outer"]],
            ),
            "right": (
                key_points_px[self.RIGHT_EYE["left_corner"]],
                key_points_px[self.RIGHT_EYE["upper_inner"]],
                key_points_px[self.RIGHT_EYE["upper_outer"]],
                key_points_px[self.RIGHT_EYE["right_corner"]],
                key_points_px[self.RIGHT_EYE["lower_outer"]],
                key_points_px[self.RIGHT_EYE["lower_inner"]],
            ),
        }

        return FaceData(
            landmarks_px=landmarks_px,
            nose_px=nose_px,
            face_center_px=face_center_px,
            eye_data=eye_data,
        )

    def _extract_landmarks(self, frame_bgr: np.ndarray):
        """Extract normalized landmarks from whichever backend is active."""
        processing_w, processing_h = self._processing_size
        if processing_w > 0 and processing_h > 0:
            frame_bgr = cv2.resize(
                frame_bgr,
                (processing_w, processing_h),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if not self._use_tasks_api:
            assert self._face_mesh is not None
            results = self._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None
            return results.multi_face_landmarks[0].landmark

        assert self._task_landmarker is not None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        current_ms = time.monotonic_ns() // 1_000_000
        self._video_timestamp_ms = max(self._video_timestamp_ms + 1, int(current_ms))
        result = self._task_landmarker.detect_for_video(
            mp_image,
            self._video_timestamp_ms,
        )
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._task_landmarker is not None:
            self._task_landmarker.close()
        self._logger.info("face_tracker_closed")
