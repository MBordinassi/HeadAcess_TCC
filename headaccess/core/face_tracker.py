"""Facial landmark tracking using MediaPipe.

Supports:
- Legacy Solution API (`mp.solutions.face_mesh`) when available.
- Tasks API (`FaceLandmarker`) for newer builds where `solutions` is absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

from config import FaceMeshConfig


@dataclass
class FaceData:
    """Normalized and pixel-space face information from one frame."""

    landmarks_px: List[Tuple[int, int]]
    nose_px: Tuple[int, int]
    face_center_px: Tuple[int, int]
    eye_data: Dict[str, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]


class FaceTracker:
    """Tracks facial landmarks and exposes key points for control logic."""

    NOSE_INDEX = 1

    LEFT_EYE = {
        "left_corner": 33,
        "right_corner": 133,
        "top": 159,
        "bottom": 145,
    }

    RIGHT_EYE = {
        "left_corner": 362,
        "right_corner": 263,
        "top": 386,
        "bottom": 374,
    }

    TASK_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

    def __init__(self, config: FaceMeshConfig) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._use_tasks_api = not hasattr(mp, "solutions")
        self._face_mesh = None
        self._task_landmarker = None

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
            running_mode=vision.RunningMode.IMAGE,
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
        model_dir = Path(__file__).resolve().parents[1] / "models"
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

    def process(self, frame_bgr: np.ndarray) -> Optional[FaceData]:
        """Process one frame and return face data if detected."""
        frame_h, frame_w = frame_bgr.shape[:2]
        landmarks = self._extract_landmarks(frame_bgr)
        if landmarks is None:
            return None

        landmarks_px = [
            (int(landmark.x * frame_w), int(landmark.y * frame_h))
            for landmark in landmarks
        ]

        nose_px = landmarks_px[self.NOSE_INDEX]
        xs = [p[0] for p in landmarks_px]
        ys = [p[1] for p in landmarks_px]
        face_center_px = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

        eye_data = {
            "left": (
                landmarks_px[self.LEFT_EYE["left_corner"]],
                landmarks_px[self.LEFT_EYE["right_corner"]],
                landmarks_px[self.LEFT_EYE["top"]],
                landmarks_px[self.LEFT_EYE["bottom"]],
            ),
            "right": (
                landmarks_px[self.RIGHT_EYE["left_corner"]],
                landmarks_px[self.RIGHT_EYE["right_corner"]],
                landmarks_px[self.RIGHT_EYE["top"]],
                landmarks_px[self.RIGHT_EYE["bottom"]],
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
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if not self._use_tasks_api:
            assert self._face_mesh is not None
            results = self._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None
            return results.multi_face_landmarks[0].landmark

        assert self._task_landmarker is not None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._task_landmarker.detect(mp_image)
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
