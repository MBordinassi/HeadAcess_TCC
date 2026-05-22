"""Central configuration for HeadAccess."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    """Camera and video processing settings."""

    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass(frozen=True)
class FaceMeshConfig:
    """MediaPipe Face Mesh settings."""

    max_num_faces: int = 1
    refine_landmarks: bool = False
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    processing_width: int = 320
    processing_height: int = 240
    include_all_landmarks: bool = False


@dataclass(frozen=True)
class MovementConfig:
    """Pointer movement behavior settings."""

    sensitivity_x: float = 1.9
    sensitivity_y: float = 1.7
    smoothing_window: int = 5
    dead_zone_px: int = 18


@dataclass(frozen=True)
class BlinkConfig:
    """Blink detection settings."""

    ear_close_threshold: float = 0.30
    ear_open_threshold: float = 0.32
    open_eye_calibration_seconds: float = 1.6
    close_ratio: float = 0.78
    open_ratio: float = 0.90
    min_closed_seconds_for_hold: float = 0.08
    min_open_seconds_for_release: float = 0.08
    click_cooldown_seconds: float = 0.20
    unilateral_margin: float = 0.015


@dataclass(frozen=True)
class AppConfig:
    """Global application settings."""

    debug_mode: bool = True
    log_level: str = "INFO"
    target_fps: int = 30
    face_lost_grace_seconds: float = 0.35
    auto_recalibration_enabled: bool = True
    auto_recalibration_interval_seconds: float = 15.0
    auto_recalibration_stable_seconds: float = 1.2
    auto_recalibration_stable_radius_px: float = 8.0


CAMERA = CameraConfig()
FACEMESH = FaceMeshConfig()
MOVEMENT = MovementConfig()
BLINK = BlinkConfig()
APP = AppConfig()
