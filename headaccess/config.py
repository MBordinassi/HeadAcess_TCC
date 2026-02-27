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
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6


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

    ear_threshold: float = 0.20
    min_consecutive_frames: int = 2
    cooldown_frames: int = 8


@dataclass(frozen=True)
class AppConfig:
    """Global application settings."""

    debug_mode: bool = True
    log_level: str = "INFO"
    target_fps: int = 30


CAMERA = CameraConfig()
FACEMESH = FaceMeshConfig()
MOVEMENT = MovementConfig()
BLINK = BlinkConfig()
APP = AppConfig()
