from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v")

# COCO class ids used by the default Ultralytics YOLO models.
VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_LABELS = ("car", "motorcycle", "bus", "truck")


@dataclass(slots=True)
class AppConfig:
    detector_model: str = "yolo11n.pt"
    helmet_model: str = str(MODEL_DIR / "helmet.pt")
    plate_model: str = str(MODEL_DIR / "plate.pt")
    confidence: float = 0.35
    iou: float = 0.45
    pixels_per_meter: float = 12.0
    density_low: int = 8
    density_medium: int = 20
    accident_seconds: float = 4.0
    stopped_speed_kmh: float = 2.0
    stop_line_ratio: float = 0.62
    lane_roi_top: float = 0.58
    lane_canny_low: int = 60
    lane_canny_high: int = 160
    lane_hough_threshold: int = 30
    lane_min_line_length: int = 35
    lane_max_line_gap: int = 55
    lane_smoothing: float = 0.2
    lane_hold_frames: int = 8
    lane_min_confidence: float = 0.35
    enable_lane_overlay: bool = True
    enable_ocr: bool = False
    enable_helmet: bool = False
    enable_plate: bool = False
