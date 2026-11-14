from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from config import VEHICLE_CLASS_IDS
from utils import crop_box, readable_plate


class ModelLoadError(RuntimeError):
    pass


class ModelBundle:
    """Loads YOLO models lazily and exposes detection helpers used by the pipeline."""

    def __init__(
        self,
        detector_model: str,
        helmet_model: str,
        plate_model: str,
        enable_ocr: bool,
        enable_helmet: bool,
        enable_plate: bool,
    ) -> None:
        try:
            self.detector = YOLO(detector_model)
        except Exception as exc:  # Ultralytics raises several exception types depending on source.
            raise ModelLoadError(f"Could not load vehicle detector '{detector_model}': {exc}") from exc

        self.helmet = self._load_optional_yolo(helmet_model, enable_helmet, "helmet")
        self.plate = self._load_optional_yolo(plate_model, enable_plate, "plate")
        self.ocr = self._load_ocr(enable_ocr)

    @staticmethod
    def _load_optional_yolo(model_path: str, enabled: bool, name: str) -> YOLO | None:
        if not enabled:
            return None
        if not Path(model_path).exists():
            return None
        try:
            return YOLO(model_path)
        except Exception as exc:
            raise ModelLoadError(f"Could not load {name} model '{model_path}': {exc}") from exc

    @staticmethod
    def _load_ocr(enabled: bool) -> Any | None:
        if not enabled:
            return None
        try:
            import easyocr
        except ImportError:
            return None
        return easyocr.Reader(["en"], gpu=False)

    def track_vehicles(self, frame: np.ndarray, confidence: float, iou: float) -> Any:
        return self.detector.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            iou=iou,
            classes=list(VEHICLE_CLASS_IDS.keys()),
            verbose=False,
        )[0]

    def detect_helmets(self, frame: np.ndarray, bike_boxes: list[tuple[int, int, int, int]], confidence: float) -> list[str]:
        if self.helmet is None:
            return []
        labels: list[str] = []
        for box in bike_boxes:
            crop = crop_box(frame, box)
            if crop.size == 0:
                continue
            result = self.helmet.predict(crop, conf=confidence, verbose=False)[0]
            names = result.names
            for cls in result.boxes.cls.cpu().numpy().astype(int):
                labels.append(str(names.get(cls, cls)).lower())
        return labels

    def read_number_plates(self, frame: np.ndarray, confidence: float) -> list[tuple[tuple[int, int, int, int], str]]:
        if self.plate is None or self.ocr is None:
            return []

        detections: list[tuple[tuple[int, int, int, int], str]] = []
        result = self.plate.predict(frame, conf=confidence, verbose=False)[0]
        for xyxy in result.boxes.xyxy.cpu().numpy().astype(int):
            box = tuple(int(v) for v in xyxy)
            crop = crop_box(frame, box)
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ocr_result = self.ocr.readtext(gray, detail=0, paragraph=False)
            plate_text = readable_plate("".join(ocr_result))
            if plate_text:
                detections.append((box, plate_text))
        return detections
