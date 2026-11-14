from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class LaneDetectionConfig:
    roi_top_ratio: float = 0.58
    canny_low: int = 60
    canny_high: int = 160
    hough_threshold: int = 30
    min_line_length: int = 35
    max_line_gap: int = 55
    min_slope: float = 0.35
    smoothing: float = 0.2
    hold_frames: int = 8
    min_confidence: float = 0.35


@dataclass(frozen=True, slots=True)
class LaneDetectionResult:
    overlay: np.ndarray
    left_boundary: tuple[tuple[int, int], tuple[int, int]] | None
    right_boundary: tuple[tuple[int, int], tuple[int, int]] | None
    confidence: float
    valid: bool
    segment_count: int


class LaneDetector:
    """Detect and temporally stabilize a two-boundary road lane."""

    def __init__(self, config: LaneDetectionConfig | None = None) -> None:
        self.config = config or LaneDetectionConfig()
        self._left_model: tuple[float, float] | None = None
        self._right_model: tuple[float, float] | None = None
        self._missing_frames = 0

    def reset(self) -> None:
        self._left_model = None
        self._right_model = None
        self._missing_frames = 0

    def detect(self, frame: np.ndarray) -> LaneDetectionResult:
        self._validate_frame(frame)
        height, width = frame.shape[:2]
        config = self.config
        roi_top = int(height * float(np.clip(config.roi_top_ratio, 0.1, 0.95)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), config.canny_low, config.canny_high)
        mask = np.zeros_like(edges)
        polygon = np.array([[(0, height), (int(width * 0.43), roi_top),
                             (int(width * 0.57), roi_top), (width, height)]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        lines = cv2.HoughLinesP(
            cv2.bitwise_and(edges, mask), 1, np.pi / 180,
            threshold=config.hough_threshold, minLineLength=config.min_line_length,
            maxLineGap=config.max_line_gap,
        )

        left_samples: list[tuple[float, float]] = []
        right_samples: list[tuple[float, float]] = []
        segment_count = 0
        if lines is not None:
            for raw_line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = map(int, raw_line)
                if x1 == x2 or abs((y2 - y1) / (x2 - x1)) < config.min_slope:
                    continue
                segment_count += 1
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                sample = (float(slope), float(intercept))
                midpoint = (x1 + x2) / 2
                (left_samples if slope < 0 and midpoint < width / 2 else right_samples).append(sample)

        left_model = self._fit(left_samples, self._left_model)
        right_model = self._fit(right_samples, self._right_model)
        fresh = bool(left_samples and right_samples)
        if fresh and self._models_are_valid(left_model, right_model, width, height, roi_top):
            self._left_model, self._right_model = left_model, right_model
            self._missing_frames = 0
        else:
            self._missing_frames += 1
            left_model, right_model = self._left_model, self._right_model
            if self._missing_frames > config.hold_frames:
                self._left_model = self._right_model = None
                left_model = right_model = None

        overlay = np.zeros_like(frame)
        left_boundary = self._boundary(left_model, roi_top, height)
        right_boundary = self._boundary(right_model, roi_top, height)
        confidence = self._confidence(left_samples, right_samples, left_boundary, right_boundary, width)
        valid = left_boundary is not None and right_boundary is not None and confidence >= config.min_confidence
        if valid:
            left_top, left_bottom = left_boundary
            right_top, right_bottom = right_boundary
            cv2.fillPoly(overlay, [np.array([left_bottom, left_top, right_top, right_bottom])], (40, 170, 70))
            cv2.line(overlay, left_top, left_bottom, (0, 220, 255), 6)
            cv2.line(overlay, right_top, right_bottom, (0, 220, 255), 6)
        return LaneDetectionResult(overlay, left_boundary, right_boundary, confidence, valid, segment_count)

    def _fit(self, samples: list[tuple[float, float]], previous: tuple[float, float] | None) -> tuple[float, float] | None:
        if not samples:
            return previous
        model = tuple(float(value) for value in np.average(np.array(samples), axis=0, weights=np.ones(len(samples))))
        if previous is None:
            return model
        alpha = float(np.clip(self.config.smoothing, 0.0, 1.0))
        return tuple(alpha * current + (1 - alpha) * old for current, old in zip(model, previous))

    @staticmethod
    def _models_are_valid(left: tuple[float, float], right: tuple[float, float], width: int, height: int, roi_top: int) -> bool:
        left_top = (roi_top - left[1]) / left[0]
        right_top = (roi_top - right[1]) / right[0]
        left_bottom = (height - left[1]) / left[0]
        right_bottom = (height - right[1]) / right[0]
        return left_top < width / 2 < right_top and right_bottom > left_bottom and right_top > left_top and left[0] < 0 and right[0] > 0

    @staticmethod
    def _boundary(model: tuple[float, float] | None, roi_top: int, height: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if model is None or abs(model[0]) < 1e-6:
            return None
        top_x = int((roi_top - model[1]) / model[0])
        bottom_x = int((height - model[1]) / model[0])
        return (top_x, roi_top), (bottom_x, height)

    @staticmethod
    def _confidence(left_samples: list[tuple[float, float]], right_samples: list[tuple[float, float]], left_boundary, right_boundary, width: int) -> float:
        if left_boundary is None or right_boundary is None:
            return 0.0
        evidence = 1.0 if not left_samples and not right_samples else min(1.0, min(len(left_samples), len(right_samples)) / 3)
        width_score = min(1.0, max(0.0, (right_boundary[1][0] - left_boundary[1][0]) / (width * 0.55)))
        return round(0.65 * evidence + 0.35 * width_score, 3)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("frame must be a non-empty BGR image")


def detect_lanes(frame: np.ndarray, roi_top_ratio: float = 0.58) -> tuple[np.ndarray, int]:
    """Backward-compatible one-frame lane overlay helper."""
    result = LaneDetector(LaneDetectionConfig(roi_top_ratio=roi_top_ratio)).detect(frame)
    return result.overlay, result.segment_count
