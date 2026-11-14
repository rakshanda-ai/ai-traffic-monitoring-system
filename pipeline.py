from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Callable

import cv2
import numpy as np

from config import AppConfig, VEHICLE_CLASS_IDS, VEHICLE_LABELS
from detectors import ModelBundle
from lane_detection import LaneDetectionConfig, LaneDetector
from traffic_logic import TrafficAnalyzer
from utils import draw_label


@dataclass
class FrameStats:
    active_counts: dict[str, int] = field(default_factory=lambda: {label: 0 for label in VEHICLE_LABELS})
    total_counts: dict[str, int] = field(default_factory=lambda: {label: 0 for label in VEHICLE_LABELS})
    density: str = "Low"
    average_speed: float = 0.0
    total_vehicles: int = 0
    accidents: int = 0
    red_light_violations: int = 0
    helmet_alerts: int = 0
    plates: list[str] = field(default_factory=list)
    lane_segments: int = 0
    lane_confidence: float = 0.0
    lane_valid: bool = False
    progress: float = 0.0


FrameCallback = Callable[[np.ndarray, FrameStats, int], None]
ProgressCallback = Callable[[float, str], None]


class TrafficPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.models = ModelBundle(
            config.detector_model,
            config.helmet_model,
            config.plate_model,
            config.enable_ocr,
            config.enable_helmet,
            config.enable_plate,
        )
        self.analyzer = TrafficAnalyzer(
            config.pixels_per_meter,
            config.stopped_speed_kmh,
            config.accident_seconds,
            config.density_low,
            config.density_medium,
        )
        self.lane_detector = LaneDetector(LaneDetectionConfig(
            roi_top_ratio=config.lane_roi_top,
            canny_low=config.lane_canny_low,
            canny_high=config.lane_canny_high,
            hough_threshold=config.lane_hough_threshold,
            min_line_length=config.lane_min_line_length,
            max_line_gap=config.lane_max_line_gap,
            smoothing=config.lane_smoothing,
            hold_frames=config.lane_hold_frames,
            min_confidence=config.lane_min_confidence,
        ))

    def process_frame(self, frame: np.ndarray, fps: float, red_light_active: bool) -> tuple[np.ndarray, FrameStats]:
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        count_line_y = int(height * self.config.stop_line_ratio)
        stats = FrameStats()
        speeds: list[float] = []
        bike_boxes: list[tuple[int, int, int, int]] = []

        if self.config.enable_lane_overlay:
            lane_result = self.lane_detector.detect(annotated)
            stats.lane_segments = lane_result.segment_count
            stats.lane_confidence = lane_result.confidence
            stats.lane_valid = lane_result.valid
            annotated = cv2.addWeighted(annotated, 1.0, lane_result.overlay, 0.35, 0)

        result = self.models.track_vehicles(frame, self.config.confidence, self.config.iou)
        boxes = result.boxes
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(boxes))

        for box, cls_id, track_id in zip(boxes.xyxy.cpu().numpy().astype(int), boxes.cls.cpu().numpy().astype(int), track_ids):
            label = VEHICLE_CLASS_IDS.get(int(cls_id))
            if not label:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            stats.active_counts[label] += 1
            if label == "motorcycle":
                bike_boxes.append((x1, y1, x2, y2))

            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            state = self.analyzer.tracks[int(track_id)]
            previous_y = state.centers[-1][1] if state.centers else None
            speed = self.analyzer.update_speed(int(track_id), center, fps, label)
            speeds.append(speed)

            counted = self.analyzer.count_crossing(int(track_id), previous_y, center[1], count_line_y, label)
            accident = self.analyzer.accident_event(int(track_id), speed, time())
            violation = self.analyzer.red_light_violation(int(track_id), previous_y, center[1], count_line_y, red_light_active)
            if accident:
                stats.accidents += 1
            if violation:
                stats.red_light_violations += 1

            color = (35, 180, 90)
            if violation:
                color = (35, 35, 230)
            elif accident:
                color = (0, 145, 255)
            elif counted:
                color = (255, 190, 40)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            draw_label(annotated, f"#{track_id} {label} {speed:.1f} km/h", x1, y1 - 6, color)

        line_color = (35, 35, 230) if red_light_active else (35, 180, 90)
        line_text = "COUNT / STOP LINE"
        cv2.line(annotated, (0, count_line_y), (width, count_line_y), line_color, 3)
        draw_label(annotated, line_text, 12, count_line_y - 8, line_color)

        helmet_labels = self.models.detect_helmets(frame, bike_boxes, self.config.confidence)
        stats.helmet_alerts = sum(1 for item in helmet_labels if "no" in item or "without" in item)

        for plate_box, plate_text in self.models.read_number_plates(frame, self.config.confidence):
            x1, y1, x2, y2 = plate_box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 170, 0), 2)
            draw_label(annotated, plate_text, x1, y1 - 6, (255, 170, 0))
            stats.plates.append(plate_text)

        active_count = sum(stats.active_counts.values())
        stats.density = self.analyzer.traffic_density(active_count)
        stats.average_speed = float(np.mean(speeds)) if speeds else 0.0
        stats.total_vehicles = self.analyzer.total_vehicles
        stats.total_counts.update({label: self.analyzer.total_counts.get(label, 0) for label in VEHICLE_LABELS})

        summary = (
            f"Counted:{stats.total_vehicles} Active:{active_count} "
            f"Avg:{stats.average_speed:.1f} km/h Density:{stats.density}"
        )
        draw_label(annotated, summary, 12, 30, (35, 35, 35))
        return annotated, stats

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        frame_callback: FrameCallback | None = None,
        progress_callback: ProgressCallback | None = None,
        red_light_active: bool = False,
        frame_stride: int = 1,
        stop_requested: Callable[[], bool] | None = None,
    ) -> FrameStats:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {input_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.lane_detector.reset()

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, fps / max(1, frame_stride)),
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {output_path}")

        aggregate = FrameStats()
        frame_index = 0
        try:
            while True:
                if stop_requested and stop_requested():
                    if progress_callback:
                        progress_callback(aggregate.progress, "Processing stopped by user.")
                    break

                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % max(1, frame_stride) != 0:
                    continue

                annotated, stats = self.process_frame(frame, fps, red_light_active)
                stats.progress = frame_index / total_frames if total_frames else 0.0
                writer.write(annotated)
                aggregate = stats

                if frame_callback:
                    frame_callback(annotated, stats, frame_index)
                if progress_callback:
                    progress_callback(stats.progress, f"Processed frame {frame_index:,}")
        finally:
            capture.release()
            writer.release()

        return aggregate
