from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from time import time

import numpy as np


@dataclass
class TrackState:
    centers: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=12))
    stopped_since: float | None = None
    violated_red: bool = False
    counted: bool = False
    label: str = ""


class TrafficAnalyzer:
    """Keeps track-level state for speed, events, density, and counting."""

    def __init__(
        self,
        pixels_per_meter: float,
        stopped_speed_kmh: float,
        accident_seconds: float,
        density_low: int,
        density_medium: int,
    ) -> None:
        self.pixels_per_meter = max(1.0, pixels_per_meter)
        self.stopped_speed_kmh = max(0.0, stopped_speed_kmh)
        self.accident_seconds = max(0.5, accident_seconds)
        self.density_low = max(1, density_low)
        self.density_medium = max(self.density_low + 1, density_medium)
        self.tracks: dict[int, TrackState] = defaultdict(TrackState)
        self.total_counts: dict[str, int] = defaultdict(int)
        self.total_vehicles = 0

    def update_speed(self, track_id: int, center: tuple[float, float], fps: float, label: str) -> float:
        state = self.tracks[track_id]
        state.label = label
        now = time()
        state.centers.append((center[0], center[1], now))
        if len(state.centers) < 2:
            return 0.0

        x0, y0, t0 = state.centers[0]
        x1, y1, t1 = state.centers[-1]
        seconds = max(t1 - t0, 1.0 / max(fps, 1.0))
        pixels = float(np.hypot(x1 - x0, y1 - y0))
        meters = pixels / self.pixels_per_meter
        return (meters / seconds) * 3.6

    def count_crossing(self, track_id: int, previous_y: float | None, current_y: float, line_y: int, label: str) -> bool:
        state = self.tracks[track_id]
        if state.counted or previous_y is None:
            return False
        crossed_down = previous_y < line_y <= current_y
        crossed_up = previous_y > line_y >= current_y
        if crossed_down or crossed_up:
            state.counted = True
            self.total_counts[label] += 1
            self.total_vehicles += 1
            return True
        return False

    def accident_event(self, track_id: int, speed_kmh: float, current_time: float) -> bool:
        state = self.tracks[track_id]
        if speed_kmh <= self.stopped_speed_kmh:
            if state.stopped_since is None:
                state.stopped_since = current_time
            return current_time - state.stopped_since >= self.accident_seconds
        state.stopped_since = None
        return False

    def red_light_violation(
        self,
        track_id: int,
        previous_y: float | None,
        current_y: float,
        stop_line_y: int,
        red_light_active: bool,
    ) -> bool:
        state = self.tracks[track_id]
        if state.violated_red:
            return True
        if red_light_active and previous_y is not None and previous_y < stop_line_y <= current_y:
            state.violated_red = True
            return True
        return False

    def traffic_density(self, active_vehicle_count: int) -> str:
        if active_vehicle_count < self.density_low:
            return "Low"
        if active_vehicle_count < self.density_medium:
            return "Medium"
        return "High"
