from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def crop_box(frame: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1 = clamp(x1, 0, width - 1)
    x2 = clamp(x2, 0, width - 1)
    y1 = clamp(y1, 0, height - 1)
    y2 = clamp(y2, 0, height - 1)
    if x2 <= x1 or y2 <= y1:
        return frame[0:0, 0:0]
    return frame[y1:y2, x1:x2]


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(text_h + 8, y)
    cv2.rectangle(frame, (x, y - text_h - baseline - 6), (x + text_w + 8, y + baseline), color, -1)
    cv2.putText(frame, text, (x + 4, y - 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def readable_plate(text: str) -> str:
    cleaned = "".join(ch for ch in text.upper() if ch.isalnum())
    return cleaned[:12]
