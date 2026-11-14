# AI Vehicle Detection, Tracking, and Counting

A production-ready Python desktop application for detecting, tracking, and counting vehicles from uploaded video files. The app uses Tkinter for the GUI, OpenCV for video processing, Ultralytics YOLO for deep-learning detection, and ByteTrack for multi-object tracking.

## Features

- Modern Tkinter desktop interface
- Upload/select MP4, AVI, MOV, MKV, WMV, or M4V videos
- YOLO vehicle detection for cars, motorcycles, buses, and trucks
- ByteTrack object tracking with stable track IDs
- Unique vehicle counting when tracks cross the configurable count line
- Live annotated preview while processing
- Annotated MP4 export in `outputs/`
- Per-class total counts and active on-screen counts
- Speed estimation with pixels-per-meter calibration
- Traffic density estimation
- Configurable classical lane detection using grayscale preprocessing, Canny edges, ROI masking, and probabilistic Hough lines
- Smoothed left/right lane boundaries with confidence scoring, filled drivable-lane corridor, and short-term fallback when markings disappear
- Optional red-light violation checks using the count/stop line
- Optional custom helmet and number plate models
- Optional EasyOCR number plate text reading
- Threaded processing so the GUI remains responsive
- Clear error handling for video, model, and output failures

## Project Structure

```text
ai_traffic_monitoring_system/
  app.py              # Tkinter GUI and worker thread orchestration
  config.py           # Paths, model defaults, labels, and app settings
  detectors.py        # YOLO model loading, vehicle tracking, optional OCR helpers
  lane_detection.py   # Stateful OpenCV lane boundary detector and overlay renderer
  pipeline.py         # Frame/video processing pipeline
  traffic_logic.py    # Track state, counting, speed, density, event logic
  utils.py            # Drawing, cropping, directory, and text helpers
  models/             # Optional custom models: helmet.pt, plate.pt
  uploads/            # Selected videos copied for processing
  outputs/            # Annotated output videos
```

## Setup

```bash
cd "C:\Users\Sadaf\python project\ai_traffic_monitoring_system"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first run with `yolo11n.pt` may download the YOLO model weights automatically through Ultralytics.

## Run

```bash
python app.py
```

Choose a video, tune the confidence and count-line settings if needed, then press **Start**. The processed video is saved to `outputs/`.

## Optional Custom Models

Place these files in `models/` and enable the matching checkbox in the GUI:

- `models/helmet.pt` for helmet/no-helmet detection
- `models/plate.pt` for number plate detection

OCR is optional. If EasyOCR is enabled but unavailable, plate reading is skipped safely.

## Calibration Notes

Speed estimation depends on the scene. Adjust **Pixels per meter** until the displayed speeds look realistic for known traffic. Move **Count line** to a road position that vehicles reliably cross once.

Lane detection assumes a forward-facing road view. The default ROI begins at 58% of the frame height; tune `lane_roi_top`, Canny/Hough thresholds, smoothing, and `lane_min_confidence` in `AppConfig` for a different camera or road surface. The detector rejects implausible boundary geometry and holds the last valid corridor for a configurable number of frames to avoid flicker.

## Troubleshooting

- If model loading fails, check the YOLO model path or internet access for the first automatic weight download.
- If the output cannot be written, make sure `outputs/` is writable and the selected video is not open in another application.
- If counting seems too high, increase confidence or place the count line where tracks are stable.
- If counting seems too low, lower confidence slightly or reduce frame stride.
