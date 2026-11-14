from __future__ import annotations

import queue
import shutil
import threading
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk, filedialog, messagebox
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from config import AppConfig, MODEL_DIR, OUTPUT_DIR, UPLOAD_DIR, VEHICLE_LABELS, VIDEO_EXTENSIONS
from pipeline import FrameStats, TrafficPipeline
from utils import ensure_dirs


class VehicleCounterApp(Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AI Vehicle Detection, Tracking, and Counting")
        self.geometry("1220x760")
        self.minsize(1060, 680)

        ensure_dirs([UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR])
        self.video_path: Path | None = None
        self.output_path: Path | None = None
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.preview_image: ImageTk.PhotoImage | None = None

        self._create_variables()
        self._configure_style()
        self._build_layout()
        self.after(100, self._drain_events)

    def _create_variables(self) -> None:
        self.detector_model = StringVar(value="yolo11n.pt")
        self.confidence = DoubleVar(value=0.35)
        self.iou = DoubleVar(value=0.45)
        self.frame_stride = IntVar(value=1)
        self.pixels_per_meter = DoubleVar(value=12.0)
        self.stop_line_ratio = DoubleVar(value=0.62)
        self.lane_roi_top = DoubleVar(value=0.58)
        self.red_light_active = BooleanVar(value=False)
        self.enable_lane_overlay = BooleanVar(value=True)
        self.enable_plate = BooleanVar(value=False)
        self.enable_ocr = BooleanVar(value=False)
        self.enable_helmet = BooleanVar(value=False)
        self.status_text = StringVar(value="Select a traffic video to begin.")
        self.file_text = StringVar(value="No video selected")
        self.output_text = StringVar(value="Output will appear in the outputs folder")
        self.progress_value = DoubleVar(value=0.0)
        self.total_text = StringVar(value="0")
        self.speed_text = StringVar(value="0.0 km/h")
        self.density_text = StringVar(value="Low")
        self.alert_text = StringVar(value="0")
        self.count_vars = {label: StringVar(value="0") for label in VEHICLE_LABELS}
        self.active_vars = {label: StringVar(value="0") for label in VEHICLE_LABELS}

    def _configure_style(self) -> None:
        self.configure(bg="#f5f7fb")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#f5f7fb")
        style.configure("Panel.TFrame", background="#ffffff", relief="flat")
        style.configure("TLabel", background="#f5f7fb", foreground="#172033", font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background="#ffffff", foreground="#172033", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#f5f7fb", foreground="#111827", font=("Segoe UI Semibold", 18))
        style.configure("Metric.TLabel", background="#ffffff", foreground="#111827", font=("Segoe UI Semibold", 20))
        style.configure("Small.TLabel", background="#ffffff", foreground="#5b6475", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI Semibold", 10), padding=(12, 8))
        style.configure("Accent.TButton", background="#2563eb", foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#1d4ed8"), ("disabled", "#9db7ef")])
        style.configure("Horizontal.TProgressbar", background="#2563eb", troughcolor="#e8edf7", bordercolor="#e8edf7")

    def _build_layout(self) -> None:
        header = ttk.Frame(self, padding=(22, 16, 22, 8))
        header.pack(fill="x")
        ttk.Label(header, text="AI Vehicle Counter", style="Title.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.status_text).pack(side="right")

        body = ttk.Frame(self, padding=(18, 8, 18, 16))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=330)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        controls = ttk.Frame(body, style="Panel.TFrame", padding=16)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Video", style="Panel.TLabel", font=("Segoe UI Semibold", 11)).grid(row=0, column=0, sticky="w")
        ttk.Button(controls, text="Choose Video", command=self._choose_video, style="Accent.TButton").grid(row=1, column=0, sticky="ew", pady=(8, 6))
        ttk.Label(controls, textvariable=self.file_text, style="Small.TLabel", wraplength=280).grid(row=2, column=0, sticky="ew")
        ttk.Label(controls, textvariable=self.output_text, style="Small.TLabel", wraplength=280).grid(row=3, column=0, sticky="ew", pady=(2, 12))

        self._add_separator(controls, 4)
        row = 5
        row = self._add_entry(controls, row, "YOLO model", self.detector_model)
        row = self._add_scale(controls, row, "Confidence", self.confidence, 0.10, 0.90)
        row = self._add_scale(controls, row, "IoU", self.iou, 0.10, 0.90)
        row = self._add_spinbox(controls, row, "Frame stride", self.frame_stride, 1, 8)
        row = self._add_scale(controls, row, "Count line", self.stop_line_ratio, 0.30, 0.90)
        row = self._add_entry(controls, row, "Pixels per meter", self.pixels_per_meter)

        self._add_separator(controls, row)
        row += 1
        for text, variable in (
            ("Red light mode", self.red_light_active),
            ("Lane overlay", self.enable_lane_overlay),
            ("Helmet model", self.enable_helmet),
            ("Plate model", self.enable_plate),
            ("OCR", self.enable_ocr),
        ):
            ttk.Checkbutton(controls, text=text, variable=variable).grid(row=row, column=0, sticky="w", pady=3)
            row += 1

        actions = ttk.Frame(controls, style="Panel.TFrame")
        actions.grid(row=row, column=0, sticky="ew", pady=(14, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(actions, text="Start", command=self._start_processing, style="Accent.TButton")
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.stop_button = ttk.Button(actions, text="Stop", command=self._stop_processing, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        main = ttk.Frame(body)
        main.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        preview_panel = ttk.Frame(main, style="Panel.TFrame", padding=12)
        preview_panel.grid(row=0, column=0, sticky="nsew")
        preview_panel.rowconfigure(0, weight=1)
        preview_panel.columnconfigure(0, weight=1)
        self.preview = ttk.Label(preview_panel, text="Video preview", anchor="center", background="#101827", foreground="#dce7ff")
        self.preview.grid(row=0, column=0, sticky="nsew")

        bottom = ttk.Frame(main)
        bottom.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.columnconfigure(2, weight=1)
        bottom.columnconfigure(3, weight=1)
        self._metric_card(bottom, 0, "Total counted", self.total_text)
        self._metric_card(bottom, 1, "Average speed", self.speed_text)
        self._metric_card(bottom, 2, "Density", self.density_text)
        self._metric_card(bottom, 3, "Alerts", self.alert_text)

        counts = ttk.Frame(main, style="Panel.TFrame", padding=12)
        counts.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        for index, label in enumerate(VEHICLE_LABELS):
            counts.columnconfigure(index, weight=1)
            text = label.replace("motorcycle", "bike").title()
            ttk.Label(counts, text=text, style="Small.TLabel").grid(row=0, column=index, sticky="w")
            ttk.Label(counts, textvariable=self.count_vars[label], style="Metric.TLabel").grid(row=1, column=index, sticky="w")
            ttk.Label(counts, textvariable=self.active_vars[label], style="Small.TLabel").grid(row=2, column=index, sticky="w")

        progress = ttk.Progressbar(main, variable=self.progress_value, maximum=100)
        progress.grid(row=3, column=0, sticky="ew", pady=(12, 0))

    def _add_separator(self, parent: ttk.Frame, row: int) -> None:
        ttk.Separator(parent).grid(row=row, column=0, sticky="ew", pady=12)

    def _add_entry(self, parent: ttk.Frame, row: int, label: str, variable) -> int:
        ttk.Label(parent, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=variable).grid(row=row + 1, column=0, sticky="ew", pady=(4, 8))
        return row + 2

    def _add_scale(self, parent: ttk.Frame, row: int, label: str, variable: DoubleVar, start: float, end: float) -> int:
        line = ttk.Frame(parent, style="Panel.TFrame")
        line.grid(row=row, column=0, sticky="ew")
        ttk.Label(line, text=label, style="Panel.TLabel").pack(side="left")
        ttk.Label(line, textvariable=variable, style="Small.TLabel").pack(side="right")
        ttk.Scale(parent, from_=start, to=end, variable=variable).grid(row=row + 1, column=0, sticky="ew", pady=(3, 8))
        return row + 2

    def _add_spinbox(self, parent: ttk.Frame, row: int, label: str, variable: IntVar, start: int, end: int) -> int:
        ttk.Label(parent, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(parent, from_=start, to=end, textvariable=variable, width=6).grid(row=row + 1, column=0, sticky="ew", pady=(4, 8))
        return row + 2

    def _metric_card(self, parent: ttk.Frame, column: int, label: str, variable: StringVar) -> None:
        card = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0))
        ttk.Label(card, text=label, style="Small.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=variable, style="Metric.TLabel").pack(anchor="w")

    def _choose_video(self) -> None:
        filetypes = [("Video files", " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)), ("All files", "*.*")]
        selected = filedialog.askopenfilename(title="Choose traffic video", filetypes=filetypes)
        if not selected:
            return
        path = Path(selected)
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            messagebox.showerror("Unsupported file", "Choose a common video file such as MP4, AVI, MOV, or MKV.")
            return
        self.video_path = path
        self.file_text.set(path.name)
        self.status_text.set("Ready to process.")
        self._show_first_frame(path)

    def _show_first_frame(self, path: Path) -> None:
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
            if ok:
                self._update_preview(frame)
        finally:
            capture.release()

    def _build_config(self) -> AppConfig:
        return AppConfig(
            detector_model=self.detector_model.get().strip() or "yolo11n.pt",
            confidence=float(self.confidence.get()),
            iou=float(self.iou.get()),
            pixels_per_meter=float(self.pixels_per_meter.get()),
            stop_line_ratio=float(self.stop_line_ratio.get()),
            lane_roi_top=float(self.lane_roi_top.get()),
            enable_lane_overlay=bool(self.enable_lane_overlay.get()),
            enable_ocr=bool(self.enable_ocr.get()),
            enable_helmet=bool(self.enable_helmet.get()),
            enable_plate=bool(self.enable_plate.get()),
        )

    def _start_processing(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if self.video_path is None:
            messagebox.showinfo("Choose video", "Select a traffic video first.")
            return

        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.stop_event.clear()
        self.progress_value.set(0)
        self._set_running(True)
        self.status_text.set("Preparing models...")

        input_path = self._copy_to_uploads(self.video_path)
        self.output_path = OUTPUT_DIR / f"annotated_{input_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        self.output_text.set(self.output_path.name)

        self.worker = threading.Thread(
            target=self._run_pipeline,
            args=(config, input_path, self.output_path, bool(self.red_light_active.get()), max(1, int(self.frame_stride.get()))),
            daemon=True,
        )
        self.worker.start()

    def _copy_to_uploads(self, source: Path) -> Path:
        target = UPLOAD_DIR / f"{source.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{source.suffix.lower()}"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target

    def _run_pipeline(self, config: AppConfig, input_path: Path, output_path: Path, red_light: bool, stride: int) -> None:
        try:
            pipeline = TrafficPipeline(config)

            def on_frame(frame, stats: FrameStats, frame_index: int) -> None:
                self.events.put(("frame", (frame, stats)))

            def on_progress(progress: float, message: str) -> None:
                self.events.put(("progress", (progress, message)))

            final_stats = pipeline.process_video(
                input_path,
                output_path,
                frame_callback=on_frame,
                progress_callback=on_progress,
                red_light_active=red_light,
                frame_stride=stride,
                stop_requested=self.stop_event.is_set,
            )
            self.events.put(("done", final_stats))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _stop_processing(self) -> None:
        self.stop_event.set()
        self.status_text.set("Stopping after current frame...")
        self.stop_button.configure(state="disabled")

    def _drain_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "frame":
                    frame, stats = payload  # type: ignore[misc]
                    self._update_preview(frame)
                    self._update_stats(stats)
                elif event == "progress":
                    progress, message = payload  # type: ignore[misc]
                    self.progress_value.set(float(progress) * 100)
                    self.status_text.set(str(message))
                elif event == "done":
                    self._update_stats(payload)  # type: ignore[arg-type]
                    self.progress_value.set(100)
                    self.status_text.set("Processing complete.")
                    self._set_running(False)
                    if self.output_path:
                        self.output_text.set(str(self.output_path))
                elif event == "error":
                    self._set_running(False)
                    self.status_text.set("Processing failed.")
                    messagebox.showerror("Processing error", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _update_preview(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.update_idletasks()
        max_w = max(320, self.preview.winfo_width() - 8)
        max_h = max(240, self.preview.winfo_height() - 8)
        image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_image, text="")

    def _update_stats(self, stats: FrameStats) -> None:
        self.total_text.set(str(stats.total_vehicles))
        self.speed_text.set(f"{stats.average_speed:.1f} km/h")
        self.density_text.set(stats.density)
        self.alert_text.set(str(stats.accidents + stats.red_light_violations + stats.helmet_alerts))
        for label in VEHICLE_LABELS:
            self.count_vars[label].set(str(stats.total_counts.get(label, 0)))
            self.active_vars[label].set(f"Active: {stats.active_counts.get(label, 0)}")

    def _set_running(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")


def main() -> None:
    app = VehicleCounterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
