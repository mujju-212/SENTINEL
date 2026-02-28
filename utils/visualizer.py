"""
MODULE 7 - Display & Visualization
Draws all bounding boxes, labels, info bars, OCR overlays,
and the FPS counter onto the video frame using OpenCV.

Color scheme:
  Green  → Known person
  Red    → Unknown person
  Orange → Non-person object
  Yellow → OCR text location
"""

import cv2
import numpy as np
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

# ── FONTS & CONSTANTS ────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


class FPSCounter:
    """Rolling average FPS tracker."""
    def __init__(self, window: int = 30):
        self._times = deque(maxlen=window)
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) < 2:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))


class Visualizer:
    """Draws all visual elements on each frame."""

    def __init__(self, config: dict):
        self.config = config
        self.thickness = config.get('box_thickness', 2)
        self.font_scale = config.get('font_scale', 0.60)
        self.show_fps = config.get('show_fps', True)
        self.show_conf = config.get('show_confidence', True)
        self.show_id = config.get('show_track_id', True)

        # Colors (BGR)
        self.color_known    = tuple(config.get('known_person_color',   [0, 255, 0]))
        self.color_unknown  = tuple(config.get('unknown_person_color', [0, 0, 255]))
        self.color_object   = tuple(config.get('object_color',         [255, 165, 0]))
        self.color_ocr      = tuple(config.get('ocr_color',            [0, 255, 255]))

        self.fps_counter = FPSCounter()
        logger.info("[Visualizer] Ready")

    # ── MAIN DRAW ────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, detections: list,
             ocr_results: list = None, source: str = 'camera') -> np.ndarray:
        """
        Draw all annotations onto frame and return the annotated copy.

        Args:
            frame:       BGR frame from OpenCV
            detections:  List of detection dicts (from tracker / detector)
            ocr_results: List of OCRResult objects (can be None)
            source:      Input source label for status bar

        Returns:
            Annotated frame (numpy array)
        """
        frame = frame.copy()
        fps = self.fps_counter.tick()

        persons = [d for d in detections if d.get('class_name') == 'person']
        objects = [d for d in detections if d.get('class_name') != 'person']

        # Draw movement trails
        for det in detections:
            self._draw_trail(frame, det.get('trail', []),
                             self._get_color(det))

        # Draw object boxes
        for det in objects:
            self._draw_box(frame, det, self.color_object)

        # Draw person boxes (on top)
        for det in persons:
            name = det.get('person_name') or 'Unknown'
            color = self.color_known if name != 'Unknown' else self.color_unknown
            self._draw_box(frame, det, color, is_person=True)

        # Draw OCR boxes
        if ocr_results:
            for r in ocr_results:
                self._draw_ocr(frame, r)

        # Draw HUD bars
        self._draw_top_bar(frame, fps, len(detections), len(persons), source)
        if ocr_results:
            self._draw_ocr_bar(frame, ocr_results)

        return frame

    # ── BOX DRAWING ──────────────────────────────────────────────────
    def _draw_box(self, frame, det: dict, color: tuple, is_person: bool = False):
        x1, y1, x2, y2 = det['bbox']
        cls = det.get('class_name', '?')
        conf = det.get('confidence', 0.0)
        tid = det.get('track_id', 0)
        name = det.get('person_name')
        face_conf = det.get('face_confidence', 0.0)
        time_in = det.get('time_in_frame', 0.0)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

        # Build label
        if is_person:
            label = name if name else 'Person'
            if name and name != 'Unknown' and self.show_conf:
                label += f" {face_conf*100:.0f}%"
        else:
            label = cls
            if self.show_conf:
                label += f" {conf*100:.0f}%"

        if self.show_id:
            label = f"#{tid} {label}"

        # Label background
        (lw, lh), _ = cv2.getTextSize(label, FONT, self.font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    FONT, self.font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        # Time in frame (persons only)
        if is_person and time_in > 1.0:
            time_label = f"{time_in:.0f}s"
            cv2.putText(frame, time_label, (x1 + 2, y2 - 4),
                        FONT, 0.45, color, 1, cv2.LINE_AA)

    # ── OCR OVERLAY ──────────────────────────────────────────────────
    def _draw_ocr(self, frame, result):
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color_ocr, 2)
        label = f'"{result.text}"'
        cv2.putText(frame, label, (x1, y1 - 6),
                    FONT, 0.50, self.color_ocr, 1, cv2.LINE_AA)

    # ── TRAIL ────────────────────────────────────────────────────────
    def _draw_trail(self, frame, trail: list, color: tuple):
        if len(trail) < 2:
            return
        pts = list(trail)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            c = tuple(int(v * alpha) for v in color)
            cv2.line(frame, pts[i-1], pts[i], c, 1)

    # ── TOP HUD BAR ──────────────────────────────────────────────────
    def _draw_top_bar(self, frame, fps: float, obj_count: int,
                      person_count: int, source: str):
        h, w = frame.shape[:2]
        bar_h = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        import datetime
        now = datetime.datetime.now().strftime('%H:%M:%S')

        info = (f"FPS: {fps:.0f}  |  Objects: {obj_count}  |  "
                f"Persons: {person_count}  |  {now}  |  Mode: {source.upper()}")
        cv2.putText(frame, info, (8, 22),
                    FONT, 0.52, (200, 255, 200), 1, cv2.LINE_AA)

    # ── OCR BAR ──────────────────────────────────────────────────────
    def _draw_ocr_bar(self, frame, ocr_results: list):
        h, w = frame.shape[:2]
        bar_h = 28
        y_start = h - bar_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_start), (w, h), (0, 40, 40), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        texts = [f'"{r.text}"' for r in ocr_results[:6]]  # show max 6
        label = "TEXT: " + "  |  ".join(texts)
        cv2.putText(frame, label, (8, h - 8),
                    FONT, 0.50, self.color_ocr, 1, cv2.LINE_AA)

    # ── HELPERS ──────────────────────────────────────────────────────
    def _get_color(self, det: dict) -> tuple:
        if det.get('class_name') == 'person':
            name = det.get('person_name') or 'Unknown'
            return self.color_known if name != 'Unknown' else self.color_unknown
        return self.color_object

    def draw_alert(self, frame: np.ndarray, message: str,
                   color=(0, 0, 255)) -> np.ndarray:
        """Flash an alert banner on the frame."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h // 2 - 25), (w, h // 2 + 25), color, -1)
        (tw, _), _ = cv2.getTextSize(message, FONT_BOLD, 0.9, 2)
        x = max(8, (w - tw) // 2)
        cv2.putText(frame, message, (x, h // 2 + 8),
                    FONT_BOLD, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def draw_image_report(self, frame: np.ndarray, detections: list,
                          ocr_results: list) -> np.ndarray:
        """Draw a full analysis overlay for static image mode."""
        annotated = self.draw(frame, detections, ocr_results, source='image')
        return annotated
