"""
MODULE 5 - OCR Text Reading
Uses EasyOCR to extract visible text from each frame.

Runs every Nth frame (default every 10th) because OCR is
computationally heavy. Results are cached between runs so
the display always shows the most recent text.
"""

import cv2
import numpy as np
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class OCRResult:
    """Holds a single detected text block."""
    def __init__(self, text: str, confidence: float, bbox: tuple):
        self.text = text
        self.confidence = confidence  # 0.0 – 1.0
        self.bbox = bbox              # (x1, y1, x2, y2)

    def __repr__(self):
        return f"OCRResult('{self.text}', conf={self.confidence:.2f})"


class OCREngine:
    """
    EasyOCR-based text extractor.

    Thread-safe - designed to be called from background thread.
    Caches the last result so the main loop can read it safely.
    """

    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.languages = config.get('languages', ['en'])
        self.gpu = config.get('gpu', True)
        self.min_confidence = config.get('min_confidence', 0.30)
        self.reader = None
        self._lock = threading.Lock()
        self._cached_results: list[OCRResult] = []

        if self.enabled:
            self._load_reader()

    # ── LOAD EASYOCR ─────────────────────────────────────────────────
    def _load_reader(self):
        try:
            import easyocr
            import os
            # Use configured cache dir, fall back to env var, then default
            model_dir = self.config.get('model_dir',
                        os.environ.get('EASYOCR_MODULE_PATH',
                        r'D:\AI_Cache\easyocr'))
            logger.info(f"[OCR] Loading EasyOCR (gpu={self.gpu}, langs={self.languages}, models={model_dir})...")
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                model_storage_directory=model_dir,
                verbose=False
            )
            logger.info("[OCR] EasyOCR ready")
        except ImportError:
            logger.error("[OCR] EasyOCR not installed → pip install easyocr")
        except Exception as e:
            logger.error(f"[OCR] Failed to load reader: {e}")

    # ── RUN OCR ON FRAME ─────────────────────────────────────────────
    def read_frame(self, frame: np.ndarray) -> list:
        """
        Extract all text from a frame.

        Args:
            frame: BGR image

        Returns:
            List of OCRResult objects (also cached for get_cached)
        """
        if not self.enabled or self.reader is None:
            return []

        try:
            raw = self.reader.readtext(frame, detail=1)
            results = []
            for item in raw:
                points, text, conf = item
                if conf < self.min_confidence or not text.strip():
                    continue
                # Convert polygon points to bounding box
                pts = np.array(points, dtype=np.int32)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                results.append(OCRResult(text.strip(), conf, (x1, y1, x2, y2)))

            with self._lock:
                self._cached_results = results

            if results:
                texts = [r.text for r in results]
                logger.debug(f"[OCR] Found: {texts}")

            return results

        except Exception as e:
            logger.warning(f"[OCR] read_frame error: {e}")
            return []

    # ── OCR ON SINGLE IMAGE FILE ─────────────────────────────────────
    def read_image_file(self, image_path: str) -> list:
        """
        Full OCR analysis on a single image file.
        Best for documents, screenshots, and detailed text extraction.

        Returns:
            List of OCRResult objects
        """
        if not self.enabled or self.reader is None:
            return []

        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"[OCR] Cannot read image: {image_path}")
                return []
            return self.read_frame(img)
        except Exception as e:
            logger.error(f"[OCR] read_image_file error: {e}")
            return []

    # ── CACHED RESULTS ───────────────────────────────────────────────
    def get_cached(self) -> list:
        """Thread-safe read of last OCR results."""
        with self._lock:
            return list(self._cached_results)

    def get_text_summary(self) -> str:
        """Return all detected text as a single string."""
        results = self.get_cached()
        if not results:
            return ""
        return "  |  ".join([r.text for r in results])

    def is_ready(self) -> bool:
        return self.reader is not None or not self.enabled
