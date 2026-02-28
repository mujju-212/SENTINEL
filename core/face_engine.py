"""
MODULE 3 - Face Recognition & Identification
Detects faces inside person bounding boxes, then matches
against the known_faces/ database using DeepFace + FaceNet.

Runs every Nth frame (configurable, default every 3rd).
Returns: person name + confidence OR 'Unknown'
"""

import cv2
import numpy as np
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class FaceEngine:
    """
    Face detection and recognition.
    Thread-safe - safe to call from background threads.
    """

    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.known_faces_dir = Path(config.get('known_faces_dir', 'database/known_faces'))
        self.threshold = config.get('recognition_threshold', 0.60)
        self.detector_backend = config.get('detector_backend', 'retinaface')
        self.model_name = config.get('model_name', 'Facenet')
        self._lock = threading.Lock()
        self.deepface = None

        if self.enabled:
            self._load_deepface()

    # ── LOAD DEEPFACE ────────────────────────────────────────────────
    def _load_deepface(self):
        try:
            import os
            # Ensure DeepFace models download to D: not C:
            deepface_home = self.config.get('model_dir',
                            os.environ.get('DEEPFACE_HOME',
                            r'D:\AI_Cache\deepface'))
            os.environ['DEEPFACE_HOME'] = deepface_home
            from deepface import DeepFace
            self.deepface = DeepFace
            logger.info(f"[FaceEngine] DeepFace loaded (models → {deepface_home})")
        except ImportError:
            logger.error("[FaceEngine] DeepFace not installed → pip install deepface")

    # ── MAIN: RECOGNIZE FACES IN FRAME ──────────────────────────────
    def recognize(self, frame: np.ndarray, persons: list) -> list:
        """
        For each detected person bbox, find + recognize their face.

        Args:
            frame:   Full BGR video frame
            persons: List of person detection dicts from ObjectDetector

        Returns:
            Same list with 'person_name' and 'face_confidence' filled
        """
        if not self.enabled or self.deepface is None or not persons:
            return persons

        # Ensure known_faces directory exists
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        person_dirs = [d for d in self.known_faces_dir.iterdir() if d.is_dir()]

        updated = []
        for person in persons:
            result = person.copy()
            result['person_name'] = 'Unknown'
            result['face_confidence'] = 0.0

            if person_dirs:
                try:
                    crop = self._crop_with_padding(frame, person['bbox'])
                    if crop is not None and crop.size > 0:
                        with self._lock:
                            name, conf = self._match(crop)
                        result['person_name'] = name
                        result['face_confidence'] = conf
                except Exception as e:
                    logger.debug(f"[FaceEngine] Recognition error: {e}")

            updated.append(result)

        return updated

    # ── CROP PERSON WITH PADDING ─────────────────────────────────────
    def _crop_with_padding(self, frame: np.ndarray, bbox: tuple, pad: int = 20):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    # ── DEEPFACE MATCH ───────────────────────────────────────────────
    def _match(self, face_img: np.ndarray) -> tuple:
        """Search database for best face match. Returns (name, confidence)."""
        try:
            results = self.deepface.find(
                img_path=face_img,
                db_path=str(self.known_faces_dir),
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )

            if results and len(results) > 0 and not results[0].empty:
                df = results[0]
                dist_cols = [c for c in df.columns if 'distance' in c.lower()]
                if dist_cols:
                    best = df.loc[df[dist_cols[0]].idxmin()]
                    distance = float(best[dist_cols[0]])
                    confidence = max(0.0, 1.0 - distance)

                    if confidence >= self.threshold:
                        name = Path(str(best['identity'])).parent.name
                        return name, round(confidence, 3)

        except Exception as e:
            logger.debug(f"[FaceEngine] DeepFace.find error: {e}")

        return 'Unknown', 0.0

    # ── ADD NEW PERSON ───────────────────────────────────────────────
    def add_person(self, name: str, photo_paths: list) -> int:
        """
        Add photos for a new person to the database.

        Args:
            name:        Person's name (becomes folder name)
            photo_paths: List of image file paths

        Returns:
            Number of photos successfully saved
        """
        person_dir = self.known_faces_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, path in enumerate(photo_paths):
            try:
                img = cv2.imread(str(path))
                if img is not None:
                    out = person_dir / f"{name}_{i:03d}.jpg"
                    cv2.imwrite(str(out), img)
                    saved += 1
                    logger.info(f"[FaceEngine] Saved: {out}")
                else:
                    logger.warning(f"[FaceEngine] Cannot read image: {path}")
            except Exception as e:
                logger.error(f"[FaceEngine] Error saving {path}: {e}")

        logger.info(f"[FaceEngine] Added {saved}/{len(photo_paths)} photos for '{name}'")
        return saved

    # ── HELPERS ──────────────────────────────────────────────────────
    def list_known_persons(self) -> list:
        """Return list of person names in the database."""
        if not self.known_faces_dir.exists():
            return []
        return sorted([d.name for d in self.known_faces_dir.iterdir() if d.is_dir()])

    def is_ready(self) -> bool:
        return self.deepface is not None or not self.enabled
