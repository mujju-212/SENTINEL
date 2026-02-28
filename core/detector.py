"""
MODULE 1 & 2 - Object Detection + Human Detection
Uses YOLOv8 Nano for detecting all 80 COCO objects including people.
Runs on every frame using GPU.
"""

import cv2
import numpy as np
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Full COCO class list (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class ObjectDetector:
    """
    YOLOv8-based detector for all objects and persons.
    Uses built-in ByteTrack from ultralytics for tracking.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = self._resolve_device()
        self.model = None
        self._load_model()
        logger.info(f"[Detector] Ready | Device: {self.device} | Model: {config.get('model', 'yolov8n.pt')}")

    # ── DEVICE SELECTION ────────────────────────────────────────────
    def _resolve_device(self) -> str:
        preference = self.config.get('device', 'auto')
        if preference == 'auto':
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                logger.info(f"[Detector] GPU found: {name}")
                return 'cuda'
            else:
                logger.warning("[Detector] No GPU found, using CPU")
                return 'cpu'
        return preference

    # ── MODEL LOADING ────────────────────────────────────────────────
    def _load_model(self):
        """Load YOLOv8 model. Auto-downloads on first run."""
        from ultralytics import YOLO

        models_dir = Path(self.config.get('models_dir', 'models'))
        models_dir.mkdir(parents=True, exist_ok=True)

        model_name = self.config.get('model', 'yolov8n.pt')
        local_path = models_dir / model_name

        if local_path.exists():
            self.model = YOLO(str(local_path))
            logger.info(f"[Detector] Loaded model: {local_path}")
        else:
            logger.info(f"[Detector] Downloading {model_name} (first run only)...")
            self.model = YOLO(model_name)
            # Save to models directory
            import shutil
            try:
                shutil.copy(model_name, str(local_path))
            except Exception:
                pass
            logger.info("[Detector] Model ready")

    # ── MAIN DETECTION ───────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> dict:
        """
        Detect all objects in a frame with tracking.

        Args:
            frame: BGR image from OpenCV

        Returns:
            dict:
                'detections' → list of all detected objects (dicts)
                'persons'    → list of person detections only
                'frame'      → the input frame unchanged
        """
        if self.model is None:
            return {'detections': [], 'persons': [], 'frame': frame}

        conf = self.config.get('confidence', 0.5)
        imgsz = self.config.get('image_size', 640)

        try:
            results = self.model.track(
                source=frame,
                persist=True,
                conf=conf,
                imgsz=imgsz,
                device=self.device,
                tracker="bytetrack.yaml",
                verbose=False
            )
        except Exception as e:
            logger.warning(f"[Detector] Tracking failed ({e}), using plain detect")
            try:
                results = self.model(frame, conf=conf, imgsz=imgsz, device=self.device, verbose=False)
            except Exception as e2:
                logger.error(f"[Detector] Detection failed: {e2}")
                return {'detections': [], 'persons': [], 'frame': frame}

        return self._parse_results(results, frame)

    # ── RESULT PARSING ───────────────────────────────────────────────
    def _parse_results(self, results, frame: np.ndarray) -> dict:
        detections = []
        persons = []

        if not results or len(results) == 0:
            return {'detections': detections, 'persons': persons, 'frame': frame}

        result = results[0]
        boxes = result.boxes

        if boxes is None:
            return {'detections': detections, 'persons': persons, 'frame': frame}

        for i, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf_score = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"obj_{cls_id}"
                track_id = int(box.id[0].cpu().numpy()) if (box.id is not None) else (i + 1)

                det = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf_score,
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'track_id': track_id,
                    'person_name': None,        # filled by FaceEngine
                    'face_confidence': 0.0,
                }

                detections.append(det)
                if cls_name == 'person':
                    persons.append(det)

            except Exception as e:
                logger.debug(f"[Detector] Box parse error: {e}")

        return {'detections': detections, 'persons': persons, 'frame': frame}

    # ── HELPERS ──────────────────────────────────────────────────────
    def get_device_info(self) -> str:
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
        return "CPU Mode"

    def is_ready(self) -> bool:
        return self.model is not None
