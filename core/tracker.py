"""
MODULE 4 - Object Tracking
ByteTrack is built into ultralytics and handled automatically
in detector.py via model.track(tracker='bytetrack.yaml').

This module provides a lightweight wrapper that:
- Manages tracking state across frames
- Calculates time-in-frame for each track
- Detects entry and exit events
- Stores movement history per track
"""

import time
import logging
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger(__name__)


class TrackRecord:
    """Data for a single tracked object."""
    def __init__(self, track_id: int, class_name: str):
        self.track_id = track_id
        self.class_name = class_name
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.frame_count = 0
        self.person_name: Optional[str] = None
        # Store last N center positions for movement trail
        self.positions = deque(maxlen=30)

    @property
    def time_in_frame(self) -> float:
        """Seconds since first seen."""
        return self.last_seen - self.first_seen

    def update(self, bbox: tuple, person_name: Optional[str] = None):
        self.last_seen = time.time()
        self.frame_count += 1
        if person_name and person_name != 'Unknown':
            self.person_name = person_name
        # Add center point to trail
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        self.positions.append((cx, cy))


class ObjectTracker:
    """
    STATE MANAGER for all tracked objects.

    Works on top of ByteTrack (run inside YOLOv8).
    Simply keeps records so we can report:
    - How long each object has been visible
    - Entry / exit events
    - Movement history
    """

    def __init__(self, config: dict):
        self.config = config
        self.max_age_seconds = 5.0      # Remove track after 5s of no detection
        self.records: dict[int, TrackRecord] = {}
        self._new_entries: list[int] = []
        self._exits: list[int] = []
        logger.info("[Tracker] ObjectTracker initialized")

    # ── MAIN UPDATE ──────────────────────────────────────────────────
    def update(self, detections: list) -> list:
        """
        Update tracker state from current frame detections.

        Args:
            detections: list of detection dicts from ObjectDetector

        Returns:
            Same list with 'time_in_frame' and 'trail' keys added
        """
        self._new_entries = []
        self._exits = []
        active_ids = set()

        for det in detections:
            tid = det['track_id']
            active_ids.add(tid)

            if tid not in self.records:
                # New object entered frame
                self.records[tid] = TrackRecord(tid, det['class_name'])
                self._new_entries.append(tid)
                logger.debug(f"[Tracker] NEW track #{tid} ({det['class_name']})")

            self.records[tid].update(det['bbox'], det.get('person_name'))

        # Detect exits (track not seen this frame)
        now = time.time()
        stale = []
        for tid, rec in self.records.items():
            if tid not in active_ids:
                age = now - rec.last_seen
                if age > self.max_age_seconds:
                    stale.append(tid)
                    self._exits.append(tid)
                    logger.debug(f"[Tracker] EXIT track #{tid} ({rec.class_name}) after {rec.time_in_frame:.1f}s")

        for tid in stale:
            del self.records[tid]

        # Attach extra info to each detection
        enriched = []
        for det in detections:
            d = det.copy()
            rec = self.records.get(det['track_id'])
            if rec:
                d['time_in_frame'] = rec.time_in_frame
                d['trail'] = list(rec.positions)
            else:
                d['time_in_frame'] = 0.0
                d['trail'] = []
            enriched.append(d)

        return enriched

    # ── STATS ────────────────────────────────────────────────────────
    def get_record(self, track_id: int) -> Optional[TrackRecord]:
        return self.records.get(track_id)

    def get_new_entries(self) -> list:
        """Track IDs that appeared THIS frame."""
        return self._new_entries.copy()

    def get_exits(self) -> list:
        """Track IDs that left THIS frame."""
        return self._exits.copy()

    def total_tracks(self) -> int:
        return len(self.records)
