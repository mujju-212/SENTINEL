"""
MODULE 6 - Database & Logging
SQLite database for storing all detections, OCR results,
and person visit logs. Auto-creates tables on first run.

Three tables:
  detections  → every detected object per frame
  ocr_results → every OCR text extraction
  person_logs → aggregated visit log per person
"""

import sqlite3
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── SQL SCHEMAS ──────────────────────────────────────────────────────
CREATE_DETECTIONS = """
CREATE TABLE IF NOT EXISTS detections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    object_type  TEXT    NOT NULL,
    person_name  TEXT,
    confidence   REAL,
    face_conf    REAL,
    track_id     INTEGER,
    x1           INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    input_source TEXT,
    time_in_frame REAL
);
"""

CREATE_OCR = """
CREATE TABLE IF NOT EXISTS ocr_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    text_found  TEXT    NOT NULL,
    confidence  REAL,
    x1          INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER
);
"""

CREATE_PERSON_LOGS = """
CREATE TABLE IF NOT EXISTS person_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name   TEXT    NOT NULL UNIQUE,
    first_seen    TEXT    NOT NULL,
    last_seen     TEXT    NOT NULL,
    total_visits  INTEGER DEFAULT 0,
    total_seconds REAL    DEFAULT 0.0
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_det_timestamp  ON detections  (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_det_person     ON detections  (person_name);",
    "CREATE INDEX IF NOT EXISTS idx_ocr_timestamp  ON ocr_results (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_log_person     ON person_logs (person_name);",
]


class DatabaseManager:
    """Thread-safe SQLite manager for all Drone Vision AI data."""

    def __init__(self, config: dict):
        self.config = config
        db_path = Path(config.get('path', 'database/vision_ai.db'))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()
        logger.info(f"[DB] Database ready: {self.db_path}")

    # ── DB INIT ──────────────────────────────────────────────────────
    def _init_db(self):
        with self._connect() as conn:
            conn.execute(CREATE_DETECTIONS)
            conn.execute(CREATE_OCR)
            conn.execute(CREATE_PERSON_LOGS)
            for idx in CREATE_INDEXES:
                conn.execute(idx)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── SAVE DETECTIONS ──────────────────────────────────────────────
    def save_detections(self, detections: list, source: str = 'camera'):
        """Save a batch of detections from the current frame."""
        if not detections:
            return

        now = datetime.now().isoformat(sep=' ', timespec='seconds')
        rows = []
        for det in detections:
            x1, y1, x2, y2 = det.get('bbox', (0, 0, 0, 0))
            rows.append((
                now,
                det.get('class_name', 'unknown'),
                det.get('person_name') or None,
                round(det.get('confidence', 0.0), 4),
                round(det.get('face_confidence', 0.0), 4),
                det.get('track_id'),
                x1, y1, x2, y2,
                source,
                round(det.get('time_in_frame', 0.0), 2)
            ))

        sql = """
            INSERT INTO detections
                (timestamp, object_type, person_name, confidence, face_conf,
                 track_id, x1, y1, x2, y2, input_source, time_in_frame)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._lock:
            with self._connect() as conn:
                conn.executemany(sql, rows)
                conn.commit()

    # ── SAVE OCR RESULTS ─────────────────────────────────────────────
    def save_ocr_results(self, ocr_results: list):
        """Save a batch of OCR text detections."""
        if not ocr_results:
            return

        now = datetime.now().isoformat(sep=' ', timespec='seconds')
        rows = []
        for r in ocr_results:
            x1, y1, x2, y2 = r.bbox
            rows.append((now, r.text, round(r.confidence, 4), x1, y1, x2, y2))

        sql = """
            INSERT INTO ocr_results (timestamp, text_found, confidence, x1, y1, x2, y2)
            VALUES (?,?,?,?,?,?,?)
        """
        with self._lock:
            with self._connect() as conn:
                conn.executemany(sql, rows)
                conn.commit()

    # ── UPDATE PERSON LOG ─────────────────────────────────────────────
    def update_person_log(self, person_name: str, time_in_frame: float = 0.0):
        """Update the aggregated person visit log."""
        if not person_name or person_name in ('Unknown', 'N/A'):
            return

        now = datetime.now().isoformat(sep=' ', timespec='seconds')

        sql_upsert = """
            INSERT INTO person_logs (person_name, first_seen, last_seen, total_visits, total_seconds)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(person_name) DO UPDATE SET
                last_seen     = excluded.last_seen,
                total_visits  = total_visits + 1,
                total_seconds = total_seconds + excluded.total_seconds
        """
        with self._lock:
            with self._connect() as conn:
                conn.execute(sql_upsert, (person_name, now, now, round(time_in_frame, 2)))
                conn.commit()

    # ── QUERIES ──────────────────────────────────────────────────────
    def get_recent_detections(self, limit: int = 50) -> list:
        sql = "SELECT * FROM detections ORDER BY id DESC LIMIT ?"
        with self._lock:
            with self._connect() as conn:
                return [dict(r) for r in conn.execute(sql, (limit,)).fetchall()]

    def get_person_log(self, name: Optional[str] = None) -> list:
        if name:
            sql = "SELECT * FROM person_logs WHERE person_name = ?"
            args = (name,)
        else:
            sql = "SELECT * FROM person_logs ORDER BY last_seen DESC"
            args = ()
        with self._lock:
            with self._connect() as conn:
                return [dict(r) for r in conn.execute(sql, args).fetchall()]

    def get_todays_stats(self) -> dict:
        today = datetime.now().strftime('%Y-%m-%d')
        with self._lock:
            with self._connect() as conn:
                det_count = conn.execute(
                    "SELECT COUNT(*) FROM detections WHERE timestamp LIKE ?", (f"{today}%",)
                ).fetchone()[0]
                person_count = conn.execute(
                    "SELECT COUNT(DISTINCT person_name) FROM detections WHERE timestamp LIKE ? AND person_name IS NOT NULL",
                    (f"{today}%",)
                ).fetchone()[0]
                ocr_count = conn.execute(
                    "SELECT COUNT(*) FROM ocr_results WHERE timestamp LIKE ?", (f"{today}%",)
                ).fetchone()[0]
        return {
            'date': today,
            'total_detections': det_count,
            'unique_persons': person_count,
            'ocr_extractions': ocr_count,
        }

    # ── CLEANUP ──────────────────────────────────────────────────────
    def cleanup_old_logs(self, keep_days: int = 30):
        """Delete records older than keep_days."""
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat(sep=' ', timespec='seconds')
        with self._lock:
            with self._connect() as conn:
                d1 = conn.execute("DELETE FROM detections  WHERE timestamp < ?", (cutoff,)).rowcount
                d2 = conn.execute("DELETE FROM ocr_results WHERE timestamp < ?", (cutoff,)).rowcount
                conn.commit()
        logger.info(f"[DB] Cleanup: removed {d1} detections + {d2} OCR records older than {keep_days} days")

    def get_db_path(self) -> str:
        return self.db_path
