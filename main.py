"""
╔══════════════════════════════════════════════════╗
║          DRONE VISION AI - MAIN ENTRY POINT      ║
║  Ties together all 8 modules into one pipeline   ║
╚══════════════════════════════════════════════════╝

Usage:
  python main.py --mode camera
  python main.py --mode video  --input path/to/video.mp4
  python main.py --mode image  --input path/to/photo.jpg
  python main.py --mode camera --config custom_config.yaml

Press Q to quit live mode.
Press S to save current frame manually.
Press R to reset alert session.
"""

import cv2
import time
import logging
import argparse
import threading
import sys
from pathlib import Path

import yaml

# ── LOCAL IMPORTS ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from core.detector   import ObjectDetector
from core.face_engine import FaceEngine
from core.tracker    import ObjectTracker
from core.ocr_engine import OCREngine
from utils.database  import DatabaseManager
from utils.visualizer import Visualizer
from utils.alert     import AlertSystem

# ── LOGGING SETUP ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('output/logs/drone_vision.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#                        PIPELINE CLASS
# ════════════════════════════════════════════════════════════════

class DroneVisionPipeline:
    """
    Main processing pipeline.
    Manages frame-by-frame logic, threading, and all modules.
    """

    def __init__(self, config: dict, mode: str, input_path: str = None):
        self.config = config
        self.mode = mode                    # 'camera' | 'video' | 'image'
        self.input_path = input_path
        self.running = False

        # ── Frame counters ──
        self.frame_idx = 0
        self.face_every   = config['face_recognition'].get('process_every_n_frames', 3)
        self.ocr_every    = config['ocr'].get('process_every_n_frames', 10)
        self.db_every     = config['database'].get('save_every_n_frames', 5)

        # ── OCR background state ──
        self._ocr_results = []
        self._ocr_lock = threading.Lock()
        self._ocr_running = False

        # ── Initialize all modules ──
        self._init_modules()

    # ── MODULE INIT ──────────────────────────────────────────────────
    def _init_modules(self):
        cfg = self.config
        logger.info("=" * 52)
        logger.info("  DRONE VISION AI  —  Initializing modules")
        logger.info("=" * 52)

        # Ensure output dirs exist
        Path('output/logs').mkdir(parents=True, exist_ok=True)
        Path('output/screenshots').mkdir(parents=True, exist_ok=True)
        Path('models').mkdir(parents=True, exist_ok=True)

        self.detector   = ObjectDetector({
            **cfg['detection'],
            'models_dir': cfg['paths']['models_dir']
        })
        self.face_engine = FaceEngine(cfg['face_recognition'])
        self.tracker    = ObjectTracker(cfg['tracking'])
        self.ocr        = OCREngine(cfg['ocr'])
        self.db         = DatabaseManager(cfg['database'])
        self.viz        = Visualizer(cfg['display'])
        self.alert      = AlertSystem(cfg['alerts'])

        logger.info("  All modules loaded ✓")
        logger.info(f"  Device : {self.detector.get_device_info()}")
        logger.info(f"  Mode   : {self.mode.upper()}")
        logger.info("=" * 52)

    # ════════════════════════════════════════════════════════════════
    #   CAMERA / VIDEO MODE
    # ════════════════════════════════════════════════════════════════

    def run_stream(self):
        """Run detection loop on live camera or video file."""
        if self.mode == 'camera':
            cam_id = self.config['camera']['device_id']
            cap = cv2.VideoCapture(cam_id)
            w = self.config['camera']['width']
            h = self.config['camera']['height']
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            source_label = f'Camera #{cam_id}'
        else:
            cap = cv2.VideoCapture(self.input_path)
            source_label = Path(self.input_path).name

        if not cap.isOpened():
            logger.error(f"Cannot open: {self.input_path or 'camera'}")
            return

        logger.info(f"[Main] Stream opened: {source_label}")
        logger.info("[Main] Press Q=quit  S=screenshot  R=reset alerts")

        self.running = True
        window_name = "Drone Vision AI"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                if self.mode == 'video':
                    logger.info("[Main] End of video")
                else:
                    logger.warning("[Main] Camera frame grab failed")
                break

            self.frame_idx += 1
            annotated = self._process_frame(frame, source_label)

            # Overlay active alert banner
            active_alert = self.alert.get_active_alert()
            if active_alert:
                color = (0, 0, 200) if 'UNKNOWN' in active_alert.alert_type else (0, 160, 0)
                annotated = self.viz.draw_alert(annotated, active_alert.message, color)

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:    # Q or ESC
                self.running = False
            elif key == ord('s'):
                self._save_screenshot(annotated, manual=True)
            elif key == ord('r'):
                self.alert.reset_session()
                logger.info("[Main] Alert session reset")

        cap.release()
        cv2.destroyAllWindows()
        self._print_session_stats()

    # ════════════════════════════════════════════════════════════════
    #   IMAGE MODE
    # ════════════════════════════════════════════════════════════════

    def run_image(self):
        """Full analysis on a single image file."""
        if not self.input_path or not Path(self.input_path).exists():
            logger.error(f"Image not found: {self.input_path}")
            return

        logger.info(f"[Main] Analyzing image: {self.input_path}")
        frame = cv2.imread(self.input_path)
        if frame is None:
            logger.error("Cannot read image file")
            return

        # Run full pipeline once (no frame-skip for images)
        det_result = self.detector.detect(frame)
        detections = det_result['detections']
        persons    = det_result['persons']

        # Face recognition on all persons
        if persons:
            persons = self.face_engine.recognize(frame, persons)
            # Update detections list with enriched person data
            person_ids = {p['track_id'] for p in persons}
            detections = [p if p['track_id'] in person_ids and p['class_name'] == 'person'
                         else d for d, p in [(d, d) for d in detections]]
            # Simpler merge:
            per_dict = {p['track_id']: p for p in persons}
            detections = [per_dict.get(d['track_id'], d) for d in detections]

        # OCR full image
        ocr_results = []
        if self.config['ocr']['enabled']:
            ocr_results = self.ocr.read_frame(frame)

        # Tracker update
        detections = self.tracker.update(detections)

        # Draw
        annotated = self.viz.draw_image_report(frame, detections, ocr_results)

        # Save result
        out_path = Path('output') / f"result_{Path(self.input_path).stem}.jpg"
        cv2.imwrite(str(out_path), annotated)
        logger.info(f"[Main] Saved result: {out_path}")

        # Print report
        self._print_image_report(detections, ocr_results)

        # Show image
        cv2.imshow("Drone Vision AI - Image Result", annotated)
        logger.info("[Main] Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ════════════════════════════════════════════════════════════════
    #   CORE FRAME PROCESSING
    # ════════════════════════════════════════════════════════════════

    def _process_frame(self, frame, source: str) -> any:
        """Run the full AI pipeline on one frame."""
        idx = self.frame_idx

        # ── MODULE 1+2: Object + Human Detection (every frame) ──────
        det_result = self.detector.detect(frame)
        detections = det_result['detections']
        persons    = det_result['persons']

        # ── MODULE 3: Face Recognition (every Nth frame) ─────────────
        if persons and idx % self.face_every == 0:
            persons = self.face_engine.recognize(frame, persons)
            # Merge updated persons back into detections
            per_dict = {p['track_id']: p for p in persons}
            detections = [per_dict.get(d['track_id'], d) if d['class_name'] == 'person'
                         else d for d in detections]

        # ── MODULE 4: Tracking (enriches detections with trail+time) ──
        detections = self.tracker.update(detections)

        # ── MODULE 5: OCR (every Nth frame, runs async) ──────────────
        if idx % self.ocr_every == 0 and not self._ocr_running:
            self._run_ocr_async(frame.copy())

        with self._ocr_lock:
            ocr_results = list(self._ocr_results)

        # ── MODULE 6: Database save (every Nth frame) ─────────────────
        if idx % self.db_every == 0:
            self.db.save_detections(detections, source=source)
            if ocr_results:
                self.db.save_ocr_results(ocr_results)
            for det in detections:
                if det.get('class_name') == 'person':
                    name = det.get('person_name') or 'Unknown'
                    if name != 'Unknown':
                        self.db.update_person_log(name, det.get('time_in_frame', 0))

        # ── MODULE 8: Alerts ─────────────────────────────────────────
        self.alert.check(detections, ocr_results, frame)

        # ── MODULE 7: Visualize ───────────────────────────────────────
        annotated = self.viz.draw(frame, detections, ocr_results, source=source)

        # ── Console log (every 30 frames) ────────────────────────────
        if idx % 30 == 0:
            self._log_status(detections, ocr_results)

        return annotated

    # ── OCR ASYNC ────────────────────────────────────────────────────
    def _run_ocr_async(self, frame):
        """Run OCR in background thread so it doesn't block the frame loop."""
        def _worker():
            self._ocr_running = True
            results = self.ocr.read_frame(frame)
            with self._ocr_lock:
                self._ocr_results = results
            self._ocr_running = False
        threading.Thread(target=_worker, daemon=True).start()

    # ── UTILS ────────────────────────────────────────────────────────
    def _save_screenshot(self, frame, manual=False):
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        label = 'manual' if manual else 'auto'
        path = Path('output/screenshots') / f"{label}_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        logger.info(f"[Main] Screenshot saved: {path}")

    def _log_status(self, detections: list, ocr_results: list):
        persons = [d for d in detections if d.get('class_name') == 'person']
        objects = [d for d in detections if d.get('class_name') != 'person']
        print("\n" + "─" * 50)
        print(f"  Frame #{self.frame_idx:05d}  |  Objects: {len(detections)}")
        for p in persons:
            name = p.get('person_name') or 'Unknown'
            conf = p.get('face_confidence', 0.0)
            conf_str = f" ({conf*100:.0f}%)" if conf > 0 else ""
            print(f"  👤 {name}{conf_str}  [Track #{p['track_id']}]")
        for o in objects:
            print(f"  📦 {o['class_name']}  conf={o['confidence']:.2f}  [Track #{o['track_id']}]")
        if ocr_results:
            texts = [f'"{r.text}"' for r in ocr_results]
            print(f"  📝 OCR: {', '.join(texts)}")

    def _print_image_report(self, detections: list, ocr_results: list):
        print("\n" + "═" * 52)
        print("  DRONE VISION AI — Image Analysis Report")
        print("═" * 52)
        persons = [d for d in detections if d.get('class_name') == 'person']
        objects = [d for d in detections if d.get('class_name') != 'person']
        print(f"\n  Total objects detected : {len(detections)}")
        print(f"  Persons               : {len(persons)}")
        print(f"  Other objects         : {len(objects)}")
        if persons:
            print("\n  PERSONS:")
            for p in persons:
                name = p.get('person_name') or 'Unknown'
                conf = p.get('face_confidence', 0.0)
                print(f"    → {name}  ({conf*100:.0f}% face confidence)")
        if objects:
            print("\n  OBJECTS:")
            for o in objects:
                print(f"    → {o['class_name']}  ({o['confidence']*100:.0f}%)")
        if ocr_results:
            print("\n  TEXT FOUND (OCR):")
            for r in ocr_results:
                print(f"    → \"{r.text}\"  ({r.confidence*100:.0f}%)")
        print("\n" + "═" * 52)

    def _print_session_stats(self):
        stats = self.db.get_todays_stats()
        print("\n" + "═" * 52)
        print("  SESSION STATS")
        print("═" * 52)
        print(f"  Total frames processed : {self.frame_idx}")
        print(f"  Detections logged      : {stats['total_detections']}")
        print(f"  Unique persons today   : {stats['unique_persons']}")
        print(f"  OCR extractions today  : {stats['ocr_extractions']}")
        print("═" * 52)


# ════════════════════════════════════════════════════════════════
#                         CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════

def load_config(config_path: str = 'config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Drone Vision AI — Real-time Object Detection, Face Recognition & OCR'
    )
    parser.add_argument('--mode', choices=['camera', 'video', 'image'],
                        default='camera', help='Input source mode')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to video or image file (for video/image modes)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode in ('video', 'image') and not args.input:
        print(f"ERROR: --input is required for mode '{args.mode}'")
        print(f"  Example: python main.py --mode {args.mode} --input path/to/file")
        sys.exit(1)

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║          🚁  DRONE VISION AI  🚁                 ║")
    print("║     Object Detection + Face ID + OCR             ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    config = load_config(args.config)
    pipeline = DroneVisionPipeline(config, mode=args.mode, input_path=args.input)

    if args.mode in ('camera', 'video'):
        pipeline.run_stream()
    else:
        pipeline.run_image()


if __name__ == '__main__':
    main()
