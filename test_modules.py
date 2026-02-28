"""
MODULE TESTER
──────────────────────────────────────────────────────
Tests each module individually before running the
full pipeline. Helps isolate installation issues.

Usage:
  python test_modules.py           ← test all
  python test_modules.py --module gpu
  python test_modules.py --module detector
  python test_modules.py --module face
  python test_modules.py --module ocr
  python test_modules.py --module database
  python test_modules.py --module camera
"""

import sys
import time
import logging
import argparse
from pathlib import Path

import yaml
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s [%(levelname)s] %(message)s')


# ────────────────────────────────────────────────────────────────────
def load_config(path='config.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def ok(msg):   print(f"  ✓  {msg}")
def fail(msg): print(f"  ✗  {msg}")
def info(msg): print(f"  ℹ  {msg}")
def section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


# ── TEST GPU ─────────────────────────────────────────────────────────
def test_gpu():
    section("TEST: GPU / CUDA")
    try:
        import torch
        ok(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            ok(f"CUDA available: YES")
            ok(f"GPU: {torch.cuda.get_device_name(0)}")
            ok(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Quick tensor op on GPU
            t = torch.randn(1000, 1000).cuda()
            _ = t @ t.T
            ok("GPU computation test passed")
        else:
            fail("CUDA NOT available — will run on CPU (slower)")
            info("If you have a GPU: install CUDA 11.8 + torch cu118")
    except ImportError:
        fail("PyTorch not installed → pip install torch torchvision")


# ── TEST DETECTOR ─────────────────────────────────────────────────────
def test_detector(config: dict):
    section("TEST: YOLOv8 Object Detector")
    try:
        from core.detector import ObjectDetector
        det = ObjectDetector({**config['detection'], 'models_dir': config['paths']['models_dir']})
        ok(f"Model loaded | {det.get_device_info()}")

        # Fake black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        result = det.detect(frame)
        elapsed = (time.perf_counter() - t0) * 1000
        ok(f"detect() ran in {elapsed:.1f}ms on blank frame")
        ok(f"Returned keys: {list(result.keys())}")

        # Test with a real image if available
        test_dir = Path(config['paths']['test_images_dir'])
        images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        if images:
            frame2 = cv2.imread(str(images[0]))
            t0 = time.perf_counter()
            result2 = det.detect(frame2)
            elapsed2 = (time.perf_counter() - t0) * 1000
            ok(f"Real image test: {elapsed2:.1f}ms  |  Detected: {len(result2['detections'])} objects")
        else:
            info(f"Place images in {test_dir} for real image test")

    except Exception as e:
        fail(f"Detector error: {e}")


# ── TEST FACE ENGINE ──────────────────────────────────────────────────
def test_face(config: dict):
    section("TEST: Face Recognition (DeepFace)")
    try:
        from core.face_engine import FaceEngine
        fe = FaceEngine(config['face_recognition'])

        if fe.deepface:
            ok("DeepFace loaded")
        else:
            fail("DeepFace not loaded → pip install deepface")
            return

        persons = fe.list_known_persons()
        if persons:
            ok(f"Known persons in DB: {persons}")
        else:
            info("No persons in database yet")
            info(f"Run: python add_person.py  to add someone")

        # Test with tiny blank crop
        blank = np.ones((120, 80, 3), dtype=np.uint8) * 128
        fake_person = {'bbox': (0, 0, 80, 120), 'confidence': 0.9,
                       'class_name': 'person', 'track_id': 1,
                       'person_name': None, 'face_confidence': 0.0}
        if not persons:
            ok("recognize() skipped (no persons in DB) — OK")
        else:
            t0 = time.perf_counter()
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            result = fe.recognize(frame, [fake_person])
            elapsed = (time.perf_counter() - t0) * 1000
            ok(f"recognize() returned: {result[0].get('person_name')} in {elapsed:.0f}ms")

    except Exception as e:
        fail(f"FaceEngine error: {e}")


# ── TEST OCR ──────────────────────────────────────────────────────────
def test_ocr(config: dict):
    section("TEST: EasyOCR Text Reader")
    try:
        from core.ocr_engine import OCREngine
        ocr = OCREngine(config['ocr'])

        if ocr.reader:
            ok("EasyOCR reader loaded")
        else:
            fail("EasyOCR not loaded → pip install easyocr")
            return

        # Create a synthetic test image with text
        img = np.ones((100, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "HELLO WORLD", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)

        t0 = time.perf_counter()
        results = ocr.read_frame(img)
        elapsed = (time.perf_counter() - t0) * 1000

        ok(f"read_frame() ran in {elapsed:.0f}ms")
        if results:
            for r in results:
                ok(f"Found text: \"{r.text}\" (conf={r.confidence:.2f})")
        else:
            info("No text detected in test image (OCR may need GPU warmed up)")

    except Exception as e:
        fail(f"OCR error: {e}")


# ── TEST DATABASE ─────────────────────────────────────────────────────
def test_database(config: dict):
    section("TEST: SQLite Database")
    try:
        from utils.database import DatabaseManager
        db = DatabaseManager(config['database'])
        ok(f"Database created: {db.get_db_path()}")

        # Insert test detection
        fake_det = [{
            'bbox': (10, 20, 100, 200),
            'class_name': 'person',
            'confidence': 0.95,
            'face_confidence': 0.88,
            'track_id': 999,
            'person_name': '_test_person_',
            'time_in_frame': 3.5
        }]
        db.save_detections(fake_det, source='test')
        ok("save_detections() OK")

        # Read back
        rows = db.get_recent_detections(limit=5)
        if rows:
            ok(f"get_recent_detections() returned {len(rows)} rows")
        else:
            fail("No rows returned after insert")

        # Stats
        stats = db.get_todays_stats()
        ok(f"Today stats: {stats}")

        # Cleanup test record
        import sqlite3
        with sqlite3.connect(db.get_db_path()) as conn:
            conn.execute("DELETE FROM detections WHERE person_name = '_test_person_'")
            conn.commit()
        ok("Test record cleaned up")

    except Exception as e:
        fail(f"Database error: {e}")


# ── TEST CAMERA ───────────────────────────────────────────────────────
def test_camera(config: dict):
    section("TEST: Camera / Webcam")
    cam_id = config['camera']['device_id']
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        fail(f"Cannot open camera #{cam_id}")
        info("Check: camera is connected and not in use by another app")
        return

    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        ok(f"Camera #{cam_id} opened: {w}x{h}")
        ok("Frame captured successfully")
        # Save test frame
        Path('output').mkdir(exist_ok=True)
        cv2.imwrite('output/camera_test.jpg', frame)
        ok("Test frame saved: output/camera_test.jpg")
    else:
        fail("Failed to capture frame from camera")

    cap.release()


# ── TEST TRACKER ──────────────────────────────────────────────────────
def test_tracker(config: dict):
    section("TEST: Object Tracker")
    try:
        from core.tracker import ObjectTracker
        tracker = ObjectTracker(config['tracking'])
        ok("ObjectTracker initialized")

        # Simulate 3 frames
        fake_dets = [
            {'track_id': 1, 'class_name': 'person', 'bbox': (10, 20, 60, 120),
             'confidence': 0.9, 'person_name': 'Alice', 'face_confidence': 0.85},
            {'track_id': 2, 'class_name': 'car', 'bbox': (100, 80, 300, 200),
             'confidence': 0.88, 'person_name': None, 'face_confidence': 0.0},
        ]
        for i in range(3):
            enriched = tracker.update(fake_dets)
            time.sleep(0.1)

        ok(f"Tracker update x3 OK | Active tracks: {tracker.total_tracks()}")
        rec = tracker.get_record(1)
        if rec:
            ok(f"Track #1: '{rec.class_name}', time={rec.time_in_frame:.2f}s, frames={rec.frame_count}")

    except Exception as e:
        fail(f"Tracker error: {e}")


# ── TEST VISUALIZER ───────────────────────────────────────────────────
def test_visualizer(config: dict):
    section("TEST: Visualizer")
    try:
        from utils.visualizer import Visualizer
        viz = Visualizer(config['display'])
        ok("Visualizer initialized")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_dets = [
            {'track_id': 1, 'class_name': 'person', 'bbox': (50, 50, 200, 350),
             'confidence': 0.93, 'person_name': 'Alice', 'face_confidence': 0.88,
             'time_in_frame': 5.2, 'trail': [(125, 200), (126, 198)]},
            {'track_id': 2, 'class_name': 'car', 'bbox': (320, 100, 580, 280),
             'confidence': 0.81, 'person_name': None, 'face_confidence': 0.0,
             'time_in_frame': 2.0, 'trail': []},
        ]
        annotated = viz.draw(frame, fake_dets, [], source='test')
        out = 'output/visualizer_test.jpg'
        cv2.imwrite(out, annotated)
        ok(f"draw() OK — saved: {out}")

    except Exception as e:
        fail(f"Visualizer error: {e}")


# ── RUN ALL ───────────────────────────────────────────────────────────
def run_all(config: dict):
    print("\n" + "═" * 50)
    print("  DRONE VISION AI — Module Tests")
    print("═" * 50)
    test_gpu()
    test_detector(config)
    test_face(config)
    test_ocr(config)
    test_database(config)
    test_camera(config)
    test_tracker(config)
    test_visualizer(config)
    print("\n" + "═" * 50)
    print("  Tests complete!")
    print("  If all ✓ → run: python main.py --mode camera")
    print("═" * 50 + "\n")


# ── MAIN ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Test Drone Vision AI modules')
    parser.add_argument('--module', type=str, default='all',
                        choices=['all', 'gpu', 'detector', 'face', 'ocr',
                                 'database', 'camera', 'tracker', 'visualizer'],
                        help='Which module to test')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    # Ensure output dir
    Path('output').mkdir(exist_ok=True)

    config = load_config(args.config)

    dispatch = {
        'gpu':        lambda: test_gpu(),
        'detector':   lambda: test_detector(config),
        'face':       lambda: test_face(config),
        'ocr':        lambda: test_ocr(config),
        'database':   lambda: test_database(config),
        'camera':     lambda: test_camera(config),
        'tracker':    lambda: test_tracker(config),
        'visualizer': lambda: test_visualizer(config),
    }

    if args.module == 'all':
        run_all(config)
    else:
        dispatch[args.module]()
        print()


if __name__ == '__main__':
    main()
