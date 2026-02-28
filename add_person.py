"""
ADD PERSON TO FACE DATABASE
────────────────────────────────────────────────────
Interactive script to add a new person to the
known_faces/ directory for face recognition.

Usage:
  python add_person.py                  ← interactive
  python add_person.py --name "John"    ← with name
  python add_person.py --list           ← list existing

You can add photos from:
  A) A folder containing photos
  B) Your webcam (live capture)
  C) A list of specific image paths
"""

import cv2
import sys
import argparse
import logging
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from core.face_engine import FaceEngine

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def load_config(path='config.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ── LIST EXISTING PERSONS ────────────────────────────────────────────
def list_persons(face_engine: FaceEngine):
    persons = face_engine.list_known_persons()
    if not persons:
        print("\n  No persons in database yet.\n")
        return
    print(f"\n  Known persons ({len(persons)}):")
    for p in persons:
        db_dir = face_engine.known_faces_dir / p
        photo_count = len(list(db_dir.glob('*.jpg')) + list(db_dir.glob('*.png')))
        print(f"    • {p}  ({photo_count} photos)")
    print()


# ── ADD FROM FOLDER ──────────────────────────────────────────────────
def add_from_folder(face_engine: FaceEngine, name: str, folder: str):
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"  ERROR: Folder not found: {folder}")
        return

    images = list(folder_path.glob('*.jpg')) + \
             list(folder_path.glob('*.jpeg')) + \
             list(folder_path.glob('*.png'))

    if not images:
        print(f"  ERROR: No images found in: {folder}")
        return

    print(f"\n  Found {len(images)} images in {folder}")
    saved = face_engine.add_person(name, [str(p) for p in images])
    print(f"  ✓  Added {saved} photos for '{name}'")


# ── ADD FROM WEBCAM ──────────────────────────────────────────────────
def add_from_webcam(face_engine: FaceEngine, name: str, count: int = 10):
    print(f"\n  Opening webcam to capture {count} photos for '{name}'")
    print("  Press SPACE to capture | Q to quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: Cannot open webcam")
        return

    person_dir = face_engine.known_faces_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)

    captured = 0
    window = f"Add Person: {name} — Press SPACE to capture"
    cv2.namedWindow(window)

    while captured < count:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw guide overlay
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.rectangle(frame, (cx - 100, cy - 130), (cx + 100, cy + 130), (0, 255, 0), 2)
        cv2.putText(frame, f"Captured: {captured}/{count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=capture  Q=done", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            path = person_dir / f"{name}_cam_{captured:03d}.jpg"
            cv2.imwrite(str(path), frame)
            captured += 1
            print(f"  📸  Captured {captured}/{count}")
        elif key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  ✓  Captured {captured} photos for '{name}'")
    print(f"     Saved to: {person_dir}")


# ── INTERACTIVE MODE ─────────────────────────────────────────────────
def interactive(face_engine: FaceEngine):
    print("\n" + "═" * 50)
    print("  ADD PERSON TO FACE DATABASE")
    print("═" * 50)

    # Get person name
    name = input("\n  Enter person's name: ").strip()
    if not name:
        print("  ERROR: Name cannot be empty")
        return

    # Normalize name for folder (no spaces → underscore)
    safe_name = name.replace(' ', '_')

    print(f"\n  Adding '{safe_name}' to database")
    print("\n  How do you want to add photos?")
    print("  [1] From a folder of images")
    print("  [2] Capture from webcam")
    choice = input("\n  Enter choice (1 or 2): ").strip()

    if choice == '1':
        folder = input("  Enter folder path: ").strip().strip('"')
        add_from_folder(face_engine, safe_name, folder)

    elif choice == '2':
        count_str = input("  How many photos to capture? [default: 10]: ").strip()
        count = int(count_str) if count_str.isdigit() else 10
        add_from_webcam(face_engine, safe_name, count)

    else:
        print("  Invalid choice")
        return

    print(f"\n  Person '{safe_name}' added successfully!")
    print("  They will be recognized in the next session.")
    print("\n  Tip: Add 5-10 varied photos for best accuracy.")
    print("  Run main.py to start the vision system.\n")


# ── MAIN ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Add person to Drone Vision AI face database')
    parser.add_argument('--name',   type=str,  help='Person name')
    parser.add_argument('--folder', type=str,  help='Folder containing photos')
    parser.add_argument('--webcam', action='store_true', help='Capture from webcam')
    parser.add_argument('--count',  type=int,  default=10, help='Number of webcam photos')
    parser.add_argument('--list',   action='store_true', help='List all known persons')
    parser.add_argument('--config', type=str,  default='config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    face_engine = FaceEngine(config['face_recognition'])

    if args.list:
        list_persons(face_engine)
        return

    if args.name and args.folder:
        add_from_folder(face_engine, args.name.replace(' ', '_'), args.folder)
    elif args.name and args.webcam:
        add_from_webcam(face_engine, args.name.replace(' ', '_'), args.count)
    elif args.name:
        # Name given but no source — prompt for source
        print(f"\n  Adding person: {args.name}")
        print("  Provide --folder or --webcam flag\n")
        parser.print_help()
    else:
        interactive(face_engine)


if __name__ == '__main__':
    main()
