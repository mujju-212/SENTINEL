# System Architecture

This document explains how SENTINEL's 8 modules are wired together, how data flows between them, and the execution model.

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         Input Sources                            │
│              Webcam │ Video File │ Image File                    │
└─────────────────────────┬────────────────────────────────────────┘
                          │ frame (BGR numpy array)
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                      MAIN LOOP  (main.py)                        │
│                   DroneVisionPipeline._process_frame()           │
│                                                                  │
│   frame_count check → decides which modules run this frame       │
└──────┬──────────────────┬─────────────────────┬─────────────────┘
       │                  │                     │
       ▼                  ▼                     ▼ (every 10th frame, async)
┌─────────────┐   ┌──────────────┐      ┌──────────────┐
│  MODULE 1   │   │  MODULE 3    │      │  MODULE 5    │
│  detector   │   │  face_engine │      │  ocr_engine  │
│  YOLOv8     │   │  DeepFace    │      │  EasyOCR     │
│  .detect()  │   │  .recognize()│      │  .read_frame │
└──────┬──────┘   └──────┬───────┘      └──────┬───────┘
       │                  │                     │
       ▼                  │                     │
┌─────────────┐           │                     │
│  MODULE 2   │           │                     │
│  Person     │           │                     │
│  Isolation  │           │                     │
│  (in detect)│           │                     │
└──────┬──────┘           │                     │
       │◀─────────────────┘                     │
       ▼                                        │
┌─────────────┐                                 │
│  MODULE 4   │                                 │
│  tracker    │                                 │
│  ByteTrack  │                                 │
│  .update()  │                                 │
└──────┬──────┘                                 │
       │                                        │
       └──────────────────┬─────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Results Bundle                                  │
│  { detections, persons+names, track_ids, ocr_text }              │
└──────┬──────────────────┬─────────────────────┬─────────────────┘
       │                  │                     │
       ▼                  ▼                     ▼
┌─────────────┐   ┌──────────────┐      ┌──────────────┐
│  MODULE 6   │   │  MODULE 7    │      │  MODULE 8    │
│  database   │   │  visualizer  │      │  alert       │
│  SQLite log │   │  OpenCV HUD  │      │  system      │
│  .save()    │   │  .draw()     │      │  .trigger()  │
└─────────────┘   └──────┬───────┘      └──────────────┘
                         │
                         ▼
                  Display Window
                  (cv2.imshow)
```

---

## Execution Model

### Frame Processing Loop

```python
for each frame from input:
    frame_count += 1

    # MODULE 1+2: Always runs
    detect_result = detector.detect(frame)

    # MODULE 3: Every 3rd frame (face_recognition.process_every_n_frames)
    if frame_count % 3 == 0:
        persons = face_engine.recognize(frame, detect_result.persons)

    # MODULE 4: Always runs
    tracks = tracker.update(detect_result.detections)

    # MODULE 5: Every 10th frame, in background thread
    if frame_count % 10 == 0:
        Thread(target=ocr_engine.read_frame, args=(frame,)).start()
    ocr_results = ocr_engine.get_cached()   # always available instantly

    # MODULE 6: Every 5th frame
    if frame_count % 5 == 0:
        database.save_detections(tracks, ocr_results)

    # MODULE 7: Every frame
    annotated = visualizer.draw(frame, tracks, ocr_results)

    # MODULE 8: Every frame
    alert_system.check_and_trigger(tracks, annotated)

    cv2.imshow("SENTINEL", annotated)
```

### Threading Model

| Module | Thread | Why |
|---|---|---|
| Object Detection | Main thread | Fastest path, must be synchronous |
| Face Recognition | Main thread (every 3rd frame) | Shares GPU with detector |
| OCR | Background thread | Very slow (~400ms), cached result used by main |
| Database | Main thread async write | Non-blocking via SQLAlchemy |
| Alert | Main thread | Needs the current annotated frame |

OCR runs in a daemon thread so it never blocks the video stream. The main loop always reads the **last cached OCR result**, so text persists on screen between OCR runs.

---

## Module Interaction Map

```
detector.py ──── provides detections ────▶ tracker.py
detector.py ──── provides persons ───────▶ face_engine.py
face_engine.py ── provides names ─────────▶ visualizer.py
tracker.py ────── provides track IDs ────▶ visualizer.py
tracker.py ────── provides tracks ───────▶ database.py
tracker.py ────── provides entry events ─▶ alert.py
ocr_engine.py ─── provides text ─────────▶ visualizer.py
ocr_engine.py ─── provides text ─────────▶ database.py
visualizer.py ─── provides annotated frame▶ alert.py (screenshot)
alert.py ─────────────────────────────────▶ sound + file
database.py ──────────────────────────────▶ vision_ai.db
```

---

## Data Structures

### Detection dict (from `detector.py`)
```python
{
    "label":      "person",        # YOLO class name
    "confidence": 0.87,            # Detection confidence 0-1
    "bbox":       (x1, y1, x2, y2),# Pixel coordinates
    "track_id":   12,              # Assigned by ByteTrack
    "is_person":  True             # True if class == 'person'
}
```

### TrackRecord (from `tracker.py`)
```python
TrackRecord {
    track_id:     12
    label:        "person"
    bbox:         (x1, y1, x2, y2)
    person_name:  "Alice"          # From face_engine, or None
    face_conf:    0.91
    time_in_frame: 4.2             # Seconds visible
    frame_count:  127              # Frames seen
    trail:        deque([(cx,cy), ...], maxlen=30)  # Movement history
    first_seen:   datetime
    last_seen:    datetime
}
```

### OCRResult (from `ocr_engine.py`)
```python
OCRResult {
    text:       "STOP",
    confidence: 0.98,
    bbox:       (x1, y1, x2, y2)
}
```

---

## File I/O Summary

| File | Read / Write | Format | Notes |
|---|---|---|---|
| `config.yaml` | Read once on startup | YAML | All settings |
| `models/yolov8n.pt` | Read on startup | PyTorch binary | Auto-downloaded if missing |
| `database/vision_ai.db` | Write every 5 frames | SQLite | 3 tables |
| `database/known_faces/` | Read on face recognition init | JPEG/PNG folders | One folder per person |
| `output/screenshots/` | Write on alert trigger | JPEG | Named by timestamp |
| `output/logs/` | Write continuously | Plain text | Colorlog output |

---

## Config Loading Flow

```
main.py
  └── argparse --config (default: config.yaml)
        └── yaml.safe_load(config_path)
              └── config dict passed to each module constructor:
                    ObjectDetector(config['detection'])
                    FaceEngine(config['face_recognition'])
                    OCREngine(config['ocr'])
                    ObjectTracker(config['tracking'])
                    DatabaseManager(config['database'])
                    Visualizer(config['display'])
                    AlertSystem(config['alerts'])
```

Each module only receives its own section of the config, keeping them decoupled.
