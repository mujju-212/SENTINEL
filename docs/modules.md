# Module Reference

Detailed documentation for all 8 SENTINEL modules.

---

## Module 1 — Object Detector

**File:** `core/detector.py`  
**Class:** `ObjectDetector`  
**Runs:** Every frame

### What it does

Uses **YOLOv8n** (nano variant) to detect objects in each video frame. Returns bounding boxes, class labels, and confidence scores for every detected object.

### Technology

- **Model:** YOLOv8n by Ultralytics (6.2 MB, 80 COCO classes)
- **Tracker:** ByteTrack (built into Ultralytics)
- **Device:** Auto-selects CUDA GPU → falls back to CPU

### Key methods

```python
detector = ObjectDetector(config['detection'])
result = detector.detect(frame)

# result keys:
# result['detections']  — list of all detected objects
# result['persons']     — subset where label == 'person'
# result['frame']       — original frame (pass-through)
```

### Configuration

```yaml
detection:
  model: "yolov8n.pt"      # Model file — auto-downloaded to models/
  confidence: 0.45          # Minimum confidence to count a detection
  iou_threshold: 0.45       # Non-maximum suppression threshold
  classes: null             # null = all 80 classes. [0] = persons only
  device: "auto"            # "auto", "cuda", "cpu"
```

### Model variants (trade speed for accuracy)

| Model | Size | Speed (RTX 2050) | mAP |
|---|---|---|---|
| yolov8n.pt | 6.2 MB | ~45 FPS | 37.3 |
| yolov8s.pt | 21.5 MB | ~35 FPS | 44.9 |
| yolov8m.pt | 49.7 MB | ~20 FPS | 50.2 |
| yolov8l.pt | 83.7 MB | ~10 FPS | 52.9 |

Change the model in `config.yaml` under `detection.model`.

---

## Module 2 — Human Detection / Person Isolation

**File:** `core/detector.py` (integrated with Module 1)  
**Runs:** Every frame

### What it does

Filters the raw detection list from Module 1 to extract only `person` class detections. These person bounding boxes are passed to the face recognition engine.

No separate model is needed — this is a post-processing step on YOLO's output.

---

## Module 3 — Face Recognition

**File:** `core/face_engine.py`  
**Class:** `FaceEngine`  
**Runs:** Every 3rd frame (configurable)

### What it does

For each person bounding box, crops that region from the frame, detects the face inside it, and matches against the enrolled face database. Returns a name and confidence score.

### Technology

- **Framework:** DeepFace
- **Model:** FaceNet (default) — also supports VGG-Face, ArcFace, Facenet512
- **Face Detector:** RetinaFace (default) — most accurate, also supports MTCNN, OpenCV
- **Similarity:** Cosine distance — threshold 0.60

### How recognition works

1. Person bbox cropped from frame
2. RetinaFace detects face landmarks inside the crop
3. FaceNet encodes the face into a 128-dim embedding vector
4. Vector compared against all enrolled embeddings via cosine distance
5. Match if distance < threshold (0.60 by default)
6. Returns name + confidence or "Unknown"

### Key methods

```python
face_engine = FaceEngine(config['face_recognition'])

# Adds known_faces_dir to the database
face_engine.add_person(name="Alice", image_paths=["alice1.jpg", "alice2.jpg"])

# Runs recognition on all persons in frame
persons = face_engine.recognize(frame, detected_persons)
# persons[i]['person_name'] — "Alice" or "Unknown"
# persons[i]['face_confidence'] — float 0-1
```

### Configuration

```yaml
face_recognition:
  enabled: true
  known_faces_dir: "database/known_faces"
  recognition_threshold: 0.60  # Lower = stricter. Try 0.50 if false positives
  process_every_n_frames: 3
  detector_backend: "retinaface" # retinaface (best) | mtcnn | opencv | ssd
  model_name: "Facenet"          # Facenet | VGG-Face | ArcFace | Facenet512
  model_dir: "D:/AI_Cache/deepface"
```

### First-run model download

DeepFace downloads model weights on first use:
- `Facenet.h5` (~90 MB)
- `retinaface.h5` (~110 MB)

After that, everything is cached locally at `D:\AI_Cache\deepface`.

---

## Module 4 — Object Tracker

**File:** `core/tracker.py`  
**Class:** `ObjectTracker`  
**Runs:** Every frame

### What it does

Maintains persistent IDs for each detected object across frames. Even if an object disappears for a few frames, it gets its ID back when it reappears. Tracks movement trails (last 30 positions).

### Technology

- **Algorithm:** ByteTrack (via Ultralytics)
- **Fallback:** Simple IoU-based tracking when `lap` is not installed

### What gets tracked

For each tracked object:
- **Track ID** — stable integer ID across the video
- **Time in frame** — how many seconds it's been visible
- **Frame count** — total frames detected
- **Trail** — deque of last 30 center-point positions
- **Entry/exit events** — fired when first appears or disappears

### Key methods

```python
tracker = ObjectTracker(config['tracking'])
tracks = tracker.update(detections_list)

# tracks — dict of {track_id: TrackRecord}
for tid, track in tracks.items():
    print(tid, track.label, track.time_in_frame, track.trail)
```

### Configuration

```yaml
tracking:
  max_age: 30        # Frames to keep a track alive when not detected
  min_hits: 3        # Minimum detections before a track is confirmed
  iou_threshold: 0.30
```

---

## Module 5 — OCR Engine

**File:** `core/ocr_engine.py`  
**Class:** `OCREngine`  
**Runs:** Every 10th frame (in background thread)

### What it does

Reads text visible in the video frame — license plates, street signs, labels, documents. Because OCR is computationally expensive (~400ms per frame), it runs in a background thread and caches its last result.

### Technology

- **Library:** EasyOCR 1.7.x
- **Backend:** PyTorch (GPU-accelerated when available)
- **Languages:** English by default, supports 80+ languages

### Threading behavior

```
Main loop          Background thread
    │                     │
    ├── frame 10 ─────────▶ ocr.read_frame(frame)  ← starts thread
    ├── frame 11            │   (running, ~400ms)
    ├── frame 12            │
    ├── frame 13 ◀──────────┘  result cached
    ├── frame 14  reads cache (instant)
    ...
    ├── frame 20 ─────────▶ ocr.read_frame(frame)  ← new thread
```

### Key methods

```python
ocr = OCREngine(config['ocr'])
results = ocr.read_frame(frame)           # Threaded, returns cached
results = ocr.read_image_file("img.jpg")  # Blocking, for single images

# results — list of OCRResult:
# result.text       — detected string
# result.confidence — float 0-1
# result.bbox       — (x1, y1, x2, y2)
```

### Configuration

```yaml
ocr:
  enabled: true
  languages: ["en"]              # Add more: ["en", "ar", "fr"]
  process_every_n_frames: 10
  gpu: true
  min_confidence: 0.30           # Ignore low-confidence text
  model_dir: "D:/AI_Cache/easyocr"
```

---

## Module 6 — Database Logger

**File:** `utils/database.py`  
**Class:** `DatabaseManager`  
**Runs:** Every 5th frame

### What it does

Logs all detections, OCR results, and person visits to a local SQLite database. Provides query methods for statistics and history.

### Schema

**`detections` table**
```sql
id            INTEGER PRIMARY KEY
timestamp     DATETIME
label         TEXT       -- 'person', 'car', etc.
confidence    REAL
track_id      INTEGER
person_name   TEXT       -- NULL if not a person or not recognized
x1,y1,x2,y2  INTEGER    -- Bounding box pixels
session_id    TEXT
```

**`ocr_results` table**
```sql
id            INTEGER PRIMARY KEY
timestamp     DATETIME
text          TEXT
confidence    REAL
x1,y1,x2,y2  INTEGER
frame_number  INTEGER
session_id    TEXT
```

**`person_logs` table**
```sql
id            INTEGER PRIMARY KEY
person_name   TEXT
first_seen    DATETIME
last_seen     DATETIME
total_visits  INTEGER
total_time_seconds REAL
session_id    TEXT
```

### Key methods

```python
db = DatabaseManager(config['database'])
db.save_detections(tracks, session_id)
db.save_ocr_results(ocr_results, frame_num, session_id)
db.update_person_log(person_name, duration, session_id)
rows = db.get_recent_detections(limit=100)
stats = db.get_todays_stats()
db.cleanup_old_logs(days=30)
```

### Viewing the database

Use [DB Browser for SQLite](https://sqlitebrowser.org/) (free) to open `database/vision_ai.db` and browse/query the data visually.

---

## Module 7 — Visualizer

**File:** `utils/visualizer.py`  
**Class:** `Visualizer`  
**Runs:** Every frame

### What it does

Draws all annotations on the video frame using OpenCV. Produces the final HUD-style display window.

### What gets drawn

- **Bounding boxes** — colored by category
  - Green: recognized person (known identity)
  - Red: unrecognized person (unknown)
  - Orange: non-person object
  - Yellow: OCR text region
- **Labels** — class name + confidence + track ID
- **Person name** — displayed above recognized persons
- **Movement trail** — line of last 30 positions
- **Top HUD bar** — FPS counter, frame count, active track count
- **Bottom OCR bar** — last detected text

### Color scheme

| Detection Type | Box Color | Label Color |
|---|---|---|
| Known person | `(0, 255, 0)` green | White |
| Unknown person | `(0, 0, 255)` red | White |
| Object | `(0, 165, 255)` orange | White |
| OCR region | `(0, 255, 255)` yellow | Black |

### Key methods

```python
visualizer = Visualizer(config['display'])
annotated_frame = visualizer.draw(frame, tracks, ocr_results)
fps = fps_counter.update()
```

---

## Module 8 — Alert System

**File:** `utils/alert.py`  
**Class:** `AlertSystem`  
**Runs:** Every frame

### What it does

Monitors the current detection state and fires configurable alerts when events match trigger conditions. Each alert type has an independent cooldown to prevent spam.

### Alert types

| Alert | Trigger | Default |
|---|---|---|
| `UNKNOWN_PERSON` | Person detected, not in DB | Enabled |
| `KNOWN_PERSON` | Specific enrolled person seen | Disabled |
| `OBJECT_ALERT` | Specific object class detected | Disabled |
| `TEXT_DETECTED` | OCR finds text | Disabled |

### Alert actions

When triggered:
1. **Sound** — plays a beep via `pygame.mixer` *(on systems with audio)*
2. **Screenshot** — saves annotated frame as JPEG to `output/screenshots/`
3. **Log** — writes entry to the log file
4. **Console** — prints colored alert message

### Cooldown system

Each alert type tracks its own last-fired timestamp. The same alert won't fire again within `cooldown_seconds` (default 30s). This prevents a stationary unknown person from triggering 100 alerts per second.

### Configuration

```yaml
alerts:
  unknown_person: true
  known_person: false
  object_detection: false
  text_detected: false
  cooldown_seconds: 30
  sound_enabled: true
  save_screenshot_on_alert: true
  screenshot_dir: "output/screenshots"
  target_objects: ["knife", "gun"]  # Only alert for these object classes
  target_persons: ["Alice"]         # Only alert when this person is seen
```
