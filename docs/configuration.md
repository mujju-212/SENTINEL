# Configuration Reference

All SENTINEL settings live in `config.yaml`. This document explains every option.

---

## Full Default config.yaml

```yaml
# ─── CAMERA / INPUT ─────────────────────────────────────────
camera:
  source: 0             # Webcam index (0 = default), or path to video file
  width: 1280
  height: 720
  fps: 30
  buffer_size: 1        # Minimize capture latency

# ─── OBJECT DETECTION ───────────────────────────────────────
detection:
  model: "yolov8n.pt"
  confidence: 0.45
  iou_threshold: 0.45
  classes: null         # null = detect all 80 COCO classes
  device: "auto"        # auto | cuda | cpu

# ─── FACE RECOGNITION ───────────────────────────────────────
face_recognition:
  enabled: true
  known_faces_dir: "database/known_faces"
  recognition_threshold: 0.60
  process_every_n_frames: 3
  detector_backend: "retinaface"
  model_name: "Facenet"
  model_dir: "D:/AI_Cache/deepface"

# ─── OCR ────────────────────────────────────────────────────
ocr:
  enabled: true
  languages: ["en"]
  process_every_n_frames: 10
  gpu: true
  min_confidence: 0.30
  model_dir: "D:/AI_Cache/easyocr"

# ─── TRACKING ───────────────────────────────────────────────
tracking:
  max_age: 30
  min_hits: 3
  iou_threshold: 0.30

# ─── DISPLAY ────────────────────────────────────────────────
display:
  show_fps: true
  show_confidence: true
  show_track_id: true
  show_trail: true
  trail_length: 30
  window_name: "SENTINEL"

# ─── ALERTS ─────────────────────────────────────────────────
alerts:
  unknown_person: true
  known_person: false
  object_detection: false
  text_detected: false
  cooldown_seconds: 30
  sound_enabled: true
  save_screenshot_on_alert: true
  screenshot_dir: "output/screenshots"
  target_objects: []
  target_persons: []

# ─── DATABASE ────────────────────────────────────────────────
database:
  path: "database/vision_ai.db"
  save_every_n_frames: 5
  cleanup_after_days: 30

# ─── PATHS ───────────────────────────────────────────────────
paths:
  models: "models"
  output: "output"
  logs: "output/logs"
```

---

## Section-by-Section Reference

### `camera`

| Key | Type | Default | Description |
|---|---|---|---|
| `source` | int or string | `0` | `0` = first webcam, `1` = second webcam, or `"path/to/video.mp4"` |
| `width` | int | `1280` | Capture width in pixels |
| `height` | int | `720` | Capture height in pixels |
| `fps` | int | `30` | Target frame rate |
| `buffer_size` | int | `1` | OpenCV capture buffer — keep at 1 to avoid lag |

**Examples:**
```yaml
# Use second webcam
camera:
  source: 1

# Use video file
camera:
  source: "D:/recordings/patrol_2026.mp4"

# Lower resolution for faster processing
camera:
  source: 0
  width: 640
  height: 480
```

---

### `detection`

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | string | `yolov8n.pt` | Model filename in `models/` — auto-downloaded if missing |
| `confidence` | float | `0.45` | Minimum detection confidence (0.0–1.0). Higher = fewer false positives |
| `iou_threshold` | float | `0.45` | Non-maximum suppression overlap threshold |
| `classes` | list or null | `null` | Filter to specific COCO class IDs. `null` = all 80 classes |
| `device` | string | `auto` | `auto` selects GPU if available, otherwise CPU |

**COCO class IDs (common ones):**
```
0 = person      2 = car         3 = motorcycle
4 = airplane    5 = bus         7 = truck
14 = bird       15 = cat        16 = dog
24 = backpack   26 = umbrella   28 = handbag
39 = bottle     41 = cup        56 = chair
62 = tv         63 = laptop     67 = phone
```

**Examples:**
```yaml
# Detect only people and cars
detection:
  classes: [0, 2]

# Higher accuracy model, lower FPS
detection:
  model: "yolov8m.pt"
  confidence: 0.50
```

---

### `face_recognition`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Disable to skip face recognition entirely |
| `known_faces_dir` | string | `database/known_faces` | Root folder of face image database |
| `recognition_threshold` | float | `0.60` | Cosine distance threshold. Lower value = stricter matching |
| `process_every_n_frames` | int | `3` | Skip frames to reduce compute load |
| `detector_backend` | string | `retinaface` | Face detector inside DeepFace |
| `model_name` | string | `Facenet` | Face embedding model |
| `model_dir` | string | — | Where DeepFace downloads weights |

**`detector_backend` options:**

| Backend | Speed | Accuracy | Notes |
|---|---|---|---|
| `retinaface` | Slow | Highest | Best for small faces |
| `mtcnn` | Medium | High | Good balance |
| `opencv` | Fast | Low | No GPU needed |
| `ssd` | Fast | Medium | Good for large faces |

**`model_name` options:**

| Model | Embedding size | Notes |
|---|---|---|
| `Facenet` | 128-dim | Good balance, default |
| `Facenet512` | 512-dim | More accurate, slower |
| `VGG-Face` | 2622-dim | Older, heavier |
| `ArcFace` | 512-dim | Best accuracy |

---

### `ocr`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Disable OCR to save resources |
| `languages` | list | `["en"]` | ISO 639-1 language codes |
| `process_every_n_frames` | int | `10` | OCR is heavy — runs every Nth frame |
| `gpu` | bool | `true` | Use GPU for OCR inference |
| `min_confidence` | float | `0.30` | Discard OCR results below this threshold |
| `model_dir` | string | — | Where EasyOCR downloads models |

**Multi-language examples:**
```yaml
ocr:
  languages: ["en", "ar"]    # English + Arabic
  languages: ["en", "fr", "de"]  # English + French + German
```

Full language list: https://www.jaided.ai/easyocr/

---

### `tracking`

| Key | Type | Default | Description |
|---|---|---|---|
| `max_age` | int | `30` | Frames to keep track alive when object disappears |
| `min_hits` | int | `3` | Detections required before track is "confirmed" |
| `iou_threshold` | float | `0.30` | Minimum IoU overlap to associate detection to track |

---

### `display`

| Key | Type | Default | Description |
|---|---|---|---|
| `show_fps` | bool | `true` | FPS counter in top-left |
| `show_confidence` | bool | `true` | Detection confidence in label |
| `show_track_id` | bool | `true` | Track ID in label |
| `show_trail` | bool | `true` | Draw movement trail |
| `trail_length` | int | `30` | Max trail positions to draw |
| `window_name` | string | `SENTINEL` | OpenCV window title |

---

### `alerts`

| Key | Type | Default | Description |
|---|---|---|---|
| `unknown_person` | bool | `true` | Alert when unrecognized person detected |
| `known_person` | bool | `false` | Alert when enrolled person detected |
| `object_detection` | bool | `false` | Alert on specific object classes |
| `text_detected` | bool | `false` | Alert when OCR finds text |
| `cooldown_seconds` | int | `30` | Minimum seconds between same-type alerts |
| `sound_enabled` | bool | `true` | Beep on alert |
| `save_screenshot_on_alert` | bool | `true` | Save JPEG screenshot |
| `screenshot_dir` | string | `output/screenshots` | Where screenshots are saved |
| `target_objects` | list | `[]` | Object classes to alert on (e.g. `["knife"]`) |
| `target_persons` | list | `[]` | Person names to alert on (e.g. `["Alice"]`) |

---

### `database`

| Key | Type | Default | Description |
|---|---|---|---|
| `path` | string | `database/vision_ai.db` | SQLite file path |
| `save_every_n_frames` | int | `5` | Write frequency |
| `cleanup_after_days` | int | `30` | Auto-delete records older than N days |

---

## Using Multiple Config Files

You can maintain separate configs for different scenarios:

```bash
python main.py --mode camera --config configs/indoor.yaml
python main.py --mode camera --config configs/nightmode.yaml
python main.py --mode video  --input footage.mp4 --config configs/fast.yaml
```
