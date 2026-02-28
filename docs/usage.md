# Usage Guide

How to run SENTINEL in every mode, with all available options.

---

## Activating the Environment

Always activate the conda environment first:

```powershell
conda activate drone_vision
```

Set cache environment variables (already automatic via conda hooks, but set manually if needed):

```powershell
$env:DEEPFACE_HOME = "D:\AI_Cache\deepface"
$env:EASYOCR_MODULE_PATH = "D:\AI_Cache\easyocr"
```

Navigate to the project folder:

```powershell
cd "D:\AVTIVE PROJ\SENTINEL"
```

---

## Running Modes

### Camera Mode (Live Webcam)

```bash
python main.py --mode camera
```

Uses `camera.source` from `config.yaml` (default: `0` = first webcam).

To use a different webcam:
```yaml
# config.yaml
camera:
  source: 1   # second webcam
```

Or pass a temporary override by editing config before running.

---

### Video Mode (Process a File)

```bash
python main.py --mode video --input "D:\recordings\patrol.mp4"
```

Processes frame-by-frame at the video's native FPS. Shows live OpenCV window.

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

---

### Image Mode (Single Photo)

```bash
python main.py --mode image --input "D:\photos\scene.jpg"
```

Analyzes a single image, displays annotated result, and saves to `output/`.

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

---

### Custom Config

```bash
python main.py --mode camera --config "D:\AVTIVE PROJ\SENTINEL\configs\nightmode.yaml"
```

---

## Command-Line Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--mode` | Yes | — | `camera`, `video`, or `image` |
| `--input` | For video/image | — | Path to video file or image |
| `--config` | No | `config.yaml` | Path to config YAML |

---

## Keyboard Controls (Live Display Window)

While the OpenCV window is open and focused:

| Key | Action |
|---|---|
| `Q` | Quit and exit |
| `S` | Save current frame as screenshot |
| `P` | Pause / Resume stream |
| `ESC` | Quit (same as Q) |

> Click on the video window to make sure it has keyboard focus before pressing keys.

---

## What You See On Screen

```
┌─────────────────────────────────────────────────────────┐
│  FPS: 24.3  │  Tracks: 3  │  Frame: 1042        SENTINEL│  ← Top HUD
├─────────────────────────────────────────────────────────┤
│                                                         │
│      ┌─────────────────┐                               │
│      │  ● Alice (0.91) │  ← Green = known person       │
│      │  Track #12      │                               │
│      │  Time: 8.4s     │                               │
│      └─────────────────┘                               │
│              •                                         │
│             •• •                                        │
│            ••   ← Trail (last 30 positions)             │
│                                                         │
│   ┌──────────────┐                                      │
│   │ Unknown      │  ← Red = unrecognized person         │
│   │ Track #7     │                                      │
│   └──────────────┘                                      │
│                                                         │
│   ┌──────────────────────┐                             │
│   │ car  0.87  Track #3  │  ← Orange = object          │
│   └──────────────────────┘                             │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  OCR: "STOP"  0.98    "ZONE 30"  0.87                   │  ← Bottom OCR bar
└─────────────────────────────────────────────────────────┘
```

---

## Running test_modules.py

Before running the full system, verify all modules work:

```bash
python test_modules.py
```

This runs 8 isolated tests in sequence:

1. **GPU/CUDA** — PyTorch + CUDA availability
2. **YOLOv8 Detector** — loads model, runs on blank frame
3. **Face Recognition** — loads DeepFace (skips recognition if no persons enrolled)
4. **EasyOCR** — loads models, reads sample text
5. **SQLite Database** — creates tables, writes + reads a test row
6. **Camera** — opens webcam, captures a frame
7. **Object Tracker** — runs update() with dummy detections
8. **Visualizer** — runs draw(), saves test image to `output/`

If any test shows `✗`, fix it before running `main.py`.

---

## First Run Notes

On the very first run:

- **YOLOv8n model** (~6 MB) downloads to `models/yolov8n.pt`
- **EasyOCR models** (~30 MB) download to `D:\AI_Cache\easyocr`
- **DeepFace / FaceNet weights** (~90 MB + ~110 MB) download to `D:\AI_Cache\deepface` on first face detection

These only download once. All subsequent runs use the local cache.

---

## Output Files

| File | Created when | Location |
|---|---|---|
| `vision_ai.db` | First run | `database/` |
| `camera_test.jpg` | `test_modules.py` | `output/` |
| `visualizer_test.jpg` | `test_modules.py` | `output/` |
| `ALERT_*.jpg` | Alert triggered | `output/screenshots/` |
| `sentinel_YYYYMMDD.log` | Every run | `output/logs/` |

---

## Stopping the System

Press `Q` in the video window, or `Ctrl+C` in the terminal.

The pipeline gracefully:
1. Flushes remaining database writes
2. Releases camera / file handle
3. Destroys OpenCV windows
4. Prints summary stats to console
