<div align="center">

# SENTINEL — Drone Vision AI

**Real-time AI vision pipeline: Object Detection · Face Recognition · OCR · Tracking · Alerts**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2BCU118-EE4C2C?logo=pytorch)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-11-8-0-download-archive)

</div>

---

## Overview

SENTINEL is an offline real-time AI vision system designed for surveillance and monitoring applications. It runs entirely on your local machine — no cloud, no internet required after setup.

Feed it a webcam, a video file, or a single image, and it continuously:

- Detects and classifies every object in the frame
- Identifies known persons from a face database
- Reads visible text (license plates, signs, labels) via OCR
- Tracks each object's movement over time
- Logs everything to a local SQLite database
- Fires alerts when specific events occur

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SENTINEL Pipeline                    │
│                                                             │
│  Camera / Video / Image                                     │
│        │                                                    │
│        ▼                                                    │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │  MODULE 1   │   │   MODULE 2   │   │   MODULE 5   │    │
│  │  YOLOv8     │   │   Person     │   │   EasyOCR    │    │
│  │  Detector   │──▶│   Isolator   │   │   Text Read  │    │
│  └─────────────┘   └──────┬───────┘   └──────────────┘    │
│                           │                                 │
│                           ▼                                 │
│                    ┌──────────────┐                        │
│                    │   MODULE 3   │                        │
│                    │  DeepFace    │                        │
│                    │  FaceNet ID  │                        │
│                    └──────┬───────┘                        │
│                           │                                 │
│       ┌───────────────────┼───────────────────┐           │
│       ▼                   ▼                   ▼           │
│  ┌─────────┐       ┌─────────────┐    ┌──────────────┐   │
│  │MODULE 4 │       │  MODULE 6   │    │   MODULE 7   │   │
│  │ByteTrack│       │  SQLite DB  │    │  OpenCV HUD  │   │
│  │Tracker  │       │  Logger     │    │  Visualizer  │   │
│  └─────────┘       └─────────────┘    └──────────────┘   │
│                                               │            │
│                                        ┌──────▼───────┐   │
│                                        │   MODULE 8   │   │
│                                        │  Alert System│   │
│                                        └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

| Module | Technology | Description |
|---|---|---|
| Object Detection | YOLOv8n (Ultralytics) | Detects 80 object classes in real-time |
| Human Detection | YOLOv8 class filter | Isolates person bounding boxes for face processing |
| Face Recognition | DeepFace + FaceNet | Matches detected faces against your known-persons database |
| Object Tracking | ByteTrack | Assigns stable IDs and trails to every detected object |
| OCR | EasyOCR | Reads text from signs, license plates, labels |
| Database | SQLite + SQLAlchemy | Logs all events (detections, persons, OCR) locally |
| Visualization | OpenCV | Live HUD with bounding boxes, labels, trails, FPS counter |
| Alerts | Pygame + Screenshot | Beep + screenshot on unknown persons, known targets, or objects |

---

## Requirements

### Hardware
- CPU: Any modern multi-core processor
- GPU: NVIDIA GPU with CUDA support (**recommended** for real-time performance)
  - Minimum: 4 GB VRAM (GTX 1650 / RTX 2050 or better)
  - Falls back to CPU automatically if no GPU found
- RAM: 8 GB minimum, 16 GB recommended
- Storage: ~5 GB for models and dependencies

### Software
- Windows 10/11 (Linux also supported)
- [Anaconda Distribution](https://www.anaconda.com/download) or Miniconda
- CUDA 11.8 (for GPU acceleration)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mujju-212/SENTINEL.git
cd SENTINEL
```

### 2. Create the conda environment

```bash
conda create -n drone_vision python=3.10 -y
conda activate drone_vision
```

### 3. Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> For CPU-only: `pip install torch torchvision torchaudio`

### 4. Install all other dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify everything works

```bash
python test_modules.py
```

All 8 modules should display `✓`. The first run will auto-download the YOLOv8n model (~6 MB) and EasyOCR detection models.

---

## Usage

### Live webcam

```bash
python main.py --mode camera
```

### Process a video file

```bash
python main.py --mode video --input path/to/video.mp4
```

### Analyze a single image

```bash
python main.py --mode image --input path/to/image.jpg
```

### Use a custom config

```bash
python main.py --mode camera --config my_config.yaml
```

**Keyboard shortcuts while running:**

| Key | Action |
|---|---|
| `Q` | Quit |
| `S` | Save screenshot |
| `P` | Pause / Resume |

---

## Adding Known Persons

To enable face recognition, add people to the database:

```bash
python add_person.py
```

This interactive tool lets you:
- **Option A** — Import from an image folder (point it at a folder of face photos)
- **Option B** — Capture directly from webcam (takes 5 photos of the person)

Enrolled faces are stored in `database/known_faces/<name>/`. Once added, the system will display the person's name and a green bounding box when they appear on camera.

---

## Configuration

All settings are in [`config.yaml`](config.yaml). Key sections:

```yaml
camera:
  source: 0           # 0 = default webcam, or path to video file
  width: 1280
  height: 720

detection:
  model: "yolov8n.pt" # yolov8s.pt for better accuracy, yolov8n.pt for speed
  confidence: 0.45

face_recognition:
  enabled: true
  recognition_threshold: 0.60  # lower = stricter
  detector_backend: "retinaface"
  model_name: "Facenet"

ocr:
  enabled: true
  languages: ["en"]
  process_every_n_frames: 10

alerts:
  unknown_person: true
  known_person: false
  cooldown_seconds: 30
```

---

## Project Structure

```
SENTINEL/
├── main.py                  # Entry point — ties all modules together
├── add_person.py            # Interactive face enrollment tool
├── test_modules.py          # Module verification suite
├── config.yaml              # All configuration settings
├── requirements.txt         # Python dependencies
│
├── core/
│   ├── detector.py          # YOLOv8 object & human detection
│   ├── face_engine.py       # DeepFace face recognition
│   ├── tracker.py           # ByteTrack object tracking
│   └── ocr_engine.py        # EasyOCR text extraction
│
├── utils/
│   ├── database.py          # SQLite logging (SQLAlchemy)
│   ├── visualizer.py        # OpenCV HUD rendering
│   └── alert.py             # Alert system (sound + screenshot)
│
├── database/
│   └── known_faces/         # Face image database (local only, not synced)
├── models/                  # Model weights (auto-downloaded, not synced)
├── input/test_images/       # Place test images here
└── output/                  # Generated outputs (not synced)
```

---

## Performance

Tested on RTX 2050 (4 GB VRAM):

| Mode | Resolution | FPS |
|---|---|---|
| Detection only | 1280×720 | ~45 FPS |
| Detection + Tracking | 1280×720 | ~40 FPS |
| Detection + Face ID | 1280×720 | ~18 FPS |
| Full pipeline (all modules) | 1280×720 | ~12 FPS |
| CPU only (no GPU) | 640×480 | ~5 FPS |

---

## Troubleshooting

**CUDA not available / running on CPU**
- Verify your CUDA installation: `nvidia-smi`
- Re-install PyTorch with the correct CUDA version for your GPU

**Face not recognized**
- Add more photos per person (5–10 images with varied angles/lighting)
- Lower `recognition_threshold` in config (try 0.50)
- Switch `detector_backend` to `mtcnn` for better detection at small sizes

**Low FPS**
- Switch to `yolov8n.pt` (fastest model)
- Reduce resolution in `config.yaml`
- Increase `process_every_n_frames` for face + OCR

**ModuleNotFoundError on first run**
- Make sure the `drone_vision` conda environment is activated
- Re-run `pip install -r requirements.txt`

---

## License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
Built with YOLOv8 · DeepFace · EasyOCR · PyTorch · OpenCV
</div>

---

## ✅ Quick Start

### Step 1 — Install Anaconda
Download from: https://www.anaconda.com/download
(Python 3.10+ environment)

### Step 2 — Setup Environment (run ONCE)
```
setup_env.bat
```
This creates a `drone_vision` conda environment and installs everything.

### Step 3 — Activate Environment
```
conda activate drone_vision
```

### Step 4 — Test Each Module
```
python test_modules.py
```
All modules should show ✓. Fix any ✗ before continuing.

### Step 5 — Run the System
```
python main.py --mode camera          # Live webcam
python main.py --mode video  --input path/to/video.mp4
python main.py --mode image  --input path/to/photo.jpg
```

---

## 👤 Add a Person to Face Database
```
python add_person.py                   # Interactive
python add_person.py --list            # See who's in DB
python add_person.py --name "John" --folder C:/photos/john
python add_person.py --name "John" --webcam --count 15
```

---

## 🎮 Keyboard Controls (live mode)
| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `S` | Save screenshot |
| `R` | Reset alert session |

---

## 📁 File Structure
```
SENTINEL/
├── main.py            ← RUN THIS
├── add_person.py      ← Add person to face DB
├── test_modules.py    ← Test all modules
├── config.yaml        ← All settings
├── requirements.txt   ← Python packages
├── setup_env.bat      ← One-time setup script
│
├── core/
│   ├── detector.py    ← YOLOv8 object detector
│   ├── face_engine.py ← DeepFace recognition
│   ├── tracker.py     ← Object tracking state
│   └── ocr_engine.py  ← EasyOCR text reading
│
├── utils/
│   ├── database.py    ← SQLite logging
│   ├── visualizer.py  ← OpenCV drawing
│   └── alert.py       ← Alert system
│
├── models/            ← YOLOv8 model (auto-download)
├── database/
│   ├── vision_ai.db   ← SQLite database (auto-created)
│   └── known_faces/   ← Face photos folder
│       └── PersonName/
│           └── photo1.jpg ...
├── input/test_images/ ← Test images
└── output/
    ├── logs/          ← Log files
    └── screenshots/   ← Saved screenshots
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `camera.device_id` | `0` | Webcam index |
| `detection.confidence` | `0.50` | Min detection confidence |
| `face_recognition.recognition_threshold` | `0.60` | Face match threshold |
| `face_recognition.process_every_n_frames` | `3` | Face ID frequency |
| `ocr.process_every_n_frames` | `10` | OCR frequency |
| `ocr.enabled` | `true` | Toggle OCR on/off |

---

## 🧩 Modules Summary

| Module | Tool | Runs every |
|--------|------|-----------|
| Object Detection | YOLOv8 Nano | Every frame |
| Object Tracking | ByteTrack (built-in) | Every frame |
| Face Recognition | DeepFace + FaceNet | Every 3rd frame |
| OCR | EasyOCR | Every 10th frame |
| Database | SQLite | Every 5th frame |

---

## 🖥️ Expected Performance (RTX 2050)
- Full pipeline: **20–25 FPS** (real-time ✅)
- Detection only: **45–55 FPS**

---

## 📊 Database Queries
The SQLite database is at `database/vision_ai.db`.
Open with [DB Browser for SQLite](https://sqlitebrowser.org/) (free tool).

Tables:
- `detections` — every detected object
- `ocr_results` — all OCR text extractions  
- `person_logs` — aggregated person visit history

---

## 🔮 Future-Ready
- Code designed for easy Jetson Nano / drone port
- Can add GPS tagging, mobile app, cloud sync later
