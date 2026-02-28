# SENTINEL — Documentation

Welcome to the official documentation for **SENTINEL**, a fully offline real-time AI vision pipeline.

---

## Table of Contents

| Document | What it covers |
|---|---|
| [Installation](installation.md) | Full setup from scratch — Anaconda, CUDA, all packages |
| [Architecture](architecture.md) | How the 8 modules connect and run together |
| [Modules](modules.md) | Deep dive into every module: what it does, how it works |
| [Configuration](configuration.md) | Every setting in `config.yaml` explained |
| [Usage](usage.md) | Running the system — camera, video, image modes |
| [Face Database](face-database.md) | Enrolling people and managing the face DB |
| [Performance](performance.md) | GPU/CPU benchmarks and optimization tips |
| [Troubleshooting](troubleshooting.md) | Common errors and how to fix them |

---

## What is SENTINEL?

SENTINEL is an **8-module AI vision pipeline** designed for real-time surveillance and monitoring. It runs **100% offline** — no API calls, no cloud, no internet once the models are downloaded.

```
Input ──▶ Detect ──▶ Track ──▶ Recognize ──▶ OCR ──▶ Log ──▶ Visualize ──▶ Alert
```

### Core capabilities

- **Object Detection** — Identifies 80 classes (people, cars, bags, etc.) every frame using YOLOv8
- **Human Isolation** — Extracts person bounding boxes for face processing
- **Face Recognition** — Matches faces against your enrolled database using DeepFace + FaceNet
- **Object Tracking** — Maintains stable IDs and movement trails with ByteTrack
- **OCR** — Reads visible text from license plates, signs, labels using EasyOCR
- **Database Logging** — Every event is saved to a local SQLite database
- **HUD Visualization** — Live annotated video with bounding boxes, labels, trails, FPS
- **Alert System** — Configurable audio + screenshot alerts for specific events

---

## Quick Start (5 steps)

```bash
# 1. Clone
git clone https://github.com/mujju-212/SENTINEL.git && cd SENTINEL

# 2. Create environment
conda create -n drone_vision python=3.10 -y
conda activate drone_vision

# 3. Install PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install everything else
pip install -r requirements.txt

# 5. Test and run
python test_modules.py
python main.py --mode camera
```

See [Installation](installation.md) for a detailed walkthrough.

---

## Project Layout

```
SENTINEL/
├── main.py              ← Run this
├── add_person.py        ← Face enrollment
├── test_modules.py      ← Verify installation
├── config.yaml          ← All settings
├── requirements.txt
├── core/                ← AI modules
│   ├── detector.py
│   ├── face_engine.py
│   ├── tracker.py
│   └── ocr_engine.py
├── utils/               ← Support modules
│   ├── database.py
│   ├── visualizer.py
│   └── alert.py
├── database/
│   └── known_faces/     ← Your face DB (private)
├── models/              ← Downloaded weights (private)
├── output/              ← Generated files (private)
└── docs/                ← You are here
```
