# Performance Guide

Benchmarks, tuning tips, and hardware considerations for SENTINEL.

---

## Benchmark Results

All tests on **RTX 2050 (4 GB VRAM)**, Windows 11, Python 3.10, PyTorch 2.7+cu118.

### FPS by Active Modules

| Modules Active | Resolution | FPS |
|---|---|---|
| Detection only (Module 1) | 1280×720 | ~45 |
| Detection + Tracking (1+4) | 1280×720 | ~42 |
| Detection + Track + Visualize (1+4+7) | 1280×720 | ~40 |
| Detection + Face ID (1+2+3) | 1280×720 | ~18 |
| Full pipeline — all 8 modules | 1280×720 | ~12 |
| Full pipeline | 640×480 | ~22 |
| CPU only, detection only | 640×480 | ~8 |
| CPU only, full pipeline | 640×480 | ~3 |

> Face recognition is the most expensive module (~35ms per person per 3 frames on GPU).
> OCR runs in a background thread so it doesn't directly reduce FPS — but it competes for GPU memory.

---

## GPU Memory Usage

| Configuration | VRAM Used |
|---|---|
| Detection only (YOLOv8n) | ~400 MB |
| + Face Recognition (FaceNet) | ~1.2 GB |
| + OCR (EasyOCR) | ~1.8 GB |
| + TensorFlow (DeepFace backend) | ~2.4 GB |
| Full pipeline | ~2.8–3.2 GB |

A 4 GB VRAM GPU (RTX 2050, GTX 1650) runs the full pipeline with ~0.8 GB headroom.

---

## Optimization Strategies

### 1. Switch to a Faster YOLO Model

The quickest FPS improvement:

```yaml
# config.yaml
detection:
  model: "yolov8n.pt"   # nano — fastest (default)
  # model: "yolov8s.pt" # small — better accuracy, ~25% slower
```

### 2. Lower Resolution

Halving resolution roughly doubles FPS:

```yaml
camera:
  width: 640
  height: 480   # from 1280x720 → ~2x FPS improvement
```

### 3. Increase Frame Skip for Face and OCR

Face recognition and OCR are the bottlenecks:

```yaml
face_recognition:
  process_every_n_frames: 5   # was 3 — reduces face ID by 40%

ocr:
  process_every_n_frames: 20  # was 10 — OCR runs half as often
  enabled: false              # or disable entirely if not needed
```

### 4. Detect Only Relevant Classes

If you only care about people:

```yaml
detection:
  classes: [0]   # COCO class 0 = person
```

This reduces the number of detections processed by all downstream modules.

### 5. Disable Unused Modules

```yaml
face_recognition:
  enabled: false   # +15 FPS on full pipeline

ocr:
  enabled: false   # +3 FPS (OCR is threaded, low impact)

display:
  show_trail: false  # Minor improvement (~2 FPS)
```

### 6. Use a Faster Face Detector Backend

```yaml
face_recognition:
  detector_backend: "opencv"   # Fastest, lower accuracy
  # detector_backend: "mtcnn"  # Balance
  # detector_backend: "retinaface"  # Most accurate, slowest
```

### 7. Reduce Display Window Size

Not in config — OpenCV scales the window after rendering, but reducing source resolution is the right lever.

---

## CPU-Only Mode

If no GPU is available, SENTINEL falls back to CPU automatically. Performance is much lower:

| Module | CPU Time |
|---|---|
| YOLOv8n detection | ~120ms / frame |
| Face recognition | ~400ms / person |
| OCR | ~2000ms / frame |

**Recommended settings for CPU-only:**

```yaml
camera:
  width: 640
  height: 480

detection:
  model: "yolov8n.pt"
  confidence: 0.50   # Higher threshold = fewer detections to process

face_recognition:
  process_every_n_frames: 10
  detector_backend: "opencv"   # Fastest on CPU

ocr:
  process_every_n_frames: 30
  gpu: false
```

---

## Memory Usage (RAM)

| State | RAM Used |
|---|---|
| Idle (just imported) | ~400 MB |
| After YOLOv8 loaded | ~800 MB |
| After DeepFace loaded | ~1.4 GB |
| After EasyOCR loaded | ~1.8 GB |
| Full pipeline running | ~2.0–2.5 GB |

Minimum 8 GB system RAM recommended. 16 GB is comfortable.

---

## Disk I/O Performance

SQLite writes every 5 frames. At 20 FPS this is 4 writes/second.

If the database is on a slow HDD, increase the write interval:

```yaml
database:
  save_every_n_frames: 15   # Reduce write frequency
```

For best performance, keep `vision_ai.db` on an SSD.

---

## Profiling Individual Modules

Run the test suite with timing:

```bash
python test_modules.py
```

Each test reports its own runtime:
```
✓  detect() ran in 1217.5ms on blank frame    ← YOLOv8 first inference (includes JIT compile)
✓  read_frame() ran in 420ms                  ← EasyOCR
```

> First inference is always slow due to CUDA JIT compilation. Steady-state FPS is significantly higher.

---

## Jetson Nano / Edge Deployment Notes

SENTINEL is designed to be portable to embedded devices:

- All models are standard PyTorch — compatible with Jetson's PyTorch build
- YOLOv8n is small enough for Jetson Nano (4 GB RAM)
- Expected FPS on Jetson Nano 4GB: ~8–12 FPS (detection only)
- For edge, disable face recognition and OCR, use 640×480, `classes: [0]`
