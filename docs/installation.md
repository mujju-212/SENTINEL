# Installation Guide

This guide walks you through everything from a blank Windows machine to a fully running SENTINEL system.

---

## Prerequisites Checklist

Before starting, verify you have:

- [ ] Windows 10 or 11 (64-bit)
- [ ] NVIDIA GPU with 4+ GB VRAM *(recommended — CPU fallback works but is slow)*
- [ ] 10+ GB free disk space
- [ ] Internet connection *(only needed during setup — system runs offline after)*

---

## Step 1 — Install NVIDIA CUDA Toolkit

CUDA enables your GPU to run AI inference. This step is **only required for GPU acceleration**.

1. First check your GPU driver version:
   ```
   nvidia-smi
   ```
   Look at the top-right — it shows the max CUDA version your driver supports.

2. Download CUDA 11.8 from:
   https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Select: Windows → x86_64 → exe (local)
   - Install with default settings

3. Verify:
   ```
   nvcc --version
   ```
   Should print `release 11.8`.

> **Note:** SENTINEL was tested on CUDA 11.8. Newer CUDA versions (12.x) also work if you install the matching PyTorch build — see Step 4.

---

## Step 2 — Install Anaconda

Anaconda manages Python environments so SENTINEL's libraries never conflict with other projects.

1. Download Anaconda Distribution (not Miniconda — full distribution recommended):
   https://www.anaconda.com/download

2. Run the installer:
   - Install for "Just Me" (not all users)
   - **Change the install path to your D: drive** if C: space is limited, e.g. `D:\anaconda`
   - Do NOT add Anaconda to PATH during install — use Anaconda Prompt or `conda init` instead

3. Open **PowerShell** and initialize conda:
   ```powershell
   D:\anaconda\Scripts\conda.exe init powershell
   ```
   Close and reopen PowerShell.

4. Verify:
   ```powershell
   conda --version
   ```
   Should print `conda 26.x.x` or similar.

---

## Step 3 — Create the drone_vision Environment

Create a dedicated Python 3.10 environment for SENTINEL:

```powershell
conda create -n drone_vision python=3.10 -y
conda activate drone_vision
```

Verify the right Python is active:
```powershell
python --version
# Python 3.10.x
```

> **Important:** Always activate this environment before running any SENTINEL command.
> If you close the terminal, run `conda activate drone_vision` again.

---

## Step 4 — Install PyTorch with CUDA

PyTorch must be installed separately because it depends on your specific CUDA version.

**For CUDA 11.8 (recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only (no GPU):**
```bash
pip install torch torchvision torchaudio
```

Verify GPU is detected:
```python
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# True
# NVIDIA GeForce RTX 2050  (or your GPU name)
```

---

## Step 5 — Install All Other Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | 8.4.x | YOLOv8 object detection |
| `deepface` | 0.0.98 | Face recognition framework |
| `easyocr` | 1.7.x | OCR text reading |
| `opencv-python` | 4.13.x | Video capture and image processing |
| `opencv-contrib-python` | 4.13.x | Extra OpenCV algorithms |
| `tensorflow` | 2.20.x | DeepFace backend |
| `keras` | 3.12.x | Deep learning layers |
| `tf-keras` | 2.20.x | TF-compatible Keras |
| `retina-face` | 0.0.17 | High-accuracy face detector |
| `sqlalchemy` | 2.0.x | Database ORM |
| `pandas` | 2.3.x | Data handling |
| `scipy` | 1.15.x | Scientific computing |
| `pyyaml` | 6.0.x | Config file parsing |
| `pygame` | 2.6.x | Alert sounds |
| `colorlog` | 6.x | Colored terminal logs |
| `tqdm` | 4.x | Progress bars |
| `lap` | 0.5.x | Linear assignment (ByteTrack) |

Total install size: ~3.5 GB (models excluded, downloaded on first run).

---

## Step 6 — Configure Storage (Important if C: Drive is Limited)

By default, AI models download to your C: drive. Redirect them to D: to save space:

```powershell
$env:DEEPFACE_HOME = "D:\AI_Cache\deepface"
$env:EASYOCR_MODULE_PATH = "D:\AI_Cache\easyocr"
```

These are already set automatically via conda activation scripts at:
```
D:\anaconda\envs\drone_vision\etc\conda\activate.d\env_vars.ps1
```

Pre-download EasyOCR models (first-time only, ~30 MB):
```python
python -c "import easyocr; easyocr.Reader(['en'], model_storage_directory='D:/AI_Cache/easyocr')"
```

---

## Step 7 — Verify the Installation

Run the full module test suite:

```bash
python test_modules.py
```

Expected output:
```
══════════════════════════════════════════════════
  DRONE VISION AI — Module Tests
══════════════════════════════════════════════════

  TEST: GPU / CUDA
  ✓  PyTorch version: 2.7.1+cu118
  ✓  CUDA available: YES
  ✓  GPU: NVIDIA GeForce RTX 2050
  ✓  VRAM: 4.3 GB

  TEST: YOLOv8 Object Detector
  ✓  Model loaded | GPU: NVIDIA GeForce RTX 2050
  ✓  detect() ran in 1217ms

  TEST: Face Recognition (DeepFace)
  ✓  DeepFace loaded
  ✓  recognize() skipped (no persons in DB) — OK

  TEST: EasyOCR Text Reader
  ✓  EasyOCR reader loaded
  ✓  Found text: "HELLO" (conf=1.00)

  TEST: SQLite Database
  ✓  Database created
  ✓  save_detections() OK

  TEST: Camera / Webcam
  ✓  Camera #0 opened: 640x480

  TEST: Object Tracker
  ✓  Tracker update x3 OK | Active tracks: 2

  TEST: Visualizer
  ✓  draw() OK

══════════════════════════════════════════════════
  If all ✓ → run: python main.py --mode camera
```

All 8 `✓` means you are ready to run.

---

## Common Installation Issues

### `conda` not recognized

Run conda with full path:
```powershell
D:\anaconda\Scripts\conda.exe init powershell
```
Then restart PowerShell.

### `pip` installs to wrong Python

Always activate the environment first:
```powershell
conda activate drone_vision
```
Verify which pip/python is active:
```powershell
Get-Command python | Select-Object Source
# Should show: D:\anaconda\envs\drone_vision\python.exe
```

### CUDA not available after install

1. Restart your computer after installing CUDA toolkit
2. Verify the driver version supports CUDA 11.8: `nvidia-smi`
3. Make sure you installed the `cu118` build of PyTorch, not the CPU build

### `lap` module warning from Ultralytics

This is harmless on first run — Ultralytics auto-installs it. Restart Python after. Or install manually:
```bash
pip install lap
```

---

## Uninstalling

To remove the entire environment:
```powershell
conda deactivate
conda env remove -n drone_vision
```

To delete all cached models:
```powershell
Remove-Item -Recurse -Force "D:\AI_Cache"
```
