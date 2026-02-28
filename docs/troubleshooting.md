# Troubleshooting

Common issues and how to fix them.

---

## Installation Issues

### `conda` is not recognized in PowerShell

**Symptom:**
```
conda : The term 'conda' is not recognized...
```

**Fix:**
```powershell
# Use full path to initialize conda for PowerShell
D:\anaconda\Scripts\conda.exe init powershell
# Then close and reopen PowerShell
```

---

### pip installs to the wrong Python / system Python

**Symptom:** Packages install but `import` fails; or packages go to a different location.

**Diagnosis:**
```powershell
Get-Command python | Select-Object Source
# Should show: D:\anaconda\envs\drone_vision\python.exe
```

**Fix:** The environment wasn't activated:
```powershell
conda activate drone_vision
# Now verify:
Get-Command python | Select-Object Source
```

If conda isn't in PATH, activate with full path:
```powershell
& "D:\anaconda\Scripts\activate.bat" drone_vision
```

---

### `pip install` fails with "No module named pip"

**Fix:**
```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

---

### TensorFlow install takes too long or fails

TensorFlow 2.20 is a large package (~500 MB). If it times out:
```bash
pip install tensorflow --timeout 300
```

Or install from a mirror if the official PyPI is slow in your region.

---

### `lap` module warning on first run

**Symptom:**
```
WARNING requirements: Ultralytics requirement ['lap>=0.5.12'] not found, attempting AutoUpdate...
[Detector] Tracking failed (No module named 'lap'), using plain detect
```

**Fix:** Ultralytics auto-installs it, but needs a restart:
```bash
pip install lap
# Then restart Python
```

---

## GPU / CUDA Issues

### CUDA not available (`torch.cuda.is_available()` returns False)

**Diagnosis:**
```python
python -c "import torch; print(torch.version.cuda)"
```

**Fix 1:** Wrong PyTorch build installed (CPU build instead of CUDA):
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Fix 2:** CUDA toolkit not installed or wrong version:
```bash
nvcc --version   # Should show 11.8
nvidia-smi       # Should show GPU info
```

**Fix 3:** GPU driver too old — update from https://www.nvidia.com/drivers

---

### Out of GPU memory (CUDA OOM)

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Fixes:**
1. Lower resolution: `camera.width: 640, height: 480`
2. Disable OCR: `ocr.enabled: false`
3. Disable face recognition: `face_recognition.enabled: false`
4. Close other GPU-heavy apps (games, browsers with hardware acceleration)

---

### DeepFace / TensorFlow ignores GPU

TensorFlow sometimes doesn't see the GPU if CUDA is not properly installed.

**Diagnosis:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**Fix:** Install cuDNN matching your CUDA version:
- CUDA 11.8 → cuDNN 8.6
- Download from https://developer.nvidia.com/cudnn

---

## Module-Specific Issues

### Face Recognition — known person not recognized

**Causes and fixes:**

| Cause | Fix |
|---|---|
| Not enough training photos | Add 5–10+ photos with varied angles/lighting |
| Threshold too strict | Raise `recognition_threshold` to 0.70 |
| Face too small in frame | Person must be closer; face >= 80px |
| Wrong detector backend | Switch from `retinaface` to `mtcnn` |
| Glasses / mask obstructing face | Add photos with same accessories |

---

### Face Recognition — wrong person identified (false positive)

**Fixes:**
- Lower `recognition_threshold` to 0.50 or 0.45
- Remove poor-quality photos from the person's folder
- Make sure each person's folder contains only that person's face

---

### DeepFace downloading models to C: instead of D:

**Symptom:** You see `.deepface` folders appearing in `C:\Users\<you>\`

**Fix:** Set the env var before starting Python:
```powershell
$env:DEEPFACE_HOME = "D:\AI_Cache\deepface"
python main.py --mode camera
```

This is already configured in the conda activation script:
```
D:\anaconda\envs\drone_vision\etc\conda\activate.d\env_vars.ps1
```

If you're not using conda activate, set it manually in your PowerShell profile.

---

### EasyOCR — downloading models every run

**Symptom:** EasyOCR says "Downloading detection model..." each time.

**Fix:** The `EASYOCR_MODULE_PATH` env var isn't set, so it can't find the cached models:
```powershell
$env:EASYOCR_MODULE_PATH = "D:\AI_Cache\easyocr"
```

Or pre-download once:
```python
python -c "import easyocr; easyocr.Reader(['en'], model_storage_directory='D:/AI_Cache/easyocr')"
```

---

### Camera doesn't open

**Symptom:**
```
[Camera] Failed to open camera source 0
```

**Fixes:**
1. Check if another app is using the camera (Teams, OBS, etc.)
2. Try a different source index:
   ```yaml
   camera:
     source: 1   # or 2
   ```
3. On Windows, check camera privacy settings: Settings → Privacy → Camera → allow apps

---

### Low FPS / frame lag

**Quickest fixes:**
1. Lower resolution to 640×480
2. Increase `face_recognition.process_every_n_frames` from 3 to 8
3. Increase `ocr.process_every_n_frames` from 10 to 25
4. Switch YOLO model to `yolov8n.pt` (already the fastest)

See [Performance Guide](performance.md) for detailed tuning.

---

### OpenCV window appears then immediately closes (image mode)

**Cause:** `cv2.waitKey(0)` needs the window to have focus when you press any key.

**Fix:** Click on the OpenCV window and press any key to close it, or modify:
```python
# In main.py image mode, increase wait time
cv2.waitKey(0)   # 0 = wait forever until keypress
```

---

## Error Messages Reference

| Error | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | OpenCV not installed | `pip install opencv-python` |
| `ModuleNotFoundError: No module named 'ultralytics'` | Wrong environment active | `conda activate drone_vision` |
| `OSError: [WinError 10060]` during download | Network timeout | Retry; check firewall |
| `AttributeError: 'NoneType' object has no attribute 'shape'` | Camera frame is None | Check camera source in config |
| `PermissionError: database/vision_ai.db` | DB opened by another process | Close DB Browser for SQLite |
| `CUDA error: device-side assert triggered` | GPU memory corruption | Restart Python; lower batch size |
| `WARNING: Restart runtime for updates to take effect` | Ultralytics auto-installed a package | Restart Python (not the whole PC) |

---

## Getting Help

If none of the above fixes your issue:

1. Run `test_modules.py` and note which test fails
2. Copy the full error message from the terminal
3. Open an issue at https://github.com/mujju-212/SENTINEL/issues
   - Include: OS, GPU, error message, and which step you're on
