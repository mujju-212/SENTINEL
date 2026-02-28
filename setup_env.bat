@echo off
REM ============================================================
REM       DRONE VISION AI - ENVIRONMENT SETUP SCRIPT
REM       Run this ONCE after installing Anaconda
REM ============================================================

echo.
echo ============================================================
echo    DRONE VISION AI - Setting Up Environment
echo ============================================================
echo.

REM ── STEP 1: Create Anaconda Environment ─────────────────────
echo [1/6] Creating conda environment (drone_vision)...
call conda create -n drone_vision python=3.10 -y
if errorlevel 1 (echo ERROR: Failed to create conda environment & pause & exit /b 1)
echo Done.

REM ── STEP 2: Activate Environment ────────────────────────────
echo [2/6] Activating environment...
call conda activate drone_vision
if errorlevel 1 (echo ERROR: Failed to activate environment & pause & exit /b 1)
echo Done.

REM ── STEP 3: Install PyTorch with CUDA ───────────────────────
echo [3/6] Installing PyTorch with CUDA 11.8 support...
echo       (This may take 5-10 minutes - large download)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo WARNING: CUDA install failed. Trying CPU version...
    pip install torch torchvision torchaudio
)
echo Done.

REM ── STEP 4: Install All Requirements ────────────────────────
echo [4/6] Installing all project libraries...
echo       (This may take 10-15 minutes)
pip install -r requirements.txt
if errorlevel 1 (echo ERROR: Some packages failed to install & echo Check errors above & pause & exit /b 1)
echo Done.

REM ── STEP 5: Create Directory Structure ──────────────────────
echo [5/6] Creating project folder structure...
if not exist "models"                    mkdir models
if not exist "database\known_faces"      mkdir database\known_faces
if not exist "input\test_images"         mkdir input\test_images
if not exist "output\logs"               mkdir output\logs
if not exist "output\screenshots"        mkdir output\screenshots
echo Done.

REM ── STEP 6: Test GPU ────────────────────────────────────────
echo [6/6] Testing GPU availability...
python -c "import torch; gpu=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if gpu else 'None'; print(f'GPU Available: {gpu}'); print(f'GPU Name: {name}')"

echo.
echo ============================================================
echo    SETUP COMPLETE!
echo ============================================================
echo.
echo To start the system:
echo   conda activate drone_vision
echo   python main.py --mode camera
echo.
echo To add a person to face database:
echo   python add_person.py
echo.
echo To test all modules:
echo   python test_modules.py
echo.
pause
