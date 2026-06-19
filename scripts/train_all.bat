@echo off
REM ─── VTES Per-Type Embedding Trainer ────────────────────────────
REM Run this from WSL bash or on Windows with Python + PyTorch.
REM
REM Prerequisites:
REM   pip install torch torchvision opencv-python pillow numpy
REM
REM Steps:
REM   1. Download card images from krcg.org
REM   2. Organize by card type
REM   3. Train one ArcFace embedding model per type
REM   4. Copy results to plugin data directory
REM ─────────────────────────────────────────────────────────────────

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set VTES_JSON=%PROJECT_DIR%\vtes.json
set CACHE_DIR=%PROJECT_DIR%\data\card-cache
set DATASET_DIR=%PROJECT_DIR%\dataset\by_type
set OUTPUT_DIR=%PROJECT_DIR%\data\per_type

echo ════════════════════════════════════════════
echo VTES Per-Type Embedding Training Pipeline
echo ════════════════════════════════════════════

REM ── Step 1: Download card images ──
echo.
echo [1/4] Downloading card images from krcg.org...
python "%SCRIPT_DIR%fetch_vtes_data.py"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Download failed
    exit /b %ERRORLEVEL%
)

REM ── Step 2: Organize by type ──
echo.
echo [2/4] Organizing images by card type...
python "%SCRIPT_DIR%organize_dataset.py" ^
    --vtes-json "%VTES_JSON%" ^
    --cache-dir "%CACHE_DIR%" ^
    --output-dir "%DATASET_DIR%"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Organization failed
    exit /b %ERRORLEVEL%
)

REM ── Step 3: Train per-type models ──
echo.
echo [3/4] Training per-type embedding models (14 types)...
echo This will take several hours with GPU.
python "%SCRIPT_DIR%train_per_type.py" ^
    --data "%DATASET_DIR%" ^
    --vtes-json "%VTES_JSON%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --epochs 200 --batch-size 64
if %ERRORLEVEL% neq 0 (
    echo ERROR: Training failed
    exit /b %ERRORLEVEL%
)

REM ── Step 4: Done ──
echo.
echo [4/4] Training complete!
echo Models are in: %OUTPUT_DIR%
echo.
echo Next steps:
echo   1. Deploy to OBS: .\vtes-grimoire.ps1 deploy
echo   2. Or package:    .\vtes-grimoire.ps1 package
echo.

pause
