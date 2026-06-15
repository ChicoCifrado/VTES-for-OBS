@echo off
REM Build script for YOLO26 migration

echo "=== Migracion a YOLO26-OBB ==="

REM 1. Limpiar build anterior
cd /mnt/c/Users/JackSuicide/VTES/vtes_obs_detect
call cmake --build build_x64 --target clean

REM 2. Rebuild con YOLO26
call cmake --build build_x64 --config Release

REM 3. Copiar modelo YOLO26 ONNX
cd /mnt/c/Users/JackSuicide/VTES/vt_card_scanner
call python vt_card_scanner/train.py export:
    - model: yolo26n-obb.pt
    - imgsz: 640
    - format: onnx
    - simplify: True
    - opset: 17
    - end2end: True  # YOLO26 one-to-one output (300, 6)

REM 4. Copiar ONNX a plugin
copy "runs\obb\exp\*.onnx" "models\yolo26_obb.onnx"

REM 5. Listar archivos
echo "Archivos YOLO26:"
dir /b "models\x*

echo "=== Build YOLO26 complete! ==="
