# VTES Card Scanner — OBS Plugin

Real-time VTES card identification plugin for OBS Studio. Detect and identify cards from a webcam feed using YOLO detection + ArcFace embedding matching + OCR fallback.

## Pipeline

```
Webcam -> YOLO OBB Detection -> Per-Card Crop -> Type Classifier
  -> Per-Type ArcFace Embedder (14 models, 3-1703 classes each)
  -> Global Embedder Fallback
  -> Tesseract OCR Fallback (name crop, fuzzy match)
```

- **Detection:** YOLO OBB (Oriented Bounding Boxes) — detects card position, angle, and type
- **Type Classification:** Vision Transformer (ViT) — classifies into 14 VTES card types
- **Card Identification:** ArcFace embedding matching — per-type matchers (14 ONNX models) + global fallback
- **OCR Fallback:** Tesseract reads card name from top 10% of card image when embedding confidence < 80%

## 14 Card Types

Action, Action Modifier, Ally, Combat, Conviction, Equipment, Event, Imbued, Master, Political Action, Power, Reaction, Retainer, Vampire

## Requirements

- **OBS Studio** 30.x+
- **Windows 10/11** (x64) or **Linux** (x86_64)
- **GPU:** NVIDIA (CUDA), AMD/Intel (DirectML), or CPU fallback
- **CMake** 3.22+
- **Visual Studio 2022+** (Windows) or **GCC** (Linux)
- **OpenCV** 5.x (system-installed or auto-downloaded)

## Quick Start (Windows)

```powershell
# Clone and build
git clone https://github.com/ChicoCifrado/VTES-for-OBS.git
cd VTES-for-OBS
.\vtes-grimoire.ps1 build
.\vtes-grimoire.ps1 deploy
```

The `vtes-grimoire.ps1` script provides an interactive TUI:

```
1. BUILD PLUGIN     — Compile the OBS plugin
2. DEPLOY TO OBS    — Copy DLL + data to OBS
3. COPY PER-TYPE    — Sync per-type models from WSL
4. INSTALL TESSERACT — Install OCR engine
5. VERIFY STATUS    — Full system diagnosis
6. DEPLOY ALL       — Build + Deploy in one step
Q. QUIT
```

### OBS Setup

1. Add a **Video Capture Device** (webcam) to your scene
2. Add **VTES Card Scanner** filter to the webcam source
3. Select inference device: **CUDA** (NVIDIA), **DirectML** (any GPU), or **CPU**
4. Point camera at VTES cards — bounding boxes with card names appear

## Building from Source

### Windows (MSVC)

```powershell
cmake --preset windows-x64 -DUSE_SYSTEM_OPENCV=ON -DOpenCV_DIR=C:/opencv/build
cmake --build --preset windows-x64 --config RelWithDebInfo --parallel
cmake --install build_x64 --prefix release/RelWithDebInfo --config RelWithDebInfo
```

### Linux

```bash
cmake --preset linux-x86_64
cmake --build --preset linux-x86_64 --config RelWithDebInfo --parallel
```

### Options

| CMake Flag | Description |
|---|---|
| `-DUSE_SYSTEM_OPENCV=ON` | Use system OpenCV instead of downloading |
| `-DUSE_SYSTEM_TESSERACT=ON` | Enable OCR via system Tesseract |
| `-DOpenCV_DIR=C:/opencv/build` | Path to OpenCV cmake config |
| `-DUSE_SYSTEM_ONNXRUNTIME=ON` | Use system ONNX Runtime (Linux only) |

## Project Structure

```
src/
  plugin-main.c              — OBS plugin entry point
  detect-filter-obb.cpp      — Main filter: YOLO OBB + embedding + OCR pipeline
  detect-filter.cpp          — Original detection filter (non-OBB)
  ort-model/ONNXRuntimeModel.cpp — ONNX Runtime wrapper with GPU provider support
  yolov8/yolov8_obb_yolo26.cpp  — YOLO26 OBB inference
  classifier/vtes_card_classifier.cpp — Vision Transformer type classifier
  ocr/vtes_ocr.cpp           — Tesseract OCR + fuzzy card name matching
  detection/contour_detector.cpp — Contour-based card detection (no ML)
cmake/
  FetchOpenCV.cmake          — OpenCV dependency (auto-download or system)
  FetchOnnxruntime.cmake     — ONNX Runtime dependency
  common/FindTesseract.cmake — Tesseract find module
vendor/                      — Vendored dependencies
scripts/                     — Training & utility scripts
vtes-grimoire.ps1            — Unified TUI/CLI for build/deploy/verify
```

## Models

Models are **not** included in the git repository. Obtain them separately:

- **Detection model:** YOLO26 OBB ONNX (from training)
- **Type classifier:** ViT ONNX
- **Per-type embedders:** 14 ArcFace ONNX models + embedding banks
- **Global embedder:** ArcFace ONNX + embedding bank

Place in `data/per_type/` or use `Invoke-CopyPerType` (option 3 in the grimorie) to sync from a WSL training environment.

## License

MIT License — see [LICENSE](LICENSE)

---

**VTES Card Scanner** — La Garra Cifrada
