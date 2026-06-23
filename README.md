# VTES Card Scanner — Plugin para OBS Studio

Identificación de cartas de **Vampire: The Eternal Struggle** en tiempo real desde cámara web. Usa detección YOLO OBB + clasificador de tipo por visión + OCR (Tesseract) como sistema de identificación principal.

## Pipeline

```
Cámara web → YOLO OBB Detection → Crop por carta → Clasificador de tipo
  → OCR Tesseract (crop del nombre, fuzzy match contra base de datos)
```

- **Detección:** YOLO OBB (Oriented Bounding Boxes) — detecta posición, ángulo y tipo de cada carta
- **Clasificación de tipo:** Vision Transformer (ViT) — clasifica en 14 tipos de carta VTES
- **Identificación:** OCR Tesseract lee el nombre desde la región superior de la carta y hace fuzzy matching contra las 4149 cartas de la base de datos
- **Cooldown:** 5 segundos de pausa tras identificar una carta para evitar bucles de detección

## 14 Tipos de Carta

Action, Action Modifier, Ally, Combat, Conviction, Equipment, Event, Imbued, Master, Political Action, Power, Reaction, Retainer, Vampire

## Requisitos

- **OBS Studio** 30.x+
- **Windows 10/11** (x64)
- **GPU:** NVIDIA (CUDA) con TensorRT 11.1, o cualquier GPU con DirectML
- **Tesseract OCR** 5.x (instalado automáticamente por el script)
- **CMake** 3.22+ y **Visual Studio 2022+** (solo para compilar desde código)

## Inicio Rápido

```powershell
git clone https://github.com/ChicoCifrado/VTES-for-OBS.git
cd VTES-for-OBS
.\vtes-grimoire.ps1 build
.\vtes-grimoire.ps1 deploy
```

El script `vtes-grimoire.ps1` provee una TUI interactiva:

```
1. BUILD PLUGIN       — Compilar el plugin de OBS
2. DEPLOY TO OBS      — Copiar DLL + datos a OBS
3. COPY PER-TYPE      — Sincronizar modelos por tipo desde WSL
4. INSTALL TESSERACT   — Instalar motor OCR
5. VERIFY STATUS      — Diagnóstico completo del sistema
6. DEPLOY ALL         — Compilar + Desplegar en un solo paso
7. PACKAGE INSTALLER  — Generar .exe (NSIS) para distribuir
Q. QUIT
```

### Comandos CLI

```
.\vtes-grimoire.ps1 build       # Compilar
.\vtes-grimoire.ps1 deploy      # Copiar a OBS
.\vtes-grimoire.ps1 package     # Generar instalador NSIS (alias: installer)
.\vtes-grimoire.ps1 verify      # Diagnóstico del sistema
.\vtes-grimoire.ps1 all         # Build + Deploy
```

## Instalador NSIS

La opción **7** (o `.ps1 package`) genera un `.exe` redistribuible con:

- DLL del plugin + todas las dependencias de ejecución (ONNX Runtime, TensorRT, OpenCV)
- Archivos de datos (modelos, clasificador, base de datos de cartas)
- Entrada en Agregar o Quitar Programas de Windows
- Desinstalador que limpia todos los archivos de VTES

El instalador usa como destino `C:\Program Files\obs-studio` y detecta la ubicación de OBS desde el registro de Windows.

## Configuración en OBS

1. Añade una fuente **Video Capture Device** (cámara web) a tu escena
2. Añade el filtro **VTES Card Scanner** a la fuente de cámara
3. Selecciona dispositivo de inferencia: **CUDA** (NVIDIA), **DirectML** (cualquier GPU), o **CPU**
4. Apunta la cámara a las cartas — aparecerán las bounding boxes con el nombre de cada carta

### Controles del filtro

| Control | Descripción |
|---|---|
| **Show Bounding Boxes** | Muestra/oculta los rectángulos y etiquetas sobre las cartas detectadas |
| **Always Active** | Mantiene la detección activa aunque las bounding boxes estén ocultas |
| **Confidence Threshold** | Umbral de confianza del detector YOLO (0.0–1.0, defecto 0.5) |
| **Detection Interval** | Intervalo mínimo entre detecciones (0–5 segundos, 0 = cada frame) |
| **Inference Device** | CUDA (NVIDIA), DirectML (GPU genérica), o CPU |

## Compilar desde Código

```powershell
cmake --preset windows-x64 -DUSE_SYSTEM_OPENCV=ON
cmake --build --preset windows-x64 --config RelWithDebInfo --parallel
cmake --install build_x64 --prefix release/RelWithDebInfo --config RelWithDebInfo
```

### Opciones de CMake

| Flag | Descripción |
|---|---|
| `-DUSE_SYSTEM_OPENCV=ON` | Usar OpenCV del sistema en vez de descargarlo |
| `-DUSE_SYSTEM_TESSERACT=ON` | Usar Tesseract del sistema |
| `-DOpenCV_DIR=C:/opencv/build` | Ruta al config de CMake de OpenCV |

## Estructura del Proyecto

```
src/
  plugin-main.c                 — Punto de entrada del plugin OBS
  detect-filter-obb.cpp         — Filtro principal: YOLO OBB + OCR pipeline
  FilterData.h                  — Estructura de datos del filtro
  detection/
    yolo_detector.cpp           — Inferencia YOLO OBB (DirectML)
    tensorrt_detector.cpp       — Inferencia YOLO OBB (TensorRT CUDA)
    contour_detector.cpp        — Detección por contornos (sin IA)
    detector_base.hpp           — Interfaz base de detectores
    detection_types.hpp         — Tipos comunes (OBBObject, etc.)
  classifier/
    vtes_card_classifier.cpp    — Clasificador de tipo por visión (ViT)
  ocr/
    vtes_ocr.cpp                — OCR Tesseract + fuzzy matching
    vtes_api_lookup.cpp         — Búsqueda de cartas por nombre
  embedding_matcher.h           — Matcher por embeddings (ArcFace, actualmente sin modelos)
cmake/
  FetchOpenCV.cmake             — Dependencia OpenCV (auto-descarga o sistema)
  FetchOnnxruntime.cmake        — Dependencia ONNX Runtime
scripts/                        — Scripts de entrenamiento y utilidades
vtes-grimoire.ps1               — TUI/CLI unificada para build/deploy/verify
```

## Modelos

Los modelos **no están incluidos** en el repositorio por tamaño. Deben obtenerse por separado:

- **Detección:** YOLO26 OBB ONNX o TensorRT engine (`vtes.engine`)
- **Clasificador de tipo:** ViT ONNX (`vtes_type_classifier.onnx`)
- **Base de datos de cartas:** `vtes.json` (incluida en el repositorio)

Colocar en `data/obs-plugins/vtes-card-scanner/models/` (o usar el script `vtes-grimoire.ps1` opción 3 para sincronizar desde WSL).

## Desinstalación

1. Cierra OBS Studio
2. Ve a **Agregar o Quitar Programas** → **VTES Card Scanner OBS Plugin** → **Desinstalar**
3. O ejecuta `uninstall-vtes-card-scanner.exe` en `obs-plugins\64bit\`

El desinstalador elimina:
- DLL del plugin + PDB
- DLLs de ejecución (ONNX Runtime, TensorRT, CUDA, OpenCV)
- Datos del plugin (modelos, base de datos, logs)
- Entrada del registro de Windows

## Licencia

GNU General Public License v2.0 — ver [LICENSE](LICENSE)

---

**VTES Card Scanner** — La Garra Cifrada
