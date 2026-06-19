# v0.1.3 — Grimorio Sanguíneo

## Cambios principales

### Detección
- Nuevo detector YOLO26m-OBB (1024×1024, GPU CUDA/DML)
- Clasificador de tipo por visión (14 tipos VTES, 99.9% accuracy en scans)
- Corrección de perspectiva en crops del clasificador de tipo (usa ángulo OBB)
- Fix: clasificador de tipo usa `Ort::Session` directamente (cv::dnn parseaba mal el ONNX)

### Web Server (card search)
- Servidor web embebido en `localhost:8080` con búsqueda multi-idioma
- `vtes.json` se descarga auto desde `https://static.krcg.org/data/vtes.json` si no está local
- Notificación nativa en OBS al iniciar el servidor (una vez por sesión)

### Misc
- Limpieza: eliminados ~106 MB de modelos no usados (EdgeYOLO, YUNet, YOLOv8n)
- Script `vtes-grimoire.ps1` unificado (build/deploy/verify/package)
- CMakePresets simplificado: solo Windows + macOS
