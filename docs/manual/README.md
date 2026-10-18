# Manual de usuario — generadores

Los manuales en PDF se generan con estos scripts (fpdf2 + Pillow):

- `build_manual.py` → `VTES-for-OBS_Manual_v0.1.7.pdf` (español)
- `build_manual_en.py` → `VTES-for-OBS_Manual_v0.1.7_EN.pdf` (inglés)

## Regenerar

```bash
python3 build_manual.py
python3 build_manual_en.py
```

El script en inglés reutiliza la infraestructura del español (clase de
documento, paleta y generación de assets), así que los cambios de diseño se
hacen una sola vez en `build_manual.py`.

## Assets

`assets/` se regenera automáticamente a partir del diseño real del overlay
(`data/obs-plugins/vtes-card-scanner/overlay-bg.png`): la portada usa el ankh
recortado y los diagramas anotados (ES/EN) se dibujan sobre el propio diseño.

## Requisitos

```bash
pip install fpdf2 Pillow
```
