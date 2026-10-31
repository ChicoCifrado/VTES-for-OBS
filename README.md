# VTES Card Scanner — Plugin para OBS Studio

Identificación de cartas de **Vampire: The Eternal Struggle** en tiempo real
desde una cámara. El detector YOLO OBB localiza cada carta y la identificación
es **embedding-first**: el recorte se compara contra el índice ArcFace de las
4.149 cartas. El OCR de Tesseract es solo un apoyo para los casos en los que el
embedding no decide y la carta ocupa suficiente altura en la imagen.

## Pipeline

```text
Cámara → YOLO OBB → recorte rectificado → embeddings (global + por tipo)
       → clasificador de tipo (ViT) como filtro
       → OCR Tesseract (apoyo, solo cartas de ≥200 px de alto)
       → identificación, zonas de jugador y estado de partida
```

- **Detección:** YOLO OBB (ONNX o TensorRT) con seguimiento SORT opcional y
  detección por contornos como alternativa sin IA.
- **Identificación:** embeddings ArcFace de 1024 dimensiones. Se usa primero el
  embedder del tipo de carta, con el embedder global como respaldo. Una
  coincidencia ambigua no se pinta: en modo realizador se ofrecen las dos
  mejores candidatas para que el operador elija.
- **OCR de apoyo:** Tesseract lee el nombre solo si el embedding no ha
  identificado la carta con seguridad y el recorte mide al menos 200 px de
  alto. Así se evita gastar tiempo de CPU en cartas pequeñas o ya resueltas.
- **Zonas de jugador:** se detectan automáticamente agrupando las 4 cartas de
  cripta boca abajo de cada jugador. El asistente `/setup` permite revisarlas,
  corregir la perspectiva de la mesa y guardarlas como cuadriláteros.
- **Overlay:** el diseño 1536×1024 se escala entero sobre la escena; la cámara
  se ajusta dentro del hueco del overlay. Las guías y zonas solo se dibujan
  durante la configuración o con el toggle de depuración activo.
- **Servidor web:** búsqueda de cartas, estado de partida, cola del modo
  realizador y asistente de configuración en el puerto **8080**.

## Requisitos

- **OBS Studio** 30.x o superior.
- **Windows 10/11 x64**.
- **GPU NVIDIA** con CUDA y TensorRT 11.1 (opcional), o cualquier GPU con
  DirectML. También hay camino CPU, más lento.
- **Tesseract OCR 5.x** (opcional; sin él la identificación por embedding sigue
  funcionando).
- **CMake 3.22+** y **Visual Studio 2022+** solo para compilar desde código.

## Inicio rápido

```powershell
git clone https://github.com/ChicoCifrado/VTES-for-OBS.git
cd VTES-for-OBS
.\vtes-grimoire.ps1 build
.\vtes-grimoire.ps1 deploy
```

El script `vtes-grimoire.ps1` incluye una TUI interactiva:

```text
1. BUILD PLUGIN       — Compilar el plugin de OBS
2. DEPLOY TO OBS      — Copiar DLL + datos a OBS
3. COPY PER-TYPE      — Sincronizar modelos por tipo desde WSL
4. INSTALL TESSERACT  — Instalar motor OCR
5. VERIFY STATUS      — Diagnóstico completo del sistema
6. DEPLOY ALL         — Compilar + desplegar en un solo paso
7. PACKAGE INSTALLER  — Generar .exe (NSIS) para distribuir
Q. QUIT
```

### Comandos CLI

```text
.\vtes-grimoire.ps1 build       # Compilar
.\vtes-grimoire.ps1 deploy      # Copiar a OBS
.\vtes-grimoire.ps1 package     # Generar instalador NSIS (alias: installer)
.\vtes-grimoire.ps1 verify      # Diagnóstico del sistema
.\vtes-grimoire.ps1 all         # Build + deploy
```

## Configuración en OBS

1. Añade una fuente **Video Capture Device** (cámara) a la escena.
2. Añade el filtro **VTES Card Scanner** a la fuente de cámara.
3. Elige el dispositivo de inferencia: **CUDA**, **DirectML** o **CPU**.
4. Abre **Abrir configuración inicial** en las propiedades del filtro (o visita
   `http://localhost:8080/setup`).

### Asistente de configuración inicial

El asistente está pensado para preparar la mesa una sola vez por evento:

1. Indica si hay **4 o 5 jugadores**.
2. Coloca **4 cartas de cripta boca abajo** delante de cada jugador, juntas y
   con las mismas fundas. Espera a que aparezcan todos los grupos.
3. Ajusta la **perspectiva de la mesa** arrastrando las cuatro esquinas
   doradas. Los controles de tamaño y ángulo mueven todas las zonas a la vez.
4. Pulsa **Centrar plano** para encuadrar la cámara en el hueco del overlay.
5. Pulsa **Aplicar y guardar**. Las zonas quedan fijas como cuadriláteros, la
   detección automática se desactiva y `show_zones` vuelve a `false`.

Durante el asistente las zonas antiguas no asignan cartas ni disparan eventos
de influencia. Al aplicar se guardan las posiciones y el encuadre en la fuente
de OBS, por lo que se conservan al cerrar y reabrir OBS.

La referencia técnica del asistente, el formato de los quads y la API completa
están en **[docs/setup.md](docs/setup.md)**.

## Controles del filtro

| Control | Descripción |
|---|---|
| **Mostrar cajas** | Dibuja las cajas y etiquetas de las cartas detectadas. |
| **Siempre activo** | Mantiene la detección aunque las cajas estén ocultas. |
| **Idioma** | Español o inglés; afecta al panel y a la web. |
| **Umbral de confianza** | Confianza mínima del detector (0,0–1,0; defecto 0,5). |
| **Modo de detección** | Modelo IA (ONNX) o forma por contornos. |
| **Dispositivo de inferencia** | CUDA, DirectML o CPU. |
| **Clasificador de tipo** | Filtra la identificación por los 14 tipos de carta. |
| **OCR** | Activa el apoyo de Tesseract para cartas grandes. |
| **Escalado NIS** | Lanczos 4x + nitidez adaptativa por CUDA. |
| **Clasificador Oval/Rect** | Reduce falsos positivos en cartas boca abajo. |
| **Intervalo de detección** | Pausa mínima entre detecciones (0 = continuo). |
| **Desplazamiento de rulings** | Scroll vertical del bloque de rulings (0–100 %). |
| **Modo realizador** | Las detecciones van a la cola de pendientes; el operador elige qué se muestra. |
| **Número de jugadores** | 4 o 5 jugadores. |
| **Mostrar zonas** | Toggle de depuración; al aplicar el setup vuelve a `false`. |
| **Zonas automáticas** | Detecta las zonas agrupando 4 cartas de cripta boca abajo. |
| **Carta grande** | Dibuja la carta principal a media pantalla (modo torneo). |
| **Fuente del overlay** | Cinzel, Grenze Gotisch o Hershey clásica. |
| **Cámara a sangre** | Usa también las bandas laterales, no solo el hueco del overlay. |
| **Zoom / offsets de cámara** | Ajuste fino del encuadre bajo el overlay. |
| **Jugador N** | Nombre, pool inicial, activo y zonas manuales de cripta/minions. |

## Servidor web

El plugin escucha en `0.0.0.0:8080` (todas las interfaces) y **no tiene
autenticación**. Úsalo en una red local de confianza y no lo expongas a
Internet.

| Ruta | Contenido |
|---|---|
| `/` | Búsqueda de cartas y filtros por tipo, clan, disciplina, capacidad… |
| `/setup` | Asistente de configuración inicial. |
| `/game` | Estado de partida: jugadores, minions, pool, recursos y eventos. |
| `/pool` | Control rápido de pool. |
| `/detections` | Historial de detecciones. |
| `/pending` | Cola del modo realizador. |

La API del asistente (`/api/setup/*`) y los parámetros de perspectiva están
documentados en `docs/setup.md`.

## Zonas automáticas y perspectiva

- El detector agrupa **4 cartas boca abajo** con geometría y color similares.
- Cuando hay un grupo estable por jugador, deriva la zona de **cripta** (el
  grupo) y la de **minions** (la fila superior), ordenando los grupos por
  asiento alrededor del centro de la mesa.
- Las zonas se guardan como **quads normalizados** (0..1) en coordenadas de la
  imagen. Si un quad no está disponible, se usa el rectángulo
  `x/y/w/h + ángulo`.
- La perspectiva es una **homografía de plano**: las cuatro esquinas de la guía
  y todas las zonas comparten el mismo plano proyectivo.
- La inclinación 3D de la guía del asistente es solo una ayuda visual de CSS;
  no modifica la detección ni las zonas guardadas.
- Una carta se asigna a la zona de minions o cripta que la contiene con mayor
  margen. Las zonas de jugadores ousted no asignan cartas.

## Compilar desde código

```powershell
cmake --preset windows-x64
cmake --build --preset windows-x64 --config RelWithDebInfo --parallel
cmake --install build_x64 --prefix release/RelWithDebInfo --config RelWithDebInfo
```

Para recompilar sin reconfigurar, en Windows también está
`build_x64/build_now.bat`.

### Compilar contra el OBS instalado (SDK local)

Por defecto el build descarga OBS y compila `libobs` desde fuente dentro de
`.deps`. Para usar el OBS ya instalado:

```powershell
scripts\make_obs_sdk.bat
cmake --preset windows-x64
```

El script crea `.deps\obs-sdk` con las cabeceras de la versión instalada,
genera `obs.lib` a partir de los exports de `obs.dll` y escribe el
`libobsConfig.cmake` que define `OBS::libobs`. `CMakePresets.json` fija
`OBS_SDK_DIR` a ese directorio; si no existe, el build vuelve al camino de
`buildspec.json`. Para volver al `libobs` compilado desde fuente, borra
`.deps\obs-sdk` o pasa `-DOBS_SDK_DIR=` vacío.

### Opciones de CMake

| Flag | Descripción |
|---|---|
| `-DUSE_SYSTEM_OPENCV=ON` | Usar OpenCV del sistema en vez de descargarlo. |
| `-DUSE_SYSTEM_TESSERACT=ON` | Usar Tesseract del sistema. |
| `-DOpenCV_DIR=C:/opencv/build` | Ruta al config de CMake de OpenCV. |

## Estructura del proyecto

```text
src/
  plugin-main.c                 — Punto de entrada del plugin OBS
  detect-filter-obb.cpp         — Filtro: detección, identificación, zonas y overlay
  FilterData.h                  — Estado del filtro, jugadores, zonas y perspectiva
  web_server.h                  — Servidor HTTP, páginas y API
  i18n.cpp / i18n.h             — Textos ES/EN
  embedding_matcher.h           — Matcher ArcFace global y por tipo
  detection/
    yolo_detector.cpp           — Inferencia YOLO OBB (ONNX/DirectML)
    tensorrt_detector.cpp       — Inferencia YOLO OBB (TensorRT/CUDA)
    contour_detector.cpp        — Detección por contornos (sin IA)
    auto_zone_detector.cpp      — Grupos de 4 cartas y geometría de zonas
    detector_base.hpp           — Interfaz base de detectores
    detection_types.hpp         — Tipos comunes (OBBObject, etc.)
  classifier/
    vtes_card_classifier.cpp    — Clasificador de tipo por visión (ViT)
  ocr/
    vtes_ocr.cpp                — OCR Tesseract + fuzzy matching
    vtes_api_lookup.cpp         — Búsqueda de cartas por nombre
  overlay/
    rulings_view.cpp            — Render del bloque de rulings
  sort/                         — Seguimiento SORT
  cuda/                         — Clasificador CUDA y kernels
data/
  models/                       — YOLO OBB y configuración del detector
  locale/                       — Textos ES/EN
  effects/                      — Efectos de desenfoque/máscara
  obs-plugins/vtes-card-scanner/— Overlay y recursos empaquetados
cmake/                          — Dependencias y helpers de build
scripts/                        — Entrenamiento, índice de embeddings y utilidades
docs/                           — Documentación técnica
vtes-grimoire.ps1               — TUI/CLI para build, deploy, verify y package
```

## Modelos y assets

El repositorio incluye los modelos pequeños de inferencia y el clasificador de
tipo, pero no los índices de embeddings ni los artefactos generados:

| Recurso | Ruta | ¿Versionado? |
|---|---|---|
| Detector YOLO OBB | `data/models/vtes.onnx` | Sí |
| Configuración del detector | `data/models/vtes.json` | Sí |
| Embedder ArcFace | `data/vtes_embedder_1024d.onnx` | Sí |
| Índice de embeddings | `data/embeddings_1024d.bin` + `_meta.json` | No (122 MB) |
| Clasificador de tipo | `data/vtes_type_classifier.onnx` + `.data` | Sí |
| Modelos por tipo | `data/per_type/` + `per_type_manifest.json` | No (opcional) |
| Base de datos de cartas | `data/vtes.json` | No (se descarga de krcg) |
| Motor TensorRT | `vtes.engine` | No (se genera con `trtexec`) |
| Fuentes del overlay | `data/fonts/` | No (assets locales) |

Los modelos por tipo se sincronizan con la opción **3** de
`vtes-grimoire.ps1`. Si faltan, el plugin usa el embedder global. Si faltan las
fuentes TTF/OTF, el texto cae en Hershey y los iconos pueden no dibujarse.

Para generar el índice de embeddings:

```bash
python3 scripts/build_embedding_index.py \
  --onnx data/vtes_embedder_1024d.onnx \
  --vtes-json data/vtes.json \
  --output data/
```

## Instalador NSIS

La opción **7** (o `.ps1 package`) genera un `.exe` redistribuible con:

- DLL del plugin y dependencias de ejecución (ONNX Runtime, TensorRT, OpenCV).
- Archivos de datos (modelos, clasificador, base de datos de cartas).
- Entrada en Agregar o Quitar Programas de Windows.
- Desinstalador que limpia los archivos de VTES.

El instalador usa como destino `C:\Program Files\obs-studio` y detecta la
ubicación de OBS desde el registro de Windows.

## Diagnóstico rápido

- **No aparecen zonas:** coloca las 4 cartas boca abajo, juntas, sin solaparse y
  con luz suficiente. Todas las cartas deben usar la misma funda.
- **Faltan zonas:** espera a que los N grupos sean estables a la vez; el
  asistente exige un grupo por jugador antes de habilitar el guardado.
- **El OCR no lee:** comprueba Tesseract y acerca la cámara. Por debajo de
  200 px de alto la carta se identifica solo por embedding.
- **No identifica cartas:** revisa que existan
  `data/embeddings_1024d.bin` y `data/embeddings_1024d_meta.json` junto al
  embedder; el log de OBS indica qué falta.
- **El puerto 8080 está ocupado:** cierra el proceso que lo use; el plugin
  avisa en el log y no arranca el servidor web.
- **Se ven zonas en la salida final:** desactiva **Mostrar zonas** o vuelve a
  aplicar el setup; al guardar se fuerza `show_zones=false`.

## Documentación

- [Configuración inicial y API](docs/setup.md)
- [Entrenamiento de modelos](docs/train_model.md)
- [Cambios por versión](CHANGELOG.md)

## Licencia

GNU General Public License v2.0 — ver [LICENSE](LICENSE).

---

**VTES Card Scanner** — La Garra Cifrada
