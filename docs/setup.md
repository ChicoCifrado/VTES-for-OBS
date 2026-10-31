# Configuración inicial: zonas, perspectiva y encuadre

Este documento describe el asistente `/setup` del plugin, el formato en el que
se guardan las zonas y la API que usa la página. Está pensado para operadores
que preparan una mesa y para quien necesite mantener el código.

## 1. Qué resuelve

El asistente prepara una mesa de **4 o 5 jugadores** una sola vez:

1. Detecta un grupo de **4 cartas de cripta boca abajo** por jugador.
2. Deriva de cada grupo la zona de **cripta** y la zona de **minions**.
3. Permite corregir el conjunto con una **homografía de perspectiva**.
4. Calcula el **encuadre de cámara** para que la mesa entre en el hueco del
   overlay.
5. Guarda las zonas como **cuadriláteros normalizados** y desactiva la
   detección automática para esa partida.

Durante el asistente no se modifican las zonas de los jugadores ni se asignan
cartas: las zonas antiguas quedan ignoradas hasta pulsar **Aplicar y guardar**.
La salida final no muestra guías, rejilla ni zonas; solo se dibujan en el
preview del filtro y en la propia página de configuración.

## 2. Preparación de la mesa

- Cámara estable, preferiblemente en trípode o cenital.
- Mesa completa dentro del encuadre.
- **4 cartas de cripta boca abajo** delante de cada jugador, juntas y sin
  solaparse.
- Las cuatro cartas de un mismo jugador deben tener la **misma funda** o un
  reverso visualmente similar.
- Iluminación suficiente y sin reflejos fuertes.
- Los jugadores deben estar sentados alrededor de la mesa antes de empezar.

El detector agrupa las cartas por geometría, tamaño y color. No necesita saber
qué carta es: solo que hay cuatro objetos repetidos formando un grupo.

## 3. Flujo del asistente

### 3.1 Abrir la página

Desde OBS: propiedades del filtro **VTES Card Scanner** → **Abrir configuración
inicial**.

O directamente en el navegador:

```text
http://localhost:8080/setup
```

El servidor escucha en todas las interfaces (`0.0.0.0:8080`) y no tiene
autenticación. No lo expongas fuera de una red local de confianza.

### 3.2 Elegir jugadores y detectar

1. Selecciona **4** o **5** jugadores.
2. Pulsa **Detectar zonas**.
3. Coloca los grupos y espera. El estado muestra `Buscando zonas: n/total`.
4. Cuando todos los grupos son estables, el estado pasa a
   `Zonas detectadas: n/total` y se habilita el resto del asistente.

El detector se ejecuta a **2 Hz** y exige que los grupos sean estables. Además
hay una espera de **1,5 s** desde el inicio para que el operador termine de
colocar las cartas antes de fijar el orden de asiento.

### 3.3 Perspectiva de la mesa

La mesa se modela como un plano. La página muestra:

- la **rejilla** del plano proyectivo;
- la **guía central** (rectángulo, cuadrado, círculo u óvalo);
- las **zonas propuestas**;
- las **cuatro esquinas doradas** que se pueden arrastrar.

Controles:

| Control | Efecto |
|---|---|
| Esquinas doradas | Mueven un vértice del cuadrilátero de perspectiva. |
| **Tamaño** | Escala el conjunto alrededor de su centro (0,5–1,5). |
| **Ángulo** | Gira el conjunto alrededor de su centro (−45°–45°). |
| **Restablecer** | Vuelve a la caja envolvente detectada, tamaño 1 y ángulo 0. |
| **Forma central** | Rectángulo, cuadrado, círculo u óvalo. |
| **Tamaño / ángulo de guía** | Ajustan solo la guía visual. |
| **Profundidad de guía** | Inclinación 3D **solo CSS**; no afecta a la detección. |

La homografía se calcula entre:

- `source`: rectángulo que envuelve todas las zonas propuestas;
- `target`: cuadrilátero `base` después de aplicar tamaño y ángulo.

Cada zona se transforma con esa homografía antes de dibujarse y antes de
guardarse. La guía central y la rejilla usan el mismo plano, de modo que lo que
se ve en pantalla coincide con las zonas que se van a guardar.

El servidor rechaza cuadriláteros degenerados (puntos repetidos, área
insuficiente, esquinas cruzadas o coordenadas no finitas) y responde
`invalid_corners` sin modificar el estado.

### 3.4 Encuadre de cámara

Pulsa **Centrar plano**. El cálculo:

- toma la unión de las zonas de cripta y minions;
- añade un 4 % de aire alrededor;
- calcula el zoom y los offsets para que esa unión entre en el hueco de cámara
  del overlay con un 8 % de margen;
- limita el zoom a 1,0–3,0 y los offsets a ±400 px de diseño.

La opción **Cámara a sangre** usa el frame completo en lugar del hueco del
overlay, incluyendo las bandas laterales.

### 3.5 Aplicar y guardar

**Aplicar y guardar** hace, en este orden:

1. Vuelve a aplicar la perspectiva a las vistas propuestas.
2. Copia las zonas a cada jugador (`player_idx`).
3. Limpia la memoria de influencia y la asignación de cartas anteriores.
4. Aplica el encuadre de cámara.
5. Desactiva `auto_zones`, marca `setup_done=true` y fuerza
   `show_zones=false`.
6. Guarda las zonas, los quads y el encuadre en la configuración de la fuente.
7. Reinicia el detector automático y limpia las vistas del asistente.

El guardado usa `obs_source_update`, por lo que los valores persisten al cerrar
y reabrir OBS.

**Repetir detección** reinicia el asistente sin tocar la configuración guardada
hasta que se vuelva a aplicar.

## 4. Modelo de datos

### 4.1 `PlayerZone`

Cada zona de un jugador se representa con:

```cpp
struct PlayerZone {
    float nx, ny, nw, nh;   // rectángulo normalizado 0..1
    float angle;            // grados, positivo = horario
    float lock_angle;       // referencia para detectar cartas giradas
    bool quad;              // true si hay perspectiva
    cv::Point2f corners[4]; // TL, TR, BR, BL normalizados
};
```

- Si `quad == false`, la zona es el rectángulo `nx/ny/nw/nh` rotado `angle`.
- Si `quad == true`, la zona es el cuadrilátero `corners` y el rectángulo
  anterior es solo su caja envolvente.
- `lock_angle` se toma de la orientación real de las cuatro cartas de cripta y
  se usa para decidir si una carta está girada.

### 4.2 Ajustes persistidos

Para cada jugador `N` (0–4) se guardan:

```text
player_N_crypt_x
player_N_crypt_y
player_N_crypt_w
player_N_crypt_h
player_N_crypt_angle
player_N_crypt_quad     # "x,y;x,y;x,y;x,y" en orden TL,TR,BR,BL
player_N_minion_x
player_N_minion_y
player_N_minion_w
player_N_minion_h
player_N_minion_angle
player_N_minion_quad
```

Y a nivel de filtro:

```text
player_count
auto_zones = false
setup_done = true
show_zones = false
cam_zoom
cam_offset_x
cam_offset_y
cam_full_frame
```

Al cargar, `loadZoneQuadString` valida el quad y comprueba que su caja
envolvente siga coincidiendo con `x/y/w/h`. Si el operador ha editado la caja a
mano desde las propiedades, el quad se descarta y se respeta la edición manual
(tolerancia 0,002 en coordenadas normalizadas).

## 5. API de configuración

Todas las rutas son `GET` y devuelven JSON.

| Ruta | Parámetros | Descripción |
|---|---|---|
| `/api/setup/status` | — | Estado completo del asistente. |
| `/api/setup/start` | `players=4\|5` | Inicia la detección y reinicia el estado. |
| `/api/setup/reset` | — | Cancela el asistente sin guardar. |
| `/api/setup/perspective` | `corners=...&size=...&angle=...` o `reset=1` | Actualiza la homografía. |
| `/api/setup/camera` | `full_frame=0\|1` | Calcula y aplica el encuadre. |
| `/api/setup/apply` | `full_frame=0\|1` | Guarda zonas y encuadre. |

### 5.1 `/api/setup/status`

Ejemplo abreviado:

```json
{
  "active": true,
  "ready": true,
  "done": false,
  "player_count": 5,
  "expected": 5,
  "frame": { "w": 1920, "h": 1080 },
  "camera": { "zoom": 128.0, "offset_x": -40, "offset_y": 18, "full_frame": false },
  "perspective": {
    "active": true,
    "size": 1.2,
    "angle": 12.0,
    "source": [[0.12, 0.42], [0.88, 0.42], [0.88, 0.95], [0.12, 0.95]],
    "base":   [[0.10, 0.41], [0.90, 0.41], [0.90, 0.96], [0.10, 0.96]],
    "target": [[0.09, 0.40], [0.91, 0.40], [0.91, 0.97], [0.09, 0.97]]
  },
  "views": [
    {
      "player": 0,
      "stable": true,
      "cards": 4,
      "crypt":  { "x": 0.16, "y": 0.70, "w": 0.18, "h": 0.16, "angle": 0, "quad": [] },
      "minion": { "x": 0.16, "y": 0.48, "w": 0.20, "h": 0.18, "angle": 0, "quad": [] }
    }
  ],
  "players": []
}
```

`views` contiene las propuestas del asistente. `players` contiene las zonas
guardadas cuando el asistente no está activo.

### 5.2 `/api/setup/perspective`

```text
/api/setup/perspective?corners=0.10,0.41;0.90,0.41;0.90,0.96;0.10,0.96&size=1.20&angle=12
```

- `corners`: 8 números separados por `,` y `;`; orden **TL, TR, BR, BL**.
  El navegador los envía con `encodeURIComponent`, el servidor decodifica la
  query y luego valida.
- `size`: 0,5–1,5; los valores no finitos se ignoran.
- `angle`: −45–45; los valores no finitos se ignoran.
- `reset=1`: vuelve a `source`, tamaño 1 y ángulo 0.

La petición se rechaza con `invalid_corners` si faltan números, no son finitos,
el área es demasiado pequeña o el cuadrilátero no es convexo.

### 5.3 Errores

| Error | Significado |
|---|---|
| `perspective_not_ready` | La detección aún no ha inicializado la perspectiva. |
| `invalid_corners` | Cuadrilátero inválido; no se ha modificado nada. |
| `not_ready` | Faltan zonas o el asistente no está activo. |
| `no_zones` | No hay vistas que guardar. |
| `camera_not_ready` | No se pudo calcular un encuadre válido. |
| `save_failed` | `obs_source_update` no pudo guardar la configuración. |
| `unknown_action` | Acción no reconocida. |

## 6. Dibujo y salida final

- Con `setup_mode=true`, el preview dibuja rejilla, guía, propuestas y zonas.
- Con `show_zones=true`, el preview dibuja las zonas guardadas aunque no haya
  asistente activo.
- Al aplicar, `show_zones` se fuerza a `false`, de modo que la salida final no
  contiene guías, rejilla ni rectángulos de zona.
- La inclinación 3D de la guía es una transformación CSS del canvas del
  navegador; no se aplica a la detección, a las zonas guardadas ni al overlay.

## 7. Comprobaciones recomendadas

Antes de dar una configuración por buena:

1. Probar con **4 y 5 jugadores**.
2. Guardar y reiniciar OBS: las zonas y el encuadre deben seguir ahí.
3. Comprobar que `auto_zones=false`, `setup_done=true` y `show_zones=false`.
4. Colocar una carta conocida en la zona de minions de cada jugador y verificar
   la asignación (incluido un jugador ousted).
5. Verificar que la salida final no muestra guías ni zonas.
6. Medir FPS y `outputSkippedFrames` con la mesa completa.
7. Revisar el log de OBS por si falta el índice de embeddings, las fuentes o
   Tesseract.

## 8. Problemas habituales

| Síntoma | Causa probable | Solución |
|---|---|---|
| `0/N` grupos | Cartas separadas, fundas distintas, poca luz o cámara inestable. | Junta las 4 cartas, usa la misma funda y estabiliza la cámara. |
| Faltan grupos | No se ven las 4 cartas de todos los jugadores a la vez. | Reencuadra o usa **Repetir detección**. |
| No se habilita guardar | `setup_ready=false` o faltan vistas. | Espera a que el contador llegue a N/N. |
| Perspectiva rara | Esquinas cruzadas o quad degenerado. | Pulsa **Restablecer** y arrastra de nuevo sin cruzar. |
| El encuadre falla | Zonas incompletas o frame aún no disponible. | Espera a que haya frame y vuelve a centrar. |
| Se ven zonas en la salida | `show_zones` activo. | Aplica el setup o desactiva **Mostrar zonas**. |
| El OCR no lee | Carta por debajo de 200 px de alto o falta Tesseract. | Acerca la cámara; el embedding sigue identificando. |
| El puerto 8080 no responde | Otro proceso ocupa el puerto. | Libera el puerto y reinicia el filtro. |
