# Changelog

Todos los cambios relevantes del proyecto. El formato sigue
[Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el versionado
[SemVer](https://semver.org/lang/es/).

## [0.1.8] — En desarrollo

### Añadido

- Asistente de configuración inicial `/setup` para 4 o 5 jugadores:
  detección de grupos de 4 cartas de cripta, propuesta de zonas, perspectiva
  global, encuadre de cámara y guardado de quads normalizados.
- Zonas automáticas por fila de cripta e influencia: la cripta se detecta por
  el grupo de cuatro cartas boca abajo y la zona de minions se deriva de la
  geometría de la mesa.
- Costes de biblioteca y recursos de minions en el estado de partida.
- Página `/game` con pool, recursos, cartas giradas y eventos de influencia.

### Mejorado

- Validación de la perspectiva: se rechazan cuadriláteros no finitos,
  degenerados o con esquinas cruzadas, tanto al recibirlos por la API como al
  cargar quads guardados.
- El modo configuración ya no usa las zonas antiguas para asignar cartas ni
  disparar influencia.
- Al guardar el setup se fuerza `show_zones=false` para que la salida final no
  muestre guías ni zonas.
- Generación segura de los textos I18N embebidos en las páginas web.
- Documentación reescrita: pipeline embedding-first, asistente, API, modelo de
  datos y diagnóstico.

### Corregido

- Default duplicado de `temporal_smoothing` en `detect_filter_obb_defaults`.
- Fugas de estado al repetir la detección o cancelar el asistente.

## [0.1.7] — 2026-10-22

### Añadido

- Release **Producción Vampírica**.
- Manual de usuario en español e inglés (v0.1.7).
- Build con TensorRT anidado, shim de DirectML y CUDA 13.

### Mejorado

- Manuales comprimidos a JPEG para reducir el tamaño de distribución.

## Versiones anteriores

Consulta los tags `v0.1.0`–`v0.1.6` del repositorio.
