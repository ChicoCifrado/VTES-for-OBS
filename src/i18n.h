#ifndef VTES_I18N_H
#define VTES_I18N_H

/*
 * Sistema de textos del plugin.
 *
 * El plugin tiene dos idiomas completos: espanol (nativo) e ingles. Los
 * textos viven en data/locale/{es-ES,en-US}.ini y aqui solo se elige cual
 * se usa: en la primera carga se pregunta al usuario con un dialogo de dos
 * botones (Espanol / English) y, despues, puede cambiarse desde las
 * propiedades del filtro.
 *
 * A diferencia de obs_module_text(), esta eleccion es independiente del
 * idioma de la interfaz de OBS.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Carga los textos y el idioma guardado (por defecto: espanol nativo). */
void tf_i18n_init(void);

/* 1 si el usuario todavia no ha elegido idioma (primera carga). */
int tf_i18n_first_run(void);

/* Dialogo de primera carga con dos botones (Espanol / English). */
void tf_i18n_prompt_if_needed(void);

/* Codigo del idioma actual: "es" o "en". */
const char *tf_i18n_code(void);

/* Cambia el idioma actual y lo persiste para las siguientes cargas. */
void tf_i18n_set(const char *code);

/* Texto de una clave en el idioma actual. Si la clave no existe en ninguno
 * de los dos idiomas, devuelve la propia clave. */
const char *tf_text(const char *key);

#ifdef __cplusplus
}
#endif

#endif /* VTES_I18N_H */
