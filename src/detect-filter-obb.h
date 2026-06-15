#ifndef DETECT_FILTER_OBB_H
#define DETECT_FILTER_OBB_H

#include <obs-module.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *detect_filter_obb_getname(void *unused);
void *detect_filter_obb_create(obs_data_t *settings, obs_source_t *source);
void detect_filter_obb_destroy(void *data);
void detect_filter_obb_update(void *data, obs_data_t *settings);
void detect_filter_obb_activate(void *data);
void detect_filter_obb_deactivate(void *data);
void detect_filter_obb_video_tick(void *data, float seconds);
void detect_filter_obb_video_render(void *data, gs_effect_t *_effect);
obs_properties_t *detect_filter_obb_properties(void *data);
void detect_filter_obb_defaults(obs_data_t *settings);

#ifdef __cplusplus
}
#endif

#endif // DETECT_FILTER_OBB_H
