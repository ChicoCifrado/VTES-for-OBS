#include "detect-filter-obb.h"

struct obs_source_info detect_filter_obb_info = {
	.id = "vtes-card-detector-obb",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = detect_filter_obb_getname,
	.create = detect_filter_obb_create,
	.destroy = detect_filter_obb_destroy,
	.get_defaults = detect_filter_obb_defaults,
	.get_properties = detect_filter_obb_properties,
	.update = detect_filter_obb_update,
	.activate = detect_filter_obb_activate,
	.deactivate = detect_filter_obb_deactivate,
	.video_tick = detect_filter_obb_video_tick,
	.video_render = detect_filter_obb_video_render,
};
