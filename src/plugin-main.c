/*
Plugin Name
Copyright (C) <Year> <Developer> <Email Address>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#include <obs-module.h>
#include <plugin-support.h>
#include <curl/curl.h>
#include "i18n.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

MODULE_EXPORT const char *obs_module_description(void)
{
	return tf_text("DetectFilterPlugin");
}

extern struct obs_source_info detect_filter_obb_info;

bool obs_module_load(void)
{
	/* Textos del plugin: espanol nativo + ingles. En la primera carga se
	 * pregunta al usuario que idioma quiere (dos botones). */
	tf_i18n_init();
	tf_i18n_prompt_if_needed();

	curl_global_init(CURL_GLOBAL_DEFAULT);

	obs_register_source(&detect_filter_obb_info);
	obs_log(LOG_INFO, "plugin loaded successfully (version %s)", PLUGIN_VERSION);
	return true;
}

void obs_module_unload(void)
{
	curl_global_cleanup();
	obs_log(LOG_INFO, "plugin unloaded");
}
