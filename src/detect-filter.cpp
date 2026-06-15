#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#undef WIN32_LEAN_AND_MEAN
#endif

#include "detect-filter.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
#endif // _WIN32

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <numeric>
#include <memory>
#include <exception>
#include <fstream>
#include <new>
#include <mutex>
#include <regex>
#include <thread>

#include <nlohmann/json.hpp>

#include <plugin-support.h>
#include "FilterData.h"
#include "consts.h"
#include "obs-utils/obs-utils.h"
#include "ort-model/utils.hpp"
#include "detect-filter-utils.h"
#include "yolov8/yolov8_onnxruntime.hpp"
#include "coco_names.hpp"
#include "base64-utils.h"

#define EXTERNAL_MODEL_SIZE "!!!EXTERNAL_MODEL!!!"
#define VTES_MODEL_SIZE "!!!VTES_MODEL!!!"

struct detect_filter : public filter_data {};

const char *detect_filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("Detect");
}

/**                   PROPERTIES                     */

void set_class_names_on_object_category(obs_property_t *object_category,
					std::vector<std::string> class_names)
{
	std::vector<std::pair<size_t, std::string>> indexed_classes;
	for (size_t i = 0; i < class_names.size(); ++i) {
		const std::string &class_name = class_names[i];
		// capitalize the first letter of the class name
		std::string class_name_cap = class_name;
		class_name_cap[0] = (char)std::toupper((int)class_name_cap[0]);
		indexed_classes.push_back({i, class_name_cap});
	}

	// sort the vector based on the class names
	std::sort(indexed_classes.begin(), indexed_classes.end(),
		  [](const std::pair<size_t, std::string> &a,
		     const std::pair<size_t, std::string> &b) { return a.second < b.second; });

	// clear the object category list
	obs_property_list_clear(object_category);

	// add the sorted classes to the property list
	obs_property_list_add_int(object_category, obs_module_text("All"), -1);

	// add the sorted classes to the property list
	for (const auto &indexed_class : indexed_classes) {
		obs_property_list_add_int(object_category, indexed_class.second.c_str(),
					  (int)indexed_class.first);
	}
}

void read_model_config_json_and_set_class_names(const char *model_file, obs_properties_t *props_,
						obs_data_t *settings, struct detect_filter *tf_)
{
	if (model_file == nullptr || model_file[0] == '\0' || strlen(model_file) == 0) {
		obs_log(LOG_ERROR, "Model file path is empty");
		return;
	}

	// read the '.json' file near the model file to find the class names
	std::string json_file = model_file;
	json_file.replace(json_file.find(".onnx"), 5, ".json");
	std::ifstream file(json_file);
	if (!file.is_open()) {
		obs_data_set_string(settings, "error", "JSON file not found");
		obs_log(LOG_ERROR, "JSON file not found: %s", json_file.c_str());
	} else {
		obs_data_set_string(settings, "error", "");
		// parse the JSON file
		nlohmann::json j;
		file >> j;
		if (j.contains("names")) {
			std::vector<std::string> labels = j["names"];
			set_class_names_on_object_category(
				obs_properties_get(props_, "object_category"), labels);
			tf_->classNames = labels;
		} else {
			obs_data_set_string(settings, "error",
					    "JSON file does not contain 'names' field");
			obs_log(LOG_ERROR, "JSON file does not contain 'names' field");
		}
	}
}

obs_properties_t *detect_filter_properties(void *data)
{
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	obs_properties_t *props = obs_properties_create();

	obs_properties_add_bool(props, "preview", obs_module_text("Preview"));

	obs_properties_add_float_slider(props, "threshold", obs_module_text("ConfThreshold"), 0.0,
					1.0, 0.025);

	obs_property_t *p_use_gpu =
		obs_properties_add_list(props, "useGPU", obs_module_text("InferenceDevice"),
				OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_use_gpu, obs_module_text("CPU"), USEGPU_CPU);
#if _WIN32
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPU (DirectML)"), USEGPU_DML);
#endif
#if defined(__linux__) && defined(__x86_64__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPU (TensorRT)"), USEGPU_TENSORRT);
#endif
#if defined(__APPLE__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("CoreML"), USEGPU_COREML);
#endif

	// add drop down option for model: VTES Model, External Model
	obs_property_t *model_size =
		obs_properties_add_list(props, "model_size", obs_module_text("ModelSize"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(model_size, obs_module_text("VTESModel"), VTES_MODEL_SIZE);
	obs_property_list_add_string(model_size, obs_module_text("ExternalModel"),
				     EXTERNAL_MODEL_SIZE);

	// add external model file path
	obs_properties_add_path(props, "external_model_file", obs_module_text("ModelPath"),
				OBS_PATH_FILE, "EdgeYOLO onnx files (*.onnx);;all files (*.*)",
				nullptr);

	// add callback to show/hide the external model file path
	obs_property_set_modified_callback2(
		model_size,
		[](void *data_, obs_properties_t *props_, obs_property_t *p, obs_data_t *settings) {
			UNUSED_PARAMETER(p);
			const char *val = obs_data_get_string(settings, "model_size");
			bool is_external = strcmp(val, EXTERNAL_MODEL_SIZE) == 0;
			obs_property_t *prop = obs_properties_get(props_, "external_model_file");
			obs_property_set_visible(prop, is_external);
			return true;
		},
		tf);

	// VTES: WebSocket server settings
	obs_properties_t *vtes_group_props = obs_properties_create();
	obs_property_t *vtes_group =
		obs_properties_add_group(props, "vtes_group", "VTES Card Server",
					 OBS_GROUP_CHECKABLE, vtes_group_props);

	obs_property_set_modified_callback(vtes_group, [](obs_properties_t *props_,
							  obs_property_t *, obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "vtes_group");
		for (auto prop_name : {"vtes_ws_host", "vtes_ws_port", "vtes_cooldown"}) {
			obs_property_t *prop = obs_properties_get(props_, prop_name);
			obs_property_set_visible(prop, enabled);
		}
		return true;
	});

	obs_properties_add_text(vtes_group_props, "vtes_ws_host", "WebSocket Host",
				OBS_TEXT_DEFAULT);
	obs_properties_add_int_slider(vtes_group_props, "vtes_ws_port", "WebSocket Port",
				      1000, 9999, 1);
	obs_properties_add_int_slider(vtes_group_props, "vtes_cooldown", "Cooldown (seconds)",
				      1, 30, 1);

	std::string basic_info =
		std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
	obs_properties_add_text(props, "info", basic_info.c_str(), OBS_TEXT_INFO);

	UNUSED_PARAMETER(data);
	return props;
}

void detect_filter_defaults(obs_data_t *settings)
{
#if _WIN32
	obs_data_set_default_string(settings, "useGPU", USEGPU_DML);
#else
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#endif
	obs_data_set_default_bool(settings, "preview", true);
	obs_data_set_default_double(settings, "threshold", 0.5);
	obs_data_set_default_string(settings, "model_size", VTES_MODEL_SIZE);

	// VTES defaults
	obs_data_set_default_bool(settings, "vtes_group", false);
	obs_data_set_default_string(settings, "vtes_ws_host", "127.0.0.1");
	obs_data_set_default_int(settings, "vtes_ws_port", 3998);
	obs_data_set_default_int(settings, "vtes_cooldown", 10);
}

void detect_filter_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "Detect filter update");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	tf->isDisabled = true;

	tf->preview = obs_data_get_bool(settings, "preview");
	tf->conf_threshold = (float)obs_data_get_double(settings, "threshold");
	tf->objectCategory = (int)obs_data_get_int(settings, "object_category");
	tf->maskingEnabled = obs_data_get_bool(settings, "masking_group");
	tf->maskingType = obs_data_get_string(settings, "masking_type");
	tf->maskingColor = (int)obs_data_get_int(settings, "masking_color");
	tf->maskingBlurRadius = (int)obs_data_get_int(settings, "masking_blur_radius");
	bool newTrackingEnabled = obs_data_get_bool(settings, "tracking_group");
	tf->zoomFactor = (float)obs_data_get_double(settings, "zoom_factor");
	tf->zoomSpeedFactor = (float)obs_data_get_double(settings, "zoom_speed_factor");
	tf->zoomObject = obs_data_get_string(settings, "zoom_object");
	tf->sortTracking = obs_data_get_bool(settings, "sort_tracking");
	size_t maxUnseenFrames = (size_t)obs_data_get_int(settings, "max_unseen_frames");
	if (tf->tracker.getMaxUnseenFrames() != maxUnseenFrames) {
		tf->tracker.setMaxUnseenFrames(maxUnseenFrames);
	}
	tf->showUnseenObjects = obs_data_get_bool(settings, "show_unseen_objects");
	tf->saveDetectionsPath = obs_data_get_string(settings, "save_detections_path");
	tf->crop_enabled = obs_data_get_bool(settings, "crop_group");
	tf->crop_left = (int)obs_data_get_int(settings, "crop_left");
	tf->crop_right = (int)obs_data_get_int(settings, "crop_right");
	tf->crop_top = (int)obs_data_get_int(settings, "crop_top");
	tf->crop_bottom = (int)obs_data_get_int(settings, "crop_bottom");

	// VTES settings
	bool vtesEnabled = obs_data_get_bool(settings, "vtes_group");
	std::string newWsHost = obs_data_get_string(settings, "vtes_ws_host");
	int newWsPort = (int)obs_data_get_int(settings, "vtes_ws_port");
	int newCooldown = (int)obs_data_get_int(settings, "vtes_cooldown");

	if (vtesEnabled) {
		tf->wsHost = newWsHost;
		tf->wsPort = newWsPort;
		tf->cooldownSeconds = newCooldown;

		if (!tf->wsClient.is_connected()) {
			if (tf->wsClient.connect(tf->wsHost, tf->wsPort)) {
				obs_log(LOG_INFO, "[VTES] WebSocket connected to %s:%d",
					tf->wsHost.c_str(), tf->wsPort);
			} else {
				obs_log(LOG_WARNING,
					"[VTES] WebSocket connection failed to %s:%d",
					tf->wsHost.c_str(), tf->wsPort);
			}
		}
	} else {
		if (tf->wsClient.is_connected()) {
			tf->wsClient.disconnect();
			obs_log(LOG_INFO, "[VTES] WebSocket disconnected");
		}
	}

	// check if tracking state has changed
	if (tf->trackingEnabled != newTrackingEnabled) {
		tf->trackingEnabled = newTrackingEnabled;
		obs_source_t *parent = obs_filter_get_parent(tf->source);
		if (!parent) {
			obs_log(LOG_ERROR, "Parent source not found");
			return;
		}
		if (tf->trackingEnabled) {
			obs_log(LOG_DEBUG, "Tracking enabled");
			// get the parent of the source
			// check if it has a crop/pad filter
			obs_source_t *crop_pad_filter =
				obs_source_get_filter_by_name(parent, "Detect Tracking");
			if (!crop_pad_filter) {
				// create a crop-pad filter
				crop_pad_filter = obs_source_create(
					"crop_filter", "Detect Tracking", nullptr, nullptr);
				// add a crop/pad filter to the source
				// set the parent of the crop/pad filter to the parent of the source
				obs_source_filter_add(parent, crop_pad_filter);
			}
			tf->trackingFilter = crop_pad_filter;
		} else {
			obs_log(LOG_DEBUG, "Tracking disabled");
			// remove the crop/pad filter
			obs_source_t *crop_pad_filter =
				obs_source_get_filter_by_name(parent, "Detect Tracking");
			if (crop_pad_filter) {
				obs_source_filter_remove(parent, crop_pad_filter);
			}
			tf->trackingFilter = nullptr;
		}
	}

	const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
	const uint32_t newNumThreads = (uint32_t)obs_data_get_int(settings, "numThreads");
	const std::string newModelSize = obs_data_get_string(settings, "model_size");

	bool reinitialize = false;
	if (tf->useGPU != newUseGpu || tf->numThreads != newNumThreads ||
	    tf->modelSize != newModelSize) {
		obs_log(LOG_INFO, "Reinitializing model");
		reinitialize = true;

		// lock modelMutex
		std::unique_lock<std::mutex> lock(tf->modelMutex);

		char *modelFilepath_rawPtr = nullptr;
		if (newModelSize == VTES_MODEL_SIZE) {
			modelFilepath_rawPtr =
				obs_module_file("models/vtes_card_yolov8n.onnx");
		} else if (newModelSize == EXTERNAL_MODEL_SIZE) {
			const char *external_model_file =
				obs_data_get_string(settings, "external_model_file");
			if (external_model_file == nullptr || external_model_file[0] == '\0' ||
			    strlen(external_model_file) == 0) {
				obs_log(LOG_ERROR, "External model file path is empty");
				tf->isDisabled = true;
				return;
			}
			modelFilepath_rawPtr = bstrdup(external_model_file);
		} else {
			obs_log(LOG_ERROR, "Invalid model size: %s", newModelSize.c_str());
			tf->isDisabled = true;
			return;
		}

		if (modelFilepath_rawPtr == nullptr) {
			obs_log(LOG_ERROR, "Unable to get model filename from plugin.");
			tf->isDisabled = true;
			return;
		}

#if _WIN32
		int outLength = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr,
						    -1, nullptr, 0);
		tf->modelFilepath = std::wstring(outLength, L'\0');
		MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr, -1,
				    tf->modelFilepath.data(), outLength);
#else
		tf->modelFilepath = std::string(modelFilepath_rawPtr);
#endif
		bfree(modelFilepath_rawPtr);

		// Re-initialize model if it's not already the selected one or switching inference device
		tf->useGPU = newUseGpu;
		tf->numThreads = newNumThreads;
		tf->modelSize = newModelSize;

		// parameters
		int onnxruntime_device_id_ = 0;
		bool onnxruntime_use_parallel_ = true;
		float nms_th_ = 0.45f;
		int num_classes_ = (int)edgeyolo_cpp::COCO_CLASSES.size();
		tf->classNames = edgeyolo_cpp::COCO_CLASSES;

		// If this is an external model - look for the config JSON file
		if (tf->modelSize == EXTERNAL_MODEL_SIZE) {
#ifdef _WIN32
			std::wstring labelsFilepath = tf->modelFilepath;
			labelsFilepath.replace(labelsFilepath.find(L".onnx"), 5, L".json");
#else
			std::string labelsFilepath = tf->modelFilepath;
			labelsFilepath.replace(labelsFilepath.find(".onnx"), 5, ".json");
#endif
			std::ifstream labelsFile(labelsFilepath);
			if (labelsFile.is_open()) {
				// Parse the JSON file
				nlohmann::json j;
				labelsFile >> j;
				if (j.contains("names")) {
					std::vector<std::string> labels = j["names"];
					num_classes_ = (int)labels.size();
					tf->classNames = labels;
				} else {
					obs_log(LOG_ERROR,
						"JSON file does not contain 'names' field");
					tf->isDisabled = true;
					tf->onnxruntimemodel.reset();
					return;
				}
			} else {
				obs_log(LOG_ERROR, "Failed to open JSON file: %s",
					labelsFilepath.c_str());
				tf->isDisabled = true;
				tf->onnxruntimemodel.reset();
				return;
			}
		}

		// Load model
		try {
			if (tf->onnxruntimemodel) {
				tf->onnxruntimemodel.reset();
			}
			tf->onnxruntimemodel =
				std::make_unique<yolov8_cpp::YOLOv8>(
					tf->modelFilepath, tf->numThreads, num_classes_,
					tf->numThreads, tf->useGPU, onnxruntime_device_id_,
					onnxruntime_use_parallel_, nms_th_,
					tf->conf_threshold);
			// clear error message
			obs_data_set_string(settings, "error", "");
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "Failed to load model: %s", e.what());
			// disable filter
			tf->isDisabled = true;
			tf->onnxruntimemodel.reset();
			return;
		}
	}

	// update threshold on YOLOv8
	if (tf->onnxruntimemodel) {
		tf->onnxruntimemodel->setBBoxConfThresh(tf->conf_threshold);
	}

	if (reinitialize) {
		// Log the currently selected options
		obs_log(LOG_INFO, "Detect Filter Options:");
		// name of the source that the filter is attached to
		obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
		obs_log(LOG_INFO, "  Inference Device: %s", tf->useGPU.c_str());
		obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
		obs_log(LOG_INFO, "  Model Size: %s", tf->modelSize.c_str());
		obs_log(LOG_INFO, "  Preview: %s", tf->preview ? "true" : "false");
		obs_log(LOG_INFO, "  Threshold: %.2f", tf->conf_threshold);
		obs_log(LOG_INFO, "  Object Category: %s",
			obs_data_get_string(settings, "object_category"));
		obs_log(LOG_INFO, "  Masking Enabled: %s",
			obs_data_get_bool(settings, "masking_group") ? "true" : "false");
		obs_log(LOG_INFO, "  Masking Type: %s",
			obs_data_get_string(settings, "masking_type"));
		obs_log(LOG_INFO, "  Masking Color: %s",
			obs_data_get_string(settings, "masking_color"));
		obs_log(LOG_INFO, "  Masking Blur Radius: %d",
			obs_data_get_int(settings, "masking_blur_radius"));
		obs_log(LOG_INFO, "  Tracking Enabled: %s",
			obs_data_get_bool(settings, "tracking_group") ? "true" : "false");
		obs_log(LOG_INFO, "  Zoom Factor: %.2f",
			obs_data_get_double(settings, "zoom_factor"));
		obs_log(LOG_INFO, "  Zoom Object: %s",
			obs_data_get_string(settings, "zoom_object"));
		obs_log(LOG_INFO, "  Disabled: %s", tf->isDisabled ? "true" : "false");
#ifdef _WIN32
		obs_log(LOG_INFO, "  Model file path: %ls", tf->modelFilepath.c_str());
#else
		obs_log(LOG_INFO, "  Model file path: %s", tf->modelFilepath.c_str());
#endif
	}

	// enable
	tf->isDisabled = false;
}

void detect_filter_activate(void *data)
{
	obs_log(LOG_INFO, "Detect filter activated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = false;
}

void detect_filter_deactivate(void *data)
{
	obs_log(LOG_INFO, "Detect filter deactivated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = true;
}

/**                   FILTER CORE                     */

void *detect_filter_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "Detect filter created");
	void *data = bmalloc(sizeof(struct detect_filter));
	struct detect_filter *tf = new (data) detect_filter();

	tf->source = source;
	tf->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	tf->lastDetectedObjectId = -1;

	// VTES initialization
	tf->wsHost = "127.0.0.1";
	tf->wsPort = 3998;
	tf->cooldownSeconds = 10;
	tf->lastSendTime = std::chrono::steady_clock::now();
	tf->wsSendInProgress = false;

	char *kawaseBlurEffectPath = obs_module_file(KAWASE_BLUR_EFFECT_PATH);
	if (!kawaseBlurEffectPath) {
		obs_log(LOG_ERROR, "Failed to get Kawase Blur effect path");
		tf->isDisabled = true;
		return tf;
	}
	char *maskingEffectPath = obs_module_file(MASKING_EFFECT_PATH);
	if (!maskingEffectPath) {
		obs_log(LOG_ERROR, "Failed to get masking effect path");
		tf->isDisabled = true;
		bfree(kawaseBlurEffectPath);
		return tf;
	}

	obs_enter_graphics();
	gs_effect_destroy(tf->kawaseBlurEffect);
	tf->kawaseBlurEffect = nullptr;
	char *error = nullptr;
	tf->kawaseBlurEffect = gs_effect_create_from_file(kawaseBlurEffectPath, &error);
	bfree(kawaseBlurEffectPath);
	if (!tf->kawaseBlurEffect || error) {
		obs_log(LOG_ERROR, "Failed to load Kawase Blur effect: %s", error);
	}
	gs_effect_destroy(tf->maskingEffect);
	tf->maskingEffect = nullptr;
	tf->maskingEffect = gs_effect_create_from_file(maskingEffectPath, &error);
	bfree(maskingEffectPath);
	if (!tf->maskingEffect || error) {
		obs_log(LOG_ERROR, "Failed to load masking effect: %s", error);
	}
	obs_leave_graphics();

	detect_filter_update(tf, settings);

	return tf;
}

void detect_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "Detect filter destroyed");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf) {
		tf->isDisabled = true;

		// VTES: disconnect WebSocket
		if (tf->wsClient.is_connected()) {
			tf->wsClient.disconnect();
		}

		obs_enter_graphics();
		gs_texrender_destroy(tf->texrender);
		if (tf->stagesurface) {
			gs_stagesurface_destroy(tf->stagesurface);
		}
		gs_effect_destroy(tf->kawaseBlurEffect);
		gs_effect_destroy(tf->maskingEffect);
		obs_leave_graphics();
		tf->~detect_filter();
		bfree(tf);
	}
}

void detect_filter_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf->isDisabled || !tf->onnxruntimemodel) {
		return;
	}

	if (!obs_source_enabled(tf->source)) {
		return;
	}

	cv::Mat imageBGRA;
	{
		std::unique_lock<std::mutex> lock(tf->inputBGRALock, std::try_to_lock);
		if (!lock.owns_lock()) {
			// No data to process
			return;
		}
		if (tf->inputBGRA.empty()) {
			// No data to process
			return;
		}
		imageBGRA = tf->inputBGRA.clone();
	}

	cv::Mat inferenceFrame;

	cv::Rect cropRect(0, 0, imageBGRA.cols, imageBGRA.rows);
	if (tf->crop_enabled) {
		cropRect = cv::Rect(tf->crop_left, tf->crop_top,
				    imageBGRA.cols - tf->crop_left - tf->crop_right,
				    imageBGRA.rows - tf->crop_top - tf->crop_bottom);
		cv::cvtColor(imageBGRA(cropRect), inferenceFrame, cv::COLOR_BGRA2BGR);
	} else {
		cv::cvtColor(imageBGRA, inferenceFrame, cv::COLOR_BGRA2BGR);
	}

	std::vector<Object> objects;

	try {
		std::unique_lock<std::mutex> lock(tf->modelMutex);
		objects = tf->onnxruntimemodel->inference(inferenceFrame);
	} catch (const Ort::Exception &e) {
		obs_log(LOG_ERROR, "ONNXRuntime Exception: %s", e.what());
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "%s", e.what());
	}

	if (tf->crop_enabled) {
		// translate the detected objects to the original frame
		for (Object &obj : objects) {
			obj.rect.x += (float)cropRect.x;
			obj.rect.y += (float)cropRect.y;
		}
	}

	// update the detected object text input
	if (objects.size() > 0) {
		if (tf->lastDetectedObjectId != objects[0].label) {
			tf->lastDetectedObjectId = objects[0].label;
			// get source settings
			obs_data_t *source_settings = obs_source_get_settings(tf->source);
			obs_data_set_string(source_settings, "detected_object",
					    tf->classNames[objects[0].label].c_str());
			// release the source settings
			obs_data_release(source_settings);
		}
	} else {
		if (tf->lastDetectedObjectId != -1) {
			tf->lastDetectedObjectId = -1;
			// get source settings
			obs_data_t *source_settings = obs_source_get_settings(tf->source);
			obs_data_set_string(source_settings, "detected_object", "");
			// release the source settings
			obs_data_release(source_settings);
		}
	}

	if (tf->objectCategory != -1) {
		std::vector<Object> filtered_objects;
		for (const Object &obj : objects) {
			if (obj.label == tf->objectCategory) {
				filtered_objects.push_back(obj);
			}
		}
		objects = filtered_objects;
	}

	if (tf->sortTracking) {
		objects = tf->tracker.update(objects);
	}

	if (!tf->showUnseenObjects) {
		objects.erase(
			std::remove_if(objects.begin(), objects.end(),
				       [](const Object &obj) { return obj.unseenFrames > 0; }),
			objects.end());
	}

	// VTES: Send card crop to Node.js server via WebSocket
	if (tf->wsClient.is_connected() && objects.size() > 0) {
		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
			now - tf->lastSendTime).count();

		if (elapsed >= tf->cooldownSeconds) {
			// Get the first detected card (highest confidence)
			const Object &card = objects[0];
			cv::Rect cardRect = cv::Rect(
				(int)card.rect.x, (int)card.rect.y,
				(int)card.rect.width, (int)card.rect.height);

			// Clamp to image bounds
			cardRect &= cv::Rect(0, 0, imageBGRA.cols, imageBGRA.rows);

			if (cardRect.width >= 60 && cardRect.height >= 60) {
				cv::Mat cardCrop = imageBGRA(cardRect).clone();

				// Convert BGRA to BGR for JPEG encoding
				cv::Mat cardCropBGR;
				cv::cvtColor(cardCrop, cardCropBGR, cv::COLOR_BGRA2BGR);

				// Extract 3 zones: top (30%), center (40%), side (30% of width)
				int h = cardCropBGR.rows;
				int w = cardCropBGR.cols;
				int topH = (int)(h * 0.30);
				int centerH = (int)(h * 0.40);
				int sideW = (int)(w * 0.30);

				if (topH < 5 || centerH < 5 || sideW < 5) {
					obs_log(LOG_WARNING, "[VTES] Zone too small for encoding (w=%d h=%d)", w, h);
				} else {
					cv::Mat zoneTop = cardCropBGR(cv::Rect(0, 0, w, topH)).clone();
					cv::Mat zoneCenter = cardCropBGR(cv::Rect(0, topH, w, centerH)).clone();
					cv::Mat zoneSide = cardCropBGR(cv::Rect(w - sideW, 0, sideW, h)).clone();

					// Encode zones as JPEG with try-catch
					try {
						std::vector<uchar> bufTop, bufCenter, bufSide;
						std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 85};
						cv::imencode(".jpg", zoneTop, bufTop, params);
						cv::imencode(".jpg", zoneCenter, bufCenter, params);
						cv::imencode(".jpg", zoneSide, bufSide, params);

						// Build JSON message with base64-encoded zones
						nlohmann::json msg;
						msg["type"] = "card_crop";
						msg["zones"]["top"] = base64_encode(bufTop);
						msg["zones"]["center"] = base64_encode(bufCenter);
						msg["zones"]["side"] = base64_encode(bufSide);
						msg["confidence"] = card.prob;
						msg["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
							std::chrono::system_clock::now().time_since_epoch()).count();

						std::string jsonStr = msg.dump();
						if (tf->wsClient.send_text(jsonStr)) {
							tf->lastSendTime = now;
							obs_log(LOG_INFO, "[VTES] Sent card crop (conf: %.2f)", card.prob);
						} else {
							obs_log(LOG_WARNING, "[VTES] Failed to send card crop");
							tf->wsClient.disconnect();
						}
					} catch (const cv::Exception &e) {
						obs_log(LOG_ERROR, "[VTES] imencode failed: %s", e.what());
					}
				}
			}
		}
	}

	if (!tf->saveDetectionsPath.empty()) {
		std::ofstream detectionsFile(tf->saveDetectionsPath);
		if (detectionsFile.is_open()) {
			nlohmann::json j;
			for (const Object &obj : objects) {
				nlohmann::json obj_json;
				obj_json["label"] = obj.label;
				obj_json["confidence"] = obj.prob;
				obj_json["rect"] = {{"x", obj.rect.x},
						    {"y", obj.rect.y},
						    {"width", obj.rect.width},
						    {"height", obj.rect.height}};
				obj_json["id"] = obj.id;
				j.push_back(obj_json);
			}
			detectionsFile << j.dump(4);
			detectionsFile.close();
		} else {
			obs_log(LOG_ERROR, "Failed to open file for writing detections: %s",
				tf->saveDetectionsPath.c_str());
		}
	}

	if (tf->preview || tf->maskingEnabled) {
		cv::Mat frame;
		cv::cvtColor(imageBGRA, frame, cv::COLOR_BGRA2BGR);

		if (tf->preview && tf->crop_enabled) {
			// draw the crop rectangle on the frame in a dashed line
			drawDashedRectangle(frame, cropRect, cv::Scalar(0, 255, 0), 5, 8, 15);
		}
		if (tf->preview && objects.size() > 0) {
			draw_objects(frame, objects, tf->classNames);
		}
		if (tf->maskingEnabled) {
			cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
			for (const Object &obj : objects) {
				cv::rectangle(mask, obj.rect, cv::Scalar(255), -1);
			}
			std::lock_guard<std::mutex> lock(tf->outputLock);
			mask.copyTo(tf->outputMask);
		}

		std::lock_guard<std::mutex> lock(tf->outputLock);
		cv::cvtColor(frame, tf->outputPreviewBGRA, cv::COLOR_BGR2BGRA);
	}

	if (tf->trackingEnabled && tf->trackingFilter) {
		const int width = imageBGRA.cols;
		const int height = imageBGRA.rows;

		cv::Rect2f boundingBox = cv::Rect2f(0, 0, (float)width, (float)height);
		// get location of the objects
		if (tf->zoomObject == "single") {
			if (objects.size() > 0) {
				boundingBox = objects[0].rect;
			}
		} else {
			// get the bounding box of all objects
			if (objects.size() > 0) {
				boundingBox = objects[0].rect;
				for (const Object &obj : objects) {
					boundingBox |= obj.rect;
				}
			}
		}
		bool lostTracking = objects.size() == 0;
		// the zooming box should maintain the aspect ratio of the image
		// with the tf->zoomFactor controlling the effective buffer around the bounding box
		// the bounding box is the center of the zooming box
		float frameAspectRatio = (float)width / (float)height;
		// calculate an aspect ratio box around the object using its height
		float boxHeight = boundingBox.height;
		// calculate the zooming box size
		float dh = (float)height - boxHeight;
		float buffer = dh * (1.0f - tf->zoomFactor);
		float zh = boxHeight + buffer;
		float zw = zh * frameAspectRatio;
		// calculate the top left corner of the zooming box
		float zx = boundingBox.x - (zw - boundingBox.width) / 2.0f;
		float zy = boundingBox.y - (zh - boundingBox.height) / 2.0f;

		if (tf->trackingRect.width == 0) {
			// initialize the trackingRect
			tf->trackingRect = cv::Rect2f(zx, zy, zw, zh);
		} else {
			// interpolate the zooming box to tf->trackingRect
			float factor = tf->zoomSpeedFactor * (lostTracking ? 0.2f : 1.0f);
			tf->trackingRect.x =
				tf->trackingRect.x + factor * (zx - tf->trackingRect.x);
			tf->trackingRect.y =
				tf->trackingRect.y + factor * (zy - tf->trackingRect.y);
			tf->trackingRect.width =
				tf->trackingRect.width + factor * (zw - tf->trackingRect.width);
			tf->trackingRect.height =
				tf->trackingRect.height + factor * (zh - tf->trackingRect.height);
		}

		// get the settings of the crop/pad filter
		obs_data_t *crop_pad_settings = obs_source_get_settings(tf->trackingFilter);
		obs_data_set_int(crop_pad_settings, "left", (int)tf->trackingRect.x);
		obs_data_set_int(crop_pad_settings, "top", (int)tf->trackingRect.y);
		// right = image width - (zx + zw)
		obs_data_set_int(
			crop_pad_settings, "right",
			(int)((float)width - (tf->trackingRect.x + tf->trackingRect.width)));
		// bottom = image height - (zy + zh)
		obs_data_set_int(
			crop_pad_settings, "bottom",
			(int)((float)height - (tf->trackingRect.y + tf->trackingRect.height)));
		// apply the settings
		obs_source_update(tf->trackingFilter, crop_pad_settings);
		obs_data_release(crop_pad_settings);
	}
}

void detect_filter_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf->isDisabled || !tf->onnxruntimemodel) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	uint32_t width, height;
	if (!getRGBAFromStageSurface(tf, width, height)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	// if preview is enabled, render the image
	if (tf->preview || tf->maskingEnabled) {
		cv::Mat outputBGRA, outputMask;
		{
			// lock the outputLock mutex
			std::lock_guard<std::mutex> lock(tf->outputLock);
			if (tf->outputPreviewBGRA.empty()) {
				obs_log(LOG_ERROR, "Preview image is empty");
				if (tf->source) {
					obs_source_skip_video_filter(tf->source);
				}
				return;
			}
			if ((uint32_t)tf->outputPreviewBGRA.cols != width ||
			    (uint32_t)tf->outputPreviewBGRA.rows != height) {
				if (tf->source) {
					obs_source_skip_video_filter(tf->source);
				}
				return;
			}
			outputBGRA = tf->outputPreviewBGRA.clone();
			outputMask = tf->outputMask.clone();
		}

		gs_texture_t *tex = gs_texture_create(width, height, GS_BGRA, 1,
						      (const uint8_t **)&outputBGRA.data, 0);
		gs_texture_t *maskTexture = nullptr;
		std::string technique_name = "Draw";
		gs_eparam_t *imageParam = gs_effect_get_param_by_name(tf->maskingEffect, "image");
		gs_eparam_t *maskParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "focalmask");
		gs_eparam_t *maskColorParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "color");

		if (tf->maskingEnabled) {
			maskTexture = gs_texture_create(width, height, GS_R8, 1,
							(const uint8_t **)&outputMask.data, 0);
			gs_effect_set_texture(maskParam, maskTexture);
			if (tf->maskingType == "output_mask") {
				technique_name = "DrawMask";
			} else if (tf->maskingType == "blur") {
				gs_texture_destroy(tex);
				tex = blur_image(tf, width, height, maskTexture);
			} else if (tf->maskingType == "transparent") {
				technique_name = "DrawSolidColor";
				gs_effect_set_color(maskColorParam, 0);
			} else if (tf->maskingType == "solid_color") {
				technique_name = "DrawSolidColor";
				gs_effect_set_color(maskColorParam, tf->maskingColor);
			}
		}

		gs_effect_set_texture(imageParam, tex);

		while (gs_effect_loop(tf->maskingEffect, technique_name.c_str())) {
			gs_draw_sprite(tex, 0, 0, 0);
		}

		gs_texture_destroy(tex);
		gs_texture_destroy(maskTexture);
	} else {
		obs_source_skip_video_filter(tf->source);
	}
	return;
}
