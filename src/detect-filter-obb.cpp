#include "detect-filter-obb.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#undef WIN32_LEAN_AND_MEAN
#include <wchar.h>
#endif

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/geometry.hpp>

static constexpr float kPI = 3.14159265358979323846f;

#include <numeric>
#include <memory>
#include <exception>
#include <fstream>
#include <new>
#include <mutex>
#include <regex>
#include <thread>
#include <deque>
#include <unordered_map>
#include <bitset>
#include <cmath>

#include <nlohmann/json.hpp>

#include <plugin-support.h>
#include "FilterData.h"
#include "consts.h"
#include "obs-utils/obs-utils.h"
#include "ort-model/utils.hpp"
#include "detect-filter-utils.h"
#include "yolov8/yolov8_obb_onnxruntime.hpp"
#include "detection/contour_detector.hpp"
#include "classifier/vtes_card_classifier.hpp"
#include "base64-utils.h"

#define VTES_OBB_MODEL "!!!VTES_OBB_MODEL!!!"

static const char *DETECT_MODE_CONTOUR = "contour";
static const char *DETECT_MODE_ONNX = "onnx";

struct detect_filter_obb : public filter_data {};

// ─── Card region extraction ─────────────────────────────────────────────────

// Perspective-correct a region defined by 4 corners into a top-down view,
// then enhance the crop for better embedding matching.
static cv::Mat extractCardRegion(const cv::Mat& frame,
                                   const cv::Point2f corners[4])
{
	// Compute card dimensions from corners
	float top = std::min({corners[0].y, corners[1].y, corners[2].y, corners[3].y});
	float bottom = std::max({corners[0].y, corners[1].y, corners[2].y, corners[3].y});
	float left = std::min({corners[0].x, corners[1].x, corners[2].x, corners[3].x});
	float right = std::max({corners[0].x, corners[1].x, corners[2].x, corners[3].x});
	int raw_w = (int)(right - left + 1);
	int raw_h = (int)(bottom - top + 1);
	if (raw_w < 16 || raw_h < 16) return cv::Mat();

	// Sort corners into TL, TR, BR, BL using sum/diff method
	cv::Point2f src_pts[4];
	{
		std::vector<cv::Point2f> pts(corners, corners + 4);
		std::vector<float> sums, diffs;
		for (auto& p : pts) { sums.push_back(p.x + p.y); diffs.push_back(p.x - p.y); }
		auto min_sum = std::min_element(sums.begin(), sums.end());
		auto max_sum = std::max_element(sums.begin(), sums.end());
		auto min_diff = std::min_element(diffs.begin(), diffs.end());
		auto max_diff = std::max_element(diffs.begin(), diffs.end());
		int tl = (int)(min_sum - sums.begin());
		int br = (int)(max_sum - sums.begin());
		int tr = (int)(max_diff - diffs.begin());
		int bl = (int)(min_diff - diffs.begin());
		src_pts[0] = pts[tl]; src_pts[1] = pts[tr];
		src_pts[2] = pts[br]; src_pts[3] = pts[bl];
	}

	// Output size: preserve VTES card aspect (63:88) with minimum quality
	constexpr double vtes_aspect = 88.0 / 63.0; // h/w
	int out_w, out_h;
	if ((double)raw_h / raw_w > vtes_aspect) {
		out_h = std::max(raw_h, 128);
		out_w = (int)(out_h / vtes_aspect + 0.5f);
	} else {
		out_w = std::max(raw_w, 96);
		out_h = (int)(out_w * vtes_aspect + 0.5f);
	}

	cv::Point2f dst_pts[4] = {
		{0, 0}, {(float)out_w - 1, 0}, {(float)out_w - 1, (float)out_h - 1}, {0, (float)out_h - 1}
	};

	cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
	cv::Mat warped;
	cv::warpPerspective(frame, warped, M, cv::Size(out_w, out_h));

	// Enhance for embedding — subtle CLAHE on L channel for better contrast
	cv::Mat enhanced;
	if (warped.channels() == 3) {
		cv::Mat lab;
		cv::cvtColor(warped, lab, cv::COLOR_BGR2Lab);
		std::vector<cv::Mat> lab_channels;
		cv::split(lab, lab_channels);
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.5, cv::Size(4, 4));
		clahe->apply(lab_channels[0], lab_channels[0]);
		cv::merge(lab_channels, lab);
		cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);
	} else {
		enhanced = warped.clone();
	}

	return enhanced;
}

// ─── Vision-based type classifier inference ─────────────────────────────
// Uses vtes_type_classifier.onnx (99.9% accurate, 14 VTES types)

static void loadVisionTypeClassifier(struct filter_data *tf)
{
	char *onnxPath = obs_module_file("vtes_type_classifier.onnx");
	char *labelsPath = obs_module_file("type_labels.json");
	if (!onnxPath || !labelsPath) {
		obs_log(LOG_WARNING, "[TypeClassifier] Missing model or labels file");
		bfree(onnxPath); bfree(labelsPath);
		return;
	}

	// Load ONNX (Windows needs wchar_t* for UTF-16 paths)
	try {
		Ort::SessionOptions opts;
		opts.SetIntraOpNumThreads(1);
		opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
#ifdef _WIN32
		int wlen = MultiByteToWideChar(CP_UTF8, 0, onnxPath, -1, nullptr, 0);
		std::wstring wpath(wlen, L'\0');
		MultiByteToWideChar(CP_UTF8, 0, onnxPath, -1, wpath.data(), wlen);
		tf->type_classifier_session = std::make_unique<Ort::Session>(
			tf->type_classifier_env, wpath.c_str(), opts);
#else
		tf->type_classifier_session = std::make_unique<Ort::Session>(
			tf->type_classifier_env, onnxPath, opts);
#endif
		obs_log(LOG_INFO, "[TypeClassifier] Loaded ONNX: %s", onnxPath);
	} catch (const Ort::Exception& e) {
		obs_log(LOG_ERROR, "[TypeClassifier] Failed to load ONNX: %s", e.what());
		bfree(onnxPath); bfree(labelsPath);
		return;
	}

	// Load type labels (14 VTES types)
	std::ifstream f(labelsPath);
	if (f.is_open()) {
		try {
			nlohmann::json j;
			f >> j;
			tf->type_labels = j.get<std::vector<std::string>>();
			obs_log(LOG_INFO, "[TypeClassifier] %zu type labels loaded", tf->type_labels.size());
		} catch (const std::exception& e) {
			obs_log(LOG_ERROR, "[TypeClassifier] Failed to parse labels: %s", e.what());
		}
	} else {
		obs_log(LOG_ERROR, "[TypeClassifier] Cannot open labels file: %s", labelsPath);
	}

	bfree(onnxPath);
	bfree(labelsPath);
}

static std::string runVisionTypeClassifier(struct filter_data *tf, const cv::Mat& card_bgr)
{
	if (!tf->type_classifier_session || tf->type_labels.empty() || card_bgr.empty())
		return "";

	// Preprocess: resize to 224x224, ImageNet normalization
	cv::Mat resized, float_img;
	cv::resize(card_bgr, resized, cv::Size(224, 224));
	resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

	static const float mean[] = {0.485f, 0.456f, 0.406f};
	static const float stdv[] = {0.229f, 0.224f, 0.225f};

	std::vector<float> input_tensor(1 * 3 * 224 * 224);
	for (int y = 0; y < 224; y++) {
		for (int x = 0; x < 224; x++) {
			cv::Vec3f p = float_img.at<cv::Vec3f>(y, x);
			input_tensor[0 * 224 * 224 + y * 224 + x] = (p[2] - mean[0]) / stdv[0]; // R
			input_tensor[1 * 224 * 224 + y * 224 + x] = (p[1] - mean[1]) / stdv[1]; // G
			input_tensor[2 * 224 * 224 + y * 224 + x] = (p[0] - mean[2]) / stdv[2]; // B
		}
	}

	try {
		Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		std::vector<int64_t> shape = {1, 3, 224, 224};
		Ort::Value input_val = Ort::Value::CreateTensor<float>(
			mem, input_tensor.data(), input_tensor.size(), shape.data(), shape.size());

		const char* in_name = "input";
		const char* out_name = "type_logits";

		std::vector<Ort::Value> output = tf->type_classifier_session->Run(
			Ort::RunOptions{nullptr}, &in_name, &input_val, 1, &out_name, 1);

		if (output.empty() || !output[0].IsTensor()) return "";

		float* logits = output[0].GetTensorMutableData<float>();
		auto shape_info = output[0].GetTensorTypeAndShapeInfo();
		size_t num_classes = shape_info.GetElementCount();

		int best = 0;
		for (size_t i = 1; i < num_classes && i < tf->type_labels.size(); i++) {
			if (logits[i] > logits[best]) best = (int)i;
		}

		return tf->type_labels[best];
	} catch (const Ort::Exception& e) {
		obs_log(LOG_ERROR, "[TypeClassifier] Inference error: %s", e.what());
		return "";
	}
}

// ─── Apply card types to embedder (call after embedder loads) ───────────

static void applyCardTypesToEmbedder(struct filter_data *tf)
{
	if (!tf->vtes_db.is_empty() && tf->embedder.is_loaded()) {
		auto type_map = tf->vtes_db.build_type_map();
		if (!type_map.empty()) {
			tf->embedder.set_card_types(type_map);
			obs_log(LOG_INFO, "Applied card types to embedder (%zu entries)", type_map.size());
		}
	}
}

// ─── Load per-type embedding models (one ONNX + embedding index per VTES type) ─

static void loadPerTypeEmbedders(struct filter_data *tf)
{
	// Load manifest
	char *manifest_path = obs_module_file("per_type_manifest.json");
	if (!manifest_path) {
		obs_log(LOG_WARNING, "[PerType] Missing manifest file");
		return;
	}

	std::ifstream f(manifest_path);
	if (!f.is_open()) {
		obs_log(LOG_WARNING, "[PerType] Cannot open manifest: %s", manifest_path);
		bfree(manifest_path);
		return;
	}

	nlohmann::json manifest;
	try {
		f >> manifest;
	} catch (const std::exception& e) {
		obs_log(LOG_ERROR, "[PerType] Failed to parse manifest: %s", e.what());
		bfree(manifest_path);
		return;
	}
	bfree(manifest_path);

	int loaded = 0;
	for (auto it = manifest.begin(); it != manifest.end(); ++it) {
		const std::string& card_type = it.key();
		const auto& entry = it.value();

		std::string onnx_name = entry.value("onnx", "");
		std::string index_name = entry.value("index", "");
		std::string meta_name = entry.value("meta", "");

		if (onnx_name.empty() || index_name.empty() || meta_name.empty()) {
			obs_log(LOG_WARNING, "[PerType] Skipping %s: incomplete manifest entry", card_type.c_str());
			continue;
		}

		char *onnx_path = obs_module_file(("per_type/" + onnx_name).c_str());
		char *index_path = obs_module_file(("per_type/" + index_name).c_str());
		char *meta_path = obs_module_file(("per_type/" + meta_name).c_str());

		if (!onnx_path || !index_path || !meta_path) {
			obs_log(LOG_WARNING, "[PerType] Skipping %s: missing files", card_type.c_str());
			bfree(onnx_path); bfree(index_path); bfree(meta_path);
			continue;
		}

		auto matcher = std::make_unique<EmbeddingMatcher>();
		bool ok = matcher->load(onnx_path, index_path, meta_path);

		bfree(onnx_path);
		bfree(index_path);
		bfree(meta_path);

		if (ok) {
			tf->per_type_matchers[card_type] = std::move(matcher);
			loaded++;
		} else {
			obs_log(LOG_WARNING, "[PerType] Failed to load embedder for type: %s", card_type.c_str());
		}
	}

	obs_log(LOG_INFO, "[PerType] Loaded %d/%zu per-type embedders", loaded, manifest.size());
}

// ─── Build card name entries for OCR fuzzy matching ────────────────────

static void buildCardNameEntries(struct filter_data *tf)
{
    tf->card_name_entries.clear();
    if (tf->vtes_db.is_empty()) return;

    for (const auto& [id_str, entry] : tf->vtes_db.all_entries()) {
        VTESCardNameEntry ce;
        ce.id = id_str;
        ce.printed_name = entry.printed_name;
        ce.full_name = entry.name;
        ce.normalized = VtesOcrReader::normalize(
            ce.printed_name.empty() ? ce.full_name : ce.printed_name);
        tf->card_name_entries.push_back(std::move(ce));
    }

    obs_log(LOG_INFO, "[OCR] Built %zu card name entries for fuzzy matching",
            tf->card_name_entries.size());
}

// ─── Initialize Tesseract OCR reader ───────────────────────────────────

static void initOcrReader(struct filter_data *tf)
{
    if (tf->ocr_reader) return; // already initialized

    // Build card name entries first (needs vtes_db to be loaded)
    if (tf->card_name_entries.empty()) {
        buildCardNameEntries(tf);
    }
    if (tf->card_name_entries.empty()) return;

    // Try to find tessdata directory
    std::vector<std::string> tessdata_candidates;

    // 1. Plugin's own data directory
    char *plugin_tessdata = obs_module_file("tessdata");
    if (plugin_tessdata) {
        // Check if the directory exists
        std::ifstream test((std::string(plugin_tessdata) + "/eng.traineddata").c_str());
        if (test.good()) {
            tessdata_candidates.push_back(plugin_tessdata);
        }
        bfree(plugin_tessdata);
    }

    // 2. Standard Tesseract install locations
    tessdata_candidates.push_back("C:/Program Files/Tesseract-OCR/tessdata");
    tessdata_candidates.push_back("C:/Program Files (x86)/Tesseract-OCR/tessdata");

    // 3. TESSDATA_PREFIX environment variable
    const char* env_tessdata = std::getenv("TESSDATA_PREFIX");
    if (env_tessdata) {
        tessdata_candidates.push_back(env_tessdata);
    }

    std::string found_path;
    for (const auto& path : tessdata_candidates) {
        if (path.empty()) continue;
        std::ifstream test((path + "/eng.traineddata").c_str());
        if (test.good()) {
            found_path = path;
            break;
        }
    }

    if (found_path.empty()) {
        obs_log(LOG_WARNING, "[OCR] eng.traineddata not found in any tessdata path. "
                "Install Tesseract or place tessdata/ in plugin data dir.");
        return;
    }

    auto reader = std::make_unique<VtesOcrReader>();
    if (reader->init(found_path, tf->card_name_entries)) {
        tf->ocr_reader = std::move(reader);
        tf->ocr_enabled = true;
        obs_log(LOG_INFO, "[OCR] Tesseract initialized with tessdata: %s", found_path.c_str());
    }
}

// ─── Card info loader (byId section from card-hash-index.json + vtes.json) ─

static bool loadCardInfo(struct filter_data *tf)
{
	if (!tf->card_info_by_id.empty()) return true;

	// 1. Load card-hash-index.json for display info (types, clans, disciplines)
	char *jsonPath = obs_module_file("card-hash-index.json");
	if (jsonPath) {
		std::ifstream f(jsonPath);
		bfree(jsonPath);
		if (f.is_open()) {
			try {
				nlohmann::json j;
				f >> j;

				if (j.contains("byId")) {
					for (auto& [cardId, cardJson] : j["byId"].items()) {
						CardInfo info;
						info.id = cardId;
						if (cardJson.contains("name")) info.name = cardJson["name"];
						if (cardJson.contains("url")) info.url = cardJson["url"];
						if (cardJson.contains("cardText")) info.cardText = cardJson["cardText"];
						if (cardJson.contains("group")) info.group = cardJson["group"];
						if (cardJson.contains("capacity") && !cardJson["capacity"].is_null())
							info.capacity = cardJson["capacity"];
						if (cardJson.contains("types"))
							for (auto& t : cardJson["types"]) info.types.push_back(t);
						if (cardJson.contains("clans"))
							for (auto& c : cardJson["clans"]) info.clans.push_back(c);
						if (cardJson.contains("disciplines"))
							for (auto& d : cardJson["disciplines"]) info.disciplines.push_back(d);
						tf->card_info_by_id[cardId] = info;
					}
				}
				obs_log(LOG_INFO, "Loaded %zu card infos from card-hash-index.json (byId)",
					tf->card_info_by_id.size());
			} catch (const std::exception &e) {
				obs_log(LOG_WARNING, "Failed to parse card-hash-index.json: %s", e.what());
			}
		}
	}

	// 2. Load vtes.json into the full card database for type-based filtering
	// (set_card_types is called after embedder loads, see detect_filter_obb_create)
	char *vtesJsonPath = obs_module_file("vtes.json");
	if (vtesJsonPath) {
		bool loaded = tf->vtes_db.load(vtesJsonPath);
		bfree(vtesJsonPath);
		if (loaded) {
			obs_log(LOG_INFO, "Loaded %zu cards from vtes.json", tf->vtes_db.size());
		} else {
			obs_log(LOG_WARNING, "Failed to load vtes.json");
		}
	} else {
		obs_log(LOG_WARNING, "vtes.json not found in plugin data directory");
	}

	return !tf->card_info_by_id.empty();
}

// ─── Temporal smoothing ─────────────────────────────────────────────────────

static void updateTemporalTracks(
	struct filter_data *tf,
	const std::vector<yolov8_obb_cpp::OBBObject>& current_detections,
	std::vector<yolov8_obb_cpp::OBBObject>& output_objects)
{
	output_objects.clear();

	// Mark all existing tracks as unseen this frame
	for (auto& [id, track] : tf->temporal_tracks) {
		track.unseen_count++;
	}

	// Match current detections to existing tracks (simple IoU-based matching)
	for (const auto& det : current_detections) {
		float det_cx = det.rect.x + det.rect.width / 2;
		float det_cy = det.rect.y + det.rect.height / 2;
		float det_area = det.rect.width * det.rect.height;

		// Find best matching track by center distance
		int best_id = -1;
		float best_dist = 1e9f;
		for (auto& [id, track] : tf->temporal_tracks) {
			float dx = det_cx - track.avg_cx;
			float dy = det_cy - track.avg_cy;
			float dist = std::sqrt(dx * dx + dy * dy);
			if (dist < best_dist && dist < std::max(det.rect.width, det.rect.height)) {
				best_dist = dist;
				best_id = id;
			}
		}

		if (best_id >= 0) {
			// Update existing track
			auto& track = tf->temporal_tracks[best_id];
			float alpha = 0.3f; // smoothing factor
			track.avg_cx = (1 - alpha) * track.avg_cx + alpha * det_cx;
			track.avg_cy = (1 - alpha) * track.avg_cy + alpha * det_cy;
			track.avg_w = (1 - alpha) * track.avg_w + alpha * det.rect.width;
			track.avg_h = (1 - alpha) * track.avg_h + alpha * det.rect.height;
			track.avg_angle = (1 - alpha) * track.avg_angle + alpha * det.angle;
			track.stable_count++;
			track.unseen_count = 0;
			track.history.push_back(true);
			if ((int)track.history.size() > filter_data::TEMPORAL_WINDOW)
				track.history.pop_front();

			// Update identity if detection has card_id
			if (!det.card_id.empty() && det.card_id == track.tracked_card_id) {
				track.identity_hits++;
			} else if (!det.card_id.empty()) {
				// New identity detected — reset if strong evidence
				if (track.identity_hits < 2) {
					track.tracked_card_id = det.card_id;
					track.tracked_card_name = det.card_name;
					track.identity_hits = 1;
				}
			}
			track.identity_total++;
			if (track.identity_total > filter_data::TrackedObject::IDENTITY_WINDOW) {
				track.identity_total = filter_data::TrackedObject::IDENTITY_WINDOW;
				if (track.identity_hits > filter_data::TrackedObject::IDENTITY_WINDOW)
					track.identity_hits = filter_data::TrackedObject::IDENTITY_WINDOW;
			}

			// Recompute confidence
			int hits = 0;
			for (bool h : track.history) if (h) hits++;
			track.confidence = (float)hits / (float)filter_data::TEMPORAL_WINDOW;
		} else {
			// Create new track
			filter_data::TrackedObject track;
			track.avg_cx = det_cx;
			track.avg_cy = det_cy;
			track.avg_w = det.rect.width;
			track.avg_h = det.rect.height;
			track.avg_angle = det.angle;
			track.stable_count = 1;
			track.unseen_count = 0;
			track.id = tf->next_track_id++;
			track.confidence = 1.0f / (float)filter_data::TEMPORAL_WINDOW;
			track.history = {true};
			track.tracked_card_id = det.card_id;
			track.tracked_card_name = det.card_name;
			track.identity_hits = det.card_id.empty() ? 0 : 1;
			track.identity_total = 1;
			tf->temporal_tracks[track.id] = track;
			best_id = track.id;
		}
	}

	// Remove stale tracks (unseen for too long)
	for (auto it = tf->temporal_tracks.begin(); it != tf->temporal_tracks.end();) {
		if (it->second.unseen_count > filter_data::TEMPORAL_WINDOW * 2) {
			it = tf->temporal_tracks.erase(it);
		} else {
			++it;
		}
	}

	// Output only tracks that meet confidence and stability thresholds
	for (auto& [id, track] : tf->temporal_tracks) {
		if (track.unseen_count > 0) continue; // not detected this frame

		if (track.stable_count >= tf->temporal_min_stable &&
		    track.confidence >= tf->temporal_min_confidence) {
			yolov8_obb_cpp::OBBObject obj;
			obj.rect.x = track.avg_cx - track.avg_w / 2;
			obj.rect.y = track.avg_cy - track.avg_h / 2;
			obj.rect.width = track.avg_w;
			obj.rect.height = track.avg_h;
			obj.angle = track.avg_angle;
			obj.label = 0;
			obj.prob = track.confidence;
			obj.id = track.id;
			obj.unseenFrames = track.unseen_count;

			// Use smoothed identity if majority of frames agree
			if (!track.tracked_card_id.empty() &&
			    track.identity_total >= 3 &&
			    (float)track.identity_hits / track.identity_total > 0.5f) {
				obj.card_id = track.tracked_card_id;
				obj.card_name = track.tracked_card_name;
			}

			output_objects.push_back(obj);
		}
	}
}

// ─── OBS filter callbacks ──────────────────────────────────────────────────

const char *detect_filter_obb_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("VTES Card Detector (OBB)");
}

obs_properties_t *detect_filter_obb_properties(void *data)
{
	obs_properties_t *props = obs_properties_create();

	obs_properties_add_bool(props, "preview", obs_module_text("Preview"));

	obs_properties_add_float_slider(props, "threshold", obs_module_text("ConfThreshold"), 0.0,
					1.0, 0.025);

	obs_property_t *p_det_mode =
		obs_properties_add_list(props, "detection_mode", obs_module_text("DetectionMode"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p_det_mode, obs_module_text("Shape (Contour)"), DETECT_MODE_CONTOUR);
	obs_property_list_add_string(p_det_mode, obs_module_text("AI Model (ONNX)"), DETECT_MODE_ONNX);

	obs_property_t *p_use_gpu =
		obs_properties_add_list(props, "useGPU", obs_module_text("InferenceDevice"),
				OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_use_gpu, obs_module_text("CPU"), USEGPU_CPU);
#if _WIN32
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPU (DirectML)"), USEGPU_DML);
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPU (CUDA)"), USEGPU_CUDA);
#endif

	// --- Card Type Classifier (for type-based embedding filter) ---
	obs_properties_add_bool(props, "classifier_enabled", "Type Classifier (filter by card type)");

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

	// Card Search Web Server
	obs_properties_add_button(props, "card_search_btn", "Open Card Search",
		[](obs_properties_t *, obs_property_t *, void *data) {
			struct detect_filter_obb *tf = reinterpret_cast<detect_filter_obb *>(data);
			if (!tf->web_server || !tf->web_server->is_running()) {
				tf->web_server = std::make_unique<WebServer>();
				tf->web_server->start(tf->web_server_port, &tf->vtes_db);
			}
			if (tf->web_server && tf->web_server->is_running()) {
				std::string url = "http://localhost:" + std::to_string(tf->web_server->port());
#ifdef _WIN32
				ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
#else
				std::string cmd = "xdg-open " + url;
				std::thread([cmd]() { system(cmd.c_str()); }).detach();
#endif
			}
			return true;
		});

	std::string basic_info =
		std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
	obs_properties_add_text(props, "info", basic_info.c_str(), OBS_TEXT_INFO);

	UNUSED_PARAMETER(data);
	return props;
}

void detect_filter_obb_defaults(obs_data_t *settings)
{
#if _WIN32
	obs_data_set_default_string(settings, "useGPU", USEGPU_DML);
#else
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#endif
	obs_data_set_default_bool(settings, "sort_tracking", false);
	obs_data_set_default_bool(settings, "preview", true);
	obs_data_set_default_double(settings, "threshold", 0.5);
	obs_data_set_default_string(settings, "detection_mode", DETECT_MODE_CONTOUR);
	obs_data_set_default_bool(settings, "temporal_smoothing", true);

	// Contour detection defaults
	obs_data_set_default_int(settings, "contour_edge_low", 50);
	obs_data_set_default_int(settings, "contour_edge_high", 150);
	obs_data_set_default_double(settings, "contour_min_area", 0.02);
	obs_data_set_default_double(settings, "contour_max_area", 0.30);
	obs_data_set_default_double(settings, "contour_aspect_tol", 0.15);

	// ONNX area filter defaults
	obs_data_set_default_int(settings, "onnx_min_area", 4000);
	obs_data_set_default_double(settings, "onnx_max_area", 0.30);

	// Temporal smoothing defaults
	obs_data_set_default_bool(settings, "temporal_smoothing", false);
	obs_data_set_default_double(settings, "temporal_min_conf", 0.3);
	obs_data_set_default_int(settings, "temporal_min_stable", 3);

	// Embedding matcher is always enabled when model is available

	// Classifier defaults
	obs_data_set_default_bool(settings, "classifier_enabled", true);
	obs_data_set_default_double(settings, "clf_oval_weight", 0.5);
	obs_data_set_default_double(settings, "clf_color_weight", 0.3);
	obs_data_set_default_double(settings, "clf_contour_weight", 0.2);
	obs_data_set_default_double(settings, "clf_oval_thresh", 0.6);
	obs_data_set_default_double(settings, "clf_color_thresh", 0.55);
	obs_data_set_default_double(settings, "clf_oval_reject_master", 0.4);

	// VTES defaults
	obs_data_set_default_bool(settings, "vtes_group", false);
	obs_data_set_default_string(settings, "vtes_ws_host", "127.0.0.1");
	obs_data_set_default_int(settings, "vtes_ws_port", 3998);
	obs_data_set_default_int(settings, "vtes_cooldown", 10);
}

void detect_filter_obb_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "VTES OBB filter update");

	struct detect_filter_obb *tf = reinterpret_cast<detect_filter_obb *>(data);

	tf->isDisabled = true;

	tf->preview = obs_data_get_bool(settings, "preview");
	tf->conf_threshold = (float)obs_data_get_double(settings, "threshold");
	tf->sortTracking = obs_data_get_bool(settings, "sort_tracking");
	size_t maxUnseenFrames = (size_t)obs_data_get_int(settings, "max_unseen_frames");
	if (tf->tracker.getMaxUnseenFrames() != maxUnseenFrames) {
		tf->tracker.setMaxUnseenFrames(maxUnseenFrames);
	}
	tf->showUnseenObjects = obs_data_get_bool(settings, "show_unseen_objects");
	tf->saveDetectionsPath = obs_data_get_string(settings, "save_detections_path");

	const std::string newDetectMode = obs_data_get_string(settings, "detection_mode");
	tf->contourEdgeLow = (int)obs_data_get_int(settings, "contour_edge_low");
	tf->contourEdgeHigh = (int)obs_data_get_int(settings, "contour_edge_high");
	tf->contourMinAreaFrac = obs_data_get_double(settings, "contour_min_area");
	tf->contourMaxAreaFrac = obs_data_get_double(settings, "contour_max_area");
	tf->contourAspectTolerance = obs_data_get_double(settings, "contour_aspect_tol");

	tf->onnxMinArea = (int)obs_data_get_int(settings, "onnx_min_area");
	tf->onnxMaxAreaFrac = obs_data_get_double(settings, "onnx_max_area");

	// Temporal smoothing
	tf->temporal_smoothing_enabled = obs_data_get_bool(settings, "temporal_smoothing");
	tf->temporal_min_confidence = (float)obs_data_get_double(settings, "temporal_min_conf");
	tf->temporal_min_stable = (int)obs_data_get_int(settings, "temporal_min_stable");

	// Embedding matcher (load once on startup)
	if (!tf->embedder.is_loaded()) {
		char *modelPath = obs_module_file("vtes_embedder_1024d.onnx");
		char *binPath = obs_module_file("embeddings_1024d.bin");
		char *metaPath = obs_module_file("embeddings_1024d_meta.json");
		if (modelPath && binPath && metaPath) {
			obs_log(LOG_INFO, "[Embedding] Loading model: %s", modelPath);
			tf->embedder.load(modelPath, binPath, metaPath);
		}
		bfree(modelPath);
		bfree(binPath);
		bfree(metaPath);
	}
	// Apply card types if not already done
	applyCardTypesToEmbedder(tf);

	// Vision-based type classifier (load once on startup)
	if (!tf->type_classifier_session) {
		loadVisionTypeClassifier(tf);
	}

	// Per-type embedding models (load once on startup)
	if (tf->per_type_matchers.empty()) {
		loadPerTypeEmbedders(tf);
	}

	// OCR reader (load once on startup, requires vtes_db from loadCardInfo)
	if (!tf->ocr_reader) {
		initOcrReader(tf);
	}

	// Classifier
	tf->classifier_enabled = obs_data_get_bool(settings, "classifier_enabled");
	if (tf->classifier_enabled) {
		tf->classifierConfig.ovalWeight = (float)obs_data_get_double(settings, "clf_oval_weight");
		tf->classifierConfig.colorWeight = (float)obs_data_get_double(settings, "clf_color_weight");
		tf->classifierConfig.contourWeight = (float)obs_data_get_double(settings, "clf_contour_weight");
		tf->classifierConfig.ovalThreshold = (float)obs_data_get_double(settings, "clf_oval_thresh");
		tf->classifierConfig.colorThreshold = (float)obs_data_get_double(settings, "clf_color_thresh");
		tf->classifierConfig.ovalRejectForMaster = (float)obs_data_get_double(settings, "clf_oval_reject_master");

		if (!tf->classifier) {
			tf->classifier = std::make_unique<vtes_classifier::VTESCardClassifier>(tf->classifierConfig);
			tf->cropExtractor = std::make_unique<vtes_classifier::CardCropExtractor>();
			obs_log(LOG_INFO, "VTES Card Classifier initialized");
		} else {
			// Update config
			tf->classifier = std::make_unique<vtes_classifier::VTESCardClassifier>(tf->classifierConfig);
		}
	} else {
		tf->classifier.reset();
		tf->cropExtractor.reset();
	}

	bool modeChanged = (tf->detectionMode != newDetectMode);
	tf->detectionMode = newDetectMode;

	if (tf->detectionMode == DETECT_MODE_CONTOUR) {
		std::unique_lock<std::mutex> lock(tf->modelMutex);
		if (tf->onnxruntimemodel) {
			tf->onnxruntimemodel.reset();
			obs_log(LOG_INFO, "Contour mode: ONNX model released");
		}
		tf->isDisabled = false;
		obs_log(LOG_INFO, "VTES OBB Filter set to Shape (Contour) mode");
		return;
	}

	// ONNX mode
	const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
	const uint32_t newNumThreads = (uint32_t)obs_data_get_int(settings, "numThreads");

	bool reinitialize = modeChanged;
	if (tf->useGPU != newUseGpu || tf->numThreads != newNumThreads) {
		reinitialize = true;
	}

	if (reinitialize) {
		obs_log(LOG_INFO, "Loading ONNX model");

		std::unique_lock<std::mutex> lock(tf->modelMutex);

		char *modelFilepath_rawPtr = obs_module_file("models/vtes.onnx");

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

		tf->useGPU = newUseGpu;
		tf->numThreads = newNumThreads;

		int onnxruntime_device_id_ = 0;
		bool onnxruntime_use_parallel_ = true;
		float nms_th_ = 0.45f;
		int num_classes_ = 1;

		try {
			if (tf->onnxruntimemodel) {
				tf->onnxruntimemodel.reset();
			}
			tf->onnxruntimemodel =
				std::make_unique<yolov8_obb_cpp::YOLOv8OBB>(
					tf->modelFilepath, tf->numThreads, num_classes_,
					tf->numThreads, tf->useGPU, onnxruntime_device_id_,
					onnxruntime_use_parallel_, nms_th_,
					tf->conf_threshold);

			char *jsonPath_rawPtr = obs_module_file("models/vtes.json");
			if (jsonPath_rawPtr) {
				std::ifstream jsonFile(jsonPath_rawPtr);
				if (jsonFile.is_open()) {
					try {
						nlohmann::json config;
						jsonFile >> config;
						if (config.contains("output_shape") &&
						    config["output_shape"].contains("shape") &&
						    config["output_shape"]["shape"].size() >= 3) {
							auto &shapeArr = config["output_shape"]["shape"];
							int num_dets = std::stoi(shapeArr[1].get<std::string>());
							int num_feats = std::stoi(shapeArr[2].get<std::string>());
							tf->onnxruntimemodel->setOutputShape(num_dets, num_feats);
							obs_log(LOG_INFO,
								"Model shape overridden from JSON: [1,%d,%d]",
								num_dets, num_feats);
						}
					} catch (const std::exception &e) {
						obs_log(LOG_WARNING, "Failed to parse vtes.json: %s", e.what());
					}
				}
				bfree(jsonPath_rawPtr);
			}

			obs_data_set_string(settings, "error", "");
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "Failed to load OBB model: %s", e.what());
			tf->isDisabled = true;
			tf->onnxruntimemodel.reset();
			return;
		}
	}

	if (tf->onnxruntimemodel) {
		tf->onnxruntimemodel->setBBoxConfThresh(tf->conf_threshold);
	}

	if (reinitialize) {
		obs_log(LOG_INFO, "VTES OBB Filter Options:");
		obs_log(LOG_INFO, "  Mode: ONNX (AI Model)");
		obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
		obs_log(LOG_INFO, "  Inference Device: %s", tf->useGPU.c_str());
		obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
		obs_log(LOG_INFO, "  Preview: %s", tf->preview ? "true" : "false");
		obs_log(LOG_INFO, "  Threshold: %.2f", tf->conf_threshold);
	}

	// VTES: WebSocket connection
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

	tf->isDisabled = false;
}

void detect_filter_obb_activate(void *data)
{
	obs_log(LOG_INFO, "VTES OBB filter activated");
	struct detect_filter_obb *tf = reinterpret_cast<detect_filter_obb *>(data);
	tf->isDisabled = false;
}

void detect_filter_obb_deactivate(void *data)
{
	obs_log(LOG_INFO, "VTES OBB filter deactivated");
	struct detect_filter_obb *tf = reinterpret_cast<detect_filter_obb *>(data);
	tf->isDisabled = true;
}

void *detect_filter_obb_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "VTES OBB filter created");
	void *data = bmalloc(sizeof(struct detect_filter_obb));
	struct detect_filter_obb *tf = new (data) detect_filter_obb();

	tf->source = source;
	tf->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	tf->lastDetectedObjectId = -1;

	// Pre-load card info and embedding model
	loadCardInfo(tf);
	if (!tf->embedder.is_loaded()) {
		char *modelPath = obs_module_file("vtes_embedder_1024d.onnx");
		char *binPath = obs_module_file("embeddings_1024d.bin");
		char *metaPath = obs_module_file("embeddings_1024d_meta.json");
		if (modelPath && binPath && metaPath) {
			obs_log(LOG_INFO, "[Embedding] Loading model on startup: %s", modelPath);
			tf->embedder.load(modelPath, binPath, metaPath);
		}
		bfree(modelPath);
		bfree(binPath);
		bfree(metaPath);
	}
	// Apply card types now that embedder is loaded (or was already loaded)
	applyCardTypesToEmbedder(tf);

	// Pre-load vision-based type classifier
	if (!tf->type_classifier_session) {
		loadVisionTypeClassifier(tf);
	}

	// Pre-load per-type embedding models
	loadPerTypeEmbedders(tf);

	// Pre-load OCR reader (requires vtes_db from loadCardInfo)
	if (!tf->ocr_reader) {
		initOcrReader(tf);
	}

	// Start embedded web server
	tf->web_server = std::make_unique<WebServer>();
	tf->web_server->start(tf->web_server_port, &tf->vtes_db);

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

	detect_filter_obb_update(tf, settings);

	return tf;
}

void detect_filter_obb_destroy(void *data)
{
	obs_log(LOG_INFO, "VTES OBB filter destroyed");

	struct detect_filter_obb *tf = reinterpret_cast<detect_filter_obb *>(data);

	if (tf) {
		tf->isDisabled = true;

		// VTES: stop web server
		if (tf->web_server) {
			tf->web_server->stop();
			tf->web_server.reset();
		}

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
		tf->~detect_filter_obb();
		bfree(tf);
	}
}

void detect_filter_obb_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);

	struct detect_filter_obb *tf = reinterpret_cast<detect_filter_obb *>(data);

	if (tf->isDisabled) {
		return;
	}
	if (tf->detectionMode == DETECT_MODE_ONNX && !tf->onnxruntimemodel) {
		return;
	}

	if (!obs_source_enabled(tf->source)) {
		return;
	}

	cv::Mat imageBGRA;
	{
		std::unique_lock<std::mutex> lock(tf->inputBGRALock, std::try_to_lock);
		if (!lock.owns_lock()) {
			obs_log(LOG_DEBUG, "OBB tick: inputBGRALock busy");
			return;
		}
		if (tf->inputBGRA.empty()) {
			obs_log(LOG_DEBUG, "OBB tick: inputBGRA empty");
			return;
		}
		obs_log(LOG_DEBUG, "OBB tick: inputBGRA %dx%d",
			tf->inputBGRA.cols, tf->inputBGRA.rows);
		imageBGRA = tf->inputBGRA.clone();
	}

	cv::Mat inferenceFrame;
	cv::cvtColor(imageBGRA, inferenceFrame, cv::COLOR_BGRA2BGR);

	std::vector<yolov8_obb_cpp::OBBObject> raw_objects;

	if (tf->detectionMode == DETECT_MODE_CONTOUR) {
		vtes_detection::ContourParams cparams;
		cparams.edge_low = tf->contourEdgeLow;
		cparams.edge_high = tf->contourEdgeHigh;
		cparams.min_area_fraction = tf->contourMinAreaFrac;
		cparams.max_area_fraction = tf->contourMaxAreaFrac;
		cparams.aspect_tolerance = tf->contourAspectTolerance;
		raw_objects = vtes_detection::detect_by_contour(inferenceFrame, cparams);
	} else {
		try {
			std::unique_lock<std::mutex> lock(tf->modelMutex);
			auto *obb_model = dynamic_cast<yolov8_obb_cpp::YOLOv8OBB *>(tf->onnxruntimemodel.get());
			if (obb_model) {
				raw_objects = obb_model->inferOBB(inferenceFrame);
			}
		} catch (const Ort::Exception &e) {
			obs_log(LOG_ERROR, "ONNXRuntime Exception: %s", e.what());
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "%s", e.what());
		}
	}

	// ─── ONNX area filtering (filter out too small/large detections) ──────
	if (tf->detectionMode == DETECT_MODE_ONNX && !raw_objects.empty()) {
		std::vector<yolov8_obb_cpp::OBBObject> filtered;
		double max_area = tf->onnxMaxAreaFrac * inferenceFrame.cols * inferenceFrame.rows;
		for (const auto& obj : raw_objects) {
			double area = obj.rect.width * obj.rect.height;
			if (area >= tf->onnxMinArea && area <= max_area) {
				filtered.push_back(obj);
			}
		}
		raw_objects = filtered;
	}

	// ─── VTES Card Classification (run BEFORE embedding to use type as filter) ─
	// Store per-object type filter string for embedding
	std::vector<std::string> per_object_type_filter(raw_objects.size(), "");
	if (tf->classifier_enabled && tf->classifier && tf->cropExtractor && !raw_objects.empty()) {
		std::vector<vtes_classifier::CardCropExtractor::DetectionInput> detections;
		detections.reserve(raw_objects.size());
		for (size_t i = 0; i < raw_objects.size(); ++i) {
			const auto& obj = raw_objects[i];
			detections.push_back({obj.rect, obj.prob, obj.label, static_cast<int>(i)});
		}

		auto crops = tf->cropExtractor->extract(inferenceFrame, detections);
		for (size_t i = 0; i < crops.size() && i < raw_objects.size(); ++i) {
			if (crops[i].crop.empty() || crops[i].crop.rows < 8 || crops[i].crop.cols < 8) {
				obs_log(LOG_DEBUG, "Card %zu: crop too small (%dx%d), skipping classification",
					i, crops[i].crop.cols, crops[i].crop.rows);
				continue;
			}

			// ─── Vision-based type classifier (vtes_type_classifier.onnx, 99.9% acc) ─
			std::string vision_type = runVisionTypeClassifier(tf, crops[i].crop);
			if (!vision_type.empty()) {
				per_object_type_filter[i] = vision_type;
				// Map vision type to display label (for color-coded overlay)
				static const std::unordered_map<std::string, int> type_to_label = {
					{"Vampire", 0}, {"Master", 1}, {"LibraryAction", 2}, {"Action", 2},
					{"Action Modifier", 2}, {"Event", 2}, {"Retainer", 2}, {"Conviction", 2},
					{"Reaction", 3}, {"Combat", 4}, {"Equipment", 5},
					{"Political Action", 6}, {"Ally", 6}, {"Power", 6}, {"Imbued", 6},
				};
				int typeIdx = 7; // Unknown
				auto it = type_to_label.find(vision_type);
				if (it != type_to_label.end()) typeIdx = it->second;
				raw_objects[i].label = 100 + typeIdx;
				obs_log(LOG_DEBUG, "Card %zu: vision_type=%s", i, vision_type.c_str());
			} else {
				// Fallback: signal-based classifier
				float typeConf = 0.0f;
				vtes_classifier::SignalScores signals;
				vtes_classifier::CardType type = tf->classifier->classify(crops[i].crop, typeConf, signals);
				per_object_type_filter[i] = VTESCardDatabase::classifier_type_to_vtes_filter(
					static_cast<int>(type));
				raw_objects[i].label = 100 + static_cast<int>(type);
				raw_objects[i].prob = typeConf;
				obs_log(LOG_DEBUG, "Card %zu: signal_type=%d(%s) conf=%.2f",
					i, static_cast<int>(type), per_object_type_filter[i].c_str(), typeConf);
			}
		}
	}

	// ─── Embedding-based card identification (per-type models, fallback to global) ─
	bool embed_available = tf->embedder.is_loaded() || !tf->per_type_matchers.empty();
	if (embed_available && !raw_objects.empty()) {
		for (size_t i = 0; i < raw_objects.size(); i++) {
			auto& obj = raw_objects[i];

			cv::Point2f center(obj.rect.x + obj.rect.width / 2,
					   obj.rect.y + obj.rect.height / 2);
			cv::Size2f size(obj.rect.width, obj.rect.height);
			cv::RotatedRect rr(center, size, obj.angle * 180.0f / kPI);
			cv::Point2f corners[4];
			rr.points(corners);

			cv::Mat card_region = extractCardRegion(inferenceFrame, corners);
			if (card_region.empty()) continue;

			std::string card_name, card_id;
			float confidence = 0.0f;
			const std::string& type_filter = per_object_type_filter[i];
			bool matched = false;
			float thresh = 0.35f;

			// Try per-type embedder first
			auto pt_it = tf->per_type_matchers.find(type_filter);
			if (pt_it != tf->per_type_matchers.end() && pt_it->second->is_loaded()) {
				matched = pt_it->second->identify(card_region, card_name, card_id, confidence);
				thresh = pt_it->second->threshold();
			}
			// Fallback to global embedder
			if (!matched && tf->embedder.is_loaded()) {
				matched = tf->embedder.identify(card_region, card_name, card_id, confidence, type_filter);
				thresh = tf->embedder.threshold();
			}

			// OCR fallback: when embedding confidence is low or no match
			if (tf->ocr_enabled && tf->ocr_reader &&
			    (!matched || confidence < thresh * 0.8f)) {
				std::string ocr_name, ocr_id;
				float ocr_conf = 0.0f;
				if (tf->ocr_reader->recognize(card_region, ocr_name, ocr_id, ocr_conf)) {
					if (ocr_conf > confidence) {
						card_name = ocr_name;
						card_id = ocr_id;
						confidence = ocr_conf;
						matched = true;
						obs_log(LOG_INFO, "[OCR] Card #%d: OCR matched '%s' (id=%s, conf=%.2f)",
							obj.id, ocr_name.c_str(), ocr_id.c_str(), ocr_conf);
					}
				}
			}

			if (matched) {
				if (confidence >= thresh) {
					obj.card_name = card_name;
				} else {
					obj.card_name = "?" + card_name;
				}
				obj.card_id = card_id;
				obj.prob = confidence;
				obs_log(LOG_INFO,
					"[Embed] Card #%d: %s (id=%s, sim=%.4f, filter=%s%s)",
					obj.id, card_name.c_str(), card_id.c_str(), confidence,
					type_filter.c_str(),
					confidence >= thresh ? "" : " ?");
			} else {
				obj.card_name = "???";
				obj.card_id = "";
				if (confidence > 0.2f) {
					obj.prob = confidence;
					obs_log(LOG_INFO,
						"[Embed] Card #%d: no match (best_sim=%.4f, filter=%s)",
						obj.id, confidence, type_filter.c_str());
				}
			}
		}
	}

	// ─── SORT tracking (optional) ────────────────────────────────────────
	if (tf->sortTracking) {
		std::vector<Object> sort_objects;
		for (const auto &obb : raw_objects) {
			Object obj;
			obj.rect = obb.rect;
			obj.label = obb.label;
			obj.prob = obb.prob;
			obj.id = obb.id;
			obj.unseenFrames = obb.unseenFrames;
			sort_objects.push_back(obj);
		}
		sort_objects = tf->tracker.update(sort_objects);
		for (size_t i = 0; i < raw_objects.size() && i < sort_objects.size(); ++i) {
			raw_objects[i].id = static_cast<int>(sort_objects[i].id);
			raw_objects[i].unseenFrames = static_cast<int>(sort_objects[i].unseenFrames);
		}
	}

	// ─── Temporal smoothing ──────────────────────────────────────────────
	std::vector<yolov8_obb_cpp::OBBObject> final_objects;
	if (tf->temporal_smoothing_enabled) {
		updateTemporalTracks(tf, raw_objects, final_objects);
	} else {
		final_objects = raw_objects;
	}

	if (!tf->showUnseenObjects) {
		final_objects.erase(
			std::remove_if(final_objects.begin(), final_objects.end(),
				       [](const yolov8_obb_cpp::OBBObject &obj) { return obj.unseenFrames > 0; }),
			final_objects.end());
	}

	if (!tf->saveDetectionsPath.empty()) {
		std::ofstream detectionsFile(tf->saveDetectionsPath);
		if (detectionsFile.is_open()) {
			nlohmann::json j;
			for (const auto &obj : final_objects) {
				nlohmann::json obj_json;
				obj_json["label"] = obj.label;
				obj_json["confidence"] = obj.prob;
				obj_json["rect"] = {{"x", obj.rect.x},
						    {"y", obj.rect.y},
						    {"width", obj.rect.width},
						    {"height", obj.rect.height}};
				obj_json["angle"] = obj.angle;
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

	// ─── VTES: Send card crop to Node.js server via WebSocket ──────────
	if (tf->wsClient.is_connected() && !final_objects.empty()) {
		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
			now - tf->lastSendTime).count();

		if (elapsed >= tf->cooldownSeconds) {
			const auto &card = final_objects[0];

			// Extract perspective-corrected card region from OBB corners
			cv::Point2f center(card.rect.x + card.rect.width / 2,
					   card.rect.y + card.rect.height / 2);
			cv::Size2f size(card.rect.width, card.rect.height);
			cv::RotatedRect rr(center, size, card.angle * 180.0f / kPI);
			cv::Point2f corners[4];
			rr.points(corners);

			cv::Mat cardImage = extractCardRegion(inferenceFrame, corners);
			if (!cardImage.empty()) {
				try {
					std::vector<uchar> buf;
					std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 85};
					cv::imencode(".jpg", cardImage, buf, params);

					nlohmann::json msg;
					msg["type"] = "card_crop";
					msg["image"] = base64_encode(buf);
					msg["confidence"] = card.prob;
					msg["angle"] = card.angle;
					msg["timestamp"] = std::chrono::duration_cast<
						std::chrono::milliseconds>(
						std::chrono::system_clock::now().time_since_epoch()).count();
					if (!card.card_name.empty()) {
						msg["card_name"] = card.card_name;
						msg["card_id"] = card.card_id;
						// Look up full card info if available
						auto it = tf->card_info_by_id.find(card.card_id);
						if (it != tf->card_info_by_id.end()) {
							nlohmann::json cardInfo;
							cardInfo["name"] = it->second.name;
							cardInfo["types"] = it->second.types;
							cardInfo["clans"] = it->second.clans;
							cardInfo["capacity"] = it->second.capacity;
							cardInfo["disciplines"] = it->second.disciplines;
							cardInfo["cardText"] = it->second.cardText;
							cardInfo["url"] = it->second.url;
							cardInfo["group"] = it->second.group;
							msg["card_info"] = cardInfo;
						}
					}

					std::string jsonStr = msg.dump();
					if (tf->wsClient.send_text(jsonStr)) {
						tf->lastSendTime = now;
						obs_log(LOG_INFO, "[VTES] Sent card crop (conf: %.2f, angle: %.2f)",
							card.prob, card.angle);
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

	if (tf->preview) {
		cv::Mat frame;
		cv::cvtColor(imageBGRA, frame, cv::COLOR_BGRA2BGR);

		if (final_objects.size() > 0) {
			for (const auto &obj : final_objects) {
				cv::Point2f center(obj.rect.x + obj.rect.width / 2,
						   obj.rect.y + obj.rect.height / 2);
				cv::Size2f size(obj.rect.width, obj.rect.height);
				cv::RotatedRect rotatedRect(center, size, obj.angle * 180.0f / kPI);

				cv::Point2f vertices[4];
				rotatedRect.points(vertices);
				for (int i = 0; i < 4; i++) {
					cv::line(frame, vertices[i], vertices[(i + 1) % 4],
						 cv::Scalar(0, 255, 0), 2);
				}

				// Determine color and label based on card type
				cv::Scalar color(0, 255, 0);
				std::string typeStr = "Card";
				if (obj.label >= 100) {
					int typeIdx = obj.label - 100;
					static const char* typeNames[] = {"Vampire", "Master", "LibraryAction", "Reaction", "Combat", "Equipment", "Political", "Unknown"};
					if (typeIdx >= 0 && typeIdx < 8) typeStr = typeNames[typeIdx];
					static cv::Scalar typeColors[] = {
						{30, 30, 160}, {30, 110, 30}, {160, 60, 10}, {140, 80, 0},
						{10, 10, 180}, {120, 80, 0}, {100, 0, 100}, {90, 90, 90}
					};
					if (typeIdx >= 0 && typeIdx < 8) color = typeColors[typeIdx];
				}

			std::string nameStr = obj.card_name.empty() ? "Unknown" : obj.card_name;
			// Draw card name centered above the bounding box
			cv::putText(frame, nameStr,
				    cv::Point2f(obj.rect.x, obj.rect.y - 10),
				    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 2);
			// Draw type + id + confidence below the name
			std::string text = typeStr + " #" + std::to_string(obj.id) +
					   " (" + std::to_string((int)(obj.prob * 1000) / 10.0) + "%)";
			cv::putText(frame, text,
				    cv::Point2f(obj.rect.x, obj.rect.y + obj.rect.height + 15),
				    cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);

			// Show additional card info if identified
			if (!obj.card_id.empty()) {
				auto it = tf->card_info_by_id.find(obj.card_id);
				if (it != tf->card_info_by_id.end()) {
					std::string info;
					if (!it->second.types.empty()) {
						for (size_t ti = 0; ti < it->second.types.size(); ti++) {
							if (ti > 0) info += "/";
							info += it->second.types[ti];
						}
					}
					if (!it->second.disciplines.empty()) {
						if (!info.empty()) info += " ";
						for (size_t di = 0; di < it->second.disciplines.size(); di++) {
							if (di > 0) info += "/";
							info += it->second.disciplines[di];
						}
					}
					if (!info.empty()) {
						cv::putText(frame, info,
							    cv::Point2f(obj.rect.x, obj.rect.y + obj.rect.height + 30),
							    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200), 1);
					}
				}
			}
			}
		}

		// ─── GPU indicator ─────────────────────────────────────────────────
		{
			std::string gpu_status;
			if (tf->detectionMode == DETECT_MODE_ONNX && tf->onnxruntimemodel) {
				const auto& prov = tf->onnxruntimemodel->getProvider();
				if (prov == "dml") gpu_status = "GPU (DML)";
				else if (prov == "cuda") gpu_status = "GPU (CUDA)";
				else gpu_status = "CPU";
			} else {
				gpu_status = "Contour";
			}
			cv::Scalar color = gpu_status.find("GPU") != std::string::npos
				? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
			cv::putText(frame, gpu_status, cv::Point2f(8, 20),
				    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
			// Show real provider name below
			if (tf->detectionMode == DETECT_MODE_ONNX && tf->onnxruntimemodel) {
				cv::putText(frame, tf->onnxruntimemodel->getProvider(),
					    cv::Point2f(8, 36),
					    cv::FONT_HERSHEY_SIMPLEX, 0.35,
					    cv::Scalar(200, 200, 200), 1);
			}
		}

		std::lock_guard<std::mutex> lock(tf->outputLock);
		cv::cvtColor(frame, tf->outputPreviewBGRA, cv::COLOR_BGR2BGRA);
	}
}

void detect_filter_obb_video_render(void *data, gs_effect_t *_effect)
{
	struct detect_filter_obb *tf = reinterpret_cast<detect_filter_obb *>(data);

	if (tf->isDisabled) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}
	if (tf->detectionMode == DETECT_MODE_ONNX && !tf->onnxruntimemodel) {
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

	if (tf->preview) {
		cv::Mat outputBGRA;
		{
			std::lock_guard<std::mutex> lock(tf->outputLock);
			if (tf->outputPreviewBGRA.empty()) {
				obs_log(LOG_ERROR, "OBB render: Preview image is empty");
				obs_source_skip_video_filter(tf->source);
				return;
			}
			if ((uint32_t)tf->outputPreviewBGRA.cols != width ||
			    (uint32_t)tf->outputPreviewBGRA.rows != height) {
				obs_log(LOG_DEBUG, "OBB render: size mismatch preview=%dx%d expect=%dx%d",
					tf->outputPreviewBGRA.cols, tf->outputPreviewBGRA.rows,
					width, height);
				obs_source_skip_video_filter(tf->source);
				return;
			}
			outputBGRA = tf->outputPreviewBGRA.clone();
		}

		if (!tf->maskingEffect) {
			obs_log(LOG_ERROR, "OBB render: maskingEffect is null");
			obs_source_skip_video_filter(tf->source);
			return;
		}

		gs_texture_t *tex = gs_texture_create(width, height, GS_BGRA, 1,
						      (const uint8_t **)&outputBGRA.data, 0);
		if (!tex) {
			obs_log(LOG_ERROR, "OBB render: gs_texture_create failed");
			obs_source_skip_video_filter(tf->source);
			return;
		}

		gs_eparam_t *imageParam = gs_effect_get_param_by_name(tf->maskingEffect, "image");
		if (imageParam) {
			gs_effect_set_texture(imageParam, tex);
		} else {
			obs_log(LOG_ERROR, "OBB render: 'image' param not found in maskingEffect");
		}

		while (gs_effect_loop(tf->maskingEffect, "Draw")) {
			gs_draw_sprite(tex, 0, 0, 0);
		}

		gs_texture_destroy(tex);
	} else {
		obs_log(LOG_DEBUG, "OBB render: preview disabled, passing through");
		obs_source_skip_video_filter(tf->source);
	}
	return;
}
