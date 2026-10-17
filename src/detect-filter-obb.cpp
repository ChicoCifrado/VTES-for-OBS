#include "detect-filter-obb.h"

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
#include <opencv2/dnn.hpp>

static constexpr float kPI = 3.14159265358979323846f;
// ─── Overlay: geometria de referencia ───────────────────────────────────────
// El diseño real es overlay-bg.png (1536x1024 = 3:2, con canal alfa) y se
// escala ENTERO para caber en la escena, dejando bandas negras a los lados
// en una escena 16:9. Todas las cajas de abajo estan medidas en pixeles del
// diseño (1536x1024), no en pixeles de la escena.
static constexpr int kOverlayRefW = 1536;
static constexpr int kOverlayRefH = 1024;
// Ventana de camara: el hueco transparente del diseño. La camara se ajusta
// (entera, sin recorte) dentro de este hueco; el diseño se dibuja ENCIMA.
static constexpr int kWinX = 289;
static constexpr int kWinY = 41;
static constexpr int kWinW = 1203;
static constexpr int kWinH = 947;
// Caja del simbolo ankh sobre la que se coloca la carta detectada. La carta
// lo tapa por completo y no sobresale del panel izquierdo.
static constexpr int kAnkhX = 28;
static constexpr int kAnkhY = 482;
static constexpr int kAnkhW = 204;
static constexpr int kAnkhH = 478;
// Casillas de jugador del panel izquierdo: una fila por jugador, con la caja
// ancha del nombre a la izquierda y la caja pequena del pool a la derecha.
// kSlotY es la coordenada Y del borde superior de cada fila (MAX_PLAYERS = 5).
static constexpr int kNameX = 32;
static constexpr int kNameW = 145;
static constexpr int kPoolX = 184;
static constexpr int kPoolW = 55;
static constexpr int kSlotH = 54;
static constexpr int kSlotY[5] = {89, 170, 249, 327, 401};

#include <numeric>
#include <memory>
#include <exception>
#include <fstream>
#include <filesystem>
#include <ctime>
#include <new>
#include <mutex>
#include <regex>
#include <thread>
#include <deque>
#include <unordered_map>
#include <bitset>
#include <cmath>

#include "detection/vampire_classifier.hpp"

#include <nlohmann/json.hpp>
#include <curl/curl.h>

#include <plugin-support.h>
#include "FilterData.h"
#include "consts.h"
#include "i18n.h"
#include "obs-utils/obs-utils.h"
#include "detect-filter-utils.h"
#include "detection/yolo_detector.hpp"
#ifdef TENSORRT_AVAILABLE
#include "detection/tensorrt_detector.hpp"
#endif
#include "detection/contour_detector.hpp"
#include "classifier/vtes_card_classifier.hpp"
#include "overlay/rulings_view.hpp"
#include "base64-utils.h"

#define VTES_OBB_MODEL "!!!VTES_OBB_MODEL!!!"

static InferenceDevice deviceStringToEnum(const std::string& s) {
    if (s == INFERENCE_CUDA || s == "cuda") return InferenceDevice::CUDA;
    if (s == INFERENCE_DML || s == "dml") return InferenceDevice::DirectML;
    return InferenceDevice::CPU;
}

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

		if (num_classes == 0) return "";
		if (num_classes > tf->type_labels.size())
			num_classes = tf->type_labels.size();

		// Debug log for first few inferences
		static std::once_flag flag;
		std::call_once(flag, [&]() {
			std::string s;
			for (size_t i = 0; i < num_classes; i++) {
				if (i) s += " ";
				s += tf->type_labels[i].substr(0, 6) + "=" +
				     std::to_string(logits[i]);
			}
			obs_log(LOG_INFO, "[VisionClassifier] First inference logits: %s", s.c_str());
		});

		int best = 0;
		for (size_t i = 1; i < num_classes; i++) {
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
		static bool logged = false;
		if (!logged) { logged = true; obs_log(LOG_WARNING, "[PerType] Missing manifest file"); }
		return;
	}

	std::ifstream f(manifest_path);
	if (!f.is_open()) {
		static bool logged = false;
		if (!logged) { logged = true; obs_log(LOG_WARNING, "[PerType] Cannot open manifest: %s", manifest_path); }
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
		matcher->set_inference_device(tf->inference_device_enum);
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

    // El indice indexa solo printed_name y name. Las name_variants que trae
    // vtes.json (hasta 7 por carta) se descartaban, pese a que el buscador
    // web (matchesCard en web_server.h) si las usa. Es la razon por la que
    // el camino /api/search identifica cartas que el fuzzy match no puede.
    // Medido: +710 entradas (+17%) sobre 4149 cartas, con 77 colisiones
    // preexistentes que ya existian solo con printed_name.
    size_t added = 0;
    size_t added_localized = 0;
    for (const auto& [id_str, entry] : tf->vtes_db.all_entries()) {
        VTESCardNameEntry ce;
        ce.id = id_str;
        ce.printed_name = entry.printed_name;
        ce.full_name = entry.name;
        ce.normalized = VtesOcrReader::normalize(
            ce.printed_name.empty() ? ce.full_name : ce.printed_name);
        if (!ce.normalized.empty())
            tf->card_name_entries.push_back(std::move(ce));

        for (const auto& variant : entry.name_variants) {
            VTESCardNameEntry ve;
            ve.id = id_str;
            ve.printed_name = variant;
            ve.full_name = entry.name;
            ve.normalized = VtesOcrReader::normalize(variant);
            if (!ve.normalized.empty()) {
                tf->card_name_entries.push_back(std::move(ve));
                added++;
            }
        }

        // Nombres localizados: la quinta edicion espanola/francesa imprime el
        // nombre traducido ("Ojos de Argos" donde la inglesa dice "Eyes of
        // Argus"). Sin estas entradas, un OCR perfecto del nombre impreso
        // puntuaba ~0.45 contra el indice ingles y se rechazaba. Se conserva
        // el printed_name canonico como nombre reportado para que el overlay
        // siga mostrando la carta en el idioma de la BD.
        for (const std::string* localized : {&entry.name_es, &entry.name_fr}) {
            if (localized->empty()) continue;
            VTESCardNameEntry le;
            le.id = id_str;
            le.printed_name = entry.printed_name.empty() ? entry.name : entry.printed_name;
            le.full_name = entry.name;
            le.normalized = VtesOcrReader::normalize(*localized);
            if (!le.normalized.empty()) {
                tf->card_name_entries.push_back(std::move(le));
                added_localized++;
            }
        }
    }
    obs_log(LOG_INFO, "[OCR] Indexed %zu name_variants for fuzzy matching", added);
    obs_log(LOG_INFO, "[OCR] Indexed %zu localized (es/fr) names for fuzzy matching",
            added_localized);
    obs_log(LOG_INFO, "[OCR] Built %zu card name entries for fuzzy matching",
            tf->card_name_entries.size());
}


// ─── Initialize Tesseract OCR reader ───────────────────────────────────

static void initOcrReader(struct filter_data *tf)
{
    obs_log(LOG_INFO, "[OCR] initOcrReader called (ocr_reader=%p, enabled=%d)",
            (void*)tf->ocr_reader.get(), (int)tf->ocr_enabled);
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
        std::string checked;
        for (const auto& p : tessdata_candidates) {
            if (!checked.empty()) checked += ", ";
            checked += p;
        }
        obs_log(LOG_WARNING, "[OCR] eng.traineddata not found. Checked: %s. "
                "Install Tesseract with language data or copy tessdata/ into plugin data dir.",
                checked.c_str());
        return;
    }

    auto reader = std::make_unique<VtesOcrReader>();
    if (reader->init(found_path, tf->card_name_entries)) {
        // Try to initialize NIS CUDA upscaler (Lanczos 4× + adaptive sharpen)
        int cuda_device = 0;
        if (reader->init_upscaler(cuda_device)) {
            obs_log(LOG_INFO, "[OCR] NIS CUDA upscaler enabled");
        } else {
            obs_log(LOG_INFO, "[OCR] NIS CUDA upscaler not available (CUDA GPU?)");
        }
        tf->ocr_reader = std::move(reader);
        tf->ocr_enabled = true;
        obs_log(LOG_INFO, "[OCR] Tesseract initialized with tessdata: %s", found_path.c_str());
    } else {
        obs_log(LOG_WARNING, "[OCR] VtesOcrReader::init() failed (tessdata: %s)", found_path.c_str());
    }
}

// ─── Download a file via HTTP GET (libcurl, required dep) ─────────────

static size_t write_cb(char* ptr, size_t size, size_t nmemb, FILE* fp) {
    return fwrite(ptr, size, nmemb, fp);
}

static bool downloadFile(const std::string& url, const std::string& dest) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    FILE* fp = fopen(dest.c_str(), "wb");
    if (!fp) { curl_easy_cleanup(curl); return false; }
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 30000L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    CURLcode res = curl_easy_perform(curl);
    fclose(fp);
    curl_easy_cleanup(curl);
    if (res != CURLE_OK) { remove(dest.c_str()); return false; }
    return true;
}

// Download image bytes to memory (for card overlay cache)
static size_t write_mem_cb(char* ptr, size_t size, size_t nmemb, std::vector<uchar>* out) {
    size_t total = size * nmemb;
    out->insert(out->end(), ptr, ptr + total);
    return total;
}

static bool downloadImageToMemory(const std::string& url, std::vector<uchar>& out_data) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    out_data.clear();
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_mem_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out_data);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 5000L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 2000L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "VTES-Card-Scanner/0.1.6");
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return res == CURLE_OK && !out_data.empty();
}

// ─── Card info loader (byId section from card-hash-index.json + vtes.json) ─

static void loadVtesJson(struct filter_data *tf)
{
	if (!tf->vtes_db.is_empty()) return;

	const char* krcg_url = "https://static.krcg.org/data/vtes.json";

	// 1. Try plugin data directory (bundled install)
	char* data_path = obs_module_file("vtes.json");
	if (data_path) {
		if (tf->vtes_db.load(data_path)) {
			obs_log(LOG_INFO, "[VTES] Loaded %zu cards from %s", tf->vtes_db.size(), data_path);
			bfree(data_path);
			return;
		}
		bfree(data_path);
	}

	// 2. Try plugin config directory (previously downloaded)
	char* config_dir = obs_module_config_path("");
	if (config_dir) {
		std::string config_vtes = std::string(config_dir) + "vtes.json";
		bfree(config_dir);
		if (!config_vtes.empty()) {
			std::ifstream test(config_vtes);
			if (test.good()) {
				test.close();
				if (tf->vtes_db.load(config_vtes)) {
					obs_log(LOG_INFO, "[VTES] Loaded %zu cards from cached %s",
						tf->vtes_db.size(), config_vtes.c_str());
					return;
				}
			}
		}
	}

	// 3. Download from krcg.org to config directory
	obs_log(LOG_INFO, "[VTES] Downloading vtes.json from %s ...", krcg_url);
	char* cfg_dir = obs_module_config_path("");
	if (!cfg_dir) {
		obs_log(LOG_WARNING, "[VTES] Cannot get config dir — vtes_db stays empty");
		return;
	}
	std::string dest = std::string(cfg_dir) + "vtes.json";
	bfree(cfg_dir);
	if (downloadFile(krcg_url, dest)) {
		if (tf->vtes_db.load(dest)) {
			obs_log(LOG_INFO, "[VTES] Downloaded & loaded %zu cards from krcg.org",
				tf->vtes_db.size());
			return;
		}
		remove(dest.c_str());
		obs_log(LOG_WARNING, "[VTES] Downloaded vtes.json but failed to parse");
	} else {
		obs_log(LOG_WARNING, "[VTES] Failed to download vtes.json from krcg.org");
	}
}

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

	// 2. Load vtes.json into the full card database (local → download fallback)
	loadVtesJson(tf);

	return !tf->card_info_by_id.empty();
}

// ─── Temporal smoothing ─────────────────────────────────────────────────────

static void updateTemporalTracks(
	struct filter_data *tf,
	const std::vector<vtes_detection::OBBObject>& current_detections,
	std::vector<vtes_detection::OBBObject>& output_objects)
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
				bool current_weak = track.identity_total < 5 ||
				    (track.identity_total > 0 &&
				     (float)track.identity_hits / track.identity_total < 0.4f);
				bool new_much_better = det.prob > track.confidence + 0.2f;
				if (current_weak || new_much_better) {
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
			vtes_detection::OBBObject obj;
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
	return tf_text("VTES Card Detector (OBB)");
}

obs_properties_t *detect_filter_obb_properties(void *data)
{
	obs_properties_t *props = obs_properties_create();

	obs_properties_add_bool(props, "preview", tf_text("Preview"));
	obs_properties_add_bool(props, "always_active", tf_text("AlwaysActive"));

	// Idioma del plugin (nativo: espanol). Se guarda para las siguientes
	// cargas y las etiquetas del panel se refrescan al reabrir el dialogo.
	obs_property_t *p_lang = obs_properties_add_list(props, "language", tf_text("Language"),
							 OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p_lang, "Español", "es");
	obs_property_list_add_string(p_lang, "English", "en");
	obs_property_set_modified_callback2(
		p_lang,
		[](void *, obs_properties_t *, obs_property_t *, obs_data_t *settings) {
			tf_i18n_set(obs_data_get_string(settings, "language"));
			return true;
		},
		nullptr);

	obs_properties_add_float_slider(props, "threshold", tf_text("ConfThreshold"), 0.0,
					1.0, 0.025);

	obs_property_t *p_det_mode =
		obs_properties_add_list(props, "detection_mode", tf_text("DetectionMode"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p_det_mode, tf_text("Shape (Contour)"), DETECT_MODE_CONTOUR);
	obs_property_list_add_string(p_det_mode, tf_text("AI Model (ONNX)"), DETECT_MODE_ONNX);

	obs_property_t *p_inf_dev =
		obs_properties_add_list(props, "inference_device", tf_text("InferenceDevice"),
				OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_inf_dev, tf_text("CPU"), INFERENCE_CPU);
#if _WIN32
	obs_property_list_add_string(p_inf_dev, tf_text("GPU (DirectML)"), INFERENCE_DML);
	obs_property_list_add_string(p_inf_dev, tf_text("GPU (CUDA)"), INFERENCE_CUDA);
#endif

	obs_properties_add_bool(props, "classifier_enabled", tf_text("TypeClassifier"));
	
	// OCR
	obs_properties_add_bool(props, "ocr_enabled", tf_text("OcrEnabled"));
	obs_properties_add_bool(props, "upscaler_enabled", tf_text("NisUpscale"));
	obs_properties_add_bool(props, "vampire_cuda_enabled", tf_text("OvalRectClassifier"));

	// Detection interval (0 = continuous, no logging)
	obs_properties_add_float_slider(props, "detection_interval_seconds",
		tf_text("DetectionInterval"),
		0.0, 5.0, 0.1);

	// Click-to-overlay: manual from web UI
	// Desplazamiento vertical de los rulings. Se expone como porcentaje y no
	// como pixeles porque el rango de OBS es fijo, mientras que lo que se
	// puede desplazar depende de cada carta: con dos rulings apenas se mueve
	// y con dieciocho se mueve mucho.
	obs_properties_add_int_slider(props, "rulings_scroll",
	                              tf_text("RulingsScroll"), 0, 100, 1);

	obs_properties_add_bool(props, "click_to_overlay",
		tf_text("ClickToOverlay"));

	// Card Search Web Server + Detection History
	// OBS 32 marca obs_properties_add_button como desusado: button2 con priv
	// nullptr es identico (si priv es nulo, el callback recibe el data de la
	// fuente, que es lo que hacia la version antigua).
	obs_properties_add_button2(props, "card_search_btn", tf_text("OpenCardSearch"),
		[](obs_properties_t *, obs_property_t *, void *data) {
			struct detect_filter_obb *tf = reinterpret_cast<detect_filter_obb *>(data);
			if (!tf->web_server || !tf->web_server->is_running()) {
				tf->web_server = std::make_unique<WebServer>();
				tf->web_server->start(tf->web_server_port, &tf->vtes_db);
			}
			if (tf->web_server && tf->web_server->is_running()) {
				std::string url = "http://localhost:" + std::to_string(tf->web_server->port());
#ifdef _WIN32
				int len = MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, nullptr, 0);
				if (len > 0) {
					std::wstring wurl(len, L'\0');
					MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, &wurl[0], len);
					HINSTANCE result = ShellExecuteW(NULL, L"open", wurl.c_str(), NULL, NULL, SW_SHOWNORMAL);
					if ((INT_PTR)result <= 32) {
						obs_log(LOG_WARNING, "Failed to open browser (error %d)", (INT_PTR)result);
					}
				}
#else
				std::string cmd = "xdg-open " + url;
				std::thread([cmd]() { system(cmd.c_str()); }).detach();
#endif
			}
			return true;
		}, nullptr);

	std::string basic_info =
		std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
	obs_properties_add_text(props, "info", basic_info.c_str(), OBS_TEXT_INFO);

	// ─── Game State: Player configuration ───────────────────────────────
	obs_properties_add_int_slider(props, "player_count", tf_text("PlayerCount"), 4, 5, 1);
	obs_properties_add_bool(props, "show_zones", tf_text("ShowZones"));
	obs_properties_add_bool(props, "big_card", tf_text("BigCard"));

	for (int i = 0; i < filter_data::MAX_PLAYERS; i++) {
		std::string prefix = "player_" + std::to_string(i);
		std::string label = std::string(tf_text("Player")) + " " + std::to_string(i + 1);

		obs_properties_t *grp = obs_properties_create();
		obs_properties_add_text(grp, (prefix + "_name").c_str(), tf_text("PlayerName"), OBS_TEXT_DEFAULT);
		obs_properties_add_int(grp, (prefix + "_pool").c_str(), tf_text("InitialPool"), 0, 99, 1);
		obs_properties_add_bool(grp, (prefix + "_active").c_str(), tf_text("PlayerActive"));

		obs_properties_add_float(grp, (prefix + "_minion_x").c_str(), "Minion X (norm)", 0.0, 1.0, 0.01);
		obs_properties_add_float(grp, (prefix + "_minion_y").c_str(), "Minion Y (norm)", 0.0, 1.0, 0.01);
		obs_properties_add_float(grp, (prefix + "_minion_w").c_str(), "Minion W (norm)", 0.0, 1.0, 0.01);
		obs_properties_add_float(grp, (prefix + "_minion_h").c_str(), "Minion H (norm)", 0.0, 1.0, 0.01);
		obs_properties_add_int_slider(grp, (prefix + "_minion_angle").c_str(), tf_text("MinionAngle"), -180, 180, 1);

		obs_properties_add_float(grp, (prefix + "_crypt_x").c_str(), "Crypt X (norm)", 0.0, 1.0, 0.01);
		obs_properties_add_float(grp, (prefix + "_crypt_y").c_str(), "Crypt Y (norm)", 0.0, 1.0, 0.01);
		obs_properties_add_float(grp, (prefix + "_crypt_w").c_str(), "Crypt W (norm)", 0.0, 1.0, 0.01);
		obs_properties_add_float(grp, (prefix + "_crypt_h").c_str(), "Crypt H (norm)", 0.0, 1.0, 0.01);
		obs_properties_add_int_slider(grp, (prefix + "_crypt_angle").c_str(), tf_text("CryptAngle"), -180, 180, 1);

		obs_properties_add_group(props, (prefix + "_group").c_str(),
			label.c_str(), OBS_GROUP_NORMAL, grp);
	}

	UNUSED_PARAMETER(data);
	return props;
}

void detect_filter_obb_defaults(obs_data_t *settings)
{
#if _WIN32
	obs_data_set_default_string(settings, "inference_device", INFERENCE_CUDA);
#else
	obs_data_set_default_string(settings, "inference_device", INFERENCE_CPU);
#endif
	obs_data_set_default_bool(settings, "sort_tracking", false);
	obs_data_set_default_bool(settings, "preview", true);
	obs_data_set_default_string(settings, "language", tf_i18n_code());
	obs_data_set_default_bool(settings, "always_active", true);
	obs_data_set_default_double(settings, "threshold", 0.5);
	obs_data_set_default_string(settings, "detection_mode", DETECT_MODE_ONNX);
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
	obs_data_set_default_bool(settings, "ocr_enabled", true);
	obs_data_set_default_bool(settings, "upscaler_enabled", false);
	obs_data_set_default_bool(settings, "vampire_cuda_enabled", true);
	obs_data_set_default_bool(settings, "click_to_overlay", true);
	obs_data_set_default_double(settings, "detection_interval_seconds", 0.0);
	obs_data_set_default_double(settings, "clf_oval_weight", 0.5);
	obs_data_set_default_double(settings, "clf_color_weight", 0.3);
	obs_data_set_default_double(settings, "clf_contour_weight", 0.2);
	obs_data_set_default_double(settings, "clf_oval_thresh", 0.6);
	obs_data_set_default_double(settings, "clf_color_thresh", 0.55);
	obs_data_set_default_double(settings, "clf_oval_reject_master", 0.4);

	// Game state defaults
	obs_data_set_default_int(settings, "player_count", 5);
	obs_data_set_default_bool(settings, "show_zones", true);
	obs_data_set_default_bool(settings, "big_card", false);
	for (int i = 0; i < filter_data::MAX_PLAYERS; i++) {
		std::string p = "player_" + std::to_string(i);
		obs_data_set_default_string(settings, (p + "_name").c_str(), ("Jugador " + std::to_string(i + 1)).c_str());
		obs_data_set_default_int(settings, (p + "_pool").c_str(), 30);
		obs_data_set_default_bool(settings, (p + "_active").c_str(), true);

		// Default zones: divide screen width into MAX_PLAYERS columns
		float col_w = 1.0f / filter_data::MAX_PLAYERS;
		float col_x = i * col_w;
		obs_data_set_default_double(settings, (p + "_minion_x").c_str(), col_x);
		obs_data_set_default_double(settings, (p + "_minion_y").c_str(), 0.0);
		obs_data_set_default_double(settings, (p + "_minion_w").c_str(), col_w);
		obs_data_set_default_double(settings, (p + "_minion_h").c_str(), 0.45);
		obs_data_set_default_double(settings, (p + "_crypt_x").c_str(), col_x);
		obs_data_set_default_double(settings, (p + "_crypt_y").c_str(), 0.55);
		obs_data_set_default_double(settings, (p + "_crypt_w").c_str(), col_w);
		obs_data_set_default_double(settings, (p + "_crypt_h").c_str(), 0.45);
		obs_data_set_default_int(settings, (p + "_minion_angle").c_str(), 0);
		obs_data_set_default_int(settings, (p + "_crypt_angle").c_str(), 0);
	}
}

void detect_filter_obb_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_DEBUG, "VTES OBB filter update");

	struct detect_filter_obb *tf = reinterpret_cast<detect_filter_obb *>(data);

	tf->isDisabled = true;

	tf->preview = obs_data_get_bool(settings, "preview");
	tf->always_active = obs_data_get_bool(settings, "always_active");
	tf->click_to_overlay = obs_data_get_bool(settings, "click_to_overlay");
	tf->conf_threshold = (float)obs_data_get_double(settings, "threshold");
	tf->rulings_scroll = (int)obs_data_get_int(settings, "rulings_scroll");
	tf->sortTracking = obs_data_get_bool(settings, "sort_tracking");
	size_t maxUnseenFrames = (size_t)obs_data_get_int(settings, "max_unseen_frames");
	if (tf->tracker.getMaxUnseenFrames() != maxUnseenFrames) {
		tf->tracker.setMaxUnseenFrames(maxUnseenFrames);
	}
	tf->showUnseenObjects = obs_data_get_bool(settings, "show_unseen_objects");
	tf->saveDetectionsPath = obs_data_get_string(settings, "save_detections_path");
	tf->detection_interval_ms = (int)(obs_data_get_double(settings, "detection_interval_seconds") * 1000.0);
	if (tf->detection_interval_ms < 0) tf->detection_interval_ms = 0;
	if (tf->detection_interval_ms > 5000) tf->detection_interval_ms = 5000;

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

	// Embedding matcher (load once on startup, log failure only once)
	{
		static bool embedder_logged_failure = false;
		if (!tf->embedder.is_loaded()) {
			char *modelPath = obs_module_file("vtes_embedder_1024d.onnx");
			char *binPath = obs_module_file("embeddings_1024d.bin");
			char *metaPath = obs_module_file("embeddings_1024d_meta.json");
			if (modelPath && binPath && metaPath) {
				obs_log(LOG_INFO, "[Embedding] Loading model: %s", modelPath);
				tf->embedder.set_inference_device(tf->inference_device_enum);
				tf->embedder.load(modelPath, binPath, metaPath);
				tf->embedder.set_threshold(0.80f);
			} else if (!embedder_logged_failure) {
				embedder_logged_failure = true;
				obs_log(LOG_WARNING, "[Embedding] Model files missing — "
					"vtes_embedder_1024d.onnx=%s embeddings_1024d.bin=%s embeddings_1024d_meta.json=%s",
					modelPath ? "OK" : "MISSING",
					binPath ? "OK" : "MISSING",
					metaPath ? "OK" : "MISSING");
			}
			bfree(modelPath);
			bfree(binPath);
			bfree(metaPath);
		}
	}
	// Apply card types if not already done
	applyCardTypesToEmbedder(tf);

	// Vision-based type classifier (load once on startup)
	if (!tf->type_classifier_session) {
		loadVisionTypeClassifier(tf);
	}

	// Per-type embedding models (load once on startup, or after embedder loaded)
	{
		static bool per_type_attempted = false;
		if (!per_type_attempted) {
			per_type_attempted = true;
			loadPerTypeEmbedders(tf);
		}
	}

	// ─── Game state: player config ─────────────────────────────────
	{
		int new_count = (int)obs_data_get_int(settings, "player_count");
		if (new_count < 4) new_count = 4;
		if (new_count > filter_data::MAX_PLAYERS) new_count = filter_data::MAX_PLAYERS;
		tf->player_count = new_count;
		tf->show_zones = obs_data_get_bool(settings, "show_zones");
		tf->big_card = obs_data_get_bool(settings, "big_card");

		for (int i = 0; i < filter_data::MAX_PLAYERS; i++) {
			std::string p = "player_" + std::to_string(i);
			tf->players[i].name = obs_data_get_string(settings, (p + "_name").c_str());
			tf->players[i].pool = (int)obs_data_get_int(settings, (p + "_pool").c_str());
			tf->players[i].minion_zone.nx = (float)obs_data_get_double(settings, (p + "_minion_x").c_str());
			tf->players[i].minion_zone.ny = (float)obs_data_get_double(settings, (p + "_minion_y").c_str());
			tf->players[i].minion_zone.nw = (float)obs_data_get_double(settings, (p + "_minion_w").c_str());
			tf->players[i].minion_zone.nh = (float)obs_data_get_double(settings, (p + "_minion_h").c_str());
			tf->players[i].crypt_zone.nx = (float)obs_data_get_double(settings, (p + "_crypt_x").c_str());
			tf->players[i].crypt_zone.ny = (float)obs_data_get_double(settings, (p + "_crypt_y").c_str());
			tf->players[i].crypt_zone.nw = (float)obs_data_get_double(settings, (p + "_crypt_w").c_str());
			tf->players[i].crypt_zone.nh = (float)obs_data_get_double(settings, (p + "_crypt_h").c_str());
			tf->players[i].minion_zone.angle = (float)obs_data_get_int(settings, (p + "_minion_angle").c_str());
			tf->players[i].crypt_zone.angle = (float)obs_data_get_int(settings, (p + "_crypt_angle").c_str());

			// Un jugador que sale de la partida (ousted) deja su sitio libre:
			// su zona deja de asignar cartas y se le quitan las ya asignadas.
			const bool was_active = tf->players[i].active;
			tf->players[i].active = obs_data_get_bool(settings, (p + "_active").c_str());
			if (was_active && !tf->players[i].active) {
				std::lock_guard<std::mutex> lock(tf->game_state_lock);
				for (auto it = tf->tracked_cards.begin(); it != tf->tracked_cards.end();) {
					if (it->second.player_idx == i)
						it = tf->tracked_cards.erase(it);
					else
						++it;
				}
			}
		}
	}

	// Ensure card database is loaded (required for OCR)
	if (tf->vtes_db.is_empty()) {
		loadCardInfo(tf);
	}

	// OCR reader: respect user setting
	{
		bool ocr_wanted = obs_data_get_bool(settings, "ocr_enabled");
		if (ocr_wanted != tf->ocr_enabled) {
			tf->ocr_enabled = ocr_wanted;
			if (ocr_wanted && !tf->ocr_reader) {
				initOcrReader(tf);
			} else if (!ocr_wanted && tf->ocr_reader) {
				tf->ocr_reader.reset();
			}
		}
	}

	// NIS CUDA upscaler toggle
	{
		bool upscaler_wanted = obs_data_get_bool(settings, "upscaler_enabled");
		if (upscaler_wanted != tf->upscaler_enabled) {
			tf->upscaler_enabled = upscaler_wanted;
			if (tf->ocr_reader) {
				tf->ocr_reader->set_upscaler_enabled(upscaler_wanted && tf->ocr_reader->is_upscaler_enabled());
			}
		}
	}

	// CUDA oval/rect vampire classifier toggle
	{
		bool vampire_cuda_wanted = obs_data_get_bool(settings, "vampire_cuda_enabled");
		if (vampire_cuda_wanted != tf->vampire_cuda_enabled) {
			tf->vampire_cuda_enabled = vampire_cuda_wanted;
			obs_log(LOG_INFO, "[CUDA] Vampire oval classifier %s",
				vampire_cuda_wanted ? "ENABLED" : "DISABLED");
		}
	}

	// Classifier (only reinit when enabled state changes or config changes)
	{
		bool clf_wanted = obs_data_get_bool(settings, "classifier_enabled");
		if (clf_wanted != tf->classifier_enabled) {
			tf->classifier_enabled = clf_wanted;
			if (clf_wanted) {
				tf->classifierConfig.inference_device = tf->inference_device_enum;
				tf->classifierConfig.ovalWeight = (float)obs_data_get_double(settings, "clf_oval_weight");
				tf->classifierConfig.colorWeight = (float)obs_data_get_double(settings, "clf_color_weight");
				tf->classifierConfig.contourWeight = (float)obs_data_get_double(settings, "clf_contour_weight");
				tf->classifierConfig.ovalThreshold = (float)obs_data_get_double(settings, "clf_oval_thresh");
				tf->classifierConfig.colorThreshold = (float)obs_data_get_double(settings, "clf_color_thresh");
				tf->classifierConfig.ovalRejectForMaster = (float)obs_data_get_double(settings, "clf_oval_reject_master");
				{
					std::unique_lock<std::mutex> lock(tf->modelMutex);
					tf->classifier = std::make_unique<vtes_classifier::VTESCardClassifier>(tf->classifierConfig);
					tf->cropExtractor = std::make_unique<vtes_classifier::CardCropExtractor>();
					obs_log(LOG_INFO, "VTES Card Classifier initialized");
				}
			} else {
				std::unique_lock<std::mutex> lock(tf->modelMutex);
				tf->classifier.reset();
				tf->cropExtractor.reset();
			}
		}
	}

	bool modeChanged = (tf->detectionMode != newDetectMode);
	tf->detectionMode = newDetectMode;

	if (tf->detectionMode == DETECT_MODE_CONTOUR) {
		if (modeChanged) {
			std::unique_lock<std::mutex> lock(tf->modelMutex);
			if (tf->yolo_detector) {
				tf->yolo_detector.reset();
				obs_log(LOG_INFO, "Contour mode: ONNX model released");
			}
			obs_log(LOG_INFO, "VTES OBB Filter set to Shape (Contour) mode");
		}
		tf->isDisabled = false;
		return;
	}

	// ONNX mode
	const std::string newInfDev = obs_data_get_string(settings, "inference_device");
	const uint32_t newNumThreads = (uint32_t)obs_data_get_int(settings, "numThreads");
	const std::string newInfDevSafe = newInfDev.empty() ? INFERENCE_DML : newInfDev;
	const uint32_t newNumThreadsSafe = newNumThreads > 0 ? newNumThreads : 0;

	// Save old values before overwriting — needed for change detection below
	std::string old_inference_device = tf->inference_device;
	uint32_t old_numThreads = tf->numThreads;

	tf->inference_device = newInfDevSafe;
	tf->inference_device_enum = deviceStringToEnum(newInfDevSafe);
	tf->numThreads = newNumThreadsSafe;

	bool reinitialize = modeChanged;
	if (old_inference_device != newInfDevSafe || old_numThreads != newNumThreadsSafe) {
		reinitialize = true;
	}

	// If model already loaded and nothing actually changed, skip reload
	if (reinitialize && tf->yolo_detector) {
		bool deviceSame = (old_inference_device == newInfDevSafe);
		bool threadsSame = (old_numThreads == newNumThreadsSafe);
		obs_log(LOG_INFO,
			"ONNX reload check: modeChanged=%d deviceSame=%d threadsSame=%d "
			"old='%s' new='%s'",
			(int)modeChanged, (int)deviceSame, (int)threadsSame,
			old_inference_device.c_str(), newInfDevSafe.c_str());
		if (!modeChanged && deviceSame && threadsSame) {
			reinitialize = false;
			obs_log(LOG_INFO, "ONNX reload: nothing changed, keeping existing model");
		} else {
			obs_log(LOG_INFO, "ONNX reload: change detected, reloading model");
		}
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

		std::string model_path(modelFilepath_rawPtr);
		bfree(modelFilepath_rawPtr);

		int onnxruntime_device_id_ = 0;

		try {
			if (tf->yolo_detector) {
				tf->yolo_detector.reset();
			}

#if defined(TENSORRT_AVAILABLE) && !defined(ORT_CPU_ONLY)
			if (tf->inference_device_enum == InferenceDevice::CUDA) {
				// Use TensorRTDetector for GPU inference (tensor cores)
				char *enginePath_rawPtr = obs_module_file("models/vtes.engine");
				if (enginePath_rawPtr == nullptr) {
					obs_log(LOG_ERROR, "TensorRT engine file not found (models/vtes.engine)");
					throw std::runtime_error("Missing TensorRT engine file");
				}
				std::string engine_path(enginePath_rawPtr);
				bfree(enginePath_rawPtr);
				tf->yolo_detector =
					std::make_unique<vtes_detection::TensorRTDetector>(
						engine_path, cv::Size(1024, 1024),
						onnxruntime_device_id_,
						tf->conf_threshold);
				obs_log(LOG_INFO, "Using TensorRT detector: %s", engine_path.c_str());
			} else {
				tf->yolo_detector =
					std::make_unique<vtes_detection::YOLODetector>(
						model_path, cv::Size(1024, 1024),
						tf->inference_device_enum,
						onnxruntime_device_id_,
						tf->conf_threshold);
			}
#else
			// TensorRT no disponible en este build — si el usuario seleccionó
			// CUDA, forzamos DirectML (YOLODetector + CUDA EP requiere DLLs
			// que no tenemos y fallaría con 'Failed to load shared library')
			{
				InferenceDevice effective_device = tf->inference_device_enum;
				if (effective_device == InferenceDevice::CUDA) {
					obs_log(LOG_WARNING,
						"TensorRT not available in this build, "
						"falling back to DirectML");
					effective_device = InferenceDevice::DirectML;
					tf->inference_device = INFERENCE_DML;
					tf->inference_device_enum = effective_device;
				}
				tf->yolo_detector =
					std::make_unique<vtes_detection::YOLODetector>(
						model_path, cv::Size(1024, 1024),
						effective_device,
						onnxruntime_device_id_,
						tf->conf_threshold);
			}
#endif

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
							tf->yolo_detector->setOutputShape(num_dets, num_feats);
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

			// Only set error to empty if it was previously non-empty
			// to avoid triggering unnecessary update callbacks
			const char* cur_err = obs_data_get_string(settings, "error");
			if (cur_err && strlen(cur_err) > 0) {
				obs_data_set_string(settings, "error", "");
			}
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "Failed to load OBB model: %s", e.what());

#if defined(TENSORRT_AVAILABLE) && !defined(ORT_CPU_ONLY)
			// If TensorRT failed, fall back to DirectML without disabling
			if (tf->inference_device_enum == InferenceDevice::CUDA) {
				obs_log(LOG_WARNING,
					"TensorRT loading failed, falling back to DirectML");
				tf->inference_device = INFERENCE_DML;
				tf->inference_device_enum = deviceStringToEnum(tf->inference_device);
				try {
					char *fallback_path = obs_module_file("models/vtes.onnx");
					if (fallback_path) {
						tf->yolo_detector =
							std::make_unique<vtes_detection::YOLODetector>(
								std::string(fallback_path),
								cv::Size(1024, 1024),
								tf->inference_device_enum, 0,
								tf->conf_threshold);
						bfree(fallback_path);
					}
				} catch (const std::exception &fallback_err) {
					obs_log(LOG_ERROR, "Fallback to DirectML also failed: %s",
						fallback_err.what());
					tf->isDisabled = true;
					tf->yolo_detector.reset();
					return;
				}
			} else {
				tf->isDisabled = true;
				tf->yolo_detector.reset();
				return;
			}
#else
			tf->isDisabled = true;
			tf->yolo_detector.reset();
			return;
#endif
		}
	}

	if (tf->yolo_detector) {
		tf->yolo_detector->setConfThreshold(tf->conf_threshold);
	}

	if (reinitialize) {
		obs_log(LOG_INFO, "VTES OBB Filter Options:");
		obs_log(LOG_INFO, "  Mode: ONNX (AI Model)");
		obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
		obs_log(LOG_INFO, "  Inference Device: %s", tf->inference_device.c_str());
		obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
		obs_log(LOG_INFO, "  Show Boxes: %s", tf->preview ? "true" : "false");
		obs_log(LOG_INFO, "  Always Active: %s", tf->always_active ? "true" : "false");
		obs_log(LOG_INFO, "  Show Zones: %s", tf->show_zones ? "true" : "false");
		obs_log(LOG_INFO, "  Big Card: %s", tf->big_card ? "true" : "false");
		obs_log(LOG_INFO, "  Threshold: %.2f", tf->conf_threshold);
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

// ─── Async OCR worker forward declarations ──────────────────────────
static void ocr_worker_func(detect_filter_obb* tf);
static void start_ocr_worker(detect_filter_obb* tf);
static void stop_ocr_worker(detect_filter_obb* tf);
static void submit_ocr_jobs(detect_filter_obb* tf,
							const std::vector<cv::Mat>& card_regions,
							const std::vector<int>& track_ids,
							const std::vector<std::string>& type_filters);
static void drain_ocr_results(detect_filter_obb* tf,
							  std::vector<vtes_detection::OBBObject>& objects);

// ─── Modo realizador: pendientes + feedback forward declarations ────
static void addPendingDetection(detect_filter_obb *tf,
								const vtes_detection::OBBObject &obj,
								const cv::Mat &frame, int64_t now_ns);
static void recordDirectorFeedback(detect_filter_obb *tf, const char *kind,
								   const filter_data::PendingDetection *chosen,
								   const std::vector<filter_data::PendingDetection> &pool);

void *detect_filter_obb_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "VTES OBB filter created");
	void *data = bmalloc(sizeof(struct detect_filter_obb));
	struct detect_filter_obb *tf = new (data) detect_filter_obb();

	tf->source = source;
	tf->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	tf->lastDetectedObjectId = -1;

	// Read inference device from settings before loading models
	tf->inference_device = obs_data_get_string(settings, "inference_device");
	if (tf->inference_device.empty()) {
#if _WIN32
		tf->inference_device = INFERENCE_DML;
#else
		tf->inference_device = INFERENCE_CPU;
#endif
	}
	tf->inference_device_enum = deviceStringToEnum(tf->inference_device);

	// Pre-load card info and embedding model
	loadCardInfo(tf);
	if (!tf->embedder.is_loaded()) {
		char *modelPath = obs_module_file("vtes_embedder_1024d.onnx");
		char *binPath = obs_module_file("embeddings_1024d.bin");
		char *metaPath = obs_module_file("embeddings_1024d_meta.json");
		if (modelPath && binPath && metaPath) {
			obs_log(LOG_INFO, "[Embedding] Loading model on startup: %s", modelPath);
			tf->embedder.set_inference_device(tf->inference_device_enum);
			tf->embedder.load(modelPath, binPath, metaPath);
			tf->embedder.set_threshold(0.80f);
		} else {
			obs_log(LOG_WARNING, "[Embedding] Model files missing at startup — "
				"vtes_embedder_1024d.onnx=%s embeddings_1024d.bin=%s embeddings_1024d_meta.json=%s",
				modelPath ? "OK" : "MISSING",
				binPath ? "OK" : "MISSING",
				metaPath ? "OK" : "MISSING");
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

	// Start embedded web server (port 8080 for card search UI + detection history)
	tf->web_server = std::make_unique<WebServer>();
	tf->web_server->get_detections_fn = [tf]() -> std::vector<DetectionEntry> {
		std::lock_guard<std::mutex> lock(tf->detection_history_lock);
		std::vector<DetectionEntry> out;
		for (const auto& d : tf->detection_history)
			out.push_back({d.card_name, d.card_id, d.card_url, d.timestamp_ns});
		return out;
	};
	tf->web_server->on_overlay_fn = [tf](const std::string& url) {
		{
			std::lock_guard<std::mutex> lock(tf->game_state_lock);
			tf->current_overlay_card_url = url;
			tf->current_overlay_card_id.clear();
		}
		obs_log(LOG_INFO, "[WebServer] Overlay manual: %s", url.c_str());
	};
	tf->web_server->on_overlay_clear_fn = [tf]() {
		{
			std::lock_guard<std::mutex> lock(tf->game_state_lock);
			tf->current_overlay_card_url.clear();
			tf->current_overlay_card_id.clear();
		}
		obs_log(LOG_INFO, "[WebServer] Overlay manual limpiado");
	};
	tf->web_server->on_pool_fn = [tf](int player_idx, int delta) {
		if (player_idx >= 0 && player_idx < tf->player_count) {
			tf->players[player_idx].pool += delta;
			if (tf->players[player_idx].pool < 0)
				tf->players[player_idx].pool = 0;
			if (tf->players[player_idx].pool > 99)
				tf->players[player_idx].pool = 99;
			obs_log(LOG_INFO, "[WebServer] Pool: %s = %d (delta %+d)",
				tf->players[player_idx].name.c_str(),
				tf->players[player_idx].pool, delta);
		}
	};
	tf->web_server->get_pool_fn = [tf]() -> std::vector<std::pair<std::string,int>> {
		std::vector<std::pair<std::string,int>> out;
		for (int i = 0; i < tf->player_count; i++)
			out.push_back({tf->players[i].name, tf->players[i].pool});
		return out;
	};
	// Estado de partida: pool + minions con su estado de giro por jugador.
	tf->web_server->get_game_fn = [tf]() -> nlohmann::json {
		nlohmann::json out;
		nlohmann::json players_json = nlohmann::json::array();
		std::lock_guard<std::mutex> lock(tf->game_state_lock);
		const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
		for (int i = 0; i < tf->player_count; i++) {
			const filter_data::PlayerState &pl = tf->players[i];
			nlohmann::json pj;
			pj["index"] = i;
			pj["name"] = pl.name;
			pj["pool"] = pl.pool;
			pj["active"] = pl.active;
			pj["zone_angle"] = pl.minion_zone.angle;
			int locked_n = 0, unlocked_n = 0, crypt_n = 0;
			nlohmann::json minions = nlohmann::json::array();
			for (const auto &kv : tf->tracked_cards) {
				const filter_data::TrackedCardState &tc = kv.second;
				if (tc.player_idx != i)
					continue;
				if (tc.in_crypt) {
					crypt_n++;
					continue;
				}
				nlohmann::json mj;
				mj["name"] = tc.card_name;
				mj["card_id"] = tc.card_id;
				mj["locked"] = (tc.lock_state == filter_data::CardLockState::Locked);
				mj["rel_angle"] = tc.last_rel_angle;
				mj["seen_ms_ago"] = (tc.last_seen_ns > 0)
					? (now_ns - tc.last_seen_ns) / 1000000LL : 0LL;
				int capacity = 0;
				auto cit = tf->card_info_by_id.find(tc.card_id);
				if (cit != tf->card_info_by_id.end())
					capacity = cit->second.capacity;
				mj["capacity"] = capacity;
				if (mj["locked"].get<bool>())
					locked_n++;
				else
					unlocked_n++;
				minions.push_back(std::move(mj));
			}
			pj["minions"] = std::move(minions);
			pj["locked_count"] = locked_n;
			pj["unlocked_count"] = unlocked_n;
			pj["crypt_count"] = crypt_n;
			players_json.push_back(std::move(pj));
		}
		out["players"] = std::move(players_json);
		return out;
	};

	// Carpeta de datos de feedback del realizador (entrenamiento futuro),
	// bajo la config del plugin: .../plugin_config/vtes-card-scanner/feedback
	{
		char *cfg_dir = obs_module_config_path("");
		if (cfg_dir) {
			tf->feedback_dir = std::string(cfg_dir) + "feedback";
			bfree(cfg_dir);
			obs_log(LOG_INFO, "[Director] Datos de feedback en: %s",
				tf->feedback_dir.c_str());
		}
	}

	// Modo realizador: cola de detecciones pendientes (la web las muestra)
	// y feedback de entrenamiento (elegidas/descartadas con recortes).
	tf->web_server->get_pending_fn = [tf]() -> nlohmann::json {
		nlohmann::json out;
		out["director_mode"] = tf->click_to_overlay;
		out["feedback_dir"] = tf->feedback_dir;
		nlohmann::json arr = nlohmann::json::array();
		const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
		std::lock_guard<std::mutex> lock(tf->pending_lock);
		for (const auto &p : tf->pending_detections) {
			nlohmann::json pj;
			pj["id"] = p.id;
			pj["card_id"] = p.card_id;
			pj["card_name"] = p.card_name;
			pj["card_url"] = p.card_url;
			pj["confidence"] = p.confidence;
			pj["hits"] = p.hits;
			pj["age_ms"] = (now_ns - p.first_seen_ns) / 1000000LL;
			pj["fresh_ms"] = (now_ns - p.last_seen_ns) / 1000000LL;
			arr.push_back(std::move(pj));
		}
		out["pending"] = std::move(arr);
		return out;
	};
	// Elegir una pendiente: se muestra en el overlay y la cola se limpia.
	tf->web_server->on_pending_select_fn = [tf](uint64_t id) -> nlohmann::json {
		nlohmann::json out;
		filter_data::PendingDetection chosen;
		std::vector<filter_data::PendingDetection> pool;
		bool found = false;
		{
			std::lock_guard<std::mutex> lock(tf->pending_lock);
			for (const auto &p : tf->pending_detections) {
				if (p.id == id) {
					chosen = p;
					found = true;
				}
			}
			if (!found) {
				out["ok"] = false;
				out["error"] = "not_found";
				return out;
			}
			pool.assign(tf->pending_detections.begin(), tf->pending_detections.end());
			tf->pending_detections.clear();
		}
		{
			std::lock_guard<std::mutex> lock(tf->game_state_lock);
			tf->current_overlay_card_url = chosen.card_url;
			tf->current_overlay_card_id = chosen.card_id;
		}
		recordDirectorFeedback(tf, "selection", &chosen, pool);
		obs_log(LOG_INFO, "[Director] Mostrado \"%s\" (elegida entre %zu; cola limpiada)",
			chosen.card_name.c_str(), pool.size());
		out["ok"] = true;
		out["card_name"] = chosen.card_name;
		return out;
	};
	// Descartar la cola sin mostrar nada (tambien genera feedback).
	tf->web_server->on_pending_clear_fn = [tf]() -> nlohmann::json {
		std::vector<filter_data::PendingDetection> pool;
		{
			std::lock_guard<std::mutex> lock(tf->pending_lock);
			pool.assign(tf->pending_detections.begin(), tf->pending_detections.end());
			tf->pending_detections.clear();
		}
		recordDirectorFeedback(tf, "dismissed", nullptr, pool);
		obs_log(LOG_INFO, "[Director] %zu detecciones descartadas (feedback guardado)",
			pool.size());
		nlohmann::json out;
		out["ok"] = true;
		out["dismissed"] = pool.size();
		return out;
	};
	tf->web_server->get_pending_crop_fn = [tf](uint64_t id) -> std::string {
		std::lock_guard<std::mutex> lock(tf->pending_lock);
		for (const auto &p : tf->pending_detections)
			if (p.id == id)
				return std::string((const char *)p.crop_jpg.data(), p.crop_jpg.size());
		return std::string();
	};
	bool ws_ok = tf->web_server->start(tf->web_server_port, &tf->vtes_db);
	if (ws_ok) {
		obs_log(LOG_INFO, "[WebServer] Started on http://localhost:%d (filter=%s)",
			tf->web_server_port, obs_source_get_name(tf->source));
		if (!tf->web_server_notified) {
			tf->web_server_notified = true;
#ifdef _WIN32
			std::string msg = std::string(tf_text("PopupAdded")) + "\n\n" +
				tf_text("PopupServer") + " http://localhost:" +
				std::to_string(tf->web_server_port) + "\n\n" +
				tf_text("PopupSearch") + "\n" +
				tf_text("PopupAlwaysActive") + "\n" +
				tf_text("PopupInterval") + "\n" +
				tf_text("PopupIntervalZero");
			// MessageBoxW + conversion explicita desde UTF-8: MessageBoxA
			// interpretaria los bytes UTF-8 ("añadido", "detección") como
			// ANSI y las tildes/eñes saldrian como mojibake (aÃ±adido).
			int wlen = MultiByteToWideChar(CP_UTF8, 0, msg.c_str(), -1, nullptr, 0);
			if (wlen > 0) {
				std::wstring wmsg(wlen, L'\0');
				MultiByteToWideChar(CP_UTF8, 0, msg.c_str(), -1, &wmsg[0], wlen);
				wmsg.resize(wlen - 1);
				MessageBoxW(NULL, wmsg.c_str(), L"VTES Card Scanner",
					MB_OK | MB_ICONINFORMATION | MB_TASKMODAL);
			}
#else
			obs_log(LOG_INFO, "[WebServer] Filter created (no MessageBox on this platform)");
#endif
		}
	} else {
		obs_log(LOG_WARNING, "[WebServer] Failed to start (port %d may be in use)", tf->web_server_port);
	}

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

	// Start background OCR worker thread
	start_ocr_worker(tf);

	return tf;
}

// ─── Async OCR worker ────────────────────────────────────────────────
// Runs Tesseract on a background thread so video_tick never blocks on OCR.
// The worker owns the ocr_reader; video_tick submits jobs and consumes results.

static void ocr_worker_func(detect_filter_obb* tf)
{
	obs_log(LOG_INFO, "[OCR Worker] Started");
	while (tf->ocr_worker_active) {
		filter_data::OcrJob job;
		{
			std::unique_lock<std::mutex> lock(tf->ocr_queue_lock);
			tf->ocr_queue_cv.wait_for(lock, std::chrono::milliseconds(100),
				[&]{ return !tf->ocr_queue.empty() || !tf->ocr_worker_active; });
			if (!tf->ocr_worker_active) break;
			if (tf->ocr_queue.empty()) continue;
			job = std::move(tf->ocr_queue.front());
			tf->ocr_queue.pop();
		}

		if (job.card_region.empty() || !tf->ocr_reader) continue;

		std::string ocr_name, ocr_id;
		float ocr_conf = 0.0f;
		bool accepted = false;
		if (tf->ocr_reader->recognize(job.card_region, ocr_name, ocr_id, ocr_conf, job.type_filter)) {
			accepted = true;
			// Vampire card validation
			auto ci_it = tf->card_info_by_id.find(ocr_id);
			if (ci_it != tf->card_info_by_id.end()) {
				bool is_vampire = false;
				for (const auto& t : ci_it->second.types) {
					if (t == "Vampire") { is_vampire = true; break; }
				}
				if (is_vampire) {
					obs_log(LOG_INFO, "[OCR] Vampire validation for '%s' (id=%s)",
						ocr_name.c_str(), ocr_id.c_str());
					bool oval_ok = false;
					if (tf->vampire_cuda_enabled) {
						oval_ok = isVampireCUDA(job.card_region);
						if (!oval_ok)
							obs_log(LOG_INFO, "[OCR] Vampire REJECTED: CUDA oval classifier says RECT");
					} else {
						oval_ok = VtesOcrReader::hasVampireOval(job.card_region);
						if (!oval_ok)
							obs_log(LOG_INFO, "[OCR] Vampire REJECTED: no oval portrait found");
					}
					if (!oval_ok) {
						accepted = false;
					} else {
						int detected_cap = tf->ocr_reader->detectVampireCapacity(job.card_region);
						int expected_cap = ci_it->second.capacity;
						if (detected_cap < 0) {
							obs_log(LOG_WARNING, "[OCR] Vampire: could not read capacity, accepting anyway");
						} else if (detected_cap != expected_cap) {
							obs_log(LOG_INFO, "[OCR] Vampire REJECTED: capacity %d != expected %d",
								detected_cap, expected_cap);
							accepted = false;
						} else {
							obs_log(LOG_INFO, "[OCR] Vampire ACCEPTED: oval + capacity %d matches",
								detected_cap);
						}
					}
				}
			}
		}

		{
			std::lock_guard<std::mutex> lock(tf->ocr_result_lock);
			tf->ocr_completed.push_back({job.id, true, accepted,
				ocr_name, ocr_id, ocr_conf});
		}
	}
	obs_log(LOG_INFO, "[OCR Worker] Stopped");
}

static void start_ocr_worker(detect_filter_obb* tf)
{
	if (tf->ocr_worker_active) return;
	tf->ocr_worker_active = true;
	tf->ocr_worker = std::thread(ocr_worker_func, tf);
}

static void stop_ocr_worker(detect_filter_obb* tf)
{
	if (!tf->ocr_worker_active) return;
	tf->ocr_worker_active = false;
	tf->ocr_queue_cv.notify_all();
	if (tf->ocr_worker.joinable())
		tf->ocr_worker.join();
}

static void submit_ocr_jobs(detect_filter_obb* tf,
							const std::vector<cv::Mat>& card_regions,
							const std::vector<int>& track_ids,
							const std::vector<std::string>& type_filters)
{
	if (!tf->ocr_worker_active || !tf->ocr_reader) return;
	for (size_t i = 0; i < card_regions.size() && i < type_filters.size(); i++) {
		if (card_regions[i].empty()) continue;
		// El ticket del job ES el track id del objeto, para que
		// drain_ocr_results pueda devolver cada nombre a su carta.
		int64_t jid = track_ids[i];
		{
			std::lock_guard<std::mutex> lock(tf->ocr_queue_lock);
			tf->ocr_queue.push({jid, card_regions[i].clone(), type_filters[i]});
		}
		tf->ocr_queue_cv.notify_one();
	}
}

static void drain_ocr_results(detect_filter_obb* tf,
							  std::vector<vtes_detection::OBBObject>& objects)
{
	std::vector<filter_data::OcrJobResult> fresh;
	{
		std::lock_guard<std::mutex> lock(tf->ocr_result_lock);
		fresh.swap(tf->ocr_completed);
	}
	for (auto& result : fresh) {
		if (!result.completed) continue;
		if (result.accepted) {
			// El resultado del OCR volvio del worker sin saber a que objeto
			// corresponde: job.id es un ticket de la cola, no el track id.
			// Antes se aplicaba "al primer objeto que siga en ???", lo que
			// con dos o mas cartas en pantalla podia poner el nombre de una
			// en la tarjeta de otra. Ahora se aplica de verdad por track_id.
			int obj_idx = -1;
			for (size_t oi = 0; oi < objects.size(); ++oi) {
				if (objects[oi].id == result.id) { obj_idx = (int)oi; break; }
			}
			if (obj_idx < 0) {
				// El objeto asociado al job ya no esta en la lista de este
				// frame (desplazamiento de camara, carta retirada): se
				// descarta el resultado en vez de aplicarlo a cualquiera.
				obs_log(LOG_INFO, "[OCR] Async result descartado: track %d ya no presente",
						result.id);
				continue;
			}
			auto& obj = objects[obj_idx];
			obj.card_name = result.card_name;
			obj.card_id = result.card_id;
			obj.prob = result.confidence;
			obs_log(LOG_INFO, "[OCR] Async result: '%s' (id=%s, conf=%.2f, track=%d)",
					result.card_name.c_str(), result.card_id.c_str(),
					result.confidence, result.id);
			}
	}
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

		// Stop OCR worker before destroying any resources it may use
		stop_ocr_worker(tf);

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

// Colocacion del diseño del overlay dentro del frame: escala uniforme para
// que quepa entero (nada se recorta) y centrado; lo que sobre son bandas.
struct OverlayFit {
	double esc;
	int    ox, oy;
};

static OverlayFit overlayFit(int fw, int fh)
{
	const double esc = std::min((double)fw / kOverlayRefW, (double)fh / kOverlayRefH);
	const int dw = (int)(kOverlayRefW * esc + 0.5);
	const int dh = (int)(kOverlayRefH * esc + 0.5);
	return {esc, (fw - dw) / 2, (fh - dh) / 2};
}

// Una caja del diseño (pixeles 1536x1024) a pixeles del frame.
static cv::Rect designRect(const OverlayFit& fit, int x, int y, int w, int h)
{
	return cv::Rect(fit.ox + (int)(x * fit.esc + 0.5), fit.oy + (int)(y * fit.esc + 0.5),
	                (int)(w * fit.esc + 0.5), (int)(h * fit.esc + 0.5));
}

// Rectangulo donde se dibuja la camara: la imagen entera ajustada (sin
// recorte) dentro del hueco del diseño, centrada. Las zonas de jugador
// (coords de camara 0..1) se mapean con este mismo rectangulo.
static cv::Rect cameraFitRect(const OverlayFit& fit, int fw, int fh)
{
	const cv::Rect win = designRect(fit, kWinX, kWinY, kWinW, kWinH);
	if (win.width < 8 || win.height < 8)
		return cv::Rect(0, 0, fw, fh);

	const double esc = std::min((double)win.width / fw, (double)win.height / fh);
	const int cw = (int)(fw * esc + 0.5), ch = (int)(fh * esc + 0.5);
	return cv::Rect(win.x + (win.width - cw) / 2, win.y + (win.height - ch) / 2, cw, ch);
}

// Texto centrado dentro de una caja, encogiendo la fuente si no cabe a lo
// ancho (los nombres de jugador son texto libre). `scale` es la escala base.
static void drawTextInBox(cv::Mat& img, const std::string& text, const cv::Rect& box,
                          const cv::Scalar& color, double scale)
{
	if (text.empty() || box.width < 6 || box.height < 6)
		return;
	int baseline = 0;
	cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
	const int avail = box.width - std::max(4, box.width / 8);
	if (ts.width > avail && ts.width > 0)
		scale *= (double)avail / ts.width;
	ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
	cv::putText(img, text,
	            cv::Point(box.x + (box.width - ts.width) / 2,
	                      box.y + (box.height + ts.height) / 2),
	            cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

// Pega una imagen (carta) rotada y centrada en `center`, escalada a w x h.
// angle en grados, positivo = horario en pantalla (convencion de las zonas).
// Usa una mascara para que las esquinas del rectangulo rotado no tapen la
// camara; sustituye al addWeighted recto cuando la zona esta girada.
[[maybe_unused]] static void pasteRotatedImage(cv::Mat& dst, const cv::Mat& img, const cv::Point2f& center,
                              float w, float h, float angle_deg)
{
	if (img.empty() || w < 2.0f || h < 2.0f)
		return;

	cv::Mat scaled;
	cv::resize(img, scaled, cv::Size((int)(w + 0.5f), (int)(h + 0.5f)));

	cv::Point2f sc(scaled.cols * 0.5f, scaled.rows * 0.5f);
	cv::Mat m = cv::getRotationMatrix2D(sc, -angle_deg, 1.0); // + = antihorario; aqui el panel usa horario
	cv::Rect bbox = cv::RotatedRect(sc, cv::Size2f((float)scaled.cols, (float)scaled.rows),
	                                angle_deg)
	                        .boundingRect();
	m.at<double>(0, 2) += bbox.width * 0.5 - scaled.cols * 0.5;
	m.at<double>(1, 2) += bbox.height * 0.5 - scaled.rows * 0.5;

	cv::Mat rot, rotmask;
	cv::warpAffine(scaled, rot, m, bbox.size());
	cv::Mat mask(scaled.size(), CV_8UC1, cv::Scalar(255));
	cv::warpAffine(mask, rotmask, m, bbox.size());

	const int x0 = (int)(center.x - bbox.width * 0.5 + 0.5f);
	const int y0 = (int)(center.y - bbox.height * 0.5 + 0.5f);
	cv::Rect dst_r(x0, y0, bbox.width, bbox.height);
	dst_r &= cv::Rect(0, 0, dst.cols, dst.rows);
	if (dst_r.width <= 0 || dst_r.height <= 0)
		return;
	const cv::Rect src_r(dst_r.x - x0, dst_r.y - y0, dst_r.width, dst_r.height);

	cv::Mat roi = dst(dst_r);
	cv::Mat blended;
	cv::addWeighted(roi, 1.0, rot(src_r), 0.85, 0.0, blended);
	blended.copyTo(roi, rotmask(src_r));
}

// ─── Estado de partida ───────────────────────────────────────────────
// Una carta está "girada" (tapped) cuando su ángulo difiere ~90° del de
// la zona minion de su jugador. Las cartas de la cripta no se giran.

struct PlayerZoneHit {
	int  player_idx = -1; // -1 = fuera de toda zona
	bool in_crypt   = false;
};

// Resuelve a qué zona (minion/cripta) de qué jugador corresponde el
// centro de una carta. Los jugadores inactivos (ousted) no asignan.
static PlayerZoneHit resolvePlayerZone(const struct detect_filter_obb *tf,
				       const cv::Point2f &c, int fw, int fh)
{
	PlayerZoneHit hit;
	for (int pi = 0; pi < tf->player_count; pi++) {
		const filter_data::PlayerState &pl = tf->players[pi];
		if (!pl.active)
			continue; // jugador fuera de partida: su sitio queda libre
		auto checkZone = [&](const filter_data::PlayerZone &z, bool crypt) -> bool {
			const float zw = z.nw * fw, zh = z.nh * fh;
			if (zw <= 0.0f || zh <= 0.0f)
				return false;
			const float zx = z.nx * fw, zy = z.ny * fh;
			float dx = c.x - (zx + zw * 0.5f);
			float dy = c.y - (zy + zh * 0.5f);
			if (z.angle != 0.0f) {
				// Zona rotada (positivo = horario): se pasa el punto
				// al sistema de la zona aplicando el giro inverso.
				const float ra = z.angle * kPI / 180.0f;
				const float ca = std::cos(ra), sa = std::sin(ra);
				const float rx = ca * dx + sa * dy;
				const float ry = -sa * dx + ca * dy;
				dx = rx;
				dy = ry;
			}
			if (std::fabs(dx) <= zw * 0.5f && std::fabs(dy) <= zh * 0.5f) {
				hit.player_idx = pi;
				hit.in_crypt = crypt;
				return true;
			}
			return false;
		};
		if (checkZone(pl.minion_zone, false))
			return hit;
		if (checkZone(pl.crypt_zone, true))
			return hit;
	}
	return hit;
}

// Votos de histéresis del estado de giro (se cuentan por evaluación).
static constexpr float kLockTiltedDeg = 55.0f; // ≥ = vota "girada"
static constexpr float kLockFlatDeg   = 35.0f; // ≤ = vota "lista"
static constexpr int   kLockVotesMax  = 8;     // votos seguidos para cambiar

// Ángulo de una carta (radianes) relativo a la referencia de su zona,
// normalizado a [-90, 90): ~0 = lista (alineada), ~±90 = girada.
static float relativeLockAngleDeg(float card_angle_rad, float base_angle_deg)
{
	const float card_deg = card_angle_rad * 180.0f / kPI;
	float rel = std::fmod(card_deg - base_angle_deg + 90.0f, 180.0f);
	if (rel < 0.0f)
		rel += 180.0f;
	return rel - 90.0f;
}

// ─── Modo realizador: pendientes + feedback de entrenamiento ────────

// Anade (o refresca) una candidata en la cola de pendientes. El recorte
// del frame se guarda como dato para futuros entrenamientos.
static void addPendingDetection(detect_filter_obb *tf, const vtes_detection::OBBObject &obj,
				const cv::Mat &frame, int64_t now_ns)
{
	if (obj.card_id.empty())
		return;
	std::lock_guard<std::mutex> lock(tf->pending_lock);

	// Limpia pendientes que llevan mucho sin volver a verse (120 s).
	const int64_t stale_ms = 120000;
	for (auto it = tf->pending_detections.begin(); it != tf->pending_detections.end();) {
		if ((now_ns - it->last_seen_ns) / 1000000LL > stale_ms)
			it = tf->pending_detections.erase(it);
		else
			++it;
	}

	for (auto &p : tf->pending_detections) {
		if (p.card_id == obj.card_id) {
			p.last_seen_ns = now_ns;
			p.hits++;
			if (obj.prob > p.confidence)
				p.confidence = obj.prob;
			return;
		}
	}

	filter_data::PendingDetection pd;
	pd.id = tf->pending_next_id++;
	pd.card_id = obj.card_id;
	pd.card_name = obj.card_name;
	auto cit = tf->card_info_by_id.find(obj.card_id);
	if (cit != tf->card_info_by_id.end())
		pd.card_url = cit->second.url;
	pd.confidence = obj.prob;
	pd.first_seen_ns = pd.last_seen_ns = now_ns;
	pd.hits = 1;

	// Recorte de la carta en el frame (evidencia + entrenamiento).
	{
		const int x0 = std::max(0, (int)obj.rect.x);
		const int y0 = std::max(0, (int)obj.rect.y);
		const int x1 = std::min(frame.cols, (int)(obj.rect.x + obj.rect.width));
		const int y1 = std::min(frame.rows, (int)(obj.rect.y + obj.rect.height));
		if (x1 > x0 && y1 > y0) {
			cv::Mat crop = frame(cv::Rect(x0, y0, x1 - x0, y1 - y0));
			if (crop.channels() == 4) {
				cv::Mat bgr;
				cv::cvtColor(crop, bgr, cv::COLOR_BGRA2BGR);
				crop = bgr;
			}
			std::vector<uchar> buf;
			const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
			if (cv::imencode(".jpg", crop, buf, params))
				pd.crop_jpg = std::move(buf);
		}
	}

	obs_log(LOG_INFO, "[Director] Pendiente: \"%s\" (conf %.2f, %zu en cola)",
		pd.card_name.c_str(), pd.confidence, tf->pending_detections.size() + 1);

	tf->pending_detections.push_back(std::move(pd));
	const size_t max_pending = 24;
	if (tf->pending_detections.size() > max_pending)
		tf->pending_detections.pop_front();
}

// Guarda un evento de feedback (elegida / descartadas) con los recortes
// de las candidatas: materia prima para entrenar futuras versiones.
static void recordDirectorFeedback(detect_filter_obb *tf, const char *kind,
				   const filter_data::PendingDetection *chosen,
				   const std::vector<filter_data::PendingDetection> &pool)
{
	if (tf->feedback_dir.empty() || pool.empty())
		return;
	std::error_code ec;
	std::filesystem::create_directories(tf->feedback_dir, ec);

	char stamp[32] = {0};
	const std::time_t t = std::time(nullptr);
	const std::tm *tmv = std::localtime(&t);
	if (tmv)
		std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", tmv);
	const std::string kind_s = kind ? kind : "event";
	const std::string ev_id = std::string(stamp) + "-" + kind_s;
	const std::string ev_dir = tf->feedback_dir + "/events/" + ev_id;
	std::filesystem::create_directories(ev_dir, ec);

	auto save_crop = [&](const filter_data::PendingDetection &p,
			     const std::string &name) -> std::string {
		if (p.crop_jpg.empty())
			return std::string();
		const std::string rel = ev_id + "/" + name + ".jpg";
		std::ofstream f(tf->feedback_dir + "/events/" + rel, std::ios::binary);
		if (!f)
			return std::string();
		f.write((const char *)p.crop_jpg.data(), (std::streamsize)p.crop_jpg.size());
		return rel;
	};

	nlohmann::json ev;
	ev["type"] = kind_s;
	ev["ts"] = stamp;
	ev["pool_size"] = pool.size();
	nlohmann::json candidates = nlohmann::json::array();
	int cand_idx = 0;
	for (const auto &p : pool) {
		if (chosen && p.id == chosen->id && kind_s == "selection")
			continue; // la elegida va aparte
		if (cand_idx >= 12)
			break; // tope de candidatas guardadas por evento
		nlohmann::json cj;
		cj["card_id"] = p.card_id;
		cj["card_name"] = p.card_name;
		cj["confidence"] = p.confidence;
		cj["hits"] = p.hits;
		const std::string saved = save_crop(p, "cand_" + std::to_string(cand_idx++));
		if (!saved.empty())
			cj["crop"] = saved;
		candidates.push_back(std::move(cj));
	}
	ev["candidates"] = std::move(candidates);
	if (chosen) {
		nlohmann::json cj;
		cj["card_id"] = chosen->card_id;
		cj["card_name"] = chosen->card_name;
		cj["card_url"] = chosen->card_url;
		cj["confidence"] = chosen->confidence;
		cj["hits"] = chosen->hits;
		const std::string saved = save_crop(*chosen, "chosen");
		if (!saved.empty())
			cj["crop"] = saved;
		ev["chosen"] = std::move(cj);
	}

	std::ofstream log(tf->feedback_dir + "/events.jsonl", std::ios::app | std::ios::binary);
	if (log)
		log << ev.dump() << "\n";
	obs_log(LOG_INFO, "[Director] Feedback guardado: %s (%zu candidatas)",
		ev_dir.c_str(), pool.size());
}

void detect_filter_obb_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);

	struct detect_filter_obb *tf = reinterpret_cast<detect_filter_obb *>(data);

	if (tf->isDisabled) {
		return;
	}
	if (tf->detectionMode == DETECT_MODE_ONNX && !tf->yolo_detector) {
		return;
	}

	if (!obs_source_enabled(tf->source)) {
		return;
	}

	// ─── Preview + Always Active: stop detection when hidden ─────────
	bool should_detect = tf->preview || tf->always_active;

	// ─── Detection interval: minimum time between detection runs ───
	// 0 = disabled (continuous), no logging
	auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count();
	if (should_detect && tf->detection_interval_ms > 0) {
		uint64_t interval_ns = (uint64_t)tf->detection_interval_ms * 1000000ULL;
		if (now_ns - tf->last_detection_time_ns < interval_ns) {
			should_detect = false;
		} else {
			tf->last_detection_time_ns = now_ns;
		}
	} else if (should_detect) {
		tf->last_detection_time_ns = now_ns;
	}

	cv::Mat imageBGRA;
	{
		std::unique_lock<std::mutex> lock(tf->inputBGRALock, std::try_to_lock);
		if (!lock.owns_lock()) {
			return;
		}
		if (tf->inputBGRA.empty()) {
			return;
		}
		imageBGRA = tf->inputBGRA.clone();
	}

	try {

	// ─── Cooldown: skip detection after successful card identification ─────
	// (Always Active bypasses cooldown entirely)
	auto _cd_now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count();
	bool in_cooldown = tf->always_active ? false :
		(tf->cooldown_until_time_ns > 0 && _cd_now_ns < tf->cooldown_until_time_ns);

	std::vector<vtes_detection::OBBObject> raw_objects;

	if (should_detect && !in_cooldown) {
	cv::Mat inferenceFrame;
	cv::cvtColor(imageBGRA, inferenceFrame, cv::COLOR_BGRA2BGR);
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
			if (tf->yolo_detector) {
				raw_objects = tf->yolo_detector->inferOBB(inferenceFrame);
			}
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "%s", e.what());
		}
	}

	// ─── ONNX area filtering (filter out too small/large detections) ──────
	if (tf->detectionMode == DETECT_MODE_ONNX && !raw_objects.empty()) {
		std::vector<vtes_detection::OBBObject> filtered;
		double max_area = tf->onnxMaxAreaFrac * inferenceFrame.cols * inferenceFrame.rows;
		for (const auto& obj : raw_objects) {
			double area = obj.rect.width * obj.rect.height;
			if (area >= tf->onnxMinArea && area <= max_area) {
				filtered.push_back(obj);
			}
		}
		raw_objects = filtered;
	}

	// ─── Extract perspective-corrected card regions once, reuse below ─
	std::vector<cv::Mat> card_regions(raw_objects.size());
	for (size_t i = 0; i < raw_objects.size(); i++) {
		const auto& obj = raw_objects[i];
		cv::Point2f center(obj.rect.x + obj.rect.width / 2,
				   obj.rect.y + obj.rect.height / 2);
		cv::Size2f size(obj.rect.width, obj.rect.height);
		cv::RotatedRect rr(center, size, obj.angle * 180.0f / kPI);
		cv::Point2f corners[4];
		rr.points(corners);
		card_regions[i] = extractCardRegion(inferenceFrame, corners);
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

			// ─── Vision-based type classifier (vtes_type_classifier.onnx) ─
			std::string vision_type;
			if (!card_regions[i].empty())
				vision_type = runVisionTypeClassifier(tf, card_regions[i]);
			if (!vision_type.empty()) {
				per_object_type_filter[i] = vision_type;

				// CUDA oval/rect validation: reject vampires that don't look oval
				if (tf->vampire_cuda_enabled && vision_type == "Vampire" &&
				    !isVampireCUDA(card_regions[i])) {
					obs_log(LOG_INFO, "[CUDA] Card %zu: type=%s but frame is RECT — rejecting",
						i, vision_type.c_str());
					per_object_type_filter[i] = "Reject";
				}
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
				obs_log(LOG_DEBUG, "Card %zu: vision_type=%s (rotated crop)", i, vision_type.c_str());
			} else {
				// Fallback: signal-based classifier (axis-aligned crop)
				float typeConf = 0.0f;
				vtes_classifier::SignalScores signals;
				vtes_classifier::CardType type = tf->classifier->classify(crops[i].crop, typeConf, signals);
				per_object_type_filter[i] = VTESCardDatabase::classifier_type_to_vtes_filter(
					static_cast<int>(type));
				raw_objects[i].label = 100 + static_cast<int>(type);
				raw_objects[i].prob = typeConf;
				obs_log(LOG_DEBUG, "Card %zu: signal_type=%d(%s) conf=%.2f (axis-aligned crop)",
					i, static_cast<int>(type), per_object_type_filter[i].c_str(), typeConf);
			}
		}
	}

	// ─── Card identification (embedding + OCR) ──────────────────────────
	// OCR runs regardless of embedding availability — don't guard with embed_available
	bool embed_available = tf->embedder.is_loaded() || !tf->per_type_matchers.empty();
	if (!raw_objects.empty()) {
		for (size_t i = 0; i < raw_objects.size(); i++) {
			auto& obj = raw_objects[i];

			cv::Mat card_region = card_regions[i];
			if (card_region.empty()) continue;

			std::string card_name, card_id;
			float confidence = 0.0f;
			const std::string& type_filter = per_object_type_filter[i];
			bool matched = false;
			float thresh = 0.35f;

			// Try per-type embedder first
	obs_log(LOG_DEBUG, "[Classify] Card #%d type_filter='%s' crop=%dx%d",
		obj.id, type_filter.c_str(), card_region.cols, card_region.rows);

	if (embed_available) {
				auto pt_it = tf->per_type_matchers.find(type_filter);
				if (pt_it != tf->per_type_matchers.end() && pt_it->second->is_loaded()) {
					matched = pt_it->second->identify(card_region, card_name, card_id, confidence);
					thresh = pt_it->second->threshold();
				}
				// Fallback to global embedder
				// NOTA: el filtro de tipo del clasificador de vision es poco fiable
				// en frames reales (misclasifica ~100%: p.ej. Reaction -> Event),
				// por lo que el matcher fusionado (embedding+NCC) identifica sin
				// filtro de tipo.
				if (!matched && tf->embedder.is_loaded()) {
					matched = tf->embedder.identify(card_region, card_name, card_id, confidence, "");
					thresh = tf->embedder.threshold();
				}
			}

			// OCR is async: runs on a background worker thread.
			// Jobs are submitted below; results are drained on the next frame.
			// (The worker handles recognize() + vampire validation.)

			if (matched) {
				if (confidence >= thresh) {
					obj.card_name = card_name;
				} else {
					obj.card_name = "?" + card_name;
				}
				obj.card_id = card_id;
				obj.prob = confidence;
				obs_log(LOG_INFO,
					"[Embed] Card #%d: '%s' (id=%s, sim=%.4f, thresh=%.2f, crop=%dx%d, filter=%s%s)",
					obj.id, card_name.c_str(), card_id.c_str(), confidence,
					thresh, card_region.cols, card_region.rows,
					type_filter.c_str(),
					confidence >= thresh ? "" : " ?");
			} else {
				obj.card_name = "???";
				obj.card_id = "";
				if (confidence > 0.2f) {
					obj.prob = confidence;
					obs_log(LOG_INFO,
						"[Embed] Card #%d: no match (best_sim=%.4f, thresh=%.2f, crop=%dx%d, filter=%s)",
						obj.id, confidence, thresh,
						card_region.cols, card_region.rows,
						type_filter.c_str());
				}
			}
		}

		// Drain async OCR results from background worker (applies to
		// objects still marked "???") and submit new jobs for this frame.
		if (tf->ocr_enabled) {
			drain_ocr_results(tf, raw_objects);
			// Pass empty type hint so name-region detection is stable
			// (the vision type classifier is too frame-unstable to trust)
			std::vector<std::string> empty_filters(card_regions.size(), "");
			// Pasar los track ids para que cada resultado de OCR vuelva
			// a su propia carta y no a la primera que siga en "???".
			std::vector<int> track_ids;
			track_ids.reserve(raw_objects.size());
			for (const auto& o : raw_objects) track_ids.push_back(o.id);
			submit_ocr_jobs(tf, card_regions, track_ids, empty_filters);
		}
	} // end if (!raw_objects.empty())
		// ─── Set cooldown on successful identification (not in Always Active mode) ─────
		if (!tf->always_active) {
			for (const auto& obj : raw_objects) {
				if (!obj.card_id.empty()) {
					auto _cd_set_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
						std::chrono::steady_clock::now().time_since_epoch()).count();
					tf->cooldown_until_time_ns = _cd_set_ns + (int64_t)(tf->card_overlay_duration * 1e9);
					obs_log(LOG_INFO, "[Cooldown] Card identified, detection paused for %.1f seconds",
						tf->card_overlay_duration);
					break;
				}
			}
		}
		// ─── Store detection history for web UI ──────────────────────
		{
			std::lock_guard<std::mutex> lock(tf->detection_history_lock);
			for (const auto& obj : raw_objects) {
				if (!obj.card_id.empty()) {
					auto it = tf->card_info_by_id.find(obj.card_id);
					std::string card_url = (it != tf->card_info_by_id.end()) ? it->second.url : "";
					tf->detection_history.push_back({obj.card_name, obj.card_id, card_url, _cd_now_ns});
					if ((int)tf->detection_history.size() > filter_data::MAX_DETECTION_HISTORY)
						tf->detection_history.pop_front();
				}
			}
		}
	} // end if (should_detect && !in_cooldown)

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
	std::vector<vtes_detection::OBBObject> final_objects;
	if (tf->temporal_smoothing_enabled) {
		updateTemporalTracks(tf, raw_objects, final_objects);
	} else {
		final_objects = raw_objects;
	}

	if (!tf->showUnseenObjects) {
		final_objects.erase(
			std::remove_if(final_objects.begin(), final_objects.end(),
				       [](const vtes_detection::OBBObject &obj) { return obj.unseenFrames > 0; }),
			final_objects.end());
	}

	// ─── Game state: lock tracking & player assignment ─────────────
	{
		int fw = imageBGRA.cols;
		int fh = imageBGRA.rows;
		auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count();

		// Increment unseen counter for all tracked cards
		{
			std::lock_guard<std::mutex> lock(tf->game_state_lock);
			for (auto& [tid, card] : tf->tracked_cards)
				card.unseen_frames++;
		}

		for (const auto& obj : final_objects) {
			std::lock_guard<std::mutex> lock(tf->game_state_lock);
			auto it = tf->tracked_cards.find(obj.id);

			if (it == tf->tracked_cards.end()) {
				// New track
				filter_data::TrackedCardState tc;
				tc.track_id = obj.id;
				tc.card_id = obj.card_id;
				tc.card_name = obj.card_name;
				tc.first_seen_ns = now_ns;
				tc.initial_angle = obj.angle;
				tc.lock_state = filter_data::CardLockState::Unlocked;
				tc.last_seen_ns = now_ns;
				tc.last_position = cv::Point2f(
					obj.rect.x + obj.rect.width / 2,
					obj.rect.y + obj.rect.height / 2);
				tc.unseen_frames = 0;

				// Look up card URL + pre-cache image
				if (!obj.card_id.empty()) {
					auto cit = tf->card_info_by_id.find(obj.card_id);
					if (cit != tf->card_info_by_id.end()) {
						tc.card_url = cit->second.url;
						if (!tc.card_url.empty() &&
							tf->card_image_cache.find(tc.card_url) == tf->card_image_cache.end()) {
							std::vector<uchar> img_data;
							if (downloadImageToMemory(tc.card_url, img_data)) {
								cv::Mat img = cv::imdecode(img_data, cv::IMREAD_COLOR);
								if (!img.empty())
									tf->card_image_cache[tc.card_url] = img;
							}
						}
						auto cit2 = tf->card_image_cache.find(tc.card_url);
						if (cit2 != tf->card_image_cache.end())
							tc.card_image = cit2->second.clone();
					}
				}

				// Assign to player by zone overlap
				PlayerZoneHit zone = resolvePlayerZone(tf, tc.last_position, fw, fh);
				tc.player_idx = zone.player_idx;
				tc.in_crypt = zone.in_crypt;

				tf->tracked_cards[obj.id] = std::move(tc);
			} else {
				// Existing track
				auto& tc = it->second;
				tc.last_seen_ns = now_ns;
				tc.last_position = cv::Point2f(
					obj.rect.x + obj.rect.width / 2,
					obj.rect.y + obj.rect.height / 2);
				tc.unseen_frames = 0;

				// Update card info if newly identified
				if (tc.card_id.empty() && !obj.card_id.empty()) {
					tc.card_id = obj.card_id;
					tc.card_name = obj.card_name;
					auto cit = tf->card_info_by_id.find(obj.card_id);
					if (cit != tf->card_info_by_id.end())
						tc.card_url = cit->second.url;
				}

				// Reasignación de zona cada frame: una carta puede cambiar
				// de zona (p.ej. de cripta a minion al completar la
				// influencia del vampiro).
				{
					PlayerZoneHit zone = resolvePlayerZone(tf, tc.last_position, fw, fh);
					if (zone.player_idx != tc.player_idx || zone.in_crypt != tc.in_crypt) {
						tc.player_idx = zone.player_idx;
						tc.in_crypt = zone.in_crypt;
						tc.lock_votes = 0; // la histéresis previa ya no aplica
					}
				}

				// ── Estado de giro: evaluación continua (una carta puede
				//    girarse/enderezarse en cualquier momento) y relativa a
				//    la zona minion del jugador. En la cripta no se gira.
				{
					const bool in_minion_zone = tc.player_idx >= 0 && !tc.in_crypt;
					const float base_deg = in_minion_zone
						? tf->players[tc.player_idx].minion_zone.angle
						: tc.initial_angle * 180.0f / kPI;
					const float rel = relativeLockAngleDeg(obj.angle, base_deg);
					tc.last_rel_angle = rel;

					int dir = 0;
					if (in_minion_zone) {
						const float a = std::fabs(rel);
						if (a >= kLockTiltedDeg)
							dir = +1;
						else if (a <= kLockFlatDeg)
							dir = -1;
					} else {
						dir = -1; // sin zona minion no se puede afirmar giro
					}
					if (dir != 0)
						tc.lock_votes = std::max(-kLockVotesMax,
									 std::min(kLockVotesMax, tc.lock_votes + dir));

					const char *cname = tc.card_name.empty() ? "?" : tc.card_name.c_str();
					if (tc.lock_state == filter_data::CardLockState::Unlocked &&
					    tc.lock_votes >= kLockVotesMax) {
						tc.lock_state = filter_data::CardLockState::Locked;
						tc.lock_changed_ns = now_ns;
						if (tc.player_idx >= 0)
							obs_log(LOG_INFO,
								"[Game] Jugador %d: \"%s\" girada (rel %+.0f deg)",
								tc.player_idx + 1, cname, rel);
					} else if (tc.lock_state == filter_data::CardLockState::Locked &&
						   tc.lock_votes <= -kLockVotesMax) {
						tc.lock_state = filter_data::CardLockState::Unlocked;
						tc.lock_changed_ns = now_ns;
						if (tc.player_idx >= 0)
							obs_log(LOG_INFO,
								"[Game] Jugador %d: \"%s\" lista (rel %+.0f deg)",
								tc.player_idx + 1, cname, rel);
					}
				}
			}

			// Modo realizador: la deteccion no se muestra sola; entra en la
			// cola de pendientes para que el operador la valide en la web.
			if (tf->click_to_overlay && !obj.card_id.empty())
				addPendingDetection(tf, obj, imageBGRA, now_ns);
		}

		// Remove stale tracks (unseen > 120 frames ≈ 4 s at 30 fps)
		{
			std::lock_guard<std::mutex> lock(tf->game_state_lock);
			auto it = tf->tracked_cards.begin();
			while (it != tf->tracked_cards.end()) {
				if (it->second.unseen_frames > 120)
					it = tf->tracked_cards.erase(it);
				else
					++it;
			}
		}
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

		// ─── Game state overlay: background, player pool, tracked cards ──
		{
			std::vector<uchar> img_data;

			// 1. Load background image if not loaded
			if (tf->overlay_bg.empty()) {
				char *bgPath = obs_module_file("overlay-bg.png");
				if (bgPath) {
					// IMREAD_UNCHANGED conserva el canal alfa: el hueco central
					// (la ventana de camara) es transparente; con IMREAD_COLOR
					// se volveria negro opaco y taparia la camara.
					tf->overlay_bg = cv::imread(bgPath, cv::IMREAD_UNCHANGED);
					bfree(bgPath);
				}
			}

			int fw = frame.cols, fh = frame.rows;
			const OverlayFit fit = overlayFit(fw, fh);

			cv::Mat display = cv::Mat::zeros(fh, fw, CV_8UC3);
			// 2. Camara DENTRO del hueco del overlay y POR DEBAJO de el: se
			// ajusta entera (sin recorte) dentro de la ventana del diseño y el
			// overlay (marco, panel, ankh) se dibuja despues, encima. Las
			// zonas de jugador (coords 0..1 de camara) se mapean a `cam`.
			const cv::Rect cam = tf->overlay_bg.empty()
				? cv::Rect(0, 0, fw, fh)
				: cameraFitRect(fit, fw, fh);
			if (cam.width > 4 && cam.height > 4 && !frame.empty()) {
				cv::Mat src_resized;
				cv::resize(frame, src_resized, cv::Size(cam.width, cam.height));
				src_resized.copyTo(display(cam));
			}

			// 2b. Zonas de jugadores (toggle): dibuja los rectangulos de minions
			// y cripta de cada jugador (con su rotacion) sobre la ventana de
			// camara para poder ajustar X/Y/W/H/angulo viendo donde caen. Las
			// zonas van en coordenadas de camara (0..1); un jugador fuera de
			// partida (ousted) se dibuja en gris y su zona no asigna cartas.
			if (tf->show_zones) {
				const cv::Scalar palette[5] = {
					cv::Scalar(60, 220, 60),   // verde
					cv::Scalar(255, 160, 0),   // azul
					cv::Scalar(0, 60, 255),    // rojo
					cv::Scalar(0, 220, 220),   // amarillo
					cv::Scalar(220, 60, 220),  // magenta
				};
				for (int pi = 0; pi < tf->player_count; pi++) {
					auto& pl = tf->players[pi];
					const cv::Scalar col = pl.active ? palette[pi % 5]
					                                 : cv::Scalar(140, 140, 140);
					auto drawZone = [&](const filter_data::PlayerZone& z, const char* tag) {
						const cv::Point2f zc(
							(float)(cam.x + (z.nx + z.nw * 0.5) * cam.width),
							(float)(cam.y + (z.ny + z.nh * 0.5) * cam.height));
						const cv::Size2f zs((float)(z.nw * cam.width),
						                    (float)(z.nh * cam.height));
						if (zs.width < 4.0f || zs.height < 4.0f)
							return;
						cv::RotatedRect rr(zc, zs, z.angle);
						cv::Point2f pts[4];
						rr.points(pts);

						// Relleno translucido dentro del poligono rotado
						cv::Rect br = rr.boundingRect() & cv::Rect(0, 0, fw, fh);
						if (br.width < 4 || br.height < 4)
							return;
						cv::Point ap[4], lp[4];
						for (int i = 0; i < 4; i++) {
							ap[i] = cv::Point((int)(pts[i].x + 0.5f),
							                  (int)(pts[i].y + 0.5f));
							lp[i] = cv::Point(ap[i].x - br.x, ap[i].y - br.y);
						}
						cv::Mat mask(display(br).size(), CV_8UC1, cv::Scalar(0));
						cv::fillConvexPoly(mask, lp, 4, cv::Scalar(255));
						cv::Mat tinted;
						cv::addWeighted(display(br), 0.88,
						                cv::Mat(br.size(), display.type(), col), 0.12,
						                0.0, tinted);
						tinted.copyTo(display(br), mask);

						// Borde del poligono
						std::vector<std::vector<cv::Point>> poly = {
							{ap[0], ap[1], ap[2], ap[3]}};
						cv::polylines(display, poly, true, col, 2);

						// Etiqueta junto al vertice superior del rectangulo
						std::string label = pl.name + " - " + tag;
						if (!pl.active)
							label += std::string(" [") + tf_text("ZoneOut") + "]";
						cv::Point2f top = pts[0];
						for (int i = 1; i < 4; i++)
							if (pts[i].y < top.y)
								top = pts[i];
						const double fs = 0.45;
						int baseline = 0;
						cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
						                              fs, 1, &baseline);
						cv::Rect lr((int)(top.x + 3.0f), (int)(top.y + 3.0f),
						            ts.width + 10, ts.height + 8);
						lr &= cv::Rect(0, 0, fw, fh);
						if (lr.width > 0 && lr.height > 0) {
							cv::Mat label_bg;
							cv::addWeighted(display(lr), 0.4,
							                cv::Mat(lr.size(), display.type(),
							                        cv::Scalar(20, 20, 20)), 0.6,
							                0.0, label_bg);
							label_bg.copyTo(display(lr));
							cv::putText(display, label,
							            cv::Point(lr.x + 6, lr.y + ts.height + 4),
							            cv::FONT_HERSHEY_SIMPLEX, fs, col, 1);
						}
					};
					drawZone(pl.minion_zone, tf_text("ZoneMinions"));
					drawZone(pl.crypt_zone, tf_text("ZoneCrypt"));
				}
			}

			// 2c. El diseño del overlay ENCIMA (marco, panel, ankh): la camara
			// queda por debajo y solo se ve por el hueco transparente. Mezcla
			// con el canal alfa real: enmascarar en binario dejaba un reborde
			// oscuro en el filo anti-aliaseado del hueco.
			if (!tf->overlay_bg.empty()) {
				const int dw = (int)(kOverlayRefW * fit.esc + 0.5);
				const int dh = (int)(kOverlayRefH * fit.esc + 0.5);
				cv::Rect dst = cv::Rect(fit.ox, fit.oy, dw, dh) & cv::Rect(0, 0, fw, fh);
				if (dst.width > 4 && dst.height > 4) {
					cv::Mat ov;
					cv::resize(tf->overlay_bg, ov, cv::Size(dw, dh), 0, 0, cv::INTER_AREA);
					if (ov.channels() == 4) {
						cv::Mat ov3;
						cv::cvtColor(ov, ov3, cv::COLOR_BGRA2BGR);
						cv::Mat af;
						cv::extractChannel(ov, af, 3);
						cv::Mat af_inv;
						cv::subtract(cv::Scalar::all(255.0), af, af_inv);
						std::vector<cv::Mat> oc, dc;
						cv::split(ov3, oc);
						cv::split(display(dst), dc);
						for (int c = 0; c < 3; c++) {
							cv::Mat t1, t2;
							cv::multiply(oc[c], af, t1, 1.0 / 255.0);
							cv::multiply(dc[c], af_inv, t2, 1.0 / 255.0);
							cv::add(t1, t2, dc[c]);
						}
						cv::merge(dc, ov3);
						ov3.copyTo(display(dst));
					} else {
						ov.copyTo(display(dst));
					}
				}
			}

			// 3. Casillas de jugador del panel izquierdo: el nombre y el valor
			// de pool van DENTRO de las casillas del diseño (el overlay ya
			// esta dibujado debajo del texto). Un jugador fuera de partida
			// (ousted) se muestra atenuado.
			for (int pi = 0; pi < tf->player_count && pi < 5; pi++) {
				auto& pl = tf->players[pi];
				const cv::Rect name_box =
					designRect(fit, kNameX, kSlotY[pi], kNameW, kSlotH);
				const cv::Rect pool_box =
					designRect(fit, kPoolX, kSlotY[pi], kPoolW, kSlotH);
				const cv::Scalar slot_col = pl.active ? cv::Scalar(235, 228, 214)
				                                      : cv::Scalar(115, 112, 108);
				drawTextInBox(display, pl.name, name_box, slot_col,
				              name_box.height * 0.42 / 22.0);
				drawTextInBox(display, std::to_string(pl.pool), pool_box, slot_col,
				              pool_box.height * 0.55 / 22.0);
			}

			// 4. Las cartas rastreadas ya no se pintan en su zona de jugador:
			// la unica carta que se renderiza es la ultima detectada, sobre el
			// ankh del panel izquierdo (bloque 4b, mas abajo).

			// 4b. Carta principal: se muestra ENTERA (el escaneo completo de la
			// cache, sin recortar los lados) sobre el ankh del panel izquierdo.
			// Con el toggle "carta grande" se dibuja a media pantalla, centrada
			// y por encima de todo, para retransmisiones vistas en un movil.
			cv::Rect shown_card;       // rect de la carta mostrada (frame px)
			double shown_metric = 1.0; // factor de metrica para el texto de rulings
			std::string shown_card_db_id; // id de la carta mostrada (para rulings)
			{
				std::lock_guard<std::mutex> lock(tf->game_state_lock);
				// Caja del ankh en pixeles del frame: la misma transformacion
				// que coloca el resto del diseño (escala + centrado).
				const cv::Rect abox = designRect(fit, kAnkhX, kAnkhY, kAnkhW, kAnkhH);

				// Carta a mostrar:
				//  - Si el operador eligio una en la web (pulsada en
				//    Pendientes), esa misma.
				//  - Si no, en modo clasico la ultima detectada (track_id
				//    mas alto); en modo realizador no se muestra nada.
				filter_data::TrackedCardState *principal = nullptr;
				cv::Mat card_img;
				if (!tf->current_overlay_card_url.empty()) {
					for (auto& [tid, tc] : tf->tracked_cards) {
						if (tc.card_url == tf->current_overlay_card_url) {
							principal = &tc;
							break;
						}
					}
					if (principal && !principal->card_image.empty()) {
						card_img = principal->card_image;
					} else {
						auto cache_it = tf->card_image_cache.find(tf->current_overlay_card_url);
						if (cache_it != tf->card_image_cache.end())
							card_img = cache_it->second;
					}
					shown_card_db_id = (principal && !principal->card_id.empty())
						? principal->card_id : tf->current_overlay_card_id;
				} else if (!tf->click_to_overlay) {
					for (auto& [tid, tc] : tf->tracked_cards) {
						if (tc.card_image.empty())
							continue;
						if (!principal || tid > principal->track_id)
							principal = &tc;
					}
					if (principal) {
						card_img = principal->card_image;
						shown_card_db_id = principal->card_id;
					}
				}

				if (!card_img.empty()) {
					const double cw = card_img.cols;
					const double ch = card_img.rows;
					if (cw > 0 && ch > 0) {
						// Carta completa (contain) centrada dentro de una caja:
						// nunca se recortan los lados.
						auto fitInto = [&](const cv::Rect& box) {
							const double esc = std::min((double)box.width / cw,
							                            (double)box.height / ch);
							const int rw = (int)(cw * esc + 0.5);
							const int rh = (int)(ch * esc + 0.5);
							return cv::Rect(box.x + (box.width - rw) / 2,
							                box.y + (box.height - rh) / 2, rw, rh);
						};
						// Referencia de metrica: la carta normal sobre el ankh
						// (su alto fija el tamaño del texto de los rulings).
						const cv::Rect normal = fitInto(abox);

						if (tf->big_card) {
							// Media pantalla: alto = 50% del frame, centrada
							// (puede tapar el resto de objetos: es lo buscado).
							int hh = (int)(fh * 0.5 + 0.5);
							int ww = (int)(hh * cw / ch + 0.5);
							const int max_w = (int)(fw * 0.9);
							if (ww > max_w) {
								ww = max_w;
								hh = (int)(ww * ch / cw + 0.5);
							}
							shown_card = cv::Rect((fw - ww) / 2, (fh - hh) / 2, ww, hh);
						} else {
							shown_card = normal;
						}

						const cv::Rect dst = shown_card & cv::Rect(0, 0, fw, fh);
						if (dst.width > 0 && dst.height > 0) {
							cv::Mat carta;
							cv::resize(card_img, carta,
							           cv::Size(shown_card.width, shown_card.height),
							           0, 0, cv::INTER_AREA);
							const cv::Rect src(dst.x - shown_card.x,
							                   dst.y - shown_card.y, dst.width,
							                   dst.height);
							carta(src).copyTo(display(dst));
						}

						if (normal.height > 0)
							shown_metric = (double)shown_card.height / normal.height;
					}
				}
			}

		// 6. Rulings de la carta principal, sobre la propia carta
		//
		// El panel no tiene ningun hueco libre: las casillas de jugador ocupan
		// y=89..455 y el ankh (donde va la carta) y=482..960 sobre el diseño
		// (1536x1024). Decision actual: los rulings se dibujan directamente
		// sobre la carta mostrada, en una caja un poco mas ancha que ella y
		// sin recuadro de fondo.
		//
		// El texto no se recorta por abajo: la caja es un viewport sobre el
		// texto completo y cuanto se ve de el lo mueve el slider
		// "Rulings scroll" del filtro. Cambiar donde se muestran es cambiar
		// esta caja.
		{
			{
				// La carta mostrada sobre el ankh (elegida en la web o, en
				// modo clasico, la ultima detectada).
				const VTESCardEntry* entry = !shown_card_db_id.empty()
					? tf->vtes_db.get_by_id(shown_card_db_id)
					: nullptr;

				if (entry && entry->has_rulings() && shown_card.width > 0 &&
				    shown_card.height > 0) {
					// La caja nace donde esta la carta mostrada (sobre el ankh
					// o a media pantalla) y es un poco mas ancha que ella,
					// porque con la caja estrecha los rulings se volvian
					// ilegibles. La metrica del texto escala con el tamaño de
					// la carta (shown_metric), no con el overlay.
					const int pad = 2;
					const int side = (int)(12 * fit.esc + 0.5);
					vtes_overlay::RulingsBox box;
					box.x = shown_card.x - side;
					box.w = shown_card.width + 2 * side;
					box.y = shown_card.y + pad;
					box.h = shown_card.height - 2 * pad;

					const double osy = fit.esc * shown_metric;
					const int lh = (int)(vtes_overlay::kRulingsLineHeight * osy + 0.5);
					const double fs = vtes_overlay::kRulingsFontScale * osy;

					// El ancho real de cada linea solo lo sabe OpenCV: putText es
					// proporcional y un avance fijo por caracter se pasa de ancho.
					// A esta escala el glifo medio mide 7.07 px con el OpenCV 5.0.0
					// que enlaza el plugin (9.15 con el 4.13 de otro entorno: un 30%
					// mas ancho por el mismo tamano de fuente), asi que no hay
					// constante que valga para todos. La unidad de rulings recibe
					vtes_overlay::RulingsMeasure measure =
						[fs](const std::string& s) -> int {
							if (s.empty())
								return 0;
							int tw = 0;
							cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX,
							                fs, 1, &tw);
							return tw;
						};

					const vtes_overlay::RulingsLayout layout =
						vtes_overlay::buildRulingsLayout(entry, box, measure);

					if (!layout.lines.empty() && lh > 0 && box.w > 0) {
						// Sin recuadro de fondo: el texto se dibuja directo
						// sobre la carta (peticion del usuario, 2026-10-17).

						// El slider es un porcentaje del desplazamiento maximo
						// de esta carta: con dos rulings el maximo es 0 y el
						// slider no hace nada, con dieciocho recorre todo.
						const int max_scroll =
							vtes_overlay::maxScroll(layout, box, lh);
						double frac = tf->rulings_scroll / 100.0;
						if (frac < 0.0)
							frac = 0.0;
						if (frac > 1.0)
							frac = 1.0;
						const int scroll_px = (int)(max_scroll * frac + 0.5);

						std::pair<size_t, size_t> range =
							vtes_overlay::visibleLineRange(layout, box,
							                               lh, scroll_px);

						// Barra de desplazamiento a la derecha de la caja: sin
						// ella no hay forma de saber cuanto texto queda detras,
						// porque el slider de OBS no muestra nada.
						if (max_scroll > 0) {
							const int track_x = box.x + box.w + 2;
							const int track_w = 3;
							const int content = vtes_overlay::contentHeight(
								layout, lh);
							cv::rectangle(display,
							              cv::Rect(track_x, box.y,
							                       track_w, box.h),
							              cv::Scalar(40, 40, 48),
							              cv::FILLED);
							int thumb_h = (int)((double)box.h * box.h /
							                        std::max(1, content) + 0.5);
							thumb_h = std::max(4, std::min(thumb_h, box.h));
							const int thumb_y = (int)((double)(box.h - thumb_h)
							                          * frac + 0.5);
							cv::rectangle(display,
							              cv::Rect(track_x,
							                       box.y + thumb_y,
							                       track_w, thumb_h),
							              cv::Scalar(120, 140, 160),
							              cv::FILLED);
						}

						const cv::Scalar kBodyColor(200, 205, 215);
						const cv::Scalar kHeadColor(130, 225, 250);

						for (size_t i = range.first; i < range.second; ++i) {
							const vtes_overlay::RulingsLine& line =
								layout.lines[i];
							// La linea ya se midio al ajustarla, pero se vuelve
							// a comprobar: si algo la deja larga, se descarta
							// antes de que se salga de la caja.
							int tw = 0;
							cv::getTextSize(line.text,
							                cv::FONT_HERSHEY_SIMPLEX,
							                fs, 1, &tw);
							if (tw <= 0 || tw > box.w)
								continue;

							const int y = box.y + lh + (int)i * lh - scroll_px;
							cv::putText(display, line.text,
							            cv::Point(box.x, y),
							            cv::FONT_HERSHEY_SIMPLEX, fs,
							            line.is_heading ? kHeadColor
							                            : kBodyColor,
							            1);
						}
					}
				}
			}
		}



			// Replace frame with display for subsequent drawing
			frame = display;
		}

		// ─── GPU indicator ─────────────────────────────────────────────────
		{
			std::string gpu_status;
			if (tf->detectionMode == DETECT_MODE_ONNX && tf->yolo_detector) {
				const auto& prov = tf->yolo_detector->provider();
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
			if (tf->detectionMode == DETECT_MODE_ONNX && tf->yolo_detector) {
				cv::putText(frame, tf->yolo_detector->provider(),
					    cv::Point2f(8, 36),
					    cv::FONT_HERSHEY_SIMPLEX, 0.35,
					    cv::Scalar(200, 200, 200), 1);
			}
		}

		std::lock_guard<std::mutex> lock(tf->outputLock);
		cv::cvtColor(frame, tf->outputPreviewBGRA, cv::COLOR_BGR2BGRA);
	}
	} catch (const cv::Exception &e) {
		obs_log(LOG_ERROR, "[video_tick] OpenCV error: %s (file=%s line=%d func=%s)",
			e.what(), e.file, e.line, e.func);
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "[video_tick] Exception: %s", e.what());
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
	if (tf->detectionMode == DETECT_MODE_ONNX && !tf->yolo_detector) {
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
// force recompile
