#include "detect-filter-obb.h"
#include "vtes_icons.h"

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
#include <cstring>
#include <cctype>

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
#include <climits>
#include <limits>
#include <iterator>

// stb_truetype (dominio publico): rasteriza los TTF de los nombres.
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

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

// En VTES ningún vampiro tiene capacidad superior a 11. Es el coste máximo
// de una influencia completa; se valida por si el dato de la base de cartas
// viniera corrupto o la carta no fuera un vampiro influenciable.
static constexpr int kMaxVampireCapacity = 11;

struct detect_filter_obb : public filter_data {};

// Forward declarations: se usan al construir el JSON de /game/status,
// mucho antes de la implementación que está junto al video tick.
static int activeAutoZonePlayers(const detect_filter_obb *tf);
static int committedAutoZonePlayers(const detect_filter_obb *tf);

// El OCR necesita bastantes pixeles por letra para ser fiable. En una mesa de
// juego la carta completa suele ocupar 50-150 px de alto y el nombre queda en
// 2-5 px: Tesseract no puede leerlo. En ese rango manda el embedding; el OCR
// se reserva para primeros planos, donde si aporta una confirmacion.
static constexpr float kMinOcrSourceHeight = 200.0f;
static constexpr float kMinCardSide = 12.0f;

// ─── Card region extraction ─────────────────────────────────────────────────

// Perspective-correct a region defined by 4 corners into a top-down view.
// Se mantiene el color original porque el indice de embeddings y el bloque
// NCC se construyen sin CLAHE; el OCR realza su propia copia por separado.
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
	if (raw_w < kMinCardSide || raw_h < kMinCardSide) return cv::Mat();

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

	// Output size: true card edges, not the axis-aligned bounding box of the
	// rotated corners. Using raw_w/raw_h on a tilted card added background
	// around it and diluted both the embedding and the NCC block.
	constexpr double vtes_aspect = 88.0 / 63.0; // h/w
	auto edge_len = [](const cv::Point2f& a, const cv::Point2f& b) {
		return std::hypot(a.x - b.x, a.y - b.y);
	};
	const float edge_a = 0.5f * (edge_len(src_pts[0], src_pts[1]) +
		                          edge_len(src_pts[3], src_pts[2]));
	const float edge_b = 0.5f * (edge_len(src_pts[0], src_pts[3]) +
		                          edge_len(src_pts[1], src_pts[2]));
	const float card_long = std::max(edge_a, edge_b);
	// No dejamos que una deteccion lejana entregue un recorte de 30-60 px al
	// embedder. El modelo trabaja a 224x224; ampliamos aqui, antes del warp,
	// para que la interpolacion no destruya bordes y texto. El limite 224 no
	// inventa detalle, pero conserva mejor la estructura que llegar a 128 y
	// volver a ampliar despues.
	int out_h = std::max((int)std::lround(card_long), 224);
	int out_w = std::max((int)std::lround(out_h / vtes_aspect), 160);

	cv::Point2f dst_pts[4] = {
		{0, 0}, {(float)out_w - 1, 0}, {(float)out_w - 1, (float)out_h - 1}, {0, (float)out_h - 1}
	};

	cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
	cv::Mat warped;
	const int interp = (out_h > (int)std::lround(card_long))
		? cv::INTER_CUBIC : cv::INTER_LINEAR;
	cv::warpPerspective(frame, warped, M, cv::Size(out_w, out_h), interp,
	                    cv::BORDER_REPLICATE);

	// El indice de embeddings se construye sin CLAHE (ver
	// scripts/build_embedding_index.py). Aplicar CLAHE solo a la consulta
	// rompia esa correspondencia y bajaba la similitud de las cartas
	// correctas. El OCR aplica su propio CLAHE en preprocessForOcr().
	return warped;
}

// dHash 9x8 del recorte. Sirve para casar el resultado asincrono del OCR
// con el recorte exacto que lo origino; un track puede reutilizarse al
// cambiar de carta y antes el nombre viejo se aplicaba a la carta nueva.
static uint64_t cardCropHash(const cv::Mat& bgr)
{
	if (bgr.empty()) return 0;
	cv::Mat gray, small;
	cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
	cv::resize(gray, small, cv::Size(9, 8), 0, 0, cv::INTER_AREA);
	uint64_t h = 0;
	for (int y = 0; y < 8; ++y) {
		for (int x = 0; x < 8; ++x) {
			h <<= 1;
			if (small.at<uchar>(y, x) > small.at<uchar>(y, x + 1)) h |= 1;
		}
	}
	return h;
}

// Varianza del laplaciano normalizada a 224x224. Sirve para diagnosticar
// capturas desenfocadas: una carta enfocada ronda 3000-8000; una captura
// borrosa de la C310 puede quedar por debajo de 100. No se usa para
// rechazar todavia; se registra junto al resultado del embedding.
static float cardRegionSharpness(const cv::Mat& bgr)
{
	if (bgr.empty()) return 0.0f;
	cv::Mat gray, small;
	cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
	cv::resize(gray, small, cv::Size(224, 224), 0, 0, cv::INTER_AREA);
	cv::Mat lap;
	cv::Laplacian(small, lap, CV_64F);
	cv::Scalar mean, stddev;
	cv::meanStdDev(lap, mean, stddev);
	return (float)(stddev[0] * stddev[0]);
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
			track.last_crop_hash = det.crop_hash;
			track.last_label = det.label;
			track.history.push_back(true);
			if ((int)track.history.size() > filter_data::TEMPORAL_WINDOW)
				track.history.pop_front();

			// Update identity if detection has card_id. La comparacion de
			// "mejor" debe usar la confianza de identidad, no track.confidence
			// (que mide estabilidad de la caja, no del nombre).
			if (!det.card_id.empty() && det.card_id == track.tracked_card_id) {
				track.identity_hits++;
				track.identity_score = std::max(track.identity_score, det.prob);
			} else if (!det.card_id.empty()) {
				const float identity_ratio = track.identity_total > 0
					? (float)track.identity_hits / (float)track.identity_total : 0.0f;
				bool current_weak = track.identity_total < 3 || identity_ratio < 0.5f;
				bool new_much_better = det.prob > track.identity_score + 0.12f;
				if (current_weak || new_much_better) {
					track.tracked_card_id = det.card_id;
					track.tracked_card_name = det.card_name;
					track.identity_hits = 1;
					track.identity_total = 0;
					track.identity_score = det.prob;
				} else if (track.identity_hits > 0) {
					track.identity_hits--;
				}
			} else if (track.identity_hits > 0) {
				// Un frame sin identificacion (borroso o ambiguo) no debe
				// mantener viva indefinidamente una identidad antigua.
				track.identity_hits--;
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
			track.last_crop_hash = det.crop_hash;
			track.last_label = det.label;
			track.tracked_card_id = det.card_id;
			track.tracked_card_name = det.card_name;
			track.identity_hits = det.card_id.empty() ? 0 : 1;
			track.identity_total = 1;
			track.identity_score = det.card_id.empty() ? 0.0f : det.prob;
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
			obj.label = track.last_label;
			obj.prob = track.confidence;
			obj.id = track.id;
			obj.unseenFrames = track.unseen_count;
			obj.crop_hash = track.last_crop_hash;

			// Use smoothed identity if majority of frames agree. Exigimos
			// al menos 3 aciertos: una unica deteccion aislada (tipica de
			// una captura borrosa) no debe entrar en la cola del realizador.
			if (!track.tracked_card_id.empty() &&
			    track.identity_total >= 3 &&
			    track.identity_hits >= 3 &&
			    (float)track.identity_hits / track.identity_total > 0.5f) {
				obj.card_id = track.tracked_card_id;
				obj.card_name = track.tracked_card_name;
				obj.prob = track.identity_score;
			}

			output_objects.push_back(obj);
		}
	}
}

// ─── OBS filter callbacks ──────────────────────────────────────────────────

namespace {

const char *tf_property_key(const char *name)
{
	if (!name)
		return nullptr;

	static const struct {
		const char *name;
		const char *key;
	} labels[] = {
		{"preview", "Preview"},
		{"always_active", "AlwaysActive"},
		{"language", "Language"},
		{"threshold", "ConfThreshold"},
		{"detection_mode", "DetectionMode"},
		{"inference_device", "InferenceDevice"},
		{"classifier_enabled", "TypeClassifier"},
		{"ocr_enabled", "OcrEnabled"},
		{"upscaler_enabled", "NisUpscale"},
		{"vampire_cuda_enabled", "OvalRectClassifier"},
		{"detection_interval_seconds", "DetectionInterval"},
		{"rulings_scroll", "RulingsScroll"},
		{"click_to_overlay", "ClickToOverlay"},
		{"card_search_btn", "OpenCardSearch"},
		{"player_count", "PlayerCount"},
		{"show_zones", "ShowZones"},
		{"auto_zones", "AutoZones"},
		{"auto_zones_hint", "AutoZonesHint"},
		{"big_card", "BigCard"},
		{"overlay_font", "OverlayFont"},
		{"cam_hint", "CamHint"},
		{"cam_full_frame", "CamFullFrame"},
		{"cam_zoom", "CamZoom"},
		{"cam_offset_x", "CamOffsetX"},
		{"cam_offset_y", "CamOffsetY"},
	};

	for (const auto &label : labels) {
		if (std::strcmp(name, label.name) == 0)
			return label.key;
	}
	return nullptr;
}

/* Separa "player_3_minion_angle" en indice 3 y campo "minion_angle". */
bool tf_player_field(const char *name, int &player, std::string &field)
{
	if (!name)
		return false;

	const std::string value = name;
	const std::string prefix = "player_";
	if (value.compare(0, prefix.size(), prefix) != 0)
		return false;

	size_t sep = value.find('_', prefix.size());
	if (sep == std::string::npos || sep == prefix.size())
		return false;

	int index = 0;
	for (size_t i = prefix.size(); i < sep; ++i) {
		if (value[i] < '0' || value[i] > '9')
			return false;
		index = index * 10 + (value[i] - '0');
	}

	player = index;
	field = value.substr(sep + 1);
	return true;
}

void tf_refresh_list(obs_property_t *property, const char *const *keys,
		     const char *const *values, size_t count)
{
	obs_property_list_clear(property);
	for (size_t i = 0; i < count; ++i)
		obs_property_list_add_string(property, tf_text(keys[i]), values[i]);
}

void tf_refresh_property_labels(obs_properties_t *properties)
{
	if (!properties)
		return;

	obs_property_t *property = obs_properties_first(properties);
	while (property) {
		const char *name = obs_property_name(property);
		int player = -1;
		std::string field;

		if (tf_player_field(name, player, field)) {
			if (field == "group") {
				const std::string label = std::string(tf_text("Player")) +
							  " " + std::to_string(player + 1);
				obs_property_set_description(property, label.c_str());
			} else if (field == "name") {
				obs_property_set_description(property, tf_text("PlayerName"));
			} else if (field == "pool") {
				obs_property_set_description(property, tf_text("InitialPool"));
			} else if (field == "active") {
				obs_property_set_description(property, tf_text("PlayerActive"));
			} else if (field == "minion_angle") {
				obs_property_set_description(property, tf_text("MinionAngle"));
			} else if (field == "crypt_angle") {
				obs_property_set_description(property, tf_text("CryptAngle"));
			}
		} else {
			const char *key = tf_property_key(name);
			if (key)
				obs_property_set_description(property, tf_text(key));
		}

		const obs_property_type type = obs_property_get_type(property);
		if (type == OBS_PROPERTY_GROUP) {
			tf_refresh_property_labels(obs_property_group_content(property));
		} else if (type == OBS_PROPERTY_LIST) {
			if (name && std::strcmp(name, "detection_mode") == 0) {
				static const char *keys[] = {"Shape (Contour)", "AI Model (ONNX)"};
				static const char *values[] = {DETECT_MODE_CONTOUR, DETECT_MODE_ONNX};
				tf_refresh_list(property, keys, values, 2);
			} else if (name && std::strcmp(name, "inference_device") == 0) {
				static const char *keys[] = {
					"CPU",
#if _WIN32
					"GPU (DirectML)", "GPU (CUDA)",
#endif
				};
				static const char *values[] = {
					INFERENCE_CPU,
#if _WIN32
					INFERENCE_DML, INFERENCE_CUDA,
#endif
				};
				tf_refresh_list(property, keys, values,
#if _WIN32
						3
#else
						1
#endif
				);
			} else if (name && std::strcmp(name, "overlay_font") == 0) {
				static const char *keys[] = {"FontCinzel", "FontGrenze",
							     "FontClassic"};
				static const char *values[] = {"cinzel", "grenze", "hershey"};
				tf_refresh_list(property, keys, values, 3);
			}
		}

		obs_property_next(&property);
	}
}

} // namespace

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
	// cargas; el callback refresca las etiquetas del panel en el acto.
	obs_property_t *p_lang = obs_properties_add_list(props, "language", tf_text("Language"),
							 OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p_lang, "Español", "es");
	obs_property_list_add_string(p_lang, "English", "en");
	obs_property_set_modified_callback2(
		p_lang,
		[](void *, obs_properties_t *props, obs_property_t *, obs_data_t *settings) {
			tf_i18n_set(obs_data_get_string(settings, "language"));
			tf_refresh_property_labels(props);
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
	obs_properties_add_bool(props, "auto_zones", tf_text("AutoZones"));
	obs_properties_add_text(props, "auto_zones_hint",
		tf_text("AutoZonesHint"), OBS_TEXT_INFO);
	obs_properties_add_bool(props, "big_card", tf_text("BigCard"));

	obs_property_t *p_font = obs_properties_add_list(
		props, "overlay_font", tf_text("OverlayFont"),
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p_font, tf_text("FontCinzel"), "cinzel");
	obs_property_list_add_string(p_font, tf_text("FontGrenze"), "grenze");
	obs_property_list_add_string(p_font, tf_text("FontClassic"), "hershey");

	// Encuadre de la camara bajo el overlay (sin barras negras)
	obs_properties_add_text(props, "cam_hint", tf_text("CamHint"), OBS_TEXT_INFO);
	obs_properties_add_bool(props, "cam_full_frame", tf_text("CamFullFrame"));
	obs_properties_add_float_slider(props, "cam_zoom", tf_text("CamZoom"),
	                                100.0, 300.0, 1.0);
	obs_properties_add_int_slider(props, "cam_offset_x", tf_text("CamOffsetX"),
	                              -400, 400, 1);
	obs_properties_add_int_slider(props, "cam_offset_y", tf_text("CamOffsetY"),
	                              -400, 400, 1);

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

	// ONNX area filter defaults. En una mesa real una carta puede ocupar solo
	// 20x30 px (600 px^2); el limite anterior de 4000 px^2 la descartaba antes
	// de llegar al embedding. El embedding sigue identificando a 40 px de alto
	// con margen amplio; dejamos el suelo en 300 y que confianza, tipo y
	// embedding filtren los falsos positivos.
	obs_data_set_default_int(settings, "onnx_min_area", 300);
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
	obs_data_set_default_bool(settings, "auto_zones", false);
	obs_data_set_default_bool(settings, "big_card", false);
	obs_data_set_default_string(settings, "overlay_font", "cinzel");
	obs_data_set_default_bool(settings, "cam_full_frame", false);
	obs_data_set_default_double(settings, "cam_zoom", 100.0);
	obs_data_set_default_int(settings, "cam_offset_x", 0);
	obs_data_set_default_int(settings, "cam_offset_y", 0);
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
		const bool count_changed = (new_count != tf->player_count);
		tf->player_count = new_count;
		tf->show_zones = obs_data_get_bool(settings, "show_zones");
		tf->big_card = obs_data_get_bool(settings, "big_card");

		// La detección automática se reinicia al activarla/desactivarla o
		// al cambiar el número de jugadores. Las zonas manuales se siguen
		// leyendo como respaldo hasta que el detector confirme un grupo.
		const bool auto_wanted = obs_data_get_bool(settings, "auto_zones");
		const bool auto_changed =
			(auto_wanted != tf->auto_zones) ||
			(auto_wanted && count_changed);
		if (auto_changed) {
			std::lock_guard<std::mutex> alock(tf->auto_zone_lock);
			tf->auto_zones = auto_wanted;
			if (!tf->auto_zone_detector)
				tf->auto_zone_detector =
					std::make_unique<vtes_detection::AutoZoneDetector>();
			else
				tf->auto_zone_detector->reset();
			tf->auto_zone_commit_count = 0;
			tf->auto_zone_last_run_ns = 0;
			tf->auto_zone_batch_start_ns = 0;
			{
				std::lock_guard<std::mutex> vlock(tf->auto_zone_view_lock);
				tf->auto_zone_views.clear();
			}
			std::lock_guard<std::mutex> glock(tf->game_state_lock);
			for (int i = 0; i < filter_data::MAX_PLAYERS; ++i) {
				tf->players[i].auto_zone = false;
				tf->players[i].auto_zone_id = -1;
			}
		} else {
			tf->auto_zones = auto_wanted;
		}

		const char *font = obs_data_get_string(settings, "overlay_font");
		const std::string font_name = font ? font : "";
		tf->overlay_font_mode = font_name == "hershey" ? 2
		                      : font_name == "grenze"  ? 1
		                                               : 0;

		tf->cam_full_frame = obs_data_get_bool(settings, "cam_full_frame");
		tf->cam_zoom = obs_data_get_double(settings, "cam_zoom") / 100.0;
		if (tf->cam_zoom < 1.0) tf->cam_zoom = 1.0;
		if (tf->cam_zoom > 3.0) tf->cam_zoom = 3.0;
		tf->cam_off_x = (int)obs_data_get_int(settings, "cam_offset_x");
		tf->cam_off_y = (int)obs_data_get_int(settings, "cam_offset_y");
		tf->cam_off_x = std::max(-400, std::min(400, tf->cam_off_x));
		tf->cam_off_y = std::max(-400, std::min(400, tf->cam_off_y));

		std::lock_guard<std::mutex> glock(tf->game_state_lock);
		for (int i = 0; i < filter_data::MAX_PLAYERS; i++) {
			std::string p = "player_" + std::to_string(i);
			tf->players[i].name = obs_data_get_string(settings, (p + "_name").c_str());
			tf->players[i].pool = (int)obs_data_get_int(settings, (p + "_pool").c_str());

			// Si la zona la ha creado el detector automático, no se pisa
			// con los valores manuales mientras la opción siga activa.
			const bool keep_auto =
				tf->auto_zones && tf->players[i].auto_zone;
			if (!keep_auto) {
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
				tf->players[i].minion_zone.lock_angle = tf->players[i].minion_zone.angle;
				tf->players[i].crypt_zone.lock_angle = tf->players[i].crypt_zone.angle;
				if (!tf->auto_zones) {
					tf->players[i].auto_zone = false;
					tf->players[i].auto_zone_id = -1;
				}
			}

			// Un jugador que sale de la partida (ousted) deja su sitio libre:
			// su zona deja de asignar cartas y se le quitan las ya asignadas.
			const bool was_active = tf->players[i].active;
			tf->players[i].active = obs_data_get_bool(settings, (p + "_active").c_str());
			if (was_active && !tf->players[i].active) {
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
		obs_log(LOG_INFO, "  Auto Zones: %s", tf->auto_zones ? "true" : "false");
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
							const std::vector<std::string>& type_filters,
							const std::vector<uint64_t>& crop_hashes,
							const std::vector<char>& ocr_allowed);
static void drain_ocr_results(detect_filter_obb* tf,
							  std::vector<vtes_detection::OBBObject>& objects);

// ─── Modo realizador: pendientes + feedback forward declarations ────
static void addPendingDetection(detect_filter_obb *tf,
								const vtes_detection::OBBObject &obj,
								const cv::Mat &frame, int64_t now_ns);
static void recordDirectorFeedback(detect_filter_obb *tf, const char *kind,
								   const filter_data::PendingDetection *chosen,
								   const std::vector<filter_data::PendingDetection> &pool);

// Registro de eventos de partida (historial web). El llamador debe tener
// game_state_lock.
static void pushGameEvent(detect_filter_obb *tf, const char *type,
						  int player_idx, const std::string &card_name,
						  int from, int to, int64_t now_ns,
						  int capacity = 0,
						  const std::string &resource = std::string());
// Ajusta el pool y gestiona auto-ousted. El llamador debe tener
// game_state_lock. Devuelve el delta realmente aplicado.
static int applyPoolDeltaLocked(detect_filter_obb *tf, int player_idx,
				int delta, int64_t now_ns);

void *detect_filter_obb_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "VTES OBB filter created");
	void *data = bmalloc(sizeof(struct detect_filter_obb));
	struct detect_filter_obb *tf = new (data) detect_filter_obb();

	tf->source = source;
	tf->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	tf->lastDetectedObjectId = -1;
	tf->auto_zone_detector =
		std::make_unique<vtes_detection::AutoZoneDetector>();

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
	tf->web_server->on_overlay_fn = [tf](const std::string& url,
	                                      const std::string& card_id) {
		{
			std::lock_guard<std::mutex> lock(tf->game_state_lock);
			tf->current_overlay_card_url = url;
			tf->current_overlay_card_id = card_id;
			tf->current_overlay_fallback.release();
		}
		obs_log(LOG_INFO, "[WebServer] Overlay manual: %s (id=%s)",
		        url.c_str(), card_id.c_str());
	};
	tf->web_server->on_overlay_clear_fn = [tf]() {
		{
			std::lock_guard<std::mutex> lock(tf->game_state_lock);
			tf->current_overlay_card_url.clear();
			tf->current_overlay_card_id.clear();
			tf->current_overlay_fallback.release();
		}
		obs_log(LOG_INFO, "[WebServer] Overlay manual limpiado");
	};
	tf->web_server->on_pool_fn = [tf](int player_idx, int delta) {
		const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
		std::lock_guard<std::mutex> lock(tf->game_state_lock);
		applyPoolDeltaLocked(tf, player_idx, delta, now_ns);
	};
	tf->web_server->on_minion_resource_fn =
		[tf](int player_idx, int track_id, const std::string &resource,
		     int delta) {
			const auto now_ns =
				std::chrono::duration_cast<std::chrono::nanoseconds>(
					std::chrono::steady_clock::now().time_since_epoch())
					.count();
			std::lock_guard<std::mutex> lock(tf->game_state_lock);
			auto it = tf->tracked_cards.find(track_id);
			if (it == tf->tracked_cards.end() ||
			    it->second.player_idx != player_idx)
				return;
			auto &tc = it->second;
			if (tc.in_crypt || tc.card_id.empty())
				return;
			// No permitir contadores que no correspondan al tipo de carta:
			// los aliados/retainers tienen vida, no sangre.
			const VTESCardEntry *entry = tf->vtes_db.get_by_id(tc.card_id);
			auto has_type = [&](const char *type) {
				if (!entry)
					return false;
				for (const auto &t : entry->types)
					if (t == type)
						return true;
				return false;
			};
			int *value = nullptr;
			if (resource == "blood" && has_type("Vampire"))
				value = &tc.blood;
			else if (resource == "life" &&
				 (has_type("Ally") || has_type("Retainer")))
				value = &tc.life;
			else if (resource == "conviction" && has_type("Imbued"))
				value = &tc.conviction;
			if (!value)
				return;
			const int old_value = *value < 0 ? 0 : *value;
			int new_value = old_value + delta;
			if (new_value < 0)
				new_value = 0;
			if (new_value > 99)
				new_value = 99;
			if (new_value == old_value && *value >= 0)
				return;
			*value = new_value;
			pushGameEvent(tf, "resource", player_idx, tc.card_name,
				      old_value, new_value, now_ns, 0, resource);
			obs_log(LOG_INFO,
				"[Game] Jugador %d: \"%s\" %s = %d (delta %+d)",
				player_idx + 1, tc.card_name.c_str(),
				resource.c_str(), new_value, delta);
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
			pj["auto_zone"] = pl.auto_zone;
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
				mj["track_id"] = tc.track_id;
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
				const VTESCardEntry *entry =
					tf->vtes_db.get_by_id(tc.card_id);
				if (capacity == 0 && entry)
					capacity = entry->capacity;
				mj["capacity"] = capacity;
				mj["blood"] = tc.blood;
				mj["life"] = tc.life;
				mj["conviction"] = tc.conviction;
				if (entry) {
					mj["types"] = entry->types;
					mj["pool_cost"] = entry->pool_cost;
					mj["blood_cost"] = entry->blood_cost;
					mj["conviction_cost"] = entry->conviction_cost;
				}
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
			pj["influence_count"] = pl.influence_count;
			pj["influence_cost"] = pl.influence_cost;
			players_json.push_back(std::move(pj));
		}
		out["auto_zones"] = tf->auto_zones;
		out["auto_zone_committed"] = committedAutoZonePlayers(tf);
		{
			nlohmann::json views = nlohmann::json::array();
			std::lock_guard<std::mutex> view_lock(tf->auto_zone_view_lock);
			for (const auto &v : tf->auto_zone_views) {
				nlohmann::json vj;
				vj["stable"] = v.stable;
				vj["player"] = v.player_idx;
				vj["cards"] = v.card_count;
				vj["score"] = v.score;
				vj["crypt"] = {{"x", v.crypt.nx}, {"y", v.crypt.ny},
					       {"w", v.crypt.nw}, {"h", v.crypt.nh},
					       {"angle", v.crypt.angle}};
				vj["minion"] = {{"x", v.minion.nx}, {"y", v.minion.ny},
						{"w", v.minion.nw}, {"h", v.minion.nh},
						{"angle", v.minion.angle}};
				views.push_back(std::move(vj));
			}
			out["auto_zone_views"] = std::move(views);
		}
		nlohmann::json events = nlohmann::json::array();
		const size_t ev_total = tf->game_events.size();
		const size_t ev_start = ev_total > 30 ? ev_total - 30 : 0;
		for (size_t i = ev_start; i < ev_total; i++) {
			const filter_data::GameEvent &ev = tf->game_events[i];
			nlohmann::json ej;
			ej["type"] = ev.type;
			ej["player"] = ev.player_idx;
			ej["card"] = ev.card_name;
			ej["from"] = ev.from;
			ej["to"] = ev.to;
			ej["capacity"] = ev.capacity;
			if (!ev.resource.empty())
				ej["resource"] = ev.resource;
			ej["age_ms"] = (now_ns - ev.ts_ns) / 1000000LL;
			events.push_back(std::move(ej));
		}
		out["events"] = std::move(events);
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
			tf->current_overlay_fallback.release();
			if (!chosen.crop_jpg.empty()) {
				cv::Mat fb = cv::imdecode(chosen.crop_jpg, cv::IMREAD_COLOR);
				if (!fb.empty())
					tf->current_overlay_fallback = fb;
			}
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
				ocr_name, ocr_id, ocr_conf, job.crop_hash});
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
							const std::vector<std::string>& type_filters,
							const std::vector<uint64_t>& crop_hashes,
							const std::vector<char>& ocr_allowed)
{
	if (!tf->ocr_worker_active || !tf->ocr_reader) return;
	// El OCR es auxiliar y mas lento que el video. Si se encola un job por
	// frame, los resultados llegan con segundos de retraso y de cartas que ya
	// no estan en pantalla. Mantener la cola corta reduce latencia y basura.
	{
		std::lock_guard<std::mutex> lock(tf->ocr_queue_lock);
		if (tf->ocr_queue.size() >= 2) return;
	}
	for (size_t i = 0; i < card_regions.size() && i < type_filters.size(); i++) {
		if (card_regions[i].empty()) continue;
		// Solo se encola lo que el OCR puede leer de verdad: primeros planos
		// que aun no tienen una identificacion segura por embedding. Con una
		// carta pequena el nombre ocupa 2-5 px y Tesseract solo puede inventar.
		if (i >= ocr_allowed.size() || !ocr_allowed[i]) continue;
		// El ticket del job ES el track id del objeto, para que
		// drain_ocr_results pueda devolver cada nombre a su carta.
		int64_t jid = track_ids[i];
		{
			std::lock_guard<std::mutex> lock(tf->ocr_queue_lock);
			tf->ocr_queue.push({jid, card_regions[i].clone(), type_filters[i],
				i < crop_hashes.size() ? crop_hashes[i] : 0});
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
			// El mismo track puede reutilizarse cuando el operador cambia de
			// carta. Sin comparar el recorte, un OCR lento de la carta anterior
			// sobrescribia la identificacion de la nueva (p. ej. Eyes of Argus
			// sobre Dust Up).
			if (result.crop_hash != obj.crop_hash) {
				obs_log(LOG_INFO, "[OCR] Async result descartado: recorte distinto (track %d)",
						result.id);
				continue;
			}
			// El embedding manda cuando ya hay una identificacion segura; el
			// OCR solo confirma o rellena huecos. Asi una lectura vieja no puede
			// cambiar una carta ya reconocida.
			if (!obj.card_id.empty() && obj.card_id != result.card_id &&
			    (obj.alt_card_id.empty() || obj.alt_card_id != result.card_id)) {
				obs_log(LOG_INFO, "[OCR] Async result '%s' descartado: embedding dice '%s' (track %d)",
						result.card_name.c_str(), obj.card_name.c_str(), result.id);
				continue;
			}
			// Empate top-1/top-2: las dos candidatas ya estan en la cola del
			// realizador. Dejar que el OCR elija una tercera carta convertiria
			// una ambiguedad conocida en un falso positivo silencioso.
			if (obj.card_id.empty() && !obj.alt_card_id.empty()) {
				obs_log(LOG_INFO, "[OCR] Async result '%s' descartado: embedding ambiguo (track %d)",
						result.card_name.c_str(), result.id);
				continue;
			}
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

// Ventana de camara = caja del area transparente del overlay (canal alfa).
// Asi el encuadre sigue al diseño si este cambia, sin tocar constantes.
static cv::Rect overlayCameraWindow(const cv::Mat& bg)
{
	if (bg.empty() || bg.channels() != 4)
		return {};

	cv::Mat alpha, mask;
	cv::extractChannel(bg, alpha, 3);
	cv::threshold(alpha, mask, 15, 255, cv::THRESH_BINARY_INV);

	std::vector<cv::Point> pts;
	cv::findNonZero(mask, pts);
	if (pts.empty())
		return {};

	cv::Rect r = cv::boundingRect(pts);
	if (r.width < 8 || r.height < 8 || r.area() >= bg.total() * 95 / 100)
		return {};
	return r;
}

// Encuadre de la camara bajo el overlay: escala tipo "cover" sobre el hueco
// del diseño (la imagen cubre el hueco entero, sin barras negras), con zoom
// y desplazamiento ajustables por el usuario. El desplazamiento se recorta
// para que la camara siga cubriendo el hueco: nunca quedan bordes negros.
struct CameraView {
	double   scale = 1.0;        // pixeles de camara -> pixeles de frame
	double   ox = 0.0, oy = 0.0; // esquina sup-izq de la imagen (px de frame)
	cv::Rect win;                // hueco de camara (px de frame)
};

static CameraView cameraView(const OverlayFit& fit, int fw, int fh,
                             const cv::Rect& win_design, double zoom,
                             int off_x, int off_y, bool full_frame)
{
	CameraView view;
	// A sangre: la camara cubre todo el frame (las bandas laterales del
	// diseno 3:2 tambien muestran camara). Si no, solo cubre el hueco.
	view.win = full_frame
		? cv::Rect(0, 0, fw, fh)
		: designRect(fit, win_design.x, win_design.y,
		             win_design.width, win_design.height);
	if (view.win.width < 8 || view.win.height < 8 || fw < 8 || fh < 8)
		return view;

	if (zoom < 1.0)
		zoom = 1.0;
	const double esc = std::max((double)view.win.width / fw,
	                            (double)view.win.height / fh) * zoom;
	const double iw = fw * esc, ih = fh * esc;
	if (iw < 8.0 || ih < 8.0)
		return view;

	// Desplazamiento en pixeles de diseño -> frame, recortado para que la
	// imagen siga cubriendo el hueco completo.
	const double max_x = std::max(0.0, (iw - view.win.width) * 0.5);
	const double max_y = std::max(0.0, (ih - view.win.height) * 0.5);
	const double dx = std::max(-max_x, std::min(max_x, off_x * fit.esc));
	const double dy = std::max(-max_y, std::min(max_y, off_y * fit.esc));

	view.scale = esc;
	view.ox = view.win.x + (view.win.width - iw) * 0.5 + dx;
	view.oy = view.win.y + (view.win.height - ih) * 0.5 + dy;
	return view;
}

// ─── Fuentes TTF para los nombres del overlay ─────────────────────────
// stb_truetype rasteriza los TTF de data/obs-plugins/…/fonts/. El texto se
// cachea ya rasterizado (mascara alfa) por texto+tamaño; el color se aplica
// al pegarlo, asi el mismo nombre vale para jugador activo u ousted.
static std::vector<int> utf8Codepoints(const std::string& s)
{
	std::vector<int> out;
	for (size_t i = 0; i < s.size();) {
		const unsigned char c = (unsigned char)s[i];
		int cp = '?';
		int n = 1;
		if (c < 0x80) {
			cp = c;
		} else if ((c & 0xE0) == 0xC0) {
			cp = c & 0x1F;
			n = 2;
		} else if ((c & 0xF0) == 0xE0) {
			cp = c & 0x0F;
			n = 3;
		} else if ((c & 0xF8) == 0xF0) {
			cp = c & 0x07;
			n = 4;
		} else {
			out.push_back('?');
			i++;
			continue;
		}
		if (i + (size_t)n > s.size()) {
			out.push_back('?');
			break;
		}
		bool bad = false;
		for (int k = 1; k < n; k++) {
			const unsigned char cc = (unsigned char)s[i + (size_t)k];
			if ((cc & 0xC0) != 0x80) {
				bad = true;
				break;
			}
			cp = (cp << 6) | (cc & 0x3F);
		}
		out.push_back(bad || cp <= 0 ? '?' : cp);
		i += (size_t)n;
	}
	return out;
}

struct TtfFont {
	const char *file;
	bool tried = false;
	bool ok = false;
	std::vector<unsigned char> ttf;
	stbtt_fontinfo info{};
	std::unordered_map<std::string, cv::Mat> cache;

	explicit TtfFont(const char *f) : file(f) {}

	bool ensure()
	{
		if (tried)
			return ok;
		tried = true;

		char *path = obs_module_file(file);
		if (!path) {
			obs_log(LOG_WARNING, "[Overlay] %s not found; using Hershey",
			        file);
			return false;
		}
		std::ifstream f(path, std::ios::binary);
		bfree(path);
		if (!f)
			return false;
		ttf.assign(std::istreambuf_iterator<char>(f),
		           std::istreambuf_iterator<char>());
		if (ttf.empty())
			return false;

		const int off = stbtt_GetFontOffsetForIndex(ttf.data(), 0);
		if (off < 0 || !stbtt_InitFont(&info, ttf.data(), off)) {
			obs_log(LOG_WARNING, "[Overlay] could not parse %s", file);
			return false;
		}
		ok = true;
		obs_log(LOG_INFO, "[Overlay] font %s loaded (%zu bytes)", file,
		        ttf.size());
		return true;
	}

	int glyphFor(int cp) const
	{
		if (cp == ' ')
			return ' ';
		return stbtt_FindGlyphIndex(&info, cp) != 0 ? cp : '?';
	}

	cv::Mat render(const std::string& text, const cv::Rect& box,
	               double target_h)
	{
		const std::vector<int> cps = utf8Codepoints(text);
		if (cps.empty())
			return {};

		// Altura visual = altura de 'H', para que el tamaño cuadre con el
		// que daba la fuente Hershey (misma caja y mismo target_h).
		int hx0 = 0, hy0 = 0, hx1 = 0, hy1 = 0;
		stbtt_GetCodepointBitmapBox(&info, 'H', 1.0f, 1.0f,
		                            &hx0, &hy0, &hx1, &hy1);
		const float cap_h = (float)(hy1 - hy0);
		if (cap_h <= 0.0f)
			return {};

		float scale = (float)target_h / cap_h;
		const int avail = box.width - std::max(4, box.width / 8);

		auto advanceAt = [&](int cp, int next) {
			int adv = 0, lsb = 0;
			stbtt_GetCodepointHMetrics(&info, glyphFor(cp), &adv, &lsb);
			double a = adv * (double)scale;
			if (next >= 0)
				a += stbtt_GetCodepointKernAdvance(&info, glyphFor(cp),
				                                  glyphFor(next)) *
				     (double)scale;
			return a;
		};

		double total = 0.0;
		for (size_t i = 0; i < cps.size(); i++)
			total += advanceAt(cps[i],
			                   i + 1 < cps.size() ? cps[i + 1] : -1);
		if (total > avail && total > 0.0)
			scale *= (float)(avail / total);

		// Caja real del texto (union de los glifos) para centrarlo sin
		// depender de ascent/descent concretos del TTF.
		struct Glyph {
			int    cp;
			double pen;
		};
		std::vector<Glyph> glyphs;
		double pen = 0.0;
		int minx = INT_MAX, miny = INT_MAX;
		int maxx = INT_MIN, maxy = INT_MIN;
		for (size_t i = 0; i < cps.size(); i++) {
			const int cp = glyphFor(cps[i]);
			glyphs.push_back({cp, pen});
			int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
			stbtt_GetCodepointBitmapBox(&info, cp, scale, scale,
			                            &x0, &y0, &x1, &y1);
			if (x1 > x0 && y1 > y0) {
				minx = std::min(minx, (int)std::floor(pen) + x0);
				maxx = std::max(maxx, (int)std::ceil(pen) + x1);
				miny = std::min(miny, y0);
				maxy = std::max(maxy, y1);
			}
			pen += advanceAt(cps[i], i + 1 < cps.size() ? cps[i + 1] : -1);
		}
		if (minx == INT_MAX || maxx <= minx || maxy <= miny)
			return {};

		const int pad = 1;
		cv::Mat mask(maxy - miny + pad * 2, maxx - minx + pad * 2,
		             CV_8UC1, cv::Scalar(0));

		for (const Glyph& g : glyphs) {
			int gw = 0, gh = 0, gx = 0, gy = 0;
			unsigned char *bmp = stbtt_GetCodepointBitmap(
				&info, scale, scale, g.cp, &gw, &gh, &gx, &gy);
			if (!bmp)
				continue;
			const int dx = (int)std::floor(g.pen) + gx - minx + pad;
			const int dy = gy - miny + pad;
			for (int y = 0; y < gh; y++) {
				const int ty = dy + y;
				if (ty < 0 || ty >= mask.rows)
					continue;
				uchar *row = mask.ptr<uchar>(ty);
				const unsigned char *src = bmp + (size_t)y * gw;
				for (int x = 0; x < gw; x++) {
					const int tx = dx + x;
					if (tx < 0 || tx >= mask.cols)
						continue;
					if (src[x] > row[tx])
						row[tx] = src[x];
				}
			}
			stbtt_FreeBitmap(bmp, nullptr);
		}
		return mask;
	}

	bool draw(cv::Mat& img, const std::string& text, const cv::Rect& box,
	          const cv::Scalar& color, double target_h)
	{
		if (!ensure() || text.empty() || box.width < 6 || box.height < 6 ||
		    target_h < 4.0)
			return false;

		const std::string key =
			text + "|" + std::to_string(box.width) + "x" +
			std::to_string(box.height) + "|" +
			std::to_string((int)std::lround(target_h * 10.0));
		auto it = cache.find(key);
		if (it == cache.end()) {
			cv::Mat mask = render(text, box, target_h);
			if (mask.empty())
				return false;
			if (cache.size() >= 256)
				cache.clear();
			it = cache.emplace(key, std::move(mask)).first;
		}

		const cv::Mat& mask = it->second;
		if (mask.cols > box.width || mask.rows > box.height)
			return false;
		const int x = box.x + (box.width - mask.cols) / 2;
		const int y = box.y + (box.height - mask.rows) / 2;
		const cv::Rect dst = cv::Rect(x, y, mask.cols, mask.rows) &
		                     cv::Rect(0, 0, img.cols, img.rows);
		if (dst.width <= 0 || dst.height <= 0)
			return false;

		const int ox = dst.x - x, oy = dst.y - y;
		const int cb = (int)color[0], cg = (int)color[1], cr = (int)color[2];
		for (int yy = 0; yy < dst.height; yy++) {
			const uchar *m = mask.ptr<uchar>(yy + oy) + ox;
			cv::Vec3b *p = img.ptr<cv::Vec3b>(dst.y + yy) + dst.x;
			for (int xx = 0; xx < dst.width; xx++) {
				const int a = m[xx];
				if (!a)
					continue;
				p[xx][0] = (uchar)((p[xx][0] * (255 - a) + cb * a) / 255);
				p[xx][1] = (uchar)((p[xx][1] * (255 - a) + cg * a) / 255);
				p[xx][2] = (uchar)((p[xx][2] * (255 - a) + cr * a) / 255);
			}
		}
		return true;
	}

	// Dibuja un unico glifo (icono) escalandolo para que llene la caja, en
	// vez de usar la altura de 'H' como referencia: los iconos de ankha2 y
	// vtes-clans ocupan casi todo el em y con draw() quedarian recortados o
	// de tamaños muy dispares.
	bool drawIcon(cv::Mat& img, const std::string& glyph, const cv::Rect& box,
	              const cv::Scalar& color)
	{
		if (!ensure() || glyph.empty() || box.width < 4 || box.height < 4)
			return false;
		auto cps = utf8Codepoints(glyph);
		if (cps.empty())
			return false;
		const int cp = glyphFor(cps[0]);
		const std::string key = "icon:" + std::to_string(cp) + "x" +
		                        std::to_string(box.width) + "x" +
		                        std::to_string(box.height);
		auto it = cache.find(key);
		if (it == cache.end()) {
			int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
			stbtt_GetCodepointBitmapBox(&info, cp, 1.0f, 1.0f, &x0, &y0,
			                            &x1, &y1);
			const int gw = x1 - x0, gh = y1 - y0;
			if (gw <= 0 || gh <= 0)
				return false;
			const double scale = std::min((double)(box.width - 2) / gw,
			                              (double)(box.height - 2) / gh);
			int w = 0, h = 0, ox = 0, oy = 0;
			unsigned char* bmp = stbtt_GetCodepointBitmap(
				&info, (float)scale, (float)scale, cp, &w, &h, &ox, &oy);
			if (!bmp || w <= 0 || h <= 0) {
				if (bmp)
					stbtt_FreeBitmap(bmp, nullptr);
				return false;
			}
			cv::Mat mask(h, w, CV_8UC1);
			for (int y = 0; y < h; y++)
				memcpy(mask.ptr<uchar>(y), bmp + (size_t)y * w, (size_t)w);
			stbtt_FreeBitmap(bmp, nullptr);
			if (cache.size() >= 256)
				cache.clear();
			it = cache.emplace(key, std::move(mask)).first;
		}

		const cv::Mat& mask = it->second;
		const int x = box.x + (box.width - mask.cols) / 2;
		const int y = box.y + (box.height - mask.rows) / 2;
		const cv::Rect dst = cv::Rect(x, y, mask.cols, mask.rows) &
		                     cv::Rect(0, 0, img.cols, img.rows);
		if (dst.width <= 0 || dst.height <= 0)
			return false;
		const int ox = dst.x - x, oy = dst.y - y;
		const int cb = (int)color[0], cg = (int)color[1], cr = (int)color[2];
		for (int yy = 0; yy < dst.height; yy++) {
			const uchar* m = mask.ptr<uchar>(yy + oy) + ox;
			cv::Vec3b* p = img.ptr<cv::Vec3b>(dst.y + yy) + dst.x;
			for (int xx = 0; xx < dst.width; xx++) {
				const int a = m[xx];
				if (!a)
					continue;
				p[xx][0] = (uchar)((p[xx][0] * (255 - a) + cb * a) / 255);
				p[xx][1] = (uchar)((p[xx][1] * (255 - a) + cg * a) / 255);
				p[xx][2] = (uchar)((p[xx][2] * (255 - a) + cr * a) / 255);
			}
		}
		return true;
	}
};
static TtfFont g_font_cinzel("fonts/Cinzel.ttf");
static TtfFont g_font_grenze("fonts/GrenzeGotisch.ttf");
// stb_truetype no soporta contornos CFF: estos .ttf son la conversion de
// los .otf oficiales de KRCP, con los mismos glifos y cmap.
static TtfFont g_font_icons("fonts/ankha2.ttf");
static TtfFont g_font_clans("fonts/vtes-clans.ttf");

// Texto centrado dentro de una caja, encogiendo la fuente si no cabe a lo
// ancho (los nombres de jugador son texto libre). `target_h` es la altura
// visual en pixeles; font_mode 0/1 usa TTF y si falla (o 2), Hershey.
static void drawTextInBox(cv::Mat& img, const std::string& text, const cv::Rect& box,
                          const cv::Scalar& color, double target_h, int font_mode)
{
	if (text.empty() || box.width < 6 || box.height < 6)
		return;
	if (font_mode != 2) {
		TtfFont *font = font_mode == 1 ? &g_font_grenze : &g_font_cinzel;
		if (font->draw(img, text, box, color, target_h))
			return;
	}

	double scale = target_h / 22.0;
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

// Fila de iconos de la carta (tipos, clan/credo y disciplinas). Se dibuja
// debajo de la carta mostrada, sobre una banda oscura semitransparente para
// que se lea tanto sobre el panel como sobre la camara (carta grande).
static void drawCardIconRow(cv::Mat& img, const std::vector<VtesIcon>& icons,
                            const cv::Rect& area, int size)
{
	if (icons.empty() || img.empty() || area.width < 8 || size < 8)
		return;

	const int n = (int)icons.size();
	int gap = std::max(2, size / 5);
	int total = n * size + (n - 1) * gap;
	while (total > area.width && size > 8) {
		--size;
		gap = std::max(2, size / 5);
		total = n * size + (n - 1) * gap;
	}
	if (size < 8)
		return;

	const int x0 = area.x + (area.width - total) / 2;
	const int y0 = area.y;
	const int pad = std::max(3, size / 4);
	cv::Rect bg(x0 - pad, y0 - pad, total + 2 * pad, size + 2 * pad);
	bg &= cv::Rect(0, 0, img.cols, img.rows);
	if (bg.width > 2 && bg.height > 2) {
		cv::Mat roi = img(bg);
		cv::Mat dark;
		roi.convertTo(dark, -1, 0.42, 0.0);
		dark.copyTo(roi);
		cv::rectangle(img, bg, cv::Scalar(45, 70, 105), 1);
	}

	const cv::Scalar col(76, 168, 201); // dorado #c9a84c (BGR)
	for (int i = 0; i < n; ++i) {
		const cv::Rect cell(x0 + i * (size + gap), y0, size, size);
		TtfFont& f = icons[i].font == 1 ? g_font_clans : g_font_icons;
		f.drawIcon(img, icons[i].glyph, cell, col);
	}
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
	float best_dist = std::numeric_limits<float>::max();

	// Las zonas automáticas de cinco jugadores convergen en el centro de la
	// mesa y pueden solaparse. Cuando un punto cae en varias, se elige la
	// zona en la que queda más centrado, no la del primer jugador. Así una
	// carta colocada en la zona de minions de un jugador no se atribuye al
	// vecino por orden de índice.
	auto checkZone = [&](const filter_data::PlayerZone &z, bool crypt,
			     int player_idx) {
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
		if (std::fabs(dx) > zw * 0.5f || std::fabs(dy) > zh * 0.5f)
			return false;
		// Distancia normalizada por el tamaño de la zona: 0 en el centro
		// y 1 en el borde. Favorece la zona más pequeña y específica.
		const float nx = dx / (zw * 0.5f);
		const float ny = dy / (zh * 0.5f);
		const float dist = nx * nx + ny * ny;
		if (dist < best_dist) {
			best_dist = dist;
			hit.player_idx = player_idx;
			hit.in_crypt = crypt;
		}
		return true;
	};

	// Primero minions (una carta en minions no debe caer en la cripta de
	// otro jugador por solapamiento); sólo si no hay minion, cripta.
	bool found_minion = false;
	for (int pi = 0; pi < tf->player_count; pi++) {
		const filter_data::PlayerState &pl = tf->players[pi];
		if (!pl.active)
			continue;
		if (checkZone(pl.minion_zone, false, pi))
			found_minion = true;
	}
	if (found_minion)
		return hit;

	best_dist = std::numeric_limits<float>::max();
	for (int pi = 0; pi < tf->player_count; pi++) {
		const filter_data::PlayerState &pl = tf->players[pi];
		if (!pl.active)
			continue;
		checkZone(pl.crypt_zone, true, pi);
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

// ─── Cache de identidad por carta visible ───────────────────────────
//
// El embedding cuesta ~34 ms por carta en CPU (indice de 4.446 x 7.168
// floats) y se ejecutaba en cada frame aunque la carta ya estuviera
// identificada. Eso bloquea el hilo de video de OBS y baja los FPS sin
// saturar la GPU. Guardamos la ultima identidad aceptada en la ficha de
// seguimiento y la reutilizamos mientras el recorte siga siendo el mismo
// (dHash parecido) y la carta siga en la misma zona. Si el operador cambia
// la carta, el hash cambia y se vuelve a identificar.

struct IdentityCacheEntry {
	std::string card_id;
	std::string card_name;
	float confidence = 0.0f;
	int label = 0;
	uint64_t crop_hash = 0;
	cv::Point2f center;
	float width = 0.0f;
	float height = 0.0f;
	int64_t last_seen_ns = 0;
};

static int hamming64(uint64_t a, uint64_t b)
{
	uint64_t v = a ^ b;
	int n = 0;
	while (v) {
		v &= v - 1;
		++n;
	}
	return n;
}

static std::vector<IdentityCacheEntry> snapshotIdentityCache(detect_filter_obb *tf)
{
	std::vector<IdentityCacheEntry> out;
	std::lock_guard<std::mutex> lock(tf->game_state_lock);
	out.reserve(tf->tracked_cards.size());
	for (const auto &[tid, tc] : tf->tracked_cards) {
		(void)tid;
		if (tc.card_id.empty() || tc.last_crop_hash == 0)
			continue;
		IdentityCacheEntry e;
		e.card_id = tc.card_id;
		e.card_name = tc.card_name;
		e.confidence = tc.identity_confidence;
		e.label = tc.last_label;
		e.crop_hash = tc.last_crop_hash;
		e.center = tc.last_position;
		e.width = tc.last_width;
		e.height = tc.last_height;
		e.last_seen_ns = tc.last_seen_ns;
		out.push_back(std::move(e));
	}
	return out;
}

static bool lookupIdentityCache(const std::vector<IdentityCacheEntry> &cache,
				const vtes_detection::OBBObject &obj,
				int64_t now_ns, IdentityCacheEntry &out)
{
	if (obj.crop_hash == 0)
		return false;

	const float cx = obj.rect.x + obj.rect.width * 0.5f;
	const float cy = obj.rect.y + obj.rect.height * 0.5f;
	const float obj_area = std::max(1.0f, obj.rect.width * obj.rect.height);
	const float max_dist = std::max(24.0f, std::max(obj.rect.width, obj.rect.height) * 0.75f);
	float best_dist = max_dist;

	for (const auto &e : cache) {
		if (now_ns - e.last_seen_ns > 1000000000LL)
			continue;

		const float dx = cx - e.center.x;
		const float dy = cy - e.center.y;
		const float dist = std::sqrt(dx * dx + dy * dy);
		if (dist >= best_dist)
			continue;

		if (e.width > 1.0f && e.height > 1.0f) {
			const float cached_area = e.width * e.height;
			const float ratio = obj_area / cached_area;
			if (ratio < 0.35f || ratio > 2.50f)
				continue;
		}

		// dHash de 64 bits. Un cambio de carta produce muchos mas bits
		// distintos que el movimiento normal de la camara; 16 bits permite
		// pequenas variaciones de luz/encuadre sin aceptar otra carta.
		if (hamming64(obj.crop_hash, e.crop_hash) > 16)
			continue;

		best_dist = dist;
		out = e;
	}
	return best_dist < max_dist;
}

// ─── Modo realizador: pendientes + feedback de entrenamiento ────────

// Anade (o refresca) una candidata en la cola de pendientes. El recorte
// del frame se guarda como dato para futuros entrenamientos.
static void addPendingDetection(detect_filter_obb *tf, const vtes_detection::OBBObject &obj,
				const cv::Mat &frame, int64_t now_ns)
{
	if (obj.card_id.empty())
		return;
	// La carta que ya esta pintada en el overlay no vuelve a la cola: tras
	// elegir una pendiente la cola debe quedar limpia aunque la carta siga
	// delante de la camara. El id se escribe desde el hilo web, asi que se
	// lee bajo el mismo mutex que el render.
	{
		std::lock_guard<std::mutex> lock(tf->game_state_lock);
		if (!tf->current_overlay_card_id.empty() &&
		    obj.card_id == tf->current_overlay_card_id)
			return;
	}
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

// ─── Registro de eventos de partida (historial en la web) ───────────

// Anade un evento al registro (max 100). Los cambios de zona de la misma
// carta se deduplican 10 s para evitar rafagas si la carta oscila.
static void pushGameEvent(detect_filter_obb *tf, const char *type,
			  int player_idx, const std::string &card_name,
			  int from, int to, int64_t now_ns, int capacity,
			  const std::string &resource)
{
	const std::string type_s = type ? type : "event";
	if (type_s == "zone" && !card_name.empty()) {
		for (auto it = tf->game_events.rbegin(); it != tf->game_events.rend(); ++it) {
			if (now_ns - it->ts_ns > 10LL * 1000000000LL)
				break;
			if (it->type == "zone" && it->player_idx == player_idx &&
			    it->card_name == card_name)
				return;
		}
	}
	filter_data::GameEvent ev;
	ev.ts_ns = now_ns;
	ev.type = type_s;
	ev.player_idx = player_idx;
	ev.card_name = card_name;
	ev.from = from;
	ev.to = to;
	ev.capacity = capacity;
	ev.resource = resource;
	tf->game_events.push_back(std::move(ev));
	const size_t max_events = 100;
	if (tf->game_events.size() > max_events)
		tf->game_events.pop_front();
}

// Ajusta el pool de un jugador y mantiene coherentes el auto-ousted y el
// registro. El llamador debe tener game_state_lock.
static int applyPoolDeltaLocked(detect_filter_obb *tf, int player_idx,
				int delta, int64_t now_ns)
{
	if (player_idx < 0 || player_idx >= tf->player_count)
		return 0;

	auto &pl = tf->players[player_idx];
	const int old_pool = pl.pool;
	int new_pool = old_pool + delta;
	if (new_pool < 0)
		new_pool = 0;
	if (new_pool > 99)
		new_pool = 99;
	if (new_pool == old_pool)
		return 0;

	pl.pool = new_pool;
	pushGameEvent(tf, "pool", player_idx, std::string(), old_pool,
		      new_pool, now_ns);

	// Auto-ousted: pool 0 = eliminado; volver a subir pool lo reactiva.
	if (pl.pool == 0 && pl.active) {
		pl.active = false;
		pushGameEvent(tf, "ousted", player_idx, std::string(), 0, 0,
			      now_ns);
		obs_log(LOG_INFO, "[Game] Jugador %d (%s) ousted: pool 0",
			player_idx + 1, pl.name.c_str());
	} else if (pl.pool > 0 && !pl.active) {
		pl.active = true;
		pushGameEvent(tf, "revived", player_idx, std::string(), 0,
			      pl.pool, now_ns);
		obs_log(LOG_INFO, "[Game] Jugador %d (%s) reactivado (pool %d)",
			player_idx + 1, pl.name.c_str(), pl.pool);
	}
	obs_log(LOG_INFO, "[Game] Pool: %s = %d (delta %+d)",
		pl.name.c_str(), pl.pool, new_pool - old_pool);
	return new_pool - old_pool;
}

// ─── Recursos y costes de cartas de biblioteca ──────────────────────
//
// La capacidad de un vampiro no es su sangre actual: al completar la
// influencia el pool gastado se convierte en la sangre inicial del vampiro.
// Los aliados/retainers usan vida y los Imbued conviccion. La camara no lee
// esos contadores; se inicializan con los valores conocidos de la carta y se
// pueden corregir desde /game.
static bool cardHasType(const VTESCardEntry *entry, const char *type)
{
	if (!entry)
		return false;
	for (const auto &t : entry->types)
		if (t == type)
			return true;
	return false;
}

static int parsePrintedCost(const std::string &value)
{
	if (value.empty() || value == "X" || value == "x")
		return 0;
	try {
		size_t used = 0;
		int cost = std::stoi(value, &used);
		return used == value.size() ? cost : 0;
	} catch (...) {
		return 0;
	}
}

// Los aliados y retainers muestran su vida en el texto ("Demon with 4
// life."). No hay un campo life en vtes.json. Se toma el primer numero que
// precede a "life"; si el texto usa X, queda desconocido.
static int parseLifeFromText(const std::string &text)
{
	size_t pos = 0;
	while ((pos = text.find("life", pos)) != std::string::npos) {
		size_t end = pos;
		while (end > 0 && std::isspace((unsigned char)text[end - 1]))
			--end;
		size_t start = end;
		while (start > 0 && std::isdigit((unsigned char)text[start - 1]))
			--start;
		if (start < end) {
			try {
				return std::stoi(text.substr(start, end - start));
			} catch (...) {
				return -1;
			}
		}
		pos += 4;
	}
	return -1;
}

// Inicializa los contadores que se pueden conocer al identificar la carta.
// No cobra nada: la sangre de un vampiro se fija al completar la influencia
// y la vida de un aliado/retainer es su valor impreso.
static void initializeTrackedResources(detect_filter_obb *tf,
				       filter_data::TrackedCardState &tc)
{
	if (tc.card_id.empty())
		return;
	// Si la identidad del track cambia, los contadores del recurso anterior
	// ya no son validos. No se resetean al perder la identidad (card_id
	// vacio): una carta tapada puede recuperar la misma identidad.
	if (tc.resource_card_id != tc.card_id) {
		tc.resource_card_id = tc.card_id;
		tc.blood = -1;
		tc.life = -1;
		tc.conviction = -1;
	}

	const VTESCardEntry *entry = tf->vtes_db.get_by_id(tc.card_id);
	if (!entry)
		return;

	if (cardHasType(entry, "Vampire")) {
		if (tc.blood < 0 && entry->capacity > 0)
			tc.blood = entry->capacity;
	} else if (cardHasType(entry, "Ally") ||
		   cardHasType(entry, "Retainer")) {
		if (tc.life < 0) {
			const int life = parseLifeFromText(entry->card_text);
			if (life >= 0)
				tc.life = life;
		}
	}
}

// Cobra automaticamente solo el coste en pool de cartas de biblioteca
// permanentes (Master, Equipment, Ally, Retainer). Es el unico caso en el
// que el pagador es inequivoco: el Matusalén. Los costes en sangre,
// conviccion y X quedan para ajuste manual porque dependen del minion que
// juega la carta o de una cantidad variable.
static void maybeChargeLibraryPoolCost(detect_filter_obb *tf,
				       filter_data::TrackedCardState &tc,
				       int64_t now_ns)
{
	if (tc.player_idx < 0 || tc.in_crypt || tc.card_id.empty())
		return;
	if (tc.library_pool_cost_paid &&
	    tc.library_pool_cost_paid_id == tc.card_id)
		return;

	const VTESCardEntry *entry = tf->vtes_db.get_by_id(tc.card_id);
	if (!entry)
		return;
	const bool permanent =
		cardHasType(entry, "Master") || cardHasType(entry, "Equipment") ||
		cardHasType(entry, "Ally") || cardHasType(entry, "Retainer");
	if (!permanent)
		return;

	const int cost = parsePrintedCost(entry->pool_cost);
	if (cost <= 0)
		return;

	// SORT puede crear un track nuevo para la misma carta fisica. Durante
	// unos segundos se recuerda el cobro por jugador+carta para no descontar
	// dos veces el pool si el track se recrea o la carta se mueve. Dos copias
	// iguales jugadas en ese margen se pueden ajustar a mano desde /pool; es
	// preferible a un doble cobro silencioso.
	const std::string cost_key =
		std::to_string(tc.player_idx) + "|" + tc.card_id;
	auto memo = tf->library_cost_memory.find(cost_key);
	if (memo != tf->library_cost_memory.end() &&
	    now_ns - memo->second < 8000000000LL) {
		tc.library_pool_cost_paid = true;
		tc.library_pool_cost_paid_id = tc.card_id;
		return;
	}

	// Marcar antes de aplicar: applyPoolDeltaLocked puede generar eventos y
	// no queremos volver a entrar en este camino en el mismo frame.
	tc.library_pool_cost_paid = true;
	tc.library_pool_cost_paid_id = tc.card_id;
	tf->library_cost_memory[cost_key] = now_ns;
	const int applied = applyPoolDeltaLocked(tf, tc.player_idx, -cost, now_ns);
	if (applied != 0) {
		pushGameEvent(tf, "cost", tc.player_idx, tc.card_name, 0,
			      -applied, now_ns, 0, "pool");
		obs_log(LOG_INFO,
			"[Game] Jugador %d: coste de biblioteca \"%s\" = %d pool",
			tc.player_idx + 1, tc.card_name.c_str(), -applied);
	}
}

// Heuristica de influencia: la camara no puede leer la sangre acumulada
// sobre la carta, pero si puede seguir la carta fisica y ver cuando pasa
// de la cripta a la zona minion. Se guardan dos claves:
//   - track: sirve para cartas boca abajo, que aun no tienen card_id;
//   - card:  sirve si SORT cambia el track_id al mover la carta.
// Al completarse el paso a minion se descuenta la capacidad del pool. La
// capacidad es el maximo de sangre del vampiro y, por tanto, el coste de
// influencia completa. No se descuenta nada por una carta que ya aparece
// en minion sin haber sido vista antes en la cripta.
static void updateInfluenceHeuristic(detect_filter_obb *tf,
				     filter_data::TrackedCardState &tc,
				     int player_idx, bool in_crypt,
				     int64_t now_ns)
{
	if (player_idx < 0 || player_idx >= filter_data::MAX_PLAYERS)
		return;

	const std::string track_key =
		"t:" + std::to_string(tc.track_id);
	std::string card_key;
	if (!tc.card_id.empty())
		card_key = "c:" + std::to_string(player_idx) + "|" +
			   tc.card_id;

	auto mark_crypt = [&](const std::string &key) {
		if (key.empty())
			return;
		auto &mem = tf->influence_memory[key];
		mem.crypt_seen_ns = now_ns;
		// Un flicker de zona no debe duplicar la influencia: solo se
		// rearma si la carta lleva un tiempo real en la cripta.
		if (mem.influenced_ns != 0 &&
		    now_ns - mem.influenced_ns > 3000000000LL)
			mem.influenced_ns = 0;
	};
	auto crypt_seen = [&](const std::string &key) {
		if (key.empty())
			return false;
		auto it = tf->influence_memory.find(key);
		return it != tf->influence_memory.end() &&
		       it->second.crypt_seen_ns != 0 &&
		       it->second.influenced_ns == 0;
	};
	auto mark_influenced = [&](const std::string &key) {
		if (!key.empty())
			tf->influence_memory[key].influenced_ns = now_ns;
	};

	if (in_crypt) {
		mark_crypt(track_key);
		mark_crypt(card_key);
		return;
	}

	// La carta acaba de entrar en minion. Si aun no tiene identidad,
	// esperamos: la capacidad se necesita para cobrar el pool y para
	// mostrar el nombre en el registro.
	if (tc.card_id.empty())
		return;
	if (!crypt_seen(track_key) && !crypt_seen(card_key))
		return;

	int capacity = 0;
	auto cit = tf->card_info_by_id.find(tc.card_id);
	if (cit != tf->card_info_by_id.end())
		capacity = cit->second.capacity;
	if (capacity == 0) {
		const VTESCardEntry *entry = tf->vtes_db.get_by_id(tc.card_id);
		if (entry)
			capacity = entry->capacity;
	}

	// La capacidad de un vampiro está entre 1 y 11. Si el dato viniera
	// fuera de rango no se debe descontar más de lo que cuesta un vampiro
	// real; una capacidad 0 tampoco cuenta como influencia completada.
	if (capacity > kMaxVampireCapacity) {
		obs_log(LOG_WARNING,
			"[Game] Capacidad anomala %d para \"%s\"; se limita a %d",
			capacity, tc.card_name.c_str(), kMaxVampireCapacity);
		capacity = kMaxVampireCapacity;
	}
	if (capacity <= 0) {
		obs_log(LOG_WARNING,
			"[Game] Sin capacidad valida para \"%s\"; no se cobra influencia",
			tc.card_name.c_str());
		// Evitar repetir el aviso en cada frame mientras la carta siga
		// en minions; el retorno a la cripta vuelve a armar la memoria.
		mark_influenced(track_key);
		mark_influenced(card_key);
		return;
	}

	auto &pl = tf->players[player_idx];
	const int charged =
		applyPoolDeltaLocked(tf, player_idx, -capacity, now_ns);
	pl.influence_count++;
	pl.influence_cost += charged < 0 ? -charged : 0;
	// La influencia completa convierte el pool gastado en la sangre con la
	// que entra el vampiro. La camara no puede leer contadores posteriores;
	// este es el valor inicial conocido y se puede corregir en /game.
	tc.blood = capacity;
	mark_influenced(track_key);
	mark_influenced(card_key);

	const char *cname = tc.card_name.empty() ? "?" : tc.card_name.c_str();
	obs_log(LOG_INFO,
		"[Game] Jugador %d: \"%s\" influencia completada (cap %d, pool %d, heuristico)",
		player_idx + 1, cname, capacity, charged);
	pushGameEvent(tf, "influence", player_idx, tc.card_name, 1,
		      charged < 0 ? -charged : 0, now_ns, capacity);
}

// ─── Zonas automáticas: cuatro cartas boca abajo ─────────────────────
//
// La cripta inicial son cuatro cartas boca abajo. No se intenta leer su
// identidad (pueden llevar fundas que el modelo VTES no reconoce): se buscan
// cuatro rectángulos parecidos, próximos y alineados. El grupo se confirma
// tras tres detecciones y se convierte en dos zonas:
//   - cripta: alrededor de las cuatro cartas;
//   - minions: hacia el centro de la mesa, delante de la cripta.
// La asignación a jugadores se hace por orden de asiento.

struct AutoZoneLayout {
	cv::Point2f centroid;
	cv::Point2f table_center;
	bool line_layout = false;
};

static filter_data::PlayerZone playerZoneFromRect(
	const cv::RotatedRect &rr, const cv::Size &frame, float lock_angle)
{
	filter_data::PlayerZone z;
	const float fw = (float)std::max(1, frame.width);
	const float fh = (float)std::max(1, frame.height);
	z.nw = rr.size.width / fw;
	z.nh = rr.size.height / fh;
	z.nx = rr.center.x / fw - z.nw * 0.5f;
	z.ny = rr.center.y / fh - z.nh * 0.5f;
	z.angle = rr.angle;
	z.lock_angle = lock_angle;
	return z;
}

static int activeAutoZonePlayers(const detect_filter_obb *tf)
{
	int n = 0;
	for (int i = 0; i < tf->player_count; ++i)
		if (tf->players[i].active)
			n++;
	return n;
}

static int committedAutoZonePlayers(const detect_filter_obb *tf)
{
	int n = 0;
	for (int i = 0; i < tf->player_count; ++i)
		if (tf->players[i].active && tf->players[i].auto_zone)
			n++;
	return n;
}

static AutoZoneLayout computeAutoZoneLayout(
	const std::vector<vtes_detection::AutoZoneCluster> &clusters,
	const cv::Size &frame)
{
	AutoZoneLayout layout;
	layout.table_center = cv::Point2f(frame.width * 0.5f, frame.height * 0.5f);
	layout.centroid = layout.table_center;

	std::vector<cv::Point2f> pts;
	for (const auto &c : clusters)
		if (c.stable || c.player_idx >= 0)
			pts.push_back(c.center);
	if (pts.empty())
		return layout;

	cv::Point2f mean(0.0f, 0.0f);
	for (const auto &p : pts)
		mean += p;
	mean /= (float)pts.size();
	layout.centroid = mean;
	layout.table_center = mean;

	if (pts.size() == 1) {
		// Con un solo grupo no hay forma de deducir la mesa. Se usa el
		// centro de la imagen para que la zona de minions quede delante
		// de la cripta (arriba o abajo), no en diagonal.
		layout.line_layout = true;
		layout.table_center = cv::Point2f(frame.width * 0.5f,
						  frame.height * 0.5f);
	} else if (pts.size() >= 2) {
		float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
		for (const auto &p : pts) {
			const cv::Point2f d = p - mean;
			sxx += d.x * d.x;
			syy += d.y * d.y;
			sxy += d.x * d.y;
		}
		const float tr = sxx + syy;
		const float disc = std::sqrt(std::max(0.0f,
			(sxx - syy) * (sxx - syy) + 4.0f * sxy * sxy));
		const float l1 = 0.5f * (tr + disc);
		const float l2 = 0.5f * (tr - disc);
		if (l1 > 1.0f && l2 / l1 < 0.12f) {
			layout.line_layout = true;
			// En una fila, "hacia el centro" es hacia el centro de la
			// imagen, no hacia la media de la propia fila.
			layout.table_center = cv::Point2f(
				frame.width * 0.5f, frame.height * 0.5f);
		}
	}
	return layout;
}

static void sortAutoZoneClusters(
	std::vector<vtes_detection::AutoZoneCluster> &clusters,
	const AutoZoneLayout &layout)
{
	if (clusters.empty())
		return;
	if (layout.line_layout) {
		float minx = clusters[0].center.x, maxx = minx;
		float miny = clusters[0].center.y, maxy = miny;
		for (const auto &c : clusters) {
			minx = std::min(minx, c.center.x);
			maxx = std::max(maxx, c.center.x);
			miny = std::min(miny, c.center.y);
			maxy = std::max(maxy, c.center.y);
		}
		if (maxx - minx >= maxy - miny)
			std::sort(clusters.begin(), clusters.end(),
				  [](const auto &a, const auto &b) {
					  return a.center.x < b.center.x;
				  });
		else
			std::sort(clusters.begin(), clusters.end(),
				  [](const auto &a, const auto &b) {
					  return a.center.y < b.center.y;
				  });
		return;
	}

	const float start = (float)(kPI * 0.5); // abajo, 6 en punto
	std::sort(clusters.begin(), clusters.end(),
		  [&](const auto &a, const auto &b) {
			  const float aa = std::atan2(a.center.y - layout.centroid.y,
						      a.center.x - layout.centroid.x);
			  const float ab = std::atan2(b.center.y - layout.centroid.y,
						      b.center.x - layout.centroid.x);
			  float ra = std::fmod(aa - start + 2.0f * kPI, 2.0f * kPI);
			  float rb = std::fmod(ab - start + 2.0f * kPI, 2.0f * kPI);
			  if (ra < 0.0f) ra += 2.0f * kPI;
			  if (rb < 0.0f) rb += 2.0f * kPI;
			  return ra < rb;
		  });
}

static void updateAutoZoneViews(
	detect_filter_obb *tf,
	const std::vector<vtes_detection::AutoZoneCluster> &clusters,
	const AutoZoneLayout &layout, const cv::Size &frame)
{
	std::vector<filter_data::AutoZoneView> views;
	views.reserve(clusters.size());

	int auto_ids[filter_data::MAX_PLAYERS];
	int auto_slots[filter_data::MAX_PLAYERS];
	{
		std::lock_guard<std::mutex> lock(tf->game_state_lock);
		for (int i = 0; i < filter_data::MAX_PLAYERS; ++i) {
			auto_ids[i] = tf->players[i].auto_zone_id;
			auto_slots[i] = tf->players[i].auto_zone ? i : -1;
		}
	}

	for (const auto &c : clusters) {
		vtes_detection::AutoZoneGeometry g =
			vtes_detection::AutoZoneDetector::deriveGeometry(
				c, layout.table_center, frame, layout.line_layout);
		if (!g.valid)
			continue;
		filter_data::AutoZoneView v;
		v.crypt = playerZoneFromRect(g.crypt, frame, g.lock_angle_deg);
		v.minion = playerZoneFromRect(g.minion, frame, g.lock_angle_deg);
		v.stable = c.stable;
		v.card_count = (int)c.cards.size();
		v.score = c.score;
		v.player_idx = -1;
		for (int i = 0; i < filter_data::MAX_PLAYERS; ++i) {
			if (auto_slots[i] >= 0 && auto_ids[i] == c.id) {
				v.player_idx = auto_slots[i];
				break;
			}
		}
		views.push_back(std::move(v));
	}

	std::lock_guard<std::mutex> lock(tf->auto_zone_view_lock);
	tf->auto_zone_views = std::move(views);
}

static bool autoZoneNearCommitted(
	const detect_filter_obb *tf, const cv::Point2f &center,
	const cv::Size &frame)
{
	for (int i = 0; i < tf->player_count; ++i) {
		const filter_data::PlayerState &pl = tf->players[i];
		if (!pl.active || !pl.auto_zone)
			continue;
		const cv::Point2f zc(
			(pl.crypt_zone.nx + pl.crypt_zone.nw * 0.5f) * frame.width,
			(pl.crypt_zone.ny + pl.crypt_zone.nh * 0.5f) * frame.height);
		const float span = std::max(pl.crypt_zone.nw * frame.width,
					    pl.crypt_zone.nh * frame.height);
		if (cv::norm(center - zc) < 0.55f * span)
			return true;
	}
	return false;
}

static void runAutoZoneDetection(detect_filter_obb *tf,
				 const cv::Mat &bgra, int64_t now_ns)
{
	if (!tf->auto_zones)
		return;

	std::lock_guard<std::mutex> auto_lock(tf->auto_zone_lock);
	if (!tf->auto_zone_detector)
		return;

	int expected = 0;
	{
		std::lock_guard<std::mutex> lock(tf->game_state_lock);
		expected = activeAutoZonePlayers(tf);
		if (expected <= 0)
			return;
		if (committedAutoZonePlayers(tf) >= expected)
			return;
	}

	const int64_t interval_ns = 500000000LL; // 2 Hz
	if (now_ns - tf->auto_zone_last_run_ns < interval_ns)
		return;
	tf->auto_zone_last_run_ns = now_ns;

	cv::Mat bgr;
	cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

	std::vector<cv::Rect2f> exclude;
	{
		std::lock_guard<std::mutex> lock(tf->game_state_lock);
		for (const auto &kv : tf->tracked_cards) {
			const auto &tc = kv.second;
			if (tc.card_id.empty() || tc.last_width <= 1.0f ||
			    tc.last_height <= 1.0f)
				continue;
			exclude.emplace_back(
				tc.last_position.x - tc.last_width * 0.5f,
				tc.last_position.y - tc.last_height * 0.5f,
				tc.last_width, tc.last_height);
		}
	}

	std::vector<vtes_detection::AutoZoneCluster> clusters =
		tf->auto_zone_detector->detect(bgr, now_ns, exclude);

	// La media de los grupos sirve para orientar las zonas en una mesa
	// circular. Si sólo hay un grupo, se usa el centro del frame.
	std::vector<vtes_detection::AutoZoneCluster> stable;
	for (const auto &c : clusters)
		if (c.stable)
			stable.push_back(c);
	AutoZoneLayout layout = computeAutoZoneLayout(stable, bgr.size());

	int committed = 0;
	{
		std::lock_guard<std::mutex> lock(tf->game_state_lock);
		committed = committedAutoZonePlayers(tf);
	}

	if (!stable.empty()) {
		// El temporizador debe arrancar también si se perdieron zonas ya
		// asignadas y hay que volver a detectar desde cero.
		if (committed == 0 && tf->auto_zone_batch_start_ns == 0)
			tf->auto_zone_batch_start_ns = now_ns;

		// Esperar un poco para que los jugadores coloquen los cuatro
		// grupos antes de fijar el orden de asiento.
		const bool settle_done =
			(tf->auto_zone_batch_start_ns > 0 &&
			 now_ns - tf->auto_zone_batch_start_ns >= 2500000000LL);
		if (committed == 0 && (int)stable.size() < expected &&
		    !settle_done) {
			updateAutoZoneViews(tf, clusters, layout, bgr.size());
			return;
		}

		sortAutoZoneClusters(stable, layout);

		{
			std::lock_guard<std::mutex> lock(tf->game_state_lock);
			const int fw = bgr.cols, fh = bgr.rows;
			int next_free = 0;
			bool first_commit = (committed == 0);
			int assigned_this_round = 0;

			for (const auto &c : stable) {
				if (autoZoneNearCommitted(tf, c.center, bgr.size()))
					continue;
				while (next_free < tf->player_count &&
				       (!tf->players[next_free].active ||
					tf->players[next_free].auto_zone))
					next_free++;
				if (next_free >= tf->player_count)
					break;

				vtes_detection::AutoZoneGeometry g =
					vtes_detection::AutoZoneDetector::deriveGeometry(
						c, layout.table_center, bgr.size(),
						layout.line_layout);
				if (!g.valid)
					continue;

				filter_data::PlayerState &pl =
					tf->players[next_free];
				pl.crypt_zone = playerZoneFromRect(
					g.crypt, cv::Size(fw, fh), g.lock_angle_deg);
				pl.minion_zone = playerZoneFromRect(
					g.minion, cv::Size(fw, fh), g.lock_angle_deg);
				pl.auto_zone = true;
				pl.auto_zone_id = c.id;
				tf->auto_zone_commit_count++;
				assigned_this_round++;

				obs_log(LOG_INFO,
					"[AutoZones] Jugador %d: %d cartas -> cripta (%.0f,%.0f) %.0fx%.0f, minions (%.0f,%.0f) %.0fx%.0f",
					next_free + 1, (int)c.cards.size(),
					g.crypt.center.x, g.crypt.center.y,
					g.crypt.size.width, g.crypt.size.height,
					g.minion.center.x, g.minion.center.y,
					g.minion.size.width, g.minion.size.height);
				next_free++;
			}

			// Al pasar de zonas manuales a automáticas se reasignan las
			// cartas y se borra la memoria de influencia: si no, el cambio
			// de rectángulo podría interpretarse como una influencia.
			if (first_commit && assigned_this_round > 0) {
				tf->influence_memory.clear();
				for (auto &kv : tf->tracked_cards) {
					kv.second.player_idx = -1;
					kv.second.in_crypt = false;
					kv.second.lock_votes = 0;
				}
				obs_log(LOG_INFO,
					"[AutoZones] Zonas automáticas aplicadas (%d/%d)",
					committedAutoZonePlayers(tf), expected);
			}
		}
		tf->auto_zone_batch_start_ns = 0;
	}

	updateAutoZoneViews(tf, clusters, layout, bgr.size());

	{
		std::lock_guard<std::mutex> lock(tf->game_state_lock);
		committed = committedAutoZonePlayers(tf);
	}
	if (committed >= expected) {
		tf->auto_zone_detector->reset();
		// Las vistas de propuesta ya no aportan nada y, si el detector
		// conservara algún track antiguo, sus etiquetas se quedarían
		// dibujadas sobre la escena. Las zonas confirmadas se dibujan
		// aparte desde PlayerZone.
		{
			std::lock_guard<std::mutex> lock(tf->auto_zone_view_lock);
			tf->auto_zone_views.clear();
		}
		obs_log(LOG_INFO,
			"[AutoZones] Detección completada: %d zonas", committed);
	}
}

void detect_filter_obb_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);

	struct detect_filter_obb *tf = reinterpret_cast<detect_filter_obb *>(data);

	if (tf->isDisabled) {
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

	// ─── Zonas automáticas: 4 cartas boca abajo ───────────────────────
	// Se ejecuta a 2 Hz y sólo hasta confirmar las zonas. No depende del
	// detector de cartas ni del modelo de embeddings.
	runAutoZoneDetection(tf, imageBGRA, now_ns);

	// ─── Cooldown: skip detection after successful card identification ─────
	// (Always Active bypasses cooldown entirely)
	auto _cd_now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count();
	bool in_cooldown = tf->always_active ? false :
		(tf->cooldown_until_time_ns > 0 && _cd_now_ns < tf->cooldown_until_time_ns);

	std::vector<vtes_detection::OBBObject> raw_objects;

	// Las zonas automáticas no necesitan el modelo de cartas. El detector
	// de cartas sí: si ONNX está seleccionado pero el modelo no cargó, se
	// omite sólo esta parte y se mantiene el escaneo de zonas.
	const bool detector_ready =
		tf->detectionMode != DETECT_MODE_ONNX || tf->yolo_detector;
	if (should_detect && !in_cooldown && detector_ready) {
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
			// obj.rect es el AABB recortado al frame. En cartas giradas o
			// pegadas al borde el area real es obb_w x obb_h; usar el AABB
			// recortado descartaba cartas pequenas que si se veian enteras.
			const double area = (obj.obb_w > 1.0f && obj.obb_h > 1.0f)
				? (double)obj.obb_w * (double)obj.obb_h
				: (double)obj.rect.width * (double)obj.rect.height;
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
		// obb_w/obb_h conservan el tamano real de la carta cuando obj.rect
		// se recorta al tocar un borde del frame.
		const float card_w = obj.obb_w > 1.0f ? obj.obb_w : obj.rect.width;
		const float card_h = obj.obb_h > 1.0f ? obj.obb_h : obj.rect.height;
		cv::Size2f size(card_w, card_h);
		cv::RotatedRect rr(center, size, obj.angle * 180.0f / kPI);
		cv::Point2f corners[4];
		rr.points(corners);
		card_regions[i] = extractCardRegion(inferenceFrame, corners);
		raw_objects[i].crop_hash = cardCropHash(card_regions[i]);
	}

	// ─── Reutilizar la identidad de una carta ya seguida ───────────────
	// Antes de gastar ~34 ms por carta en el embedding, comprobamos si la
	// misma carta ya estaba identificada en la ficha de seguimiento. La
	// comprobacion es barata (dHash + distancia entre centros) y evita
	// bloquear el hilo de video en cada frame.
	std::vector<char> identity_cached(raw_objects.size(), 0);
	if (!raw_objects.empty()) {
		const auto identity_cache = snapshotIdentityCache(tf);
		for (size_t i = 0; i < raw_objects.size(); i++) {
			IdentityCacheEntry hit;
			if (!lookupIdentityCache(identity_cache, raw_objects[i], now_ns, hit))
				continue;
			auto &obj = raw_objects[i];
			obj.card_id = hit.card_id;
			obj.card_name = hit.card_name;
			obj.prob = hit.confidence;
			obj.label = hit.label;
			obj.alt_card_id.clear();
			obj.alt_card_name.clear();
			identity_cached[i] = 1;
		}
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
			if (i < identity_cached.size() && identity_cached[i])
				continue; // identidad ya resuelta: no repetir clasificador
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
		const int64_t identify_now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
		for (size_t i = 0; i < raw_objects.size(); i++) {
			auto& obj = raw_objects[i];

			// Ya identificada en un frame anterior y el recorte sigue siendo
			// el mismo: no repetir el embedding. La ficha de seguimiento se
			// actualiza mas abajo con la posicion/hash de este frame.
			if (i < identity_cached.size() && identity_cached[i])
				continue;

			cv::Mat card_region = card_regions[i];
			if (card_region.empty()) continue;

			std::string card_name, card_id;
			std::string second_name, second_id;
			float confidence = 0.0f;
			float second_confidence = -1.0f;
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
				if ((!matched || confidence < thresh) && tf->embedder.is_loaded()) {
					matched = tf->embedder.identify(card_region, card_name, card_id, confidence, "",
					                                &second_name, &second_id, &second_confidence);
					thresh = tf->embedder.threshold();
				}
			}

			// OCR is async: runs on a background worker thread.
			// Jobs are submitted below; results are drained on the next frame.
			// (The worker handles recognize() + vampire validation.)

			const bool confident = matched && confidence >= thresh;
			const bool has_second = confident && !second_id.empty() &&
				second_confidence >= 0.0f;
			obj.alt_card_id.clear();
			obj.alt_card_name.clear();
			if (has_second) {
				obj.alt_card_id = second_id;
				obj.alt_card_name = second_name;
			}
			const float margin = has_second ? confidence - second_confidence : confidence;
			const bool ambiguous = confident && has_second &&
				margin < tf->embed_min_margin;
			const float sharpness = cardRegionSharpness(card_region);
			const float angle_deg = obj.angle * 180.0f / kPI;

			if (confident && !ambiguous) {
				obj.card_name = card_name;
				obj.card_id = card_id;
				obj.prob = confidence;
				obs_log(LOG_INFO,
					"[Embed] Card #%d: '%s' (id=%s, sim=%.4f, second=%.4f, margin=%.4f, thresh=%.2f, crop=%dx%d, angle=%.1f, obb=%.1fx%.1f, sharp=%.0f, filter=%s)",
					obj.id, card_name.c_str(), card_id.c_str(), confidence,
					has_second ? second_confidence : -1.0f, margin,
					thresh, card_region.cols, card_region.rows,
					angle_deg, obj.obb_w, obj.obb_h, sharpness,
					type_filter.c_str());
			} else {
				// No aceptamos automaticamente una identificacion ambigua.
				// En modo realizador se ofrecen las dos mejores cartas en la
				// cola para que el operador desambigue con un clic. En modo
				// clasico no se pinta nada antes que pintar una carta erronea.
				obj.card_name = confident ? ("?" + card_name) : "???";
				obj.card_id.clear();
				if (confidence > 0.2f) obj.prob = confidence;

				obs_log(LOG_INFO,
					"[Embed] Card #%d: %s (best='%s' sim=%.4f, second='%s' sim=%.4f, margin=%.4f, thresh=%.2f, crop=%dx%d, angle=%.1f, obb=%.1fx%.1f, sharp=%.0f, filter=%s)",
					obj.id, ambiguous ? "AMBIGUA" : (matched ? "BAJA CONFIANZA" : "sin match"),
					card_name.c_str(), confidence, second_name.c_str(), second_confidence,
					margin, thresh, card_region.cols, card_region.rows,
					angle_deg, obj.obb_w, obj.obb_h, sharpness,
					type_filter.c_str());

				if (ambiguous && tf->click_to_overlay) {
					vtes_detection::OBBObject best_candidate = obj;
					best_candidate.card_name = card_name;
					best_candidate.card_id = card_id;
					best_candidate.prob = confidence;
					addPendingDetection(tf, best_candidate, inferenceFrame, identify_now_ns);

					vtes_detection::OBBObject alt_candidate = obj;
					alt_candidate.card_name = second_name;
					alt_candidate.card_id = second_id;
					alt_candidate.prob = second_confidence;
					addPendingDetection(tf, alt_candidate, inferenceFrame, identify_now_ns);
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
			std::vector<uint64_t> crop_hashes;
			std::vector<char> ocr_allowed;
			track_ids.reserve(raw_objects.size());
			crop_hashes.reserve(raw_objects.size());
			ocr_allowed.reserve(raw_objects.size());
			for (const auto& o : raw_objects) {
				track_ids.push_back(o.id);
				crop_hashes.push_back(o.crop_hash);
				const float source_h = o.obb_h > 1.0f ? o.obb_h
				                                      : (float)o.rect.height;
				// Una identificacion segura ya no necesita OCR. Si el
				// embedding dejo dos candidatas (ambigua), el operador elige
				// en la web; el OCR no debe meter una tercera carta.
				const bool ambiguous_pair = o.card_id.empty() &&
					!o.alt_card_id.empty();
				ocr_allowed.push_back(
					o.card_id.empty() && !ambiguous_pair &&
					source_h >= kMinOcrSourceHeight ? 1 : 0);
			}
			submit_ocr_jobs(tf, card_regions, track_ids, empty_filters,
			                crop_hashes, ocr_allowed);
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
	} // end if (should_detect && !in_cooldown && detector_ready)

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
			std::unique_lock<std::mutex> lock(tf->game_state_lock);
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
				tc.last_crop_hash = obj.crop_hash;
				tc.identity_confidence = obj.prob;
				tc.last_label = obj.label;
				tc.last_width = obj.rect.width;
				tc.last_height = obj.rect.height;

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
				initializeTrackedResources(tf, tc);
				updateInfluenceHeuristic(tf, tc, zone.player_idx,
							 zone.in_crypt, now_ns);
				maybeChargeLibraryPoolCost(tf, tc, now_ns);

				tf->tracked_cards[obj.id] = std::move(tc);
			} else {
				// Existing track
				auto& tc = it->second;
				tc.last_seen_ns = now_ns;
				tc.last_position = cv::Point2f(
					obj.rect.x + obj.rect.width / 2,
					obj.rect.y + obj.rect.height / 2);
				tc.unseen_frames = 0;

				// Actualizar la identidad si es nueva o si el operador ha
				// cambiado la carta en la misma posicion. Guardamos tambien
				// el hash/confianza para poder reutilizarla en el proximo
				// frame sin repetir el embedding.
				if (!obj.card_id.empty()) {
					if (tc.card_id.empty() || tc.card_id != obj.card_id) {
						tc.card_id = obj.card_id;
						tc.card_name = obj.card_name;
						auto cit = tf->card_info_by_id.find(obj.card_id);
						if (cit != tf->card_info_by_id.end())
							tc.card_url = cit->second.url;
					}
					tc.last_crop_hash = obj.crop_hash;
					tc.identity_confidence = obj.prob;
					tc.last_label = obj.label;
					tc.last_width = obj.rect.width;
					tc.last_height = obj.rect.height;
				}

				// Reasignación de zona cada frame: una carta puede cambiar
				// de zona (p.ej. de cripta a minion al completar la
				// influencia del vampiro). El paso cripta<->minion del
				// mismo jugador queda registrado como influencia.
				{
					PlayerZoneHit zone = resolvePlayerZone(tf, tc.last_position, fw, fh);
					if (zone.player_idx != tc.player_idx || zone.in_crypt != tc.in_crypt) {
						const int prev_player = tc.player_idx;
						const bool was_crypt = tc.in_crypt;
						tc.player_idx = zone.player_idx;
						tc.in_crypt = zone.in_crypt;
						tc.lock_votes = 0; // la histéresis previa ya no aplica
						// La cripta→minion la registra la heurística
						// de influencia; aquí solo el retorno a cripta.
						if (zone.player_idx >= 0 && zone.player_idx == prev_player &&
						    !was_crypt && zone.in_crypt)
							pushGameEvent(tf, "zone", zone.player_idx,
								tc.card_name, 0, 1, now_ns);
					}
				}
				// Mantener la memoria cripta/minion actualizada cada
				// frame, incluso cuando SORT ha cambiado el track_id.
				updateInfluenceHeuristic(tf, tc, tc.player_idx,
							 tc.in_crypt, now_ns);
				initializeTrackedResources(tf, tc);
				maybeChargeLibraryPoolCost(tf, tc, now_ns);

				// ── Estado de giro: evaluación continua (una carta puede
				//    girarse/enderezarse en cualquier momento) y relativa a
				//    la zona minion del jugador. En la cripta no se gira.
				{
					const bool in_minion_zone = tc.player_idx >= 0 && !tc.in_crypt;
					const float base_deg = in_minion_zone
						? tf->players[tc.player_idx].minion_zone.lock_angle
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
						if (tc.player_idx >= 0) {
							obs_log(LOG_INFO,
								"[Game] Jugador %d: \"%s\" girada (rel %+.0f deg)",
								tc.player_idx + 1, cname, rel);
							pushGameEvent(tf, "lock", tc.player_idx,
								tc.card_name, 0, 0, now_ns);
						}
					} else if (tc.lock_state == filter_data::CardLockState::Locked &&
						   tc.lock_votes <= -kLockVotesMax) {
						tc.lock_state = filter_data::CardLockState::Unlocked;
						tc.lock_changed_ns = now_ns;
						if (tc.player_idx >= 0) {
							obs_log(LOG_INFO,
								"[Game] Jugador %d: \"%s\" lista (rel %+.0f deg)",
								tc.player_idx + 1, cname, rel);
							pushGameEvent(tf, "unlock", tc.player_idx,
								tc.card_name, 0, 0, now_ns);
						}
					}
				}
			}

			// Modo realizador: la deteccion no se muestra sola; entra en la
			// cola de pendientes para que el operador la valide en la web.
			if (tf->click_to_overlay && !obj.card_id.empty()) {
				lock.unlock();
				addPendingDetection(tf, obj, imageBGRA, now_ns);
			}
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
					tf->overlay_cam_window = overlayCameraWindow(tf->overlay_bg);
					if (tf->overlay_cam_window.width >= 8) {
						obs_log(LOG_INFO,
						        "[Overlay] camera window %dx%d at (%d,%d)",
						        tf->overlay_cam_window.width,
						        tf->overlay_cam_window.height,
						        tf->overlay_cam_window.x,
						        tf->overlay_cam_window.y);
					}
				}
			}

			int fw = frame.cols, fh = frame.rows;
			const OverlayFit fit = overlayFit(fw, fh);

			cv::Mat display = cv::Mat::zeros(fh, fw, CV_8UC3);
			// 2. Camara DENTRO del hueco del overlay y POR DEBAJO de el. La
			// imagen cubre el hueco entero (escala "cover"): sin barras
			// negras. Zoom y posicion son ajustables; el desplazamiento se
			// recorta para que la camara siga cubriendo el hueco. Las zonas
			// de jugador (coords 0..1 de camara) se mapean con `cam`.
			const cv::Rect win_design =
				tf->overlay_cam_window.width >= 8
					? tf->overlay_cam_window
					: cv::Rect(kWinX, kWinY, kWinW, kWinH);
			CameraView cam;
			cam.win = cv::Rect(0, 0, fw, fh);
			if (!tf->overlay_bg.empty()) {
				cam = cameraView(fit, fw, fh, win_design, tf->cam_zoom,
				                 tf->cam_off_x, tf->cam_off_y,
				                 tf->cam_full_frame);
			}
			const cv::Rect cam_win = cam.win & cv::Rect(0, 0, fw, fh);
			if (cam_win.width > 4 && cam_win.height > 4 && !frame.empty() &&
			    cam.scale > 0.0) {
				// Solo se reescala la parte visible del hueco (no la imagen
				// entera): con zoom alto el recorte ahorra mucho trabajo.
				const int x0 = std::max(0, (int)std::floor(
					(cam_win.x - cam.ox) / cam.scale));
				const int y0 = std::max(0, (int)std::floor(
					(cam_win.y - cam.oy) / cam.scale));
				const int x1 = std::min(fw, (int)std::ceil(
					(cam_win.x + cam_win.width - cam.ox) / cam.scale));
				const int y1 = std::min(fh, (int)std::ceil(
					(cam_win.y + cam_win.height - cam.oy) / cam.scale));
				const cv::Rect src(x0, y0, x1 - x0, y1 - y0);
				if (src.width > 1 && src.height > 1) {
					if (src.size() == cam_win.size()) {
						frame(src).copyTo(display(cam_win));
					} else {
						cv::Mat scaled;
						const int interp = cam.scale < 1.0 ? cv::INTER_AREA
						                                   : cv::INTER_LINEAR;
						cv::resize(frame(src), scaled, cam_win.size(), 0, 0, interp);
						scaled.copyTo(display(cam_win));
					}
				}
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

				// Zonas propuestas por el detector de 4 cartas boca abajo
				// que todavía no se han asignado a un jugador. Se dibujan
				// en blanco para distinguirlas de las zonas manuales.
				if (tf->auto_zones) {
					std::vector<filter_data::AutoZoneView> views;
					{
						std::lock_guard<std::mutex> lock(
							tf->auto_zone_view_lock);
						views = tf->auto_zone_views;
					}
					for (const auto &v : views) {
						if (v.player_idx >= 0)
							continue;
						const cv::Scalar col = v.stable
							? cv::Scalar(255, 255, 255)
							: cv::Scalar(180, 180, 180);
						auto drawProposal = [&](const filter_data::PlayerZone &z) {
							const cv::Point2f zc(
								(float)(cam.ox + (z.nx + z.nw * 0.5) * fw * cam.scale),
								(float)(cam.oy + (z.ny + z.nh * 0.5) * fh * cam.scale));
							const cv::Size2f zs(
								(float)(z.nw * fw * cam.scale),
								(float)(z.nh * fh * cam.scale));
							if (zs.width < 4.0f || zs.height < 4.0f)
								return;
							cv::RotatedRect rr(zc, zs, z.angle);
							cv::Point2f pts[4];
							rr.points(pts);
							std::vector<std::vector<cv::Point>> poly = {{
								cv::Point((int)(pts[0].x + 0.5f), (int)(pts[0].y + 0.5f)),
								cv::Point((int)(pts[1].x + 0.5f), (int)(pts[1].y + 0.5f)),
								cv::Point((int)(pts[2].x + 0.5f), (int)(pts[2].y + 0.5f)),
								cv::Point((int)(pts[3].x + 0.5f), (int)(pts[3].y + 0.5f))}};
							cv::polylines(display, poly, true, col, 1);
						};
						drawProposal(v.crypt);
						drawProposal(v.minion);
						std::string label = tf_text("AutoZoneCards");
						if (!v.stable)
							label += "...";
						const cv::Point2f top(
							(float)(v.crypt.nx * fw * cam.scale + cam.ox),
							(float)(v.crypt.ny * fh * cam.scale + cam.oy));
						cv::putText(display, label,
							cv::Point((int)top.x + 3, (int)top.y + 14),
							cv::FONT_HERSHEY_SIMPLEX, 0.42,
							col, 1);
					}
				}

				for (int pi = 0; pi < tf->player_count; pi++) {
					auto& pl = tf->players[pi];
					const cv::Scalar col = pl.active ? palette[pi % 5]
					                                 : cv::Scalar(140, 140, 140);
					auto drawZone = [&](const filter_data::PlayerZone& z, const char* tag) {
						const cv::Point2f zc(
							(float)(cam.ox + (z.nx + z.nw * 0.5) * fw * cam.scale),
							(float)(cam.oy + (z.ny + z.nh * 0.5) * fh * cam.scale));
						const cv::Size2f zs((float)(z.nw * fw * cam.scale),
						                    (float)(z.nh * fh * cam.scale));
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
						if (pl.auto_zone)
							label += std::string(" [") +
								 tf_text("ZoneAuto") + "]";
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
				              name_box.height * 0.42, tf->overlay_font_mode);
				drawTextInBox(display, std::to_string(pl.pool), pool_box, slot_col,
				              pool_box.height * 0.55, tf->overlay_font_mode);
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
					if (card_img.empty() && !tf->current_overlay_fallback.empty())
						card_img = tf->current_overlay_fallback;
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

		// 4c. Iconos oficiales de la carta (tipos, clan/credo y disciplinas)
		// justo debajo de la carta mostrada: en el ankh o bajo la carta
		// grande. La fila se encoge sola si la carta tiene muchos simbolos.
		const VTESCardEntry* shown_entry =
			!shown_card_db_id.empty() ? tf->vtes_db.get_by_id(shown_card_db_id)
			                          : nullptr;
		if (shown_entry && shown_card.width > 0) {
			const std::vector<VtesIcon> icons =
				vtes_icons::forCard(*shown_entry);
			if (!icons.empty()) {
				const cv::Rect abox =
					designRect(fit, kAnkhX, kAnkhY, kAnkhW, kAnkhH);
				cv::Rect area;
				int size = 0;
				if (tf->big_card) {
					size = (int)(42 * fit.esc + 0.5);
					const int gap = (int)(12 * fit.esc + 0.5);
					int y = shown_card.y + shown_card.height + gap;
					if (y + size > fh - 4)
						y = shown_card.y - size - gap;
					area = cv::Rect(shown_card.x, y,
					                shown_card.width, size);
				} else {
					size = (int)(26 * fit.esc + 0.5);
					const int gap = (int)(6 * fit.esc + 0.5);
					int y = shown_card.y + shown_card.height + gap;
					const int max_y = abox.y + abox.height - size;
					if (y > max_y)
						y = max_y;
					area = cv::Rect(abox.x, y, abox.width, size);
				}
				drawCardIconRow(display, icons, area, size);
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
				const VTESCardEntry* entry = shown_entry;

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
