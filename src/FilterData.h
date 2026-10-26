#ifndef FILTERDATA_H
#define FILTERDATA_H

#include <obs-module.h>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include "detection/detector_base.hpp"
#include "detection/auto_zone_detector.hpp"
#include "sort/Sort.h"
#include "classifier/vtes_card_classifier.hpp"
#include "embedding_matcher.h"
#include "vtes_database.hpp"
#include "ocr/vtes_ocr.hpp"
#include "web_server.h"
#include <chrono>
#include <atomic>
#include <deque>
#include <queue>
#include <condition_variable>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <mutex>

// Card info (name, types, disciplines) loaded from byId section of card-hash-index.json
struct CardInfo {
    std::string id;
    std::string name;
    std::vector<std::string> types;
    std::vector<std::string> clans;
    int capacity = 0;
    std::vector<std::string> disciplines;
    std::string url;
    std::string cardText;
    std::string group;
};

/**
  * @brief The filter_data struct
  *
  * Base data for ORT filters + VTES-specific fields.
*/
struct filter_data {
	std::string inference_device;
	InferenceDevice inference_device_enum = InferenceDevice::CPU;
	uint32_t numThreads;
	float conf_threshold;
	std::string modelSize;

	int objectCategory;
	bool maskingEnabled;
	std::string maskingType;
	int maskingColor;
	int maskingBlurRadius;
	bool trackingEnabled;
	float zoomFactor;
	float zoomSpeedFactor;
	std::string zoomObject;
	obs_source_t *trackingFilter;
	cv::Rect2f trackingRect;
	int lastDetectedObjectId;
	bool sortTracking;
	bool showUnseenObjects;
	std::string saveDetectionsPath;
	bool crop_enabled;
	int crop_left;
	int crop_right;
	int crop_top;
	int crop_bottom;

	// create SORT tracker
	Sort tracker;

	obs_source_t *source;
	gs_texrender_t *texrender;
	gs_stagesurf_t *stagesurface;
	gs_effect_t *kawaseBlurEffect;
	gs_effect_t *maskingEffect;

	cv::Mat inputBGRA;
	cv::Mat outputPreviewBGRA;
	cv::Mat outputMask;

	bool isDisabled;
	bool preview;
	bool always_active;
	bool show_zones; // dibuja las zonas de jugadores sobre el preview
	bool big_card;   // dibuja la carta principal a media pantalla (modo torneo)
	// Encuadre de la camara bajo el overlay: la imagen cubre el hueco entero
	// (sin barras negras). zoom 1.0 = cubrir justo; offsets en px de diseno.
	double cam_zoom = 1.0;
	int    cam_off_x = 0;
	int    cam_off_y = 0;
	bool   cam_full_frame = false; // a sangre: cubre todo el frame, no solo el hueco
	// Fuente de nombres/pool: 0 = Cinzel, 1 = Grenze Gotisch, 2 = Hershey.
	int    overlay_font_mode = 0;

	std::mutex inputBGRALock;
	std::mutex outputLock;
	std::mutex modelMutex;

	std::unique_ptr<vtes_detection::DetectorBase> yolo_detector;
	std::vector<std::string> classNames;

#if _WIN32
	std::wstring modelFilepath;
#else
	std::string modelFilepath;
#endif

	// Detection mode: "onnx" or "contour"
	std::string detectionMode;

	// Contour detection parameters
	int contourEdgeLow;
	int contourEdgeHigh;
	double contourMinAreaFrac;
	double contourMaxAreaFrac;
	double contourAspectTolerance;

	// ONNX detection area filters
	int onnxMinArea;
	double onnxMaxAreaFrac;

	// --- Temporal smoothing (mtg_card_detector style) ---
	static constexpr int TEMPORAL_WINDOW = 10; // frames

	// Tracked object: position averaged across frames
	struct TrackedObject {
		float avg_cx, avg_cy, avg_w, avg_h, avg_angle;
		float confidence;       // 0..1 fraction of frames detected
		int stable_count;       // consecutive frames seen
		int unseen_count;       // consecutive frames missed
		int id;

		// Identity tracking
		std::string tracked_card_id;
		std::string tracked_card_name;
		int identity_hits = 0;         // frames where this identity was matched
		int identity_total = 0;         // total frames tracked
		float identity_score = 0.0f;    // confianza de la identificacion
		uint64_t last_crop_hash = 0;    // dHash del ultimo recorte asociado
		int last_label = 0;             // etiqueta de tipo para el preview
		static constexpr int IDENTITY_WINDOW = 10;

		std::deque<bool> history; // sliding window of detections
	};

	std::unordered_map<int, TrackedObject> temporal_tracks;
	int next_track_id = 1;
	bool temporal_smoothing_enabled = true;
	float temporal_min_confidence = 0.3f; // require 30% of window
	int temporal_min_stable = 3;          // require 3 consecutive detections

	// --- Vision-based type classifier (vtes_type_classifier.onnx) ---
	Ort::Env type_classifier_env{ORT_LOGGING_LEVEL_WARNING, "type-classifier"};
	std::unique_ptr<Ort::Session> type_classifier_session;
	std::vector<std::string> type_labels;  // from type_labels.json

	// --- VTES Card Database (vtes.json in memory) ---
	VTESCardDatabase vtes_db;

	// --- Embedding-based card identification ---
	EmbeddingMatcher embedder;
	// Margen minimo entre la mejor carta y la segunda carta distinta para
	// aceptar una identificacion automatica. En capturas borrosas No Trace
	// y Dust Up quedan a pocas milesimas; sin este margen la cola del
	// realizador se llenaba de falsos positivos.
	float embed_min_margin = 0.06f;
	std::unordered_map<std::string, CardInfo> card_info_by_id;    // card ID -> card info

	// --- VTES Card Classifier ---
	bool classifier_enabled = true;
	vtes_classifier::VTESCardClassifier::Config classifierConfig;
	std::unique_ptr<vtes_classifier::VTESCardClassifier> classifier;
	std::unique_ptr<vtes_classifier::CardCropExtractor> cropExtractor;

	// --- Per-type embedding matchers (one ONNX session + index per card type) ---
	std::unordered_map<std::string, std::unique_ptr<EmbeddingMatcher>> per_type_matchers;

	// --- OCR card name reader (Tesseract, optional) ---
	std::unique_ptr<VtesOcrReader> ocr_reader;
	bool upscaler_enabled = false;
	std::vector<VTESCardNameEntry> card_name_entries;  // built from vtes_db for fuzzy matching
	bool ocr_enabled = false;

	// --- Async OCR worker (background thread, doesn't block video_tick) ---
	struct OcrJob {
		int64_t id;
		cv::Mat card_region;
		std::string type_filter;
		uint64_t crop_hash = 0;
	};
	struct OcrJobResult {
		int64_t id;
		bool completed = false;
		bool accepted = false;
		std::string card_name;
		std::string card_id;
		float confidence = 0.0f;
		uint64_t crop_hash = 0;
	};

	std::thread ocr_worker;
	std::mutex ocr_queue_lock;
	std::condition_variable ocr_queue_cv;
	std::queue<OcrJob> ocr_queue;
	std::atomic<bool> ocr_worker_active{false};
	std::atomic<int64_t> ocr_next_job_id{1};

	// Completed results, produced by worker thread, consumed by video_tick
	std::mutex ocr_result_lock;
	std::vector<OcrJobResult> ocr_completed;

	// --- Embedded web server for card search UI ---
	std::unique_ptr<WebServer> web_server;
	int web_server_port = 8080;

	// --- Card image cache for overlay display ---
	std::unordered_map<std::string, cv::Mat> card_image_cache;
	// Carta elegida por el operador (modo realizador): se pinta en el ankh
	// hasta que se elija otra o se limpie desde la web.
	std::string current_overlay_card_url;
	std::string current_overlay_card_id;
	// Recorte de la pendiente elegida, usado si la caratula oficial no se
	// pudo descargar/cachear. Asi el clic siempre pinta algo.
	cv::Mat current_overlay_fallback;
	float card_overlay_duration = 5.0f;

	// --- Performance: minimum detection interval ---
	std::atomic<int> detection_interval_ms{0}; // 0 = process every frame
	uint64_t last_detection_time_ns = 0;

	// --- Cooldown: skip detection for N ns after successful identification ---
	int64_t cooldown_until_time_ns = 0;

	// --- CUDA oval/rect portrait classifier ---
	bool vampire_cuda_enabled = true;

	// --- Web server notification flag (show popup once per filter instance) ---
	bool web_server_notified = false;

	// --- Modo realizador: las detecciones no se muestran solas; se acumulan
	// como pendientes y el operador elige cual se muestra desde la web. ---
	bool click_to_overlay = true;
	struct PendingDetection {
		uint64_t id = 0;
		std::string card_id, card_name, card_url;
		float confidence = 0.0f;     // deteccion (objetos) o identificacion (OCR)
		int64_t first_seen_ns = 0;
		int64_t last_seen_ns = 0;
		int hits = 0;                // veces re-detectada mientras esperaba
		std::vector<uchar> crop_jpg; // recorte del frame (datos de entrenamiento)
	};
	std::deque<PendingDetection> pending_detections;
	uint64_t pending_next_id = 1;
	std::mutex pending_lock;
	// Carpeta de datos de entrenamiento (feedback del realizador).
	std::string feedback_dir;

	// --- Registro de eventos de partida (historial para la web) ---
	struct GameEvent {
		int64_t ts_ns = 0;
		std::string type; // lock | unlock | zone | influence | pool | ousted | revived | cost | resource
		int player_idx = -1;
		std::string card_name;
		int from = 0;     // pool anterior / zona: 1 cripta, 0 minion
		int to = 0;       // pool nuevo / zona; influence: pool gastado
		int capacity = 0; // influencia inferida: capacidad de la carta
		std::string resource; // blood | life | conviction (eventos de recurso)
	};
	std::deque<GameEvent> game_events; // max 100; el mas reciente al final

	// --- Detection history for web server ---
	struct DetectedCard {
		std::string card_name;
		std::string card_id;
		std::string card_url;
		int64_t timestamp_ns;
	};
	std::mutex detection_history_lock;
	std::deque<DetectedCard> detection_history;
	static constexpr int MAX_DETECTION_HISTORY = 100;

	// ================================================================
	// v0.1.6 — VTES Game State (player zones, lock tracking, overlay)
	// ================================================================

	// Per-player zone on the source frame (normalized 0..1)
	struct PlayerZone {
		float nx = 0.0f, ny = 0.0f, nw = 0.0f, nh = 0.0f;
		float angle = 0.0f; // grados; positivo = horario en pantalla
		// Referencia para detectar si una carta está girada. En las zonas
		// manuales coincide con `angle`; en las automáticas se toma de la
		// orientación real de las cuatro cartas de cripta.
		float lock_angle = 0.0f;
	};

	// Per-player persistent state
	struct PlayerState {
		std::string name = "Jugador";
		int pool = 30;
		bool active = true; // en partida; un ousted deja su sitio libre
		PlayerZone minion_zone;  // far (top) zone
		PlayerZone crypt_zone;   // near (bottom) zone
		// Zona creada por el detector de 4 cartas boca abajo. Se conserva
		// aunque luego desaparezcan las cartas (influyeron o se movieron).
		bool auto_zone = false;
		int auto_zone_id = -1;
		// Heuristica de influencia: cartas que pasaron de cripta a minion
		// y suma de capacidades (coste de pool inferido).
		int influence_count = 0;
		int influence_cost = 0;
	};

	// Zona propuesta por el detector automático, para dibujarla en el
	// overlay mientras se confirma el grupo de cuatro cartas.
	struct AutoZoneView {
		PlayerZone crypt;
		PlayerZone minion;
		bool stable = false;
		int player_idx = -1;
		int card_count = 0;
		float score = 0.0f;
	};

	// Memoria por track fisico (cartas boca abajo) y por
	// (jugador, card_id), para reconocer la influencia aunque SORT cambie
	// el track_id al mover la carta de la cripta a minion.
	struct InfluenceMemory {
		int64_t crypt_seen_ns = 0;
		int64_t influenced_ns = 0;
	};

	// Lock state — cards are Unlocked by default (straight/enderezada)
	// and become Locked only when rotated ~90° (girada)
	enum class CardLockState : uint8_t {
		Unlocked = 0,
		Locked   = 1
	};

	// Per-track persistent card state
	struct TrackedCardState {
		int track_id = -1;
		std::string card_id, card_name, card_url;
		cv::Mat card_image;

		// Última identificación aceptada. Se usa para no repetir el
		// embedding en cada frame mientras la carta siga delante de la
		// cámara con el mismo recorte (dHash). El embedding es la parte
		// más cara del filtro (~34 ms por carta en CPU) y bloquear el
		// hilo de vídeo con él baja los FPS aunque la GPU esté libre.
		uint64_t last_crop_hash = 0;
		float    identity_confidence = 0.0f;
		int      last_label = 0;
		float    last_width = 0.0f;
		float    last_height = 0.0f;

		// Lock state: una carta está girada (tapped) cuando su ángulo
		// difiere ~90° del de la zona minion de su jugador.
		int64_t first_seen_ns   = 0;
		float   initial_angle   = 0.0f;  // radianes; baseline si no hay zona
		float   last_rel_angle  = 0.0f;  // grados normalizados a [-90, 90)
		CardLockState lock_state = CardLockState::Unlocked;
		int     lock_votes      = 0;     // histéresis (+ girada / - lista)
		int64_t lock_changed_ns = 0;     // último cambio de estado

		// Player assignment (determined by zone overlap)
		int  player_idx = -1;   // -1 = unassigned
		bool in_crypt   = false; // false = minion zone

		// Recursos del minion. La camara no puede leer los contadores, asi
		// que se inicializan con el valor conocido de la carta y se ajustan
		// a mano desde /game. -1 = desconocido/no aplica.
		int blood = -1;      // vampire: sangre actual (capacidad al influir)
		int life = -1;       // ally/retainer: vida
		int conviction = -1; // Imbued: conviccion
		std::string resource_card_id; // identidad a la que pertenecen esos valores

		// Evita cobrar dos veces el coste en pool de una carta de
		// biblioteca que permanece en la mesa con el mismo track. Los
		// costes en sangre/vida se ajustan aparte porque la camara no
		// sabe que vampiro paga.
		bool library_pool_cost_paid = false;
		std::string library_pool_cost_paid_id;

		// Lifecycle
		cv::Point2f last_position;
		int64_t     last_seen_ns  = 0;
		int         unseen_frames = 0;
	};

	static constexpr int MAX_PLAYERS = 5;

	// Game-state config (read from OBS settings)
	int                    player_count = 4;

	// Vertical scroll of the rulings block, as a percentage (0..100) of
	// the maximum scroll of the current card. It is a percentage and not a
	// pixel offset so that the OBS slider keeps a fixed range: a card with
	// two rulings barely scrolls and one with eighteen scrolls a lot, and
	// the property cannot depend on what is detected.
	int                    rulings_scroll = 0;
	PlayerState            players[MAX_PLAYERS];

	// Detección automática de zonas a partir de 4 cartas boca abajo.
	// El estado vive separado de las zonas manuales: cuando una zona se
	// confirma, se copia a players[i] y se ignora la zona manual de ese
	// jugador mientras `auto_zones` siga activo.
	bool                   auto_zones = false;
	int                    auto_zone_commit_count = 0;
	int64_t                auto_zone_last_run_ns = 0;
	int64_t                auto_zone_batch_start_ns = 0;
	std::unique_ptr<vtes_detection::AutoZoneDetector> auto_zone_detector;
	std::mutex             auto_zone_lock;
	std::vector<AutoZoneView> auto_zone_views;
	std::mutex             auto_zone_view_lock;

	// Per-track state (track_id → TrackedCardState)
	std::unordered_map<int, TrackedCardState> tracked_cards;
	// Historial cripta/minion por carta, independiente del track_id.
	std::unordered_map<std::string, InfluenceMemory> influence_memory;
	// Anti-duplicado del coste de biblioteca: SORT puede cambiar el track_id
	// de una carta que sigue en la mesa. Se recuerda el ultimo cobro por
	// jugador+carta durante unos segundos para no descontar dos veces el pool.
	std::unordered_map<std::string, int64_t> library_cost_memory;
	std::mutex                                game_state_lock;

	// Overlay background (loaded from data/obs-plugins/…/overlay-bg.png)
	cv::Mat overlay_bg;

	// Ventana de camara del diseno, leida del canal alfa del overlay
	// (coords de diseno 1536x1024). Vacia = usar las constantes kWin*.
	cv::Rect overlay_cam_window;
};
#endif /* FILTERDATA_H */
