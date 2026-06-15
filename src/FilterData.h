#ifndef FILTERDATA_H
#define FILTERDATA_H

#include <obs-module.h>
#include "ort-model/ONNXRuntimeModel.h"
#include "sort/Sort.h"
#include "ws-client.h"
#include "classifier/vtes_card_classifier.hpp"
#include "embedding_matcher.h"
#include "vtes_database.hpp"
#include "ocr/vtes_ocr.hpp"
#include <chrono>
#include <atomic>
#include <deque>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

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
	std::string useGPU;
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

	std::mutex inputBGRALock;
	std::mutex outputLock;
	std::mutex modelMutex;

	std::unique_ptr<ONNXRuntimeModel> onnxruntimemodel;
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

	// VTES-specific: WebSocket client for sending card crops to Node.js server
	WebSocketClient wsClient;
	std::string wsHost;
	int wsPort;
	std::chrono::steady_clock::time_point lastSendTime;
	int cooldownSeconds;
	std::atomic<bool> wsSendInProgress;

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
		static constexpr int IDENTITY_WINDOW = 10;

		std::deque<bool> history; // sliding window of detections
	};

	std::unordered_map<int, TrackedObject> temporal_tracks;
	int next_track_id = 1;
	bool temporal_smoothing_enabled = true;
	float temporal_min_confidence = 0.3f; // require 30% of window
	int temporal_min_stable = 3;          // require 3 consecutive detections

	// --- Vision-based type classifier (vtes_type_classifier.onnx) ---
	std::unique_ptr<Ort::Session> type_classifier_session;
	Ort::Env type_classifier_env{ OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "vtes-type-clf" };
	std::vector<std::string> type_labels;  // from type_labels.json

	// --- VTES Card Database (vtes.json in memory) ---
	VTESCardDatabase vtes_db;

	// --- Embedding-based card identification ---
	EmbeddingMatcher embedder;
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
	std::vector<VTESCardNameEntry> card_name_entries;  // built from vtes_db for fuzzy matching
	bool ocr_enabled = false;

};
#endif /* FILTERDATA_H */
