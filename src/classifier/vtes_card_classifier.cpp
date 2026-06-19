// /mnt/c/Users/JackSuicide/VTES/vtes_obs_detect/src/classifier/vtes_card_classifier.cpp

#include "vtes_card_classifier.hpp"
#include "plugin-support.h"
#include <obs.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace vtes_classifier {

// ============================================================
// CardCropExtractor
// ============================================================

std::vector<CardCropExtractor::DetectionOutput> 
CardCropExtractor::extract(const cv::Mat& image, const std::vector<DetectionInput>& detections) const {
    std::vector<DetectionOutput> results;
    int img_w = image.cols, img_h = image.rows;

    for (const auto& det : detections) {
        int x1 = std::max(0, det.bbox.x);
        int y1 = std::max(0, det.bbox.y);
        int x2 = std::min(img_w, det.bbox.x + det.bbox.width);
        int y2 = std::min(img_h, det.bbox.y + det.bbox.height);

        if (x2 <= x1 || y2 <= y1) continue;

        cv::Mat raw = image(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
        cv::Mat crop = rectify(raw);

        results.push_back({det.bbox, det.conf, det.classId, det.idx, std::move(crop)});
    }
    return results;
}

cv::Mat CardCropExtractor::rectify(const cv::Mat& rawCrop) const {
    if (contourCb_) {
        try {
            auto contour = contourCb_(rawCrop);
            if (contour && contour->size() >= 4) {
                std::vector<cv::Point2f> pts(contour->begin(), contour->end());
                if (pts.size() > 4) {
                    std::vector<int> hull;
                    cv::convexHull(cv::Mat(pts), hull, false, false);
                    std::vector<cv::Point2f> hullPts;
                    for (int idx : hull) hullPts.push_back(pts[idx]);
                    double eps = 0.05 * cv::arcLength(hullPts, true);
                    std::vector<cv::Point2f> approx;
                    cv::approxPolyDP(hullPts, approx, eps, true);
                    if (approx.size() == 4) pts = approx;
                    else pts = {pts[0], pts[1], pts[2], pts[3]};
                }
                auto ordered = orderPoints(pts);
                cv::Point2f dst[4] = {{0,0}, {CROP_W-1,0}, {CROP_W-1,CROP_H-1}, {0,CROP_H-1}};
                cv::Mat M = cv::getPerspectiveTransform(ordered.data(), dst);
                cv::Mat warped;
                cv::warpPerspective(rawCrop, warped, M, {CROP_W, CROP_H});
                return warped;
            }
        } catch (...) {}
    }
    cv::Mat resized;
    cv::resize(rawCrop, resized, {CROP_W, CROP_H}, 0, 0, cv::INTER_LINEAR);
    return resized;
}

std::vector<cv::Point2f> CardCropExtractor::orderPoints(const std::vector<cv::Point2f>& pts) {
    std::vector<float> sums, diffs;
    for (auto& p : pts) { sums.push_back(p.x + p.y); diffs.push_back(p.x - p.y); }
    std::vector<cv::Point2f> rect(4);
    rect[0] = pts[std::min_element(sums.begin(), sums.end()) - sums.begin()];
    rect[2] = pts[std::max_element(sums.begin(), sums.end()) - sums.begin()];
    rect[1] = pts[std::min_element(diffs.begin(), diffs.end()) - diffs.begin()];
    rect[3] = pts[std::max_element(diffs.begin(), diffs.end()) - diffs.begin()];
    return rect;
}

// ============================================================
// VTESCardClassifier
// ============================================================

VTESCardClassifier::VTESCardClassifier(const Config& config, ContourCallback contourCb)
    : config_(config), contourCb_(std::move(contourCb)) {
    if (config_.useOnnxClassifier && !config_.onnxModelPath.empty()) {
        initializeOnnxSession();
    }
}

VTESCardClassifier::~VTESCardClassifier() = default;

CardType VTESCardClassifier::classify(const cv::Mat& crop, float& outConfidence, SignalScores& outSignals) const {
    if (crop.empty() || crop.rows < 8 || crop.cols < 8) {
        outConfidence = 0.0f;
        outSignals = {0.0f, 0.0f, 0.0f};
        return CardType::Unknown;
    }

    float oval = signalOval(crop);
    float color = signalColor(crop);
    float contour = signalContour(crop);

    outSignals = {oval, color, contour};

    // ONNX classifier first (primary)
    if (config_.useOnnxClassifier && !onnxNet_.empty()) {
        CardType onnxType = classifyLibrarySubtype(crop, outConfidence);
        if (onnxType != CardType::Unknown) {
            return onnxType;
        }
    }

    // Fallback to heuristic
    return fuse(oval, color, contour, crop, outConfidence);
}

float VTESCardClassifier::signalOval(const cv::Mat& crop) const {
    if (crop.empty() || crop.rows < 16 || crop.cols < 16) return 0.0f;

    int h = crop.rows, w = crop.cols;
    int zone_h = static_cast<int>(h * 0.60f);
    if (zone_h < 8) return 0.0f;
    cv::Mat zone = crop(cv::Rect(0, 0, w, zone_h));

    cv::Mat gray, blurred, edges;
    try {
        cv::cvtColor(zone, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, {5,5}, 0);
        cv::Canny(blurred, edges, 30, 100);
    } catch (...) {
        return 0.0f;
    }

    std::vector<std::vector<cv::Point>> contours;
    try {
        cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    } catch (...) {
        return 0.0f;
    }

    float zone_area = static_cast<float>(zone_h) * static_cast<float>(w);
    float min_area = zone_area * 0.15f;
    float cx_center = static_cast<float>(w) / 2.0f;
    float cx_tol = static_cast<float>(w) * 0.20f;

    float best = 0.0f;
    for (auto& cnt : contours) {
        if (cnt.size() < 5) continue;
        float area = static_cast<float>(cv::contourArea(cnt));
        if (area < min_area) continue;

        try {
            cv::RotatedRect ell = cv::fitEllipse(cnt);
            float ma = std::max(ell.size.width, ell.size.height);
            float mi = std::min(ell.size.width, ell.size.height);
            if (ma == 0) continue;
            float ratio = mi / ma;
            if (ratio < 0.50f || ratio > 0.85f) continue;
            if (std::abs(ell.center.x - cx_center) > cx_tol) continue;

            float ratio_score = 1.0f - std::abs(ratio - 0.68f) / 0.18f;
            float center_score = 1.0f - std::abs(ell.center.x - cx_center) / cx_tol;
            float score = std::clamp(ratio_score * 0.6f + center_score * 0.4f, 0.0f, 1.0f);
            best = std::max(best, score);
        } catch (...) {}
    }
    return best;
}

float VTESCardClassifier::signalColor(const cv::Mat& crop) const {
    if (crop.empty() || crop.rows < 8 || crop.cols < 8) return 0.0f;

    int h = crop.rows, w = crop.cols;
    int bh = std::max(2, static_cast<int>(h * 0.08f));
    int bw = std::max(2, static_cast<int>(w * 0.08f));

    if (bh >= h || bw >= w) return 0.0f;

    cv::Mat top = crop(cv::Rect(0, 0, w, bh));
    cv::Mat bottom = crop(cv::Rect(0, h-bh, w, bh));
    cv::Mat left = crop(cv::Rect(0, 0, bw, h));
    cv::Mat right = crop(cv::Rect(w-bw, 0, bw, h));

    if (top.empty() || bottom.empty() || left.empty() || right.empty()) return 0.0f;

    std::vector<cv::Mat> bands = {top, bottom, left, right};
    cv::Mat band;
    try {
        cv::vconcat(bands, band);
    } catch (...) {
        return 0.0f;
    }

    if (band.empty()) return 0.0f;

    cv::Mat band_hsv;
    try {
        cv::cvtColor(band, band_hsv, cv::COLOR_BGR2HSV);
    } catch (...) {
        return 0.0f;
    }

    if (band_hsv.empty()) return 0.0f;

    std::vector<cv::Mat> hsv_ch;
    try {
        cv::split(band_hsv, hsv_ch);
    } catch (...) {
        return 0.0f;
    }

    if (hsv_ch.size() < 3) return 0.0f;

    cv::Mat H = hsv_ch[0];
    cv::Mat S = hsv_ch[1];
    cv::Mat V = hsv_ch[2];

    if (H.empty() || S.empty() || V.empty()) return 0.0f;

    cv::Mat green_mask;
    try {
        green_mask = (H >= 18) & (H <= 43) & (S > 76) & (V > 76); // S>0.3*255, V>0.3*255
    } catch (...) {
        return 0.0f;
    }

    if (green_mask.empty()) return 0.0f;

    float ratio = 0.0f;
    try {
        ratio = cv::countNonZero(green_mask) / static_cast<float>(green_mask.total());
    } catch (...) {
        return 0.0f;
    }

    return std::clamp(ratio / 0.30f, 0.0f, 1.0f);
}

float VTESCardClassifier::signalContour(const cv::Mat& crop) const {
    if (!contourCb_) return 0.5f;
    if (crop.empty() || crop.rows < 8 || crop.cols < 8) return 0.3f;

    try {
        auto contour = contourCb_(crop);
        if (!contour || contour->size() < 4) return 0.3f;

        std::vector<cv::Point2f> pts(contour->begin(), contour->end());
        if (pts.size() > 4) {
            std::vector<int> hull;
            cv::convexHull(cv::Mat(pts), hull, false, false);
            std::vector<cv::Point2f> hullPts;
            for (int idx : hull) hullPts.push_back((*contour)[idx]);
            double eps = 0.05 * cv::arcLength(hullPts, true);
            std::vector<cv::Point2f> approx;
            cv::approxPolyDP(hullPts, approx, eps, true);
            if (approx.size() == 4) pts = approx;
            else pts = {pts[0], pts[1], pts[2], pts[3]};
        }

        cv::Mat ptsMat(pts);
        std::vector<int> hullIdx;
        cv::convexHull(ptsMat, hullIdx, false, false);
        std::vector<cv::Point> hullPts;
        for (int idx : hullIdx) hullPts.push_back(cv::Point(pts[idx]));
        float hullArea = static_cast<float>(cv::contourArea(hullPts));
        float cntArea = static_cast<float>(cv::contourArea(pts));
        if (hullArea < 1) return 0.3f;
        float convexity = cntArea / hullArea;

        cv::Rect br = cv::boundingRect(pts);
        if (br.height == 0) return 0.3f;
        float aspect = br.width / static_cast<float>(br.height);
        float aspectScore = std::clamp(1.0f - std::abs(aspect - 0.70f) / 0.30f, 0.0f, 1.0f);

        return std::clamp(convexity * 0.5f + aspectScore * 0.5f, 0.0f, 1.0f);
    } catch (...) { return 0.5f; }
}

CardType VTESCardClassifier::fuse(float oval, float color, float contour, const cv::Mat&, float& outConfidence) const {
    if (oval >= config_.ovalThreshold) {
        outConfidence = std::clamp((oval * config_.ovalWeight + contour * config_.contourWeight) / (config_.ovalWeight + config_.contourWeight), 0.0f, 1.0f);
        return CardType::Vampire;
    }
    if (color >= config_.colorThreshold && oval < config_.ovalRejectForMaster) {
        outConfidence = std::clamp((color * config_.colorWeight + contour * config_.contourWeight) / (config_.colorWeight + config_.contourWeight), 0.0f, 1.0f);
        return CardType::Master;
    }
    float combined = oval * config_.ovalWeight + color * config_.colorWeight + contour * config_.contourWeight;
    if (combined < 0.25f) {
        outConfidence = combined;
        return CardType::Unknown;
    }
    outConfidence = 0.40f;
    return CardType::LibraryAction;
}

CardType VTESCardClassifier::classifyLibrarySubtype(const cv::Mat& crop, float& outConfidence) const {
    if (onnxNet_.empty()) {
        outConfidence = 0.40f;
        return CardType::LibraryAction;
    }

    try {
        cv::Mat inputTensor = preprocessForOnnx(crop);
        std::vector<float> logits = runOnnxInference(inputTensor);

        // Debug: log all logits
        if (logits.size() >= 7) {
            obs_log(LOG_INFO, "[TypeClassifier] Logits: v=%.2f m=%.2f eq=%.2f co=%.2f la=%.2f re=%.2f po=%.2f",
                    logits[0], logits[1], logits[2], logits[3], logits[4], logits[5], logits[6]);
        }

        // Find max logit (7 classes: vampire, master, equipment, combat, library_action, reaction, political)
        auto maxIt = std::max_element(logits.begin(), logits.end());
        int maxIdx = static_cast<int>(std::distance(logits.begin(), maxIt));
        float maxLogit = *maxIt;

        // Softmax for confidence
        float sumExp = 0.0f;
        for (float l : logits) sumExp += std::exp(l - maxLogit);
        outConfidence = std::exp(maxLogit - maxLogit) / sumExp; // = 1.0 / sumExp

        // Map ONNX output index to CardType
        // ONNX model classes: 0=vampire, 1=master, 2=equipment, 3=combat, 4=library_action, 5=reaction, 6=political
        static const CardType idxToType[7] = {
            CardType::Vampire,
            CardType::Master,
            CardType::Equipment,
            CardType::Combat,
            CardType::LibraryAction,
            CardType::Reaction,
            CardType::Political
        };

        if (maxIdx >= 0 && maxIdx < 7) {
            return idxToType[maxIdx];
        }
    } catch (...) {
        // Fallback to heuristic
    }

    outConfidence = 0.40f;
    return CardType::LibraryAction;
}

// ============================================================
// ONNX Runtime helper methods
// ============================================================

bool VTESCardClassifier::initializeOnnxSession() {
    try {
        onnxNet_ = cv::dnn::readNetFromONNX(config_.onnxModelPath);
        if (onnxNet_.empty()) return false;

        onnxNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        onnxNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        // Get input name from the model
        std::vector<cv::String> in_names = onnxNet_.getUnconnectedOutLayersNames();
        // Use a dummy forward pass to discover input name
        cv::Mat dummy(224, 224, CV_8UC3);
        cv::Mat blob = cv::dnn::blobFromImage(dummy, 1.0, cv::Size(224, 224),
                                               cv::Scalar(), true, false);
        onnxNet_.setInput(blob);
        // The input layer name can be obtained from the net
        onnxInputName_ = ""; // will be resolved at runtime by setInput

        return true;
    } catch (const std::exception&) {
        return false;
    }
}

cv::Mat VTESCardClassifier::preprocessForOnnx(const cv::Mat& crop) const {
    // Resize to 224x224 (letterbox to maintain aspect ratio)
    cv::Mat resized;
    cv::Size targetSize(224, 224);

    float scale = std::min(static_cast<float>(targetSize.width) / crop.cols,
                           static_cast<float>(targetSize.height) / crop.rows);
    int newW = static_cast<int>(crop.cols * scale);
    int newH = static_cast<int>(crop.rows * scale);

    cv::Mat scaled;
    cv::resize(crop, scaled, cv::Size(newW, newH));

    // Create letterboxed image (ImageNet mean/std normalization)
    cv::Mat input(targetSize, CV_32FC3, cv::Scalar(0, 0, 0));
    int dx = (targetSize.width - newW) / 2;
    int dy = (targetSize.height - newH) / 2;
    cv::Mat roi = input(cv::Rect(dx, dy, newW, newH));
    scaled.convertTo(roi, CV_32FC3, 1.0 / 255.0);

    // Normalize with ImageNet mean/std
    cv::subtract(roi, cv::Scalar(0.485, 0.456, 0.406), roi);
    cv::divide(roi, cv::Scalar(0.229, 0.224, 0.225), roi);

    // Convert to NCHW format (1, 3, 224, 224)
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.0, targetSize, cv::Scalar(), true, false);

    return blob;
}

std::vector<float> VTESCardClassifier::runOnnxInference(const cv::Mat& inputTensor) const {
    std::vector<float> output;

    try {
        onnxNet_.setInput(inputTensor);
        cv::Mat result = onnxNet_.forward();

        if (result.empty()) return output;

        size_t outputSize = result.total();
        output.resize(outputSize);
        std::copy(result.ptr<float>(), result.ptr<float>() + outputSize, output.begin());

    } catch (const std::exception&) {
        // Return empty on error
    }

    return output;
}

// ============================================================
// VTESDetectionPipeline
// ============================================================

VTESDetectionPipeline::VTESDetectionPipeline(const Config& config) : config_(config), extractor_(config.contourCb), classifier_(config.classifierConfig, config.contourCb) {
    // YOLO model initialization is done externally and passed via config
    // This pipeline assumes the YOLO model is already loaded in the OBS filter
}

VTESDetectionPipeline::~VTESDetectionPipeline() = default;

std::vector<CardResult> VTESDetectionPipeline::run(const cv::Mat& image) const {
    std::vector<CardResult> results;

    // This pipeline expects the YOLO detections to be passed externally
    // The actual YOLO inference is done in the OBS filter (detect-filter-obb.cpp)
    // and then passed to this pipeline for classification

    // Placeholder - in production, this would be called with pre-detected OBB objects
    // For now, return empty - the OBS filter handles the full pipeline
    return results;
}

} // namespace vtes_classifier