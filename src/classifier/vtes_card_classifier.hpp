#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <functional>
#include <optional>
#include <array>
#include <memory>

namespace vtes_classifier {

enum class CardType {
    Vampire,
    Master,
    LibraryAction,
    Reaction,
    Combat,
    Equipment,
    Political,
    Unknown
};

inline std::string cardTypeToString(CardType type) {
    switch (type) {
        case CardType::Vampire:       return "vampire";
        case CardType::Master:        return "master";
        case CardType::LibraryAction: return "library_action";
        case CardType::Reaction:      return "reaction";
        case CardType::Combat:        return "combat";
        case CardType::Equipment:     return "equipment";
        case CardType::Political:     return "political";
        default:                      return "unknown";
    }
}

struct SignalScores {
    float oval = 0.0f;
    float color = 0.0f;
    float contour = 0.0f;
};

struct CardResult {
    cv::Rect bbox;
    float detectionConf = 0.0f;
    cv::Mat crop;
    CardType type = CardType::Unknown;
    float typeConfidence = 0.0f;
    SignalScores signals;
};

using ContourCallback = std::function<std::optional<std::vector<cv::Point2f>>(const cv::Mat& crop)>;

class CardCropExtractor {
public:
    static constexpr int CROP_W = 224;
    static constexpr int CROP_H = 320;

    explicit CardCropExtractor(ContourCallback contourCb = nullptr)
        : contourCb_(std::move(contourCb)) {}

    struct DetectionInput {
        cv::Rect bbox;
        float conf = 0.0f;
        int classId = 0;
        int idx = 0;
    };

    struct DetectionOutput {
        cv::Rect bbox;
        float conf = 0.0f;
        int classId = 0;
        int idx = 0;
        cv::Mat crop;
    };

    std::vector<DetectionOutput> extract(const cv::Mat& image, const std::vector<DetectionInput>& detections) const;

private:
    ContourCallback contourCb_;

    cv::Mat rectify(const cv::Mat& rawCrop) const;
    static std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& pts);
};

class VTESCardClassifier {
public:
    struct Config {
        float ovalWeight = 0.45f;
        float colorWeight = 0.35f;
        float contourWeight = 0.20f;
        float ovalThreshold = 0.60f;
        float colorThreshold = 0.55f;
        float ovalRejectForMaster = 0.40f;
        std::string onnxModelPath = "";  // Path to classifier.onnx
        bool useOnnxClassifier = false;  // Enable ONNX classifier for library subtypes
    };

    explicit VTESCardClassifier(const Config& config = {}, ContourCallback contourCb = nullptr);
    ~VTESCardClassifier();

    CardType classify(const cv::Mat& crop, float& outConfidence, SignalScores& outSignals) const;

    // Delete copy/move (ONNX session not copyable)
    VTESCardClassifier(const VTESCardClassifier&) = delete;
    VTESCardClassifier& operator=(const VTESCardClassifier&) = delete;
    VTESCardClassifier(VTESCardClassifier&&) = default;
    VTESCardClassifier& operator=(VTESCardClassifier&&) = default;

private:
    Config config_;
    ContourCallback contourCb_;

    // ONNX Runtime for library subtype classification
    std::unique_ptr<Ort::Session> onnxSession_;
    std::unique_ptr<Ort::Env> onnxEnv_;
    std::string onnxInputName_;
    std::vector<int64_t> onnxInputShape_ = {1, 3, 224, 224};

    float signalOval(const cv::Mat& crop) const;
    float signalColor(const cv::Mat& crop) const;
    float signalContour(const cv::Mat& crop) const;
    CardType fuse(float oval, float color, float contour, const cv::Mat& crop, float& outConfidence) const;
    CardType classifyLibrarySubtype(const cv::Mat& crop, float& outConfidence) const;

    // ONNX helper methods
    bool initializeOnnxSession();
    cv::Mat preprocessForOnnx(const cv::Mat& crop) const;
    std::vector<float> runOnnxInference(const cv::Mat& inputTensor) const;
};

class VTESDetectionPipeline {
public:
    struct Config {
        std::string modelPath;
        cv::Size inputSize = {640, 640};
        float confThreshold = 0.35f;
        float iouThreshold = 0.45f;
        VTESCardClassifier::Config classifierConfig;
        ContourCallback contourCb = nullptr;
    };

    explicit VTESDetectionPipeline(const Config& config);
    ~VTESDetectionPipeline();

    std::vector<CardResult> run(const cv::Mat& image) const;

private:
    Config config_;
    void* session_ = nullptr;  // ONNX Runtime session (opaque)
    std::string inputName_;
    CardCropExtractor extractor_;
    VTESCardClassifier classifier_;

    // Non-copyable
    VTESDetectionPipeline(const VTESDetectionPipeline&) = delete;
    VTESDetectionPipeline& operator=(const VTESDetectionPipeline&) = delete;
};

} // namespace vtes_classifier