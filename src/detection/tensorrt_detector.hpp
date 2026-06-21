#ifndef TENSORRT_DETECTOR_HPP
#define TENSORRT_DETECTOR_HPP

#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include <string>

#include "detector_base.hpp"
#include "detection_types.hpp"

namespace vtes_detection {

struct TensorRTDetectorImpl;

class TensorRTDetector : public DetectorBase {
public:
    TensorRTDetector(const std::string& engine_path,
                     cv::Size input_size = cv::Size(640, 640),
                     int device_id = 0,
                     float conf_threshold = 0.3f);

    ~TensorRTDetector() override;

    TensorRTDetector(TensorRTDetector&&) noexcept = default;
    TensorRTDetector& operator=(TensorRTDetector&&) noexcept = default;

    std::vector<OBBObject> inferOBB(const cv::Mat& frame) override;

    void setConfThreshold(float thresh) override { conf_threshold_ = thresh; }
    float confThreshold() const override { return conf_threshold_; }
    cv::Size inputSize() const override { return input_size_; }
    std::string provider() const override;
    void setOutputShape(int num_detections, int /*features_per_det*/) override { num_detections_ = num_detections; }

private:
    std::unique_ptr<TensorRTDetectorImpl> impl_;
    cv::Size input_size_;
    int num_detections_ = 300;
    float conf_threshold_;
    int device_id_;
    bool graphCaptured_ = false;

    cv::Mat static_resize(const cv::Mat& img) const;
    void fill_pinned_input(const cv::Mat& img, float* dst) const;
    void decode_outputs(const float* prob, std::vector<OBBObject>& objects,
                        float conf_th, float scale,
                        int img_w, int img_h,
                        float pad_x = 0.0f, float pad_y = 0.0f) const;
};

} // namespace vtes_detection

#endif
