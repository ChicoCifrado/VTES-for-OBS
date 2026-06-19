#ifndef YOLO_DETECTOR_HPP
#define YOLO_DETECTOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>

#include "detection_types.hpp"

namespace vtes_detection {

struct OBBObject
{
    cv::Rect2f rect;
    float angle;
    int label;
    float prob;
    int id;
    int unseenFrames;
    std::string card_id;
    std::string card_name;
};

class YOLODetector {
public:
    YOLODetector(const std::string& model_path,
                 cv::Size input_size = cv::Size(640, 640),
                 InferenceDevice device = InferenceDevice::CPU,
                 int device_id = 0,
                 float conf_threshold = 0.3f);

    ~YOLODetector();

    std::vector<OBBObject> inferOBB(const cv::Mat& frame);

    void setConfThreshold(float thresh) { conf_threshold_ = thresh; }
    float confThreshold() const { return conf_threshold_; }
    cv::Size inputSize() const { return input_size_; }
    std::string provider() const;

    void setOutputShape(int num_detections, int features_per_det);

private:
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};
    Ort::Env env_{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "yolo"};
    cv::Size input_size_;
    int num_detections_ = 300;
    float conf_threshold_;
    InferenceDevice device_;
    int device_id_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<Ort::AllocatedStringPtr> input_names_ptr_;
    std::vector<Ort::AllocatedStringPtr> output_names_ptr_;

    cv::Mat static_resize(const cv::Mat& img);
    void decode_outputs(const float* prob, std::vector<OBBObject>& objects,
                        float conf_th, float scale,
                        int img_w, int img_h,
                        float pad_x = 0.0f, float pad_y = 0.0f);
};

} // namespace vtes_detection

#endif
