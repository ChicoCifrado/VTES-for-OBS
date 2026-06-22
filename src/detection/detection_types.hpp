#ifndef DETECTION_TYPES_HPP
#define DETECTION_TYPES_HPP

#include <string>
#include <opencv2/core/types.hpp>
#include <opencv2/video/tracking.hpp>

enum class InferenceDevice { CPU, CUDA, DirectML };

inline const char* inferenceDeviceToString(InferenceDevice dev) {
    switch (dev) {
        case InferenceDevice::CUDA: return "cuda";
        case InferenceDevice::DirectML: return "dml";
        default: return "cpu";
    }
}

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
    uint64_t id;
    uint64_t unseenFrames;
    cv::KalmanFilter kf;
};

#endif
