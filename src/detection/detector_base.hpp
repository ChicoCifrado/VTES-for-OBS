#ifndef DETECTOR_BASE_HPP
#define DETECTOR_BASE_HPP

#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include <string>

#include "detection_types.hpp"

namespace vtes_detection {

class DetectorBase {
public:
    virtual ~DetectorBase() = default;

    virtual std::vector<OBBObject> inferOBB(const cv::Mat& frame) = 0;
    virtual void setConfThreshold(float thresh) = 0;
    virtual float confThreshold() const = 0;
    virtual cv::Size inputSize() const = 0;
    virtual std::string provider() const = 0;
    virtual void setOutputShape(int num_detections, int features_per_det) = 0;
};

} // namespace vtes_detection

#endif
