#ifndef YOLOV8_OBB_HPP
#define YOLOV8_OBB_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>
#include <vector>

#include "ort-model/ONNXRuntimeModel.h"

namespace yolov8_obb_cpp {

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

class YOLOv8OBB : public ONNXRuntimeModel
{
public:
    YOLOv8OBB(file_name_t path_to_model, int intra_op_num_threads, int num_classes,
              int inter_op_num_threads = 1, const std::string& use_gpu_ = "",
              int device_id = 0, bool use_parallel = false, float nms_th = 0.45f,
              float conf_th = 0.3f)
        : ONNXRuntimeModel(path_to_model, intra_op_num_threads, num_classes,
                           inter_op_num_threads, use_gpu_, device_id, use_parallel, nms_th,
                           conf_th)
    {
        this->input_normalize_scale_ = 1.0f / 255.0f;
        num_detections_ = 300;
        if (this->output_shapes_.size() > 0 && this->output_shapes_[0].size() >= 3) {
            auto& shape = this->output_shapes_[0];
            int64_t dets = (shape[2] <= 10 && shape[2] > 0) ? shape[1] : shape[2];
            if (dets > 0 && dets < 20000)
                num_detections_ = static_cast<int>(dets);
        }
    }

    // YOLO26 one-to-one NMS-free decode: [1, N, 7] = [x, y, w, h, score, class_id, angle]
    void decode_outputs(const float* prob, std::vector<OBBObject>& objects,
                        const float conf_th, const float scale,
                        const int img_w, const int img_h,
                        const float pad_x = 0.0f, const float pad_y = 0.0f);

    using ONNXRuntimeModel::inference;
    std::vector<OBBObject> inferOBB(const cv::Mat& frame);
    std::vector<Object> inference(const cv::Mat& frame) override;

    void setOutputShape(int num_detections, int features_per_det) override
    {
        this->num_detections_ = num_detections;
        (void)features_per_det;
    }

protected:
    int num_detections_;
};

} // namespace yolov8_obb_cpp

#endif // YOLOV8_OBB_HPP
