#include "yolov8_obb_onnxruntime.hpp"
#include "plugin-support.h"
#include <obs.h>
#include <algorithm>
#include <cmath>

namespace yolov8_obb_cpp
{

// YOLO26 one-to-one NMS-free decode
// Output format: [1, N, 7] = [x, y, w, h, score, class_id, angle]
// score and class_id are already sigmoid-activated (0-1) from end2end export
void YOLOv8OBB::decode_outputs(const float* prob, std::vector<OBBObject>& objects,
                        const float conf_th, const float scale,
                        const int img_w, const int img_h,
                        const float pad_x, const float pad_y)
{
    objects.clear();
    const int max_dets = this->num_detections_;
    const int stride = 7;

    for (int i = 0; i < max_dets; ++i)
    {
        float x = prob[i * stride + 0];
        float y = prob[i * stride + 1];
        float w = prob[i * stride + 2];
        float h = prob[i * stride + 3];
        float score = prob[i * stride + 4];
        float angle = prob[i * stride + 6];

        if (score < 0.0f) score = 0.0f;
        if (score > 1.0f) score = 1.0f;

        if (score > conf_th)
        {
            float x0 = (x - pad_x - w * 0.5f) / scale;
            float y0 = (y - pad_y - h * 0.5f) / scale;
            float x1 = (x - pad_x + w * 0.5f) / scale;
            float y1 = (y - pad_y + h * 0.5f) / scale;

            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            if (x1 <= x0 || y1 <= y0) continue;

            OBBObject obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.angle = angle;
            obj.label = 0;
            obj.prob = score;
            obj.unseenFrames = 0;
            obj.id = static_cast<int>(objects.size()) + 1;
            objects.push_back(obj);
        }
    }
}

std::vector<OBBObject> YOLOv8OBB::inferOBB(const cv::Mat& frame)
{
    ONNXRuntimeModel::inference(frame, 0);

    if (this->output_shapes_.size() > 0 && this->output_shapes_[0].size() >= 3) {
        auto& shape = this->output_shapes_[0];
        int64_t dets = (shape[2] <= 10 && shape[2] > 0) ? shape[1] : shape[2];
        if (dets > 0 && dets < 20000)
            this->num_detections_ = static_cast<int>(dets);
    }

    if (this->output_buffer_.empty() || !this->output_buffer_[0]) {
        obs_log(LOG_ERROR, "[OBB] inferOBB: output buffer is null after inference");
        return {};
    }
    float* net_pred = (float*)this->output_buffer_[0].get();
    float scale = std::fminf((float)input_w_[0] / (float)frame.cols,
                             (float)input_h_[0] / (float)frame.rows);
    float unpad_w_f = scale * (float)frame.cols;
    float unpad_h_f = scale * (float)frame.rows;
    int pad_x_int = (int)(((float)input_w_[0] - unpad_w_f) * 0.5f);
    int pad_y_int = (int)(((float)input_h_[0] - unpad_h_f) * 0.5f);

    std::vector<OBBObject> objects;
    decode_outputs(net_pred, objects, this->bbox_conf_thresh_, scale,
                   frame.cols, frame.rows, (float)pad_x_int, (float)pad_y_int);

    if (objects.empty() && this->output_shapes_.size() > 0) {
        const int stride = 7;
        float max_score = 0.0f;
        int max_idx = 0;
        float score_sum = 0.0f;
        int n = std::min(this->num_detections_, 300);
        for (int i = 0; i < n; ++i) {
            float s = net_pred[i * stride + 4];
            score_sum += s;
            if (s > max_score) { max_score = s; max_idx = i; }
        }
        obs_log(LOG_INFO,
            "[OBB] Zero detections — max_score=%.4f (idx %d), mean=%.4f, conf_thresh=%.2f, num_dets=%d",
            max_score, max_idx, score_sum / n, this->bbox_conf_thresh_, this->num_detections_);
        if (max_idx >= 0) {
            int off = max_idx * stride;
            obs_log(LOG_INFO,
                "[OBB] Best probe: [x=%.2f y=%.2f w=%.2f h=%.2f score=%.4f _ angle=%.4f]",
                net_pred[off+0], net_pred[off+1], net_pred[off+2], net_pred[off+3],
                net_pred[off+4], net_pred[off+6]);
        }
    }

    return objects;
}

std::vector<Object> YOLOv8OBB::inference(const cv::Mat& frame)
{
    auto obb_objects = inferOBB(frame);
    std::vector<Object> objects;
    for (const auto& obb : obb_objects)
    {
        Object obj;
        obj.rect = obb.rect;
        obj.label = obb.label;
        obj.prob = obb.prob;
        obj.id = obb.id;
        obj.unseenFrames = obb.unseenFrames;
        objects.push_back(obj);
    }
    return objects;
}

} // namespace yolov8_obb_cpp
