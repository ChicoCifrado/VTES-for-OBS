#include "yolov8_onnxruntime.hpp"

#include <algorithm>
#include <cmath>

namespace yolov8_cpp {

std::vector<Object> YOLOv8::inference(const cv::Mat &frame)
{
	ONNXRuntimeModel::inference(frame, 0);

	float *net_pred = (float *)this->output_buffer_[0].get();

	// YOLOv8 output: [batch, channels, detections]
	// channels = num_classes + 4 (x, y, w, h)
	float scale = std::fminf((float)input_w_[0] / (float)frame.cols,
				 (float)input_h_[0] / (float)frame.rows);

	std::vector<Object> objects;
	decode_outputs(net_pred, objects, this->bbox_conf_thresh_, scale,
		       frame.cols, frame.rows);
	return objects;
}

void YOLOv8::decode_outputs(const float *prob, std::vector<Object> &objects,
			    const float bbox_conf_thresh, const float scale,
			    const int img_w, const int img_h)
{
	const int channels = this->num_classes_ + 4;
	std::vector<Object> proposals;

	for (int i = 0; i < this->num_detections_; ++i) {
		// Find the class with the highest score
		int class_id = 0;
		float max_class_score = 0.0f;

		for (int c = 0; c < this->num_classes_; ++c) {
			// YOLOv8 stores: [x, y, w, h, cls0, cls1, ...] per detection
			// Layout is [channels, detections], so: prob[channel * num_detections + det]
			float score = prob[(4 + c) * this->num_detections_ + i];

			// Apply sigmoid
			score = 1.0f / (1.0f + std::exp(-score));

			if (score > max_class_score) {
				max_class_score = score;
				class_id = c;
			}
		}

		if (max_class_score > bbox_conf_thresh) {
			// Extract bbox coordinates (also need sigmoid for YOLOv8)
			float cx = prob[0 * this->num_detections_ + i];
			float cy = prob[1 * this->num_detections_ + i];
			float w = prob[2 * this->num_detections_ + i];
			float h = prob[3 * this->num_detections_ + i];

			// YOLOv8 outputs are already in image space after DFL, no sigmoid needed for boxes
			// But they are relative to the 640x640 input, so we scale them
			float x0 = (cx - w * 0.5f) / scale;
			float y0 = (cy - h * 0.5f) / scale;
			float x1 = (cx + w * 0.5f) / scale;
			float y1 = (cy + h * 0.5f) / scale;

			// Clip to image bounds
			x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
			y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
			x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
			y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

			Object obj;
			obj.rect.x = x0;
			obj.rect.y = y0;
			obj.rect.width = x1 - x0;
			obj.rect.height = y1 - y0;
			obj.label = class_id;
			obj.prob = max_class_score;
			obj.unseenFrames = 0;
			obj.id = 0;
			proposals.push_back(obj);
		}
	}

	// Sort by confidence
	qsort_descent_inplace(proposals);

	// NMS
	std::vector<int> picked;
	nms_sorted_bboxes(proposals, picked, nms_thresh_);

	objects.clear();
	for (int i = 0; i < (int)picked.size(); ++i) {
		objects.push_back(proposals[picked[i]]);
		objects.back().id = objects.size();
	}
}

} // namespace yolov8_cpp
