#ifndef YOLOV8_HPP
#define YOLOV8_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ort-model/ONNXRuntimeModel.h"

namespace yolov8_cpp {

class YOLOv8 : public ONNXRuntimeModel {
public:
	YOLOv8(file_name_t path_to_model, int intra_op_num_threads, int num_classes,
	       int inter_op_num_threads = 1, const std::string &use_gpu_ = "",
	       int device_id = 0, bool use_parallel = false, float nms_th = 0.45f,
	       float conf_th = 0.3f)
		: ONNXRuntimeModel(path_to_model, intra_op_num_threads, num_classes,
				   inter_op_num_threads, use_gpu_, device_id, use_parallel, nms_th,
				   conf_th)
	{
		// YOLOv8 output shape: [batch, num_classes+4, num_detections]
		// e.g. [1, 5, 8400] for 1 class
		this->num_detections_ = 8400; // standard YOLOv8
	}

	std::vector<Object> inference(const cv::Mat &frame) override;

protected:
	int num_detections_;

	void decode_outputs(const float *prob, std::vector<Object> &objects,
			    const float bbox_conf_thresh, const float scale,
			    const int img_w, const int img_h);
};

} // namespace yolov8_cpp

#endif // YOLOV8_HPP
