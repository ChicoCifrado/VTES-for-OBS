#ifdef _WIN32
#define NOMINMAX
#endif
#include "yolo_detector.hpp"
#include "plugin-support.h"
#include <obs.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#endif

namespace vtes_detection {

static Ort::SessionOptions makeSessionOptions(InferenceDevice device, int device_id) {
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    opts.SetIntraOpNumThreads(1);
    (void)device_id;
    if (device == InferenceDevice::CUDA) {
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = device_id;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
    }
#ifdef _WIN32
    if (device == InferenceDevice::DirectML) {
        // Load DML provider DLL dynamically — the header dml_provider_factory.h
        // is only in the NuGet package, not the ORT GPU zip release
        HMODULE hDml = LoadLibraryW(L"onnxruntime_providers_dml.dll");
        if (hDml) {
            using DmlFn = OrtStatus*(__stdcall*)(OrtSessionOptions*, int);
            auto dmlFn = (DmlFn)GetProcAddress(hDml, "OrtSessionOptionsAppendExecutionProvider_DML");
            if (dmlFn) {
                OrtStatus* status = dmlFn(opts, device_id);
                if (status) {
                    obs_log(LOG_WARNING, "[YOLODetector] DML provider unavailable, fallback to CPU");
                }
            }
            // Keep DLL loaded — ORT needs it during inference
        } else {
            obs_log(LOG_WARNING, "[YOLODetector] Failed to load onnxruntime_providers_dml.dll");
        }
    }
#endif
    return opts;
}

YOLODetector::YOLODetector(const std::string& model_path,
                           cv::Size input_size,
                           InferenceDevice device,
                           int device_id,
                           float conf_threshold)
    : input_size_(input_size)
    , conf_threshold_(conf_threshold)
    , device_(device)
    , device_id_(device_id)
    , memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    auto opts = makeSessionOptions(device, device_id);
#ifdef _WIN32
    std::wstring wpath(model_path.begin(), model_path.end());
    session_ = Ort::Session(env_, wpath.c_str(), opts);
#else
    session_ = Ort::Session(env_, model_path.c_str(), opts);
#endif

    // Cache input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_in = session_.GetInputCount();
    size_t num_out = session_.GetOutputCount();

    input_names_ptr_.reserve(num_in);
    input_names_.reserve(num_in);
    for (size_t i = 0; i < num_in; ++i) {
        auto ptr = session_.GetInputNameAllocated(i, allocator);
        input_names_.push_back(ptr.get());
        input_names_ptr_.push_back(std::move(ptr));
    }

    output_names_ptr_.reserve(num_out);
    output_names_.reserve(num_out);
    for (size_t i = 0; i < num_out; ++i) {
        auto ptr = session_.GetOutputNameAllocated(i, allocator);
        output_names_.push_back(ptr.get());
        output_names_ptr_.push_back(std::move(ptr));
    }

    // Determine output shape
    auto type_info = session_.GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    if (shape.size() >= 3) {
        int dets = static_cast<int>(shape[shape.size() - 2]);
        if (dets > 0 && dets < 20000)
            num_detections_ = dets;
        obs_log(LOG_INFO, "[YOLODetector] Model output shape: %d detections, %d features",
                num_detections_, static_cast<int>(shape[shape.size() - 1]));
    }

    obs_log(LOG_INFO, "[YOLODetector] Model loaded: %s (%d x %d, %d detections, backend: %s)",
            model_path.c_str(), input_size_.width, input_size_.height, num_detections_,
            inferenceDeviceToString(device_));
}

YOLODetector::~YOLODetector() {
    session_ = Ort::Session{nullptr};
}

std::string YOLODetector::provider() const {
    return inferenceDeviceToString(device_);
}

void YOLODetector::setOutputShape(int num_detections, int /*features_per_det*/) {
    num_detections_ = num_detections;
}

cv::Mat YOLODetector::static_resize(const cv::Mat& img) {
    float r = std::fminf((float)input_size_.width / (float)img.cols,
                         (float)input_size_.height / (float)img.rows);
    float unpad_w_f = r * (float)img.cols;
    float unpad_h_f = r * (float)img.rows;
    int unpad_w = (int)unpad_w_f;
    int unpad_h = (int)unpad_h_f;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(input_size_.height, input_size_.width, CV_8UC3,
                cv::Scalar(114, 114, 114));
    int dx = (int)(((float)out.cols - unpad_w_f) * 0.5f);
    int dy = (int)(((float)out.rows - unpad_h_f) * 0.5f);
    re.copyTo(out(cv::Rect(dx, dy, re.cols, re.rows)));
    return out;
}

void YOLODetector::decode_outputs(const float* prob, std::vector<OBBObject>& objects,
                                  float conf_th, float scale,
                                  int img_w, int img_h,
                                  float pad_x, float pad_y) {
    objects.clear();
    const int max_dets = num_detections_;
    const int stride = 7;

    for (int i = 0; i < max_dets; ++i) {
        float x = prob[i * stride + 0];
        float y = prob[i * stride + 1];
        float w = prob[i * stride + 2];
        float h = prob[i * stride + 3];
        float score = prob[i * stride + 4];
        float angle = prob[i * stride + 6];

        if (score < 0.0f) score = 0.0f;
        if (score > 1.0f) score = 1.0f;

        if (score > conf_th) {
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

std::vector<OBBObject> YOLODetector::inferOBB(const cv::Mat& frame) {
    cv::Mat pr_img = static_resize(frame);

    cv::Mat blob = cv::dnn::blobFromImage(pr_img, 1.0 / 255.0, input_size_,
                                            cv::Scalar(), true, false);

    // Create ORT input tensor from blob data
    std::vector<int64_t> input_shape = {1, 3, input_size_.height, input_size_.width};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, blob.ptr<float>(), static_cast<size_t>(blob.total()),
        input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session_.Run(Ort::RunOptions{},
                                       input_names_.data(), &input_tensor, input_names_.size(),
                                       output_names_.data(), output_names_.size());

    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        obs_log(LOG_ERROR, "[YOLODetector] Inference returned empty output");
        return {};
    }

    // Get output shape
    auto out_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto out_shape = out_info.GetShape();
    if (out_shape.size() >= 3) {
        int dets = static_cast<int>(out_shape[out_shape.size() - 2]);
        if (dets > 0 && dets < 20000)
            num_detections_ = dets;
    }

    float* net_pred = output_tensors[0].GetTensorMutableData<float>();

    float scale = std::fminf((float)input_size_.width / (float)frame.cols,
                             (float)input_size_.height / (float)frame.rows);
    float unpad_w_f = scale * (float)frame.cols;
    float unpad_h_f = scale * (float)frame.rows;
    int pad_x_int = (int)(((float)input_size_.width - unpad_w_f) * 0.5f);
    int pad_y_int = (int)(((float)input_size_.height - unpad_h_f) * 0.5f);

    std::vector<OBBObject> objects;
    decode_outputs(net_pred, objects, conf_threshold_, scale,
                   frame.cols, frame.rows, (float)pad_x_int, (float)pad_y_int);

    return objects;
}

} // namespace vtes_detection
