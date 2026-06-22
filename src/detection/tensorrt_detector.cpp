#include "tensorrt_detector.hpp"
#include "plugin-support.h"

#include <obs.h>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

namespace vtes_detection {

namespace {

class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            obs_log(LOG_INFO, "[TensorRT] %s", msg);
        }
    }
};

TrtLogger gLogger;

size_t volume(const nvinfer1::Dims& dims) {
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        v *= static_cast<size_t>(dims.d[i] > 0 ? dims.d[i] : 1);
    }
    return v;
}

} // namespace

struct TensorRTDetectorImpl {
    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime{
        nullptr, [](nvinfer1::IRuntime* p) { delete p; }
    };
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine{
        nullptr, [](nvinfer1::ICudaEngine* p) { delete p; }
    };
    std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> context{
        nullptr, [](nvinfer1::IExecutionContext* p) { delete p; }
    };

    std::string inputName;
    std::string outputName;
    nvinfer1::Dims inputDims{};
    nvinfer1::Dims outputDims{};
    size_t inputSize = 0;
    size_t outputSize = 0;

    void* deviceInput = nullptr;
    void* deviceOutput = nullptr;
    float* hostInput = nullptr;
    float* hostOutput = nullptr;

    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphInstance = nullptr;

    ~TensorRTDetectorImpl() {
        if (graphInstance) cudaGraphExecDestroy(graphInstance);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
        if (hostOutput) cudaFreeHost(hostOutput);
        if (hostInput) cudaFreeHost(hostInput);
        if (deviceOutput) cudaFree(deviceOutput);
        if (deviceInput) cudaFree(deviceInput);
    }
};

TensorRTDetector::TensorRTDetector(const std::string& engine_path,
                                   cv::Size input_size,
                                   int device_id,
                                   float conf_threshold)
    : input_size_(input_size)
    , conf_threshold_(conf_threshold)
    , device_id_(device_id)
{
    impl_ = std::make_unique<TensorRTDetectorImpl>();

    cudaSetDevice(device_id_);

    // Read engine file
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        obs_log(LOG_ERROR, "[TensorRTDetector] Failed to open engine file: %s", engine_path.c_str());
        return;
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);
    file.close();

    // Create runtime
    auto* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        obs_log(LOG_ERROR, "[TensorRTDetector] Failed to create runtime");
        return;
    }
    impl_->runtime.reset(runtime);

    // Required for version-compatible (VC) engines which embed the lean runtime
    impl_->runtime->setEngineHostCodeAllowed(true);

    // Deserialize engine
    auto* engine = impl_->runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) {
        obs_log(LOG_ERROR, "[TensorRTDetector] Failed to deserialize engine");
        return;
    }
    impl_->engine.reset(engine);

    // Create execution context
    auto* context = impl_->engine->createExecutionContext();
    if (!context) {
        obs_log(LOG_ERROR, "[TensorRTDetector] Failed to create execution context");
        return;
    }
    impl_->context.reset(context);

    // Query I/O tensor names and raw dims from engine (may have -1 for dynamic dims)
    for (int i = 0; i < impl_->engine->getNbIOTensors(); i++) {
        auto name = impl_->engine->getIOTensorName(i);
        auto dims = impl_->engine->getTensorShape(name);
        bool isInput = impl_->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;

        if (isInput) {
            impl_->inputName = name;
            impl_->inputDims = dims;
        } else {
            impl_->outputName = name;
            impl_->outputDims = dims;
        }
    }

    // Resolve dynamic input dims from our expected input size, then set shape on context
    // so the context can resolve output dims before we allocate buffers.
    nvinfer1::Dims fixedInDims = impl_->inputDims;
    bool inHasDynamic = false;
    for (int i = 0; i < fixedInDims.nbDims; i++) {
        if (fixedInDims.d[i] < 1) {
            inHasDynamic = true;
            if (fixedInDims.nbDims == 4) {
                if (i == 2)      fixedInDims.d[i] = input_size_.height;
                else if (i == 3) fixedInDims.d[i] = input_size_.width;
                else             fixedInDims.d[i] = 1;
            } else {
                fixedInDims.d[i] = 1;
            }
        }
    }
    impl_->inputDims = fixedInDims;
    impl_->inputSize = volume(fixedInDims) * sizeof(float);

    if (inHasDynamic) {
        impl_->context->setInputShape(impl_->inputName.c_str(), fixedInDims);
    }

    // Now query the resolved output dims from the context and fix any remaining
    // dynamic dims (use a safe upper bound for detection-count dimensions).
    impl_->outputDims = impl_->context->getTensorShape(impl_->outputName.c_str());
    {
        std::string dimStr;
        for (int i = 0; i < impl_->outputDims.nbDims; i++) {
            if (i) dimStr += "x";
            dimStr += std::to_string(impl_->outputDims.d[i]);
        }
        obs_log(LOG_INFO, "[TensorRTDetector] Raw output dims from context: %s", dimStr.c_str());
    }
    for (int i = 0; i < impl_->outputDims.nbDims; i++) {
        if (impl_->outputDims.d[i] < 1) {
            impl_->outputDims.d[i] = 1;
        }
    }
    impl_->outputSize = volume(impl_->outputDims) * sizeof(float);

    obs_log(LOG_INFO, "[TensorRTDetector] Buffer sizes — input=%zu bytes, output=%zu bytes "
            "(input_shape=%dx%dx%d, input_dynamic=%d)",
            impl_->inputSize, impl_->outputSize,
            input_size_.width, input_size_.height, 3, (int)inHasDynamic);

    // Allocate device memory
    cudaMalloc(&impl_->deviceInput, impl_->inputSize);
    cudaMalloc(&impl_->deviceOutput, impl_->outputSize);

    // Allocate pinned (page-locked) host memory for zero-copy DMA
    cudaHostAlloc(&impl_->hostInput, impl_->inputSize, cudaHostAllocDefault);
    cudaHostAlloc(&impl_->hostOutput, impl_->outputSize, cudaHostAllocDefault);

    // Create CUDA stream
    cudaStreamCreate(&impl_->stream);

    // Set tensor addresses
    impl_->context->setTensorAddress(impl_->inputName.c_str(), impl_->deviceInput);
    impl_->context->setTensorAddress(impl_->outputName.c_str(), impl_->deviceOutput);

    obs_log(LOG_INFO, "[TensorRTDetector] Engine loaded: %s (%d x %d, input: %s, output: %s)",
            engine_path.c_str(),
            input_size.width, input_size.height,
            impl_->inputName.c_str(), impl_->outputName.c_str());
}

TensorRTDetector::~TensorRTDetector() = default;

std::string TensorRTDetector::provider() const {
    return "tensorrt";
}

cv::Mat TensorRTDetector::static_resize(const cv::Mat& img) const {
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

void TensorRTDetector::fill_pinned_input(const cv::Mat& img, float* dst) const {
    const int H = input_size_.height;
    const int W = input_size_.width;
    const float inv = 1.0f / 255.0f;
    for (int h = 0; h < H; h++) {
        const uchar* row = img.ptr<uchar>(h);
        for (int w = 0; w < W; w++) {
            // NCHW format with RB swap (BGR -> RGB)
            dst[0 * H * W + h * W + w] = row[w * 3 + 2] * inv;
            dst[1 * H * W + h * W + w] = row[w * 3 + 1] * inv;
            dst[2 * H * W + h * W + w] = row[w * 3 + 0] * inv;
        }
    }
}

void TensorRTDetector::decode_outputs(const float* prob, std::vector<OBBObject>& objects,
                                      float conf_th, float scale,
                                      int img_w, int img_h,
                                      float pad_x, float pad_y) const {
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

std::vector<OBBObject> TensorRTDetector::inferOBB(const cv::Mat& frame) {
    if (!impl_ || !impl_->context) {
        obs_log(LOG_ERROR, "[TensorRTDetector] Not initialized");
        return {};
    }

    cv::Mat pr_img = static_resize(frame);

    // Fill pinned host buffer directly (no intermediate blobFromImage)
    fill_pinned_input(pr_img, impl_->hostInput);

    // H2D copy — always outside graph because data changes every frame
    cudaMemcpyAsync(impl_->deviceInput, impl_->hostInput, impl_->inputSize,
                    cudaMemcpyHostToDevice, impl_->stream);

    if (graphCaptured_) {
        // Replay captured graph (inference + D2H)
        cudaGraphLaunch(impl_->graphInstance, impl_->stream);
    } else {
        // Capture graph on first inference
        cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeGlobal);

        if (!impl_->context->enqueueV3(impl_->stream)) {
            obs_log(LOG_ERROR, "[TensorRTDetector] Inference failed");
            cudaStreamEndCapture(impl_->stream, &impl_->graph);
            cudaStreamSynchronize(impl_->stream);
            return {};
        }

        // D2H copy inside graph — fixed src/dst addresses
        cudaMemcpyAsync(impl_->hostOutput, impl_->deviceOutput, impl_->outputSize,
                        cudaMemcpyDeviceToHost, impl_->stream);

        cudaStreamEndCapture(impl_->stream, &impl_->graph);

        cudaGraphInstantiate(&impl_->graphInstance, impl_->graph, 0);
        graphCaptured_ = true;
    }

    cudaStreamSynchronize(impl_->stream);

    // Dynamic output shape detection (updates num_detections_ for next decode)
    auto outDims = impl_->context->getTensorShape(impl_->outputName.c_str());
    if (outDims.nbDims >= 3) {
        int dets = static_cast<int>(outDims.d[outDims.nbDims - 2]);
        if (dets > 0 && dets < 20000)
            num_detections_ = dets;
    }

    // Safety: clamp num_detections_ so decode_outputs never reads past the buffer.
    // Stride is 7 (x, y, w, h, score, class_skip, angle).
    {
        constexpr int kStride = 7;
        size_t maxFloats = impl_->outputSize / sizeof(float);
        int safeDets = (int)(maxFloats / kStride);
        if (num_detections_ > safeDets) {
            obs_log(LOG_WARNING, "[TensorRTDetector] Clamping num_detections %d -> %d (buffer=%zu floats, stride=%d)",
                    num_detections_, safeDets, maxFloats, kStride);
            num_detections_ = safeDets;
        }
    }

    float scale = std::fminf((float)input_size_.width / (float)frame.cols,
                             (float)input_size_.height / (float)frame.rows);
    float unpad_w_f = scale * (float)frame.cols;
    float unpad_h_f = scale * (float)frame.rows;
    int pad_x_int = (int)(((float)input_size_.width - unpad_w_f) * 0.5f);
    int pad_y_int = (int)(((float)input_size_.height - unpad_h_f) * 0.5f);

    std::vector<OBBObject> objects;
    decode_outputs(impl_->hostOutput, objects, conf_threshold_, scale,
                   frame.cols, frame.rows, (float)pad_x_int, (float)pad_y_int);

    return objects;
}

} // namespace vtes_detection
