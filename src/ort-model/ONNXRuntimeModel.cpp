#include "ONNXRuntimeModel.h"

#include "plugin-support.h"
#include <obs.h>

#ifdef _WIN32
#include <windows.h>
// DML provider types are NOT in the basic onnxruntime-win-x64 package.
// They live in the separate onnxruntime_providers_shared.h, which is only
// included in the NuGet / GPU-specific packages. We define them manually
// and retrieve the DML API via GetExecutionProviderApi at runtime.
#include <onnxruntime_cxx_api.h>

typedef struct OrtDMLProviderOptions {
    uint32_t device_id;
} OrtDMLProviderOptions;

typedef struct OrtDmlApi {
    OrtStatus*(ORT_API_CALL* SessionOptionsAppendExecutionProvider_DML)(
        _In_ OrtSessionOptions* ortSessionOptions,
        _In_ const OrtDMLProviderOptions* dmlProviderOptions);
} OrtDmlApi;

// Pre-load ORT provider DLLs from the plugin directory so onnxruntime can
// discover the DML, CUDA etc. providers at session creation time.
// onnxruntime.dll calls LoadLibrary("onnxruntime_providers_shared.dll")
// internally, but the plugin directory is not in the default DLL search path.
static bool preload_ort_providers() {
    static bool tried = false;
    static bool ok = false;
    if (tried) return ok;
    tried = true;

    HMODULE self = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)&preload_ort_providers, &self)) {
        obs_log(LOG_WARNING, "[ORT] GetModuleHandleEx failed (%lu)", GetLastError());
        return false;
    }

    wchar_t path[MAX_PATH];
    DWORD len = GetModuleFileNameW(self, path, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) {
        obs_log(LOG_WARNING, "[ORT] GetModuleFileName failed (%lu)", GetLastError());
        return false;
    }

    std::wstring dir(path);
    auto pos = dir.find_last_of(L"\\/");
    dir = (pos == std::wstring::npos) ? L"." : dir.substr(0, pos);

    static const wchar_t* dlls[] = {
        L"onnxruntime_providers_shared.dll",
        L"DirectML.dll",
    };

    bool all_ok = true;
    for (auto name : dlls) {
        std::wstring full = dir + L"\\" + name;
        HMODULE h = LoadLibraryW(full.c_str());
        if (h) {
            obs_log(LOG_INFO, "[ORT] Pre-loaded: %ls", full.c_str());
        } else {
            obs_log(LOG_WARNING, "[ORT] Failed to load %ls (error %lu)",
                    full.c_str(), GetLastError());
            all_ok = false;
        }
    }
    ok = all_ok;
    return ok;
}
#endif

ONNXRuntimeModel::ONNXRuntimeModel(file_name_t path_to_model, int intra_op_num_threads,
				   int num_classes, int inter_op_num_threads,
				   const std::string &use_gpu_, int device_id, bool use_parallel,
				   float nms_th, float conf_th)
	: intra_op_num_threads_(intra_op_num_threads),
	  inter_op_num_threads_(inter_op_num_threads),
	  use_gpu(use_gpu_),
	  device_id_(device_id),
	  use_parallel_(use_parallel),
	  nms_thresh_(nms_th),
	  bbox_conf_thresh_(conf_th),
	  num_classes_(num_classes)
{
	try {
		Ort::SessionOptions session_options;

		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		if (this->use_parallel_) {
			session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
			session_options.SetInterOpNumThreads(this->inter_op_num_threads_);
		} else {
			session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
		}
		session_options.SetIntraOpNumThreads(this->intra_op_num_threads_);

#ifdef _WIN32
		if (this->use_gpu == "cuda") {
			OrtCUDAProviderOptions cuda_option{};
			cuda_option.device_id = static_cast<int>(this->device_id_);
			try {
				session_options.AppendExecutionProvider_CUDA(cuda_option);
				obs_log(LOG_INFO, "[ORT] CUDA provider appended successfully");
			} catch (const Ort::Exception &e) {
				obs_log(LOG_WARNING,
				     "[ORT] CUDA provider not available (%s), falling back to CPU",
				     e.what());
				this->use_gpu = "cpu";
			}
		} else if (this->use_gpu == "dml") {
			// Pre-load ORT provider DLLs so onnxruntime can discover DML
			preload_ort_providers();

			// Retrieve DML API via GetExecutionProviderApi (runtime)
			try {
				const void* dml_api_ptr = nullptr;
				// DML has its own API version (1), NOT ORT_API_VERSION (~18-20).
				// Passing ORT_API_VERSION here would cause DML provider to reject
				// the request as unsupported version, silently falling back to CPU.
				constexpr uint32_t DML_API_VERSION = 1;
				Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi(
					"DML", DML_API_VERSION, &dml_api_ptr));
				const auto* dml_api =
					static_cast<const OrtDmlApi*>(dml_api_ptr);
				OrtDMLProviderOptions dml_options{};
				dml_options.device_id =
					static_cast<uint32_t>(this->device_id_);
				Ort::ThrowOnError(
					dml_api->SessionOptionsAppendExecutionProvider_DML(
						session_options, &dml_options));
				obs_log(LOG_INFO, "[ORT] DML provider appended successfully (v%u)",
					DML_API_VERSION);
			} catch (const Ort::Exception &e) {
				obs_log(LOG_WARNING,
				     "[ORT] DML provider not available (%s), falling back to CPU",
				     e.what());
				this->use_gpu = "cpu";
			}
		}

		// Log available providers for diagnostics
		try {
			auto providers = Ort::GetAvailableProviders();
			std::string provs;
			for (auto &p : providers) {
				if (!provs.empty()) provs += ", ";
				provs += p;
			}
			obs_log(LOG_INFO, "[ORT] Requested provider: %s, Available: %s",
			     this->use_gpu.c_str(), provs.c_str());
		} catch (const std::exception &e2) {
			obs_log(LOG_WARNING, "[ORT] Failed to get available providers: %s", e2.what());
		}
#endif

		this->session_ = Ort::Session(this->env_, path_to_model.c_str(), session_options);
		obs_log(LOG_INFO, "[ORT] Model loaded with provider: %s", this->use_gpu.c_str());
	} catch (std::exception &e) {
		obs_log(LOG_ERROR, "Cannot load model: %s", e.what());
		throw e;
	}

	Ort::AllocatorWithDefaultOptions ort_alloc;

	// number of inputs
	size_t num_input = this->session_.GetInputCount();

	for (size_t i = 0; i < num_input; i++) {
		auto input_info = this->session_.GetInputTypeInfo(i);
		auto input_shape_info = input_info.GetTensorTypeAndShapeInfo();
		auto input_shape = input_shape_info.GetShape();
		this->input_tensor_type_ = input_shape_info.GetElementType();

		// Handle dynamic dimensions (-1) - replace with concrete values
		std::vector<int64_t> fixed_input_shape = input_shape;
		for (size_t d = 0; d < fixed_input_shape.size(); d++) {
			if (fixed_input_shape[d] <= 0) {
				if (d == 0) fixed_input_shape[d] = 1;       // batch size
				else if (d == 1) fixed_input_shape[d] = 3;   // channels
				else fixed_input_shape[d] = 640;             // H/W dimensions
			}
		}

		// assume input shape is NCHW
		this->input_h_.push_back((int)(fixed_input_shape[2]));
		this->input_w_.push_back((int)(fixed_input_shape[3]));

		// Allocate input memory buffer
		this->input_name_.push_back(
			std::string(this->session_.GetInputNameAllocated(i, ort_alloc).get()));
		size_t input_element_count = 1;
		for (auto dim : fixed_input_shape) {
			input_element_count *= (size_t)dim;
		}
		size_t input_byte_count = sizeof(float) * input_element_count;
		std::unique_ptr<uint8_t[]> input_buffer =
			std::make_unique<uint8_t[]>(input_byte_count);
		auto input_memory_info =
			Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

		this->input_tensor_.push_back(Ort::Value::CreateTensor(
			input_memory_info, input_buffer.get(), input_byte_count, fixed_input_shape.data(),
			fixed_input_shape.size(), this->input_tensor_type_));
		this->input_buffer_.push_back(std::move(input_buffer));

		obs_log(LOG_INFO, "Input name: %s", this->input_name_[i].c_str());
		obs_log(LOG_INFO, "Input shape: %d %d %d %d", (int)fixed_input_shape[0],
			fixed_input_shape.size() > 1 ? (int)fixed_input_shape[1] : 0,
			fixed_input_shape.size() > 2 ? (int)fixed_input_shape[2] : 0,
			fixed_input_shape.size() > 3 ? (int)fixed_input_shape[3] : 0);
	}

	// number of outputs - store metadata but DON'T pre-allocate tensors
	// (dynamic shapes + GPU providers require runtime allocation)
	size_t num_output = this->session_.GetOutputCount();

	for (size_t i = 0; i < num_output; i++) {
		auto output_info = this->session_.GetOutputTypeInfo(i);
		auto output_shape_info = output_info.GetTensorTypeAndShapeInfo();
		auto output_shape = output_shape_info.GetShape();

		this->output_shapes_.push_back(output_shape);

		this->output_name_.push_back(
			std::string(this->session_.GetOutputNameAllocated(i, ort_alloc).get()));

		// Allocate a default CPU output buffer (will be resized during inference)
		this->output_buffer_.push_back(nullptr);
		this->output_buffer_size_.push_back(0);
		this->output_tensor_.push_back(Ort::Value{nullptr});

		obs_log(LOG_INFO, "Output %zu: shape [%lld, %lld, %lld], name: %s",
			i,
			output_shape.size() > 0 ? output_shape[0] : 0,
			output_shape.size() > 1 ? output_shape[1] : 0,
			output_shape.size() > 2 ? output_shape[2] : 0,
			this->output_name_[i].c_str());
	}
}

cv::Mat ONNXRuntimeModel::static_resize(const cv::Mat &img, const int input_index)
{
	float r = std::fminf((float)input_w_[input_index] / (float)img.cols,
			     (float)input_h_[input_index] / (float)img.rows);
	float unpad_w_f = r * (float)img.cols;
	float unpad_h_f = r * (float)img.rows;
	int unpad_w = (int)unpad_w_f;
	int unpad_h = (int)unpad_h_f;
	cv::Mat re(unpad_h, unpad_w, CV_8UC3);
	cv::resize(img, re, re.size());
	cv::Mat out(input_h_[input_index], input_w_[input_index], CV_8UC3,
		    cv::Scalar(114, 114, 114));
	int dx = (int)(((float)out.cols - unpad_w_f) * 0.5f);
	int dy = (int)(((float)out.rows - unpad_h_f) * 0.5f);
	re.copyTo(out(cv::Rect(dx, dy, re.cols, re.rows)));
	return out;
}

// for NCHW
void ONNXRuntimeModel::blobFromImage(const cv::Mat &img, float *blob_data)
{
	size_t channels = 3;
	size_t img_h = img.rows;
	size_t img_w = img.cols;
	const float scale = this->input_normalize_scale_;
	for (size_t c = 0; c < channels; ++c) {
		for (size_t h = 0; h < img_h; ++h) {
			for (size_t w = 0; w < img_w; ++w) {
				blob_data[(int)(c * img_w * img_h + h * img_w + w)] =
					(float)img.ptr<cv::Vec3b>((int)h)[(int)w][(int)c] * scale;
			}
		}
	}
}

// for NHWC
void ONNXRuntimeModel::blobFromImage_nhwc(const cv::Mat &img, float *blob_data)
{
	size_t channels = 3;
	size_t img_h = img.rows;
	size_t img_w = img.cols;
	const float scale = this->input_normalize_scale_;
	for (size_t i = 0; i < img_h * img_w; ++i) {
		for (size_t c = 0; c < channels; ++c) {
			blob_data[i * channels + c] = (float)img.data[i * channels + c] * scale;
		}
	}
}

float ONNXRuntimeModel::intersection_area(const Object &a, const Object &b)
{
	cv::Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

void ONNXRuntimeModel::qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right)
{
	int i = left;
	int j = right;
	float p = faceobjects[(left + right) / 2].prob;

	while (i <= j) {
		while (faceobjects[i].prob > p)
			++i;

		while (faceobjects[j].prob < p)
			--j;

		if (i <= j) {
			std::swap(faceobjects[i], faceobjects[j]);

			++i;
			--j;
		}
	}
	if (left < j)
		qsort_descent_inplace(faceobjects, left, j);
	if (i < right)
		qsort_descent_inplace(faceobjects, i, right);
}

void ONNXRuntimeModel::qsort_descent_inplace(std::vector<Object> &objects)
{
	if (objects.empty())
		return;

	qsort_descent_inplace(objects, 0, (int)(objects.size() - 1));
}

void ONNXRuntimeModel::nms_sorted_bboxes(const std::vector<Object> &objects,
					 std::vector<int> &picked, const float nms_threshold)
{
	picked.clear();

	const size_t n = objects.size();

	std::vector<float> areas(n);
	for (size_t i = 0; i < n; ++i) {
		areas[i] = objects[i].rect.area();
	}

	for (size_t i = 0; i < n; ++i) {
		const Object &a = objects[i];
		const size_t picked_size = picked.size();

		int keep = 1;
		for (size_t j = 0; j < picked_size; ++j) {
			const Object &b = objects[picked[j]];

			// intersection over union
			float inter_area = this->intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back((int)i);
	}
}

void ONNXRuntimeModel::inference(const cv::Mat &frame, const int input_index)
{
	// preprocess
	cv::Mat pr_img = this->static_resize(frame, input_index);

	float *blob_data = (float *)(this->input_buffer_[input_index].get());
	blobFromImage(pr_img, blob_data);

	// input names
	std::vector<const char *> input_names;
	for (size_t i = 0; i < this->input_name_.size(); i++) {
		input_names.push_back(this->input_name_[i].c_str());
	}

	// output names
	std::vector<const char *> output_names;
	for (size_t i = 0; i < this->output_name_.size(); i++) {
		output_names.push_back(this->output_name_[i].c_str());
	}

	// Inference with dynamic output allocation (supports GPU providers & dynamic shapes)
	Ort::RunOptions run_options;
	auto output_values = this->session_.Run(run_options, input_names.data(),
		this->input_tensor_.data(), this->input_tensor_.size(),
		output_names.data(), output_names.size());

	// Copy outputs back to CPU buffers
	for (size_t i = 0; i < output_values.size() && i < this->output_buffer_.size(); i++) {
		auto out_info = output_values[i].GetTensorTypeAndShapeInfo();
		auto actual_shape = out_info.GetShape();
		this->output_shapes_[i] = actual_shape;
		size_t element_count = 1;
		for (auto dim : actual_shape) element_count *= (size_t)dim;
		size_t byte_count = sizeof(float) * element_count;

		// Reallocate CPU buffer if needed
		if (!this->output_buffer_[i] || byte_count > this->output_buffer_size_[i]) {
			this->output_buffer_[i] = std::make_unique<uint8_t[]>(byte_count);
			this->output_buffer_size_[i] = byte_count;
		}

		// Copy data from GPU/CPU output tensor to our CPU buffer
		memcpy(this->output_buffer_[i].get(), output_values[i].GetTensorMutableData<uint8_t>(), byte_count);

		// Update output tensor to point to our CPU buffer with the actual shape
		this->output_tensor_[i] = Ort::Value::CreateTensor(
			Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
			this->output_buffer_[i].get(), byte_count,
			actual_shape.data(), actual_shape.size(),
			out_info.GetElementType());
	}
}
