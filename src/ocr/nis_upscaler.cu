// Kernel declarations (defined in nis_upscaler_kernels.cu, linked at device-link time)
__global__ void lanczosUpscaleKernel(
    const float* src, int srcW, int srcH,
    float* dst, int dstW, int dstH,
    int channels);

__global__ void adaptiveSharpenKernel(
    const float* src,
    float* dst,
    int W, int H, int channels);

#include "nis_upscaler.hpp"
#include "cuda_debug.hpp"
#include <opencv2/imgproc.hpp>

NISUpscaler::NISUpscaler() = default;

NISUpscaler::~NISUpscaler()
{
    free_buffers();
}

bool NISUpscaler::init(int device_id)
{
    device_id_ = device_id;

    static bool log_env_set = false;
    if (!log_env_set) {
#ifdef _WIN32
        _putenv_s("CUDA_LOG_FILE", "stderr");
#else
        setenv("CUDA_LOG_FILE", "stderr", 0);
#endif
        log_env_set = true;
    }

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return false;

    int dev = device_id;
    if (dev >= count) dev = 0;

    cudaSetDevice(dev);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return false;

    available_ = true;
    return true;
}

bool NISUpscaler::ensure_buffers(int h, int w)
{
    int out_h = h * scale();
    int out_w = w * scale();
    int max_in = h * w * 3 * sizeof(float);
    int max_out = out_h * out_w * 3 * sizeof(float);

    if (max_in > alloc_h_ * alloc_w_ * 3) {
        free_buffers();
        CUDA_SAFE(cudaMalloc(&d_input_, max_in));
        CUDA_SAFE(cudaMalloc(&d_temp_, max_in));
        CUDA_SAFE(cudaMalloc(&d_output_, max_out));
        alloc_h_ = h;
        alloc_w_ = w;
    }
    return true;
}

void NISUpscaler::free_buffers()
{
    if (d_input_)  { cudaFree(d_input_);  d_input_ = nullptr; }
    if (d_temp_)   { cudaFree(d_temp_);   d_temp_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
    alloc_h_ = alloc_w_ = 0;
}

bool NISUpscaler::upscale(const cv::Mat& bgr_input, cv::Mat& bgr_output)
{
    if (!available_ || bgr_input.empty()) return false;

    int h = bgr_input.rows;
    int w = bgr_input.cols;
    int out_h = h * scale();
    int out_w = w * scale();

    if (!ensure_buffers(h, w)) return false;

    cv::Mat rgb_float;
    cv::cvtColor(bgr_input, rgb_float, cv::COLOR_BGR2RGB);
    rgb_float.convertTo(rgb_float, CV_32FC3, 1.0f / 255.0f);

    size_t buf_size = h * w * 3 * sizeof(float);
    CUDA_SAFE(cudaMemcpy(d_input_, rgb_float.ptr<float>(), buf_size, cudaMemcpyHostToDevice));

    {
        dim3 block(16, 16);
        dim3 grid((out_w + 15) / 16, (out_h + 15) / 16);
        CUDA_LAUNCH(lanczosUpscaleKernel, grid, block,
            (const float*)d_input_, w, h,
            (float*)d_output_, out_w, out_h, 3);
    }

    {
        dim3 block(16, 16);
        dim3 grid((out_w + 15) / 16, (out_h + 15) / 16);
        CUDA_SAFE(cudaMemcpy(d_temp_, d_output_, out_h * out_w * 3 * sizeof(float),
                            cudaMemcpyDeviceToDevice));
        CUDA_LAUNCH(adaptiveSharpenKernel, grid, block,
            (const float*)d_temp_,
            (float*)d_output_,
            out_w, out_h, 3);
    }

    cv::Mat out_rgb(out_h, out_w, CV_32FC3);
    CUDA_SAFE(cudaMemcpy(out_rgb.ptr<float>(), d_output_,
                        out_h * out_w * 3 * sizeof(float),
                        cudaMemcpyDeviceToHost));

    out_rgb.convertTo(out_rgb, CV_8UC3, 255.0f);
    cv::cvtColor(out_rgb, bgr_output, cv::COLOR_RGB2BGR);
    return true;
}
