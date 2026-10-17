#include "nis_upscaler.hpp"
#include <opencv2/imgproc.hpp>

#ifdef HAVE_CUDA_NIS
#include "cuda/vtes_cuda_kernels_ptx.h"
#include "cuda_debug.hpp"
#endif

NISUpscaler::NISUpscaler() = default;

NISUpscaler::~NISUpscaler()
{
#ifdef HAVE_CUDA_NIS
    free_buffers();
#endif
}

bool NISUpscaler::init(int device_id)
{
#ifdef HAVE_CUDA_NIS
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

    if (!launcher_.init(VTES_CUDA_PTX)) {
        fprintf(stderr, "[NIS] Failed to load CUDA kernel PTX\n");
        return false;
    }

    available_ = true;
    return true;
#else
    (void)device_id;
    return false;
#endif
}

#ifdef HAVE_CUDA_NIS
bool NISUpscaler::ensure_buffers(int h, int w)
{
    const size_t in_bytes = (size_t)h * w * 3 * sizeof(float);
    const size_t out_bytes = (size_t)h * scale() * w * scale() * 3 * sizeof(float);

    // Reusar solo si los tres buffers existen y el area de entrada no crecio.
    // (La condicion anterior mezclaba bytes con pixeles y re-asignaba en cada
    // frame; ademas d_temp_ se asignaba con tamano de ENTRADA pero recibe la
    // copia del resultado 4x — 16x mayor — desbordando la memoria CUDA y
    // corrompiendo las asignaciones vecinas, la propia imagen incluida.)
    if (d_input_ && d_temp_ && d_output_ && (size_t)h * w <= (size_t)alloc_h_ * alloc_w_)
        return true;

    free_buffers();
    CUDA_SAFE(cudaMalloc(&d_input_, in_bytes));
    CUDA_SAFE(cudaMalloc(&d_temp_, out_bytes));
    CUDA_SAFE(cudaMalloc(&d_output_, out_bytes));
    alloc_h_ = h;
    alloc_w_ = w;
    return true;
}

void NISUpscaler::free_buffers()
{
    if (d_input_)  { cudaFree(d_input_);  d_input_ = nullptr; }
    if (d_temp_)   { cudaFree(d_temp_);   d_temp_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
    alloc_h_ = alloc_w_ = 0;
}
#endif

bool NISUpscaler::upscale(const cv::Mat& bgr_input, cv::Mat& bgr_output)
{
#ifdef HAVE_CUDA_NIS
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

    if (!launcher_.lanczos_upscale((const float*)d_input_, w, h,
                                    (float*)d_output_, out_w, out_h, 3))
        return false;

    CUDA_SAFE(cudaMemcpy(d_temp_, d_output_, out_h * out_w * 3 * sizeof(float),
                        cudaMemcpyDeviceToDevice));

    if (!launcher_.adaptive_sharpen((const float*)d_temp_,
                                     (float*)d_output_,
                                     out_w, out_h, 3))
        return false;

    cv::Mat out_rgb(out_h, out_w, CV_32FC3);
    CUDA_SAFE(cudaMemcpy(out_rgb.ptr<float>(), d_output_,
                        out_h * out_w * 3 * sizeof(float),
                        cudaMemcpyDeviceToHost));

    out_rgb.convertTo(out_rgb, CV_8UC3, 255.0f);
    cv::cvtColor(out_rgb, bgr_output, cv::COLOR_RGB2BGR);
    return true;
#else
    (void)bgr_input;
    (void)bgr_output;
    return false;
#endif
}
