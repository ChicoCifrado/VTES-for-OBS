#include "nis_upscaler.hpp"
#include "cuda_debug.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

// ─── Lanczos 4× upscale kernel ─────────────────────────────────────────
// Each thread handles one output pixel across all channels.
// Uses Lanczos-3 (6-tap window) for high-quality text upscaling.

__global__ void lanczosUpscaleKernel(
    const float* __restrict__ src, int srcW, int srcH,
    float* __restrict__ dst, int dstW, int dstH,
    int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    float sx = (x + 0.5f) * srcW / dstW - 0.5f;
    float sy = (y + 0.5f) * srcH / dstH - 0.5f;

    int ix = floorf(sx);
    int iy = floorf(sy);
    float fx = sx - ix;
    float fy = sy - iy;

    const int a = 3;
    const int radius = a;

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        float norm = 0.0f;

        for (int dy = -radius + 1; dy <= radius; dy++) {
            int sy_ = iy + dy;
            if (sy_ < 0) sy_ = 0;
            if (sy_ >= srcH) sy_ = srcH - 1;

            float wy = (dy == 0) ? 1.0f : (
                (sinf(M_PI * (fy - dy)) * sinf(M_PI * (fy - dy) / a)) /
                (M_PI * (fy - dy) * M_PI * (fy - dy) / a));

            for (int dx = -radius + 1; dx <= radius; dx++) {
                int sx_ = ix + dx;
                if (sx_ < 0) sx_ = 0;
                if (sx_ >= srcW) sx_ = srcW - 1;

                float wx = (dx == 0) ? 1.0f : (
                    (sinf(M_PI * (fx - dx)) * sinf(M_PI * (fx - dx) / a)) /
                    (M_PI * (fx - dx) * M_PI * (fx - dx) / a));

                float w = wx * wy;
                sum += w * src[(sy_ * srcW + sx_) * channels + c];
                norm += w;
            }
        }

        dst[(y * dstW + x) * channels + c] = (norm > 0.0f) ? (sum / norm) : 0.0f;
    }
}

// ─── Adaptive contrast-aware sharpen kernel ────────────────────────────
// Inspired by NVIDIA Image Scaling (NIS) adaptive sharpening.
// Sharpen is stronger in medium-contrast areas, weaker in flat/noisy areas.

__global__ void adaptiveSharpenKernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int W, int H, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int radius = 2;
    int count = 0;

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        count = 0;

        for (int dy = -radius; dy <= radius; dy++) {
            int ny = y + dy;
            if (ny < 0 || ny >= H) continue;
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                if (nx < 0 || nx >= W) continue;
                sum += src[(ny * W + nx) * channels + c];
                count++;
            }
        }

        float mean = sum / count;
        float center = src[(y * W + x) * channels + c];
        float contrast = center - mean;
        float abs_contrast = fabsf(contrast);

        // Adaptive gain: stronger in medium contrast, weak in flat/very high
        float gain = fminf(abs_contrast * 2.5f, 1.0f) * 0.45f;

        float out = center + gain * contrast;
        dst[(y * W + x) * channels + c] = fminf(fmaxf(out, 0.0f), 1.0f);
    }
}

// ─── C++ wrapper implementation ────────────────────────────────────────

NISUpscaler::NISUpscaler() = default;

NISUpscaler::~NISUpscaler()
{
    free_buffers();
}

bool NISUpscaler::init(int device_id)
{
    device_id_ = device_id;

    // Enable CUDA driver error log before any context is created
    // All CUDA driver errors (invalid block dims, illegal access, etc.)
    // will appear as [CUDA] lines in OBS stderr automatically.
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

    // Convert BGR to float RGB planar on GPU
    cv::Mat rgb_float;
    cv::cvtColor(bgr_input, rgb_float, cv::COLOR_BGR2RGB);
    rgb_float.convertTo(rgb_float, CV_32FC3, 1.0f / 255.0f);

    size_t buf_size = h * w * 3 * sizeof(float);
    CUDA_SAFE(cudaMemcpy(d_input_, rgb_float.ptr<float>(), buf_size, cudaMemcpyHostToDevice));

    // Launch Lanczos upscale: 16×16 blocks
    {
        dim3 block(16, 16);
        dim3 grid((out_w + 15) / 16, (out_h + 15) / 16);
        CUDA_LAUNCH(lanczosUpscaleKernel, grid, block,
            (const float*)d_input_, w, h,
            (float*)d_output_, out_w, out_h, 3);
    }

    // Adaptive sharpen on upscaled result
    {
        dim3 block(16, 16);
        dim3 grid((out_w + 15) / 16, (out_h + 15) / 16);
        // Copy upscaled to temp, sharpen into output
        CUDA_SAFE(cudaMemcpy(d_temp_, d_output_, out_h * out_w * 3 * sizeof(float),
                            cudaMemcpyDeviceToDevice));
        CUDA_LAUNCH(adaptiveSharpenKernel, grid, block,
            (const float*)d_temp_,
            (float*)d_output_,
            out_w, out_h, 3);
    }

    // Copy back to host
    cv::Mat out_rgb(out_h, out_w, CV_32FC3);
    CUDA_SAFE(cudaMemcpy(out_rgb.ptr<float>(), d_output_,
                        out_h * out_w * 3 * sizeof(float),
                        cudaMemcpyDeviceToHost));

    out_rgb.convertTo(out_rgb, CV_8UC3, 255.0f);
    cv::cvtColor(out_rgb, bgr_output, cv::COLOR_RGB2BGR);
    return true;
}
