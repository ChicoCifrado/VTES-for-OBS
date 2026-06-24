#include <cuda_runtime.h>

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// ─── Lanczos 4× upscale kernel ─────────────────────────────────────────

__device__ float lanczos3(float x)
{
    x = fabsf(x);
    if (x >= 3.0f) return 0.0f;
    if (x < 1e-6f) return 1.0f;
    float pix = M_PI_F * x;
    return 3.0f * sinf(pix) * sinf(pix / 3.0f) / (pix * pix);
}

__global__ void lanczosUpscaleKernel(
    const float* src, int srcW, int srcH,
    float* dst, int dstW, int dstH,
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

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        float norm = 0.0f;

        for (int dy = -2; dy <= 3; dy++) {
            int sy_ = iy + dy;
            sy_ = max(0, min(sy_, srcH - 1));

            float wy = lanczos3(fy - dy);

            for (int dx = -2; dx <= 3; dx++) {
                int sx_ = ix + dx;
                sx_ = max(0, min(sx_, srcW - 1));

                float wx = lanczos3(fx - dx);
                float w = wx * wy;
                sum += w * src[(sy_ * srcW + sx_) * channels + c];
                norm += w;
            }
        }

        dst[(y * dstW + x) * channels + c] = (norm > 1e-10f) ? (sum / norm) : 0.0f;
    }
}

// ─── Adaptive contrast-aware sharpen kernel ────────────────────────────

__global__ void adaptiveSharpenKernel(
    const float* src,
    float* dst,
    int W, int H, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        int count = 0;

        for (int dy = -2; dy <= 2; dy++) {
            int ny = y + dy;
            if (ny < 0 || ny >= H) continue;
            for (int dx = -2; dx <= 2; dx++) {
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

        float gain = fminf(abs_contrast * 2.5f, 1.0f) * 0.45f;

        float out = center + gain * contrast;
        dst[(y * W + x) * channels + c] = fminf(fmaxf(out, 0.0f), 1.0f);
    }
}

// ─── C-linkage launch wrappers (callable from regular C++ host code) ───

extern "C" cudaError_t nv_lanczos_upscale(const float* d_in, int w, int h,
                                          float* d_out, int out_w, int out_h,
                                          int channels)
{
    dim3 block(16, 16);
    dim3 grid((out_w + 15) / 16, (out_h + 15) / 16);
    lanczosUpscaleKernel<<<grid, block>>>(d_in, w, h, d_out, out_w, out_h, channels);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

extern "C" cudaError_t nv_adaptive_sharpen(const float* d_src, float* d_dst,
                                           int W, int H, int channels)
{
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    adaptiveSharpenKernel<<<grid, block>>>(d_src, d_dst, W, H, channels);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}
