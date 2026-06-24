#include <cuda_runtime.h>

// ─── Common CUDA math constants ────────────────────────────────────────
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// ─── Lanczos 3-lobe weight function ────────────────────────────────────
__device__ float lanczos3(float x)
{
    x = fabsf(x);
    if (x >= 3.0f) return 0.0f;
    if (x < 1e-6f) return 1.0f;
    float pix = M_PI_F * x;
    return 3.0f * sinf(pix) * sinf(pix / 3.0f) / (pix * pix);
}

// ─── Lanczos 4× upscale ───────────────────────────────────────────────
extern "C" __global__ void lanczosUpscaleKernel(
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

// ─── Adaptive contrast-aware sharpen ──────────────────────────────────
extern "C" __global__ void adaptiveSharpenKernel(
    const float* src, float* dst,
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

// ─── Edge orientation histogram for oval/rect portrait classifier ─────
#define NUM_BINS 36

extern "C" __global__ void edgeOrientationHistogramKernel(
    const unsigned char* gray, int stride,
    int W, int H,
    int roi_x, int roi_y, int roi_w, int roi_h,
    float edge_threshold,
    int* global_hist)
{
    extern __shared__ int shared_hist[];

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x)
        shared_hist[i] = 0;
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int gx = roi_x + x;
    int gy = roi_y + y;

    if (x < roi_w && y < roi_h &&
        gx >= 1 && gx < W - 1 && gy >= 1 && gy < H - 1) {

        float gx_val =
            -1 * (float)gray[(gy-1)*stride + (gx-1)] + 1 * (float)gray[(gy-1)*stride + (gx+1)] +
            -2 * (float)gray[(gy)  *stride + (gx-1)] + 2 * (float)gray[(gy)  *stride + (gx+1)] +
            -1 * (float)gray[(gy+1)*stride + (gx-1)] + 1 * (float)gray[(gy+1)*stride + (gx+1)];

        float gy_val =
            -1 * (float)gray[(gy-1)*stride + (gx-1)] + -2 * (float)gray[(gy-1)*stride + (gx)] + -1 * (float)gray[(gy-1)*stride + (gx+1)] +
            +1 * (float)gray[(gy+1)*stride + (gx-1)] +  2 * (float)gray[(gy+1)*stride + (gx)] +  1 * (float)gray[(gy+1)*stride + (gx+1)];

        float mag = sqrtf(gx_val*gx_val + gy_val*gy_val);
        if (mag > edge_threshold) {
            float angle = atan2f(gy_val, gx_val) * 180.0f / M_PI_F;
            if (angle < 0.0f) angle += 180.0f;
            int bin = min((int)(angle * NUM_BINS / 180.0f), NUM_BINS - 1);
            atomicAdd(&shared_hist[bin], 1);
        }
    }

    __syncthreads();
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x)
        if (shared_hist[i] > 0)
            atomicAdd(&global_hist[i], shared_hist[i]);
}
