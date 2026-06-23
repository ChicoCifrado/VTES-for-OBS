#include <cuda_runtime.h>

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

#define NUM_BINS 36

__global__ void edgeOrientationHistogramKernel(
    const uint8_t* gray, int stride,
    int W, int H,
    int roi_x, int roi_y, int roi_w, int roi_h,
    float edge_threshold,
    int* global_hist)
{
    extern __shared__ int shared_hist[];

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int gx = roi_x + x;
    int gy = roi_y + y;

    if (x < roi_w && y < roi_h &&
        gx >= 1 && gx < W - 1 && gy >= 1 && gy < H - 1) {

        float gx_val =
            -1 * (float)gray[(gy - 1) * stride + (gx - 1)] +
            +1 * (float)gray[(gy - 1) * stride + (gx + 1)] +
            -2 * (float)gray[(gy)     * stride + (gx - 1)] +
            +2 * (float)gray[(gy)     * stride + (gx + 1)] +
            -1 * (float)gray[(gy + 1) * stride + (gx - 1)] +
            +1 * (float)gray[(gy + 1) * stride + (gx + 1)];

        float gy_val =
            -1 * (float)gray[(gy - 1) * stride + (gx - 1)] +
            -2 * (float)gray[(gy - 1) * stride + (gx)]     +
            -1 * (float)gray[(gy - 1) * stride + (gx + 1)] +
            +1 * (float)gray[(gy + 1) * stride + (gx - 1)] +
            +2 * (float)gray[(gy + 1) * stride + (gx)]     +
            +1 * (float)gray[(gy + 1) * stride + (gx + 1)];

        float mag = sqrtf(gx_val * gx_val + gy_val * gy_val);

        if (mag > edge_threshold) {
            float angle = atan2f(gy_val, gx_val) * 180.0f / M_PI_F;
            if (angle < 0.0f) angle += 180.0f;
            int bin = min((int)(angle * NUM_BINS / 180.0f), NUM_BINS - 1);
            atomicAdd(&shared_hist[bin], 1);
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        if (shared_hist[i] > 0) {
            atomicAdd(&global_hist[i], shared_hist[i]);
        }
    }
}
