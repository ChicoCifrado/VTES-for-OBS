#include "vampire_classifier.hpp"
#include "ocr/cuda_debug.hpp"
#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>
#include <cmath>

// ─── Edge orientation histogram kernel ─────────────────────────────────
// Each block computes a local histogram in shared memory, then atomically
// reduces to global memory. Gradients are binned into 36 bins (5° each).
//
// Rectangular art frames → energy concentrated in bins 0-2 (0°/180°) and
// bins 16-20 (80°-100°). Oval frames → energy spread across all bins.

#define NUM_BINS 36
#define BIN_HIST_SIZE NUM_BINS

__global__ void edgeOrientationHistogramKernel(
    const uint8_t* __restrict__ gray, int stride,
    int W, int H,
    int roi_x, int roi_y, int roi_w, int roi_h,
    float edge_threshold,
    int* __restrict__ global_hist)
{
    extern __shared__ int shared_hist[];

    // Initialize shared memory histogram (one thread per bin)
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

        // 3×3 Sobel X
        float gx_val =
            -1 * (float)gray[(gy - 1) * stride + (gx - 1)] +
            +1 * (float)gray[(gy - 1) * stride + (gx + 1)] +
            -2 * (float)gray[(gy)     * stride + (gx - 1)] +
            +2 * (float)gray[(gy)     * stride + (gx + 1)] +
            -1 * (float)gray[(gy + 1) * stride + (gx - 1)] +
            +1 * (float)gray[(gy + 1) * stride + (gx + 1)];

        // 3×3 Sobel Y
        float gy_val =
            -1 * (float)gray[(gy - 1) * stride + (gx - 1)] +
            -2 * (float)gray[(gy - 1) * stride + (gx)]     +
            -1 * (float)gray[(gy - 1) * stride + (gx + 1)] +
            +1 * (float)gray[(gy + 1) * stride + (gx - 1)] +
            +2 * (float)gray[(gy + 1) * stride + (gx)]     +
            +1 * (float)gray[(gy + 1) * stride + (gx + 1)];

        float mag = sqrtf(gx_val * gx_val + gy_val * gy_val);

        if (mag > edge_threshold) {
            float angle = atan2f(gy_val, gx_val) * 180.0f / (float)M_PI;
            if (angle < 0.0f) angle += 180.0f;
            int bin = min((int)(angle * NUM_BINS / 180.0f), NUM_BINS - 1);
            atomicAdd(&shared_hist[bin], 1);
        }
    }

    __syncthreads();

    // Reduce from shared to global memory
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        if (shared_hist[i] > 0) {
            atomicAdd(&global_hist[i], shared_hist[i]);
        }
    }
}

// ─── C++ wrapper ──────────────────────────────────────────────────────

float computeVampireScoreCUDA(const cv::Mat& card_bgr,
                              float edge_threshold,
                              int device_id)
{
    if (card_bgr.empty()) return -1.0f;

    int H = card_bgr.rows;
    int W = card_bgr.cols;

    // Portrait ROI (same as VtesOcrReader::hasVampireOval)
    int roi_x = (int)(W * 0.12f);
    int roi_y = (int)(H * 0.08f);
    int roi_w = (int)(W * 0.76f);
    int roi_h = (int)(H * 0.45f);

    if (roi_w < 16 || roi_h < 16) return -1.0f;

    // Convert to grayscale on CPU (simple, cheap)
    cv::Mat gray;
    if (card_bgr.channels() == 3)
        cv::cvtColor(card_bgr, gray, cv::COLOR_BGR2GRAY);
    else
        gray = card_bgr;

    // Copy grayscale to GPU
    cudaSetDevice(device_id);
    uint8_t* d_gray = nullptr;
    CUDA_SAFE(cudaMalloc(&d_gray, H * W));
    CUDA_SAFE(cudaMemcpy(d_gray, gray.data, H * W, cudaMemcpyHostToDevice));

    // Allocate + zero histogram on GPU
    int* d_hist = nullptr;
    CUDA_SAFE(cudaMalloc(&d_hist, NUM_BINS * sizeof(int)));
    CUDA_SAFE(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((roi_w + 15) / 16, (roi_h + 15) / 16);
    size_t shared_bytes = NUM_BINS * sizeof(int);

    edgeOrientationHistogramKernel<<<grid, block, shared_bytes>>>(
        d_gray, W, W, H,
        roi_x, roi_y, roi_w, roi_h,
        edge_threshold,
        d_hist);
    CUDA_SAFE(cudaGetLastError());
    CUDA_SAFE(cudaDeviceSynchronize());

    // Copy histogram back to CPU
    int hist[NUM_BINS];
    CUDA_SAFE(cudaMemcpy(hist, d_hist, NUM_BINS * sizeof(int),
                         cudaMemcpyDeviceToHost));

    cudaFree(d_gray);
    cudaFree(d_hist);

    // ─── Compute vampire score from histogram ─────────────────────────
    // Rectangular frames concentrate energy at 0° (horizontal borders)
    // and 90° (vertical borders). Oval frames distribute energy evenly.

    float total = 0.0f;
    for (int i = 0; i < NUM_BINS; i++)
        total += (float)hist[i];

    if (total < 10.0f) return 0.5f; // too few edges, uncertain

    // Rectangular peaks: 0° → bins 0-2, 90° → bins 16-20 (each bin = 5°)
    float horizontal_energy = 0.0f; // bins around 0°
    for (int i = 0; i <= 2; i++)  horizontal_energy += (float)hist[i];
    float vertical_energy = 0.0f;   // bins around 90°
    for (int i = 16; i <= 20; i++) vertical_energy += (float)hist[i];

    float rect_energy = (horizontal_energy + vertical_energy) / total;

    // Vampire score: low rect_energy → oval (vampire)
    // Scale: 0 (strong rect) → 1 (strong oval)
    float score = 1.0f - fminf(rect_energy * 1.5f, 1.0f);

    return score;
}

bool isVampireCUDA(const cv::Mat& card_bgr,
                   float threshold,
                   float edge_threshold,
                   int device_id)
{
    float score = computeVampireScoreCUDA(card_bgr, edge_threshold, device_id);
    if (score < 0.0f) {
        // Fallback: use CPU method if CUDA unavailable
        return false;
    }
    return score > threshold;
}
