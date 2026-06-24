#include "vampire_classifier.hpp"
#include "vampire_classifier_kernels.h"
#include "ocr/cuda_debug.hpp"
#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>
#include <cmath>

float computeVampireScoreCUDA(const cv::Mat& card_bgr,
                              float edge_threshold,
                              int device_id)
{
    if (card_bgr.empty()) return -1.0f;

    int H = card_bgr.rows;
    int W = card_bgr.cols;

    int roi_x = (int)(W * 0.12f);
    int roi_y = (int)(H * 0.08f);
    int roi_w = (int)(W * 0.76f);
    int roi_h = (int)(H * 0.45f);

    if (roi_w < 16 || roi_h < 16) return -1.0f;

    cv::Mat gray;
    if (card_bgr.channels() == 3)
        cv::cvtColor(card_bgr, gray, cv::COLOR_BGR2GRAY);
    else
        gray = card_bgr;

    cudaSetDevice(device_id);
    uint8_t* d_gray = nullptr;
    CUDA_SAFE(cudaMalloc(&d_gray, H * W));
    CUDA_SAFE(cudaMemcpy(d_gray, gray.data, H * W, cudaMemcpyHostToDevice));

    int* d_hist = nullptr;
    CUDA_SAFE(cudaMalloc(&d_hist, NUM_BINS * sizeof(int)));
    CUDA_SAFE(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));

    CUDA_SAFE(nv_edge_orientation_histogram(
        d_gray, W, W, H,
        roi_x, roi_y, roi_w, roi_h,
        edge_threshold,
        d_hist));

    int hist[NUM_BINS];
    CUDA_SAFE(cudaMemcpy(hist, d_hist, NUM_BINS * sizeof(int),
                         cudaMemcpyDeviceToHost));

    cudaFree(d_gray);
    cudaFree(d_hist);

    float total = 0.0f;
    for (int i = 0; i < NUM_BINS; i++)
        total += (float)hist[i];

    if (total < 10.0f) return 0.5f;

    float horizontal_energy = 0.0f;
    for (int i = 0; i <= 2; i++)  horizontal_energy += (float)hist[i];
    float vertical_energy = 0.0f;
    for (int i = 16; i <= 20; i++) vertical_energy += (float)hist[i];

    float rect_energy = (horizontal_energy + vertical_energy) / total;

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
        return false;
    }
    return score > threshold;
}
