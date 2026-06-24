#ifndef NIS_UPSCALER_KERNELS_H
#define NIS_UPSCALER_KERNELS_H

#include <cuda_runtime.h>

extern "C" {
cudaError_t nv_lanczos_upscale(const float* d_in, int w, int h,
                               float* d_out, int out_w, int out_h, int channels);
cudaError_t nv_adaptive_sharpen(const float* d_src, float* d_dst,
                                int W, int H, int channels);
}

#endif
