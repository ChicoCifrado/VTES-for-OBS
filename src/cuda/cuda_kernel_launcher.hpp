#ifndef VTES_CUDA_KERNEL_LAUNCHER_H
#define VTES_CUDA_KERNEL_LAUNCHER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_HISTOGRAM_BINS 36

// ─── PTX module loader for CUDA kernels ───────────────────────────────
// Loads PTX at runtime via CUDA Driver API, bypassing nvcc/cudafe++.
// All errors logged to stderr for OBS capture.

class CudaKernelLauncher {
public:
    CudaKernelLauncher();
    ~CudaKernelLauncher();

    // Load PTX from string; call once after cudaSetDevice().
    // Returns true on success.
    bool init(const char* ptx_source);

    bool is_available() const { return module_ != nullptr; }

    // ─── Kernel launches (return false on error) ───────────────────────

    // Lanczos 4× upscale: d_in (HWC float) → d_out (outH×outW×C float)
    bool lanczos_upscale(const float* d_in, int w, int h,
                         float* d_out, int out_w, int out_h, int channels);

    // Adaptive sharpen: d_src (HWC float) → d_dst (same dims)
    bool adaptive_sharpen(const float* d_src, float* d_dst,
                          int W, int H, int channels);

    // Edge orientation histogram for oval/rect classifier
    bool edge_orientation_histogram(const unsigned char* gray, int stride,
                                    int imgW, int imgH,
                                    int roi_x, int roi_y,
                                    int roi_w, int roi_h,
                                    float edge_threshold,
                                    int* global_hist);

private:
    CUmodule module_ = nullptr;
    CUfunction func_upscale_ = nullptr;
    CUfunction func_sharpen_ = nullptr;
    CUfunction func_histogram_ = nullptr;

    bool get_function(CUfunction* func, const char* name);
    bool launch(CUfunction func, dim3 grid, dim3 block,
                void** args, unsigned int shared_mem = 0);
};

#endif
