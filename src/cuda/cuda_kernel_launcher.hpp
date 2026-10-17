#ifndef VTES_CUDA_KERNEL_LAUNCHER_H
#define VTES_CUDA_KERNEL_LAUNCHER_H

// ─── PTX module loader for CUDA kernels ───────────────────────────────
// Loads PTX at runtime via CUDA Driver API, bypassing nvcc/cudafe++.
// All errors logged to stderr for OBS capture.
//
// CMake only compiles src/cuda/cuda_kernel_launcher.cpp and links the CUDA
// runtime when the toolkit is present, and in that case defines
// HAVE_CUDA_NIS. This header is included unconditionally from
// nis_upscaler.hpp and vampire_classifier.cpp, so without this guard a
// machine with no CUDA toolkit fails to build on the <cuda.h> include.
// When the toolkit is missing a no-op stub with the same interface keeps
// the callers compiling and the CUDA path simply disabled at runtime.

#ifdef HAVE_CUDA_NIS

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_HISTOGRAM_BINS 36

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

#else  // !HAVE_CUDA_NIS — no CUDA toolkit available

#include <cstdio>

class CudaKernelLauncher {
public:
    CudaKernelLauncher() = default;
    ~CudaKernelLauncher() = default;

    // Always unavailable: the PTX module is never loaded.
    bool init(const char*) { return false; }
    bool is_available() const { return false; }

    bool lanczos_upscale(const float*, int, int, float*, int, int, int)
    { return false; }
    bool adaptive_sharpen(const float*, float*, int, int, int)
    { return false; }
    bool edge_orientation_histogram(const unsigned char*, int, int, int,
                                    int, int, int, int, float, int*)
    { return false; }
};

#endif  // HAVE_CUDA_NIS

#endif  // VTES_CUDA_KERNEL_LAUNCHER_H
