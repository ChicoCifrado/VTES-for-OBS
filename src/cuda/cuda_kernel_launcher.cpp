#include "cuda_kernel_launcher.hpp"

CudaKernelLauncher::CudaKernelLauncher() = default;

CudaKernelLauncher::~CudaKernelLauncher()
{
    if (module_) {
        cuModuleUnload(module_);
        module_ = nullptr;
    }
}

bool CudaKernelLauncher::init(const char* ptx_source)
{
    if (!ptx_source) return false;

    CUresult err = cuModuleLoadData(&module_, ptx_source);
    if (err != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(err, &name);
        fprintf(stderr, "[CUDA LAUNCHER] cuModuleLoadData failed: %s (code %d)\n",
                name ? name : "unknown", (int)err);
        module_ = nullptr;
        return false;
    }

    // Cache kernel function handles
    get_function(&func_upscale_, "lanczosUpscaleKernel");
    get_function(&func_sharpen_, "adaptiveSharpenKernel");
    get_function(&func_histogram_, "edgeOrientationHistogramKernel");

    bool ok = (func_upscale_ && func_sharpen_ && func_histogram_);
    if (!ok) {
        fprintf(stderr, "[CUDA LAUNCHER] Failed to resolve one or more kernel functions\n");
        cuModuleUnload(module_);
        module_ = nullptr;
    }
    return ok;
}

bool CudaKernelLauncher::get_function(CUfunction* func, const char* name)
{
    CUresult err = cuModuleGetFunction(func, module_, name);
    if (err != CUDA_SUCCESS) {
        const char* ename = nullptr;
        cuGetErrorName(err, &ename);
        fprintf(stderr, "[CUDA LAUNCHER] cuModuleGetFunction(%s) failed: %s\n",
                name, ename ? ename : "unknown");
        *func = nullptr;
        return false;
    }
    return true;
}

bool CudaKernelLauncher::launch(CUfunction func, dim3 grid, dim3 block,
                                 void** args, unsigned int shared_mem)
{
    CUresult err = cuLaunchKernel(func,
                                   grid.x, grid.y, grid.z,
                                   block.x, block.y, block.z,
                                   shared_mem, nullptr, args, nullptr);
    if (err != CUDA_SUCCESS) {
        const char* ename = nullptr;
        cuGetErrorName(err, &ename);
        fprintf(stderr, "[CUDA LAUNCHER] cuLaunchKernel failed: %s\n",
                ename ? ename : "unknown");
        return false;
    }

    err = cuCtxSynchronize();
    if (err != CUDA_SUCCESS) {
        const char* ename = nullptr;
        cuGetErrorName(err, &ename);
        fprintf(stderr, "[CUDA LAUNCHER] cuCtxSynchronize failed: %s\n",
                ename ? ename : "unknown");
        return false;
    }
    return true;
}

// ─── Lanczos upscale ──────────────────────────────────────────────────

bool CudaKernelLauncher::lanczos_upscale(
    const float* d_in, int w, int h,
    float* d_out, int out_w, int out_h, int channels)
{
    dim3 block(16, 16);
    dim3 grid((out_w + 15) / 16, (out_h + 15) / 16);

    void* args[] = {
        &d_in, &w, &h, &d_out, &out_w, &out_h, &channels
    };
    return launch(func_upscale_, grid, block, args);
}

// ─── Adaptive sharpen ─────────────────────────────────────────────────

bool CudaKernelLauncher::adaptive_sharpen(
    const float* d_src, float* d_dst,
    int W, int H, int channels)
{
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    void* args[] = {
        &d_src, &d_dst, &W, &H, &channels
    };
    return launch(func_sharpen_, grid, block, args);
}

// ─── Edge orientation histogram ───────────────────────────────────────

bool CudaKernelLauncher::edge_orientation_histogram(
    const unsigned char* gray, int stride,
    int imgW, int imgH,
    int roi_x, int roi_y, int roi_w, int roi_h,
    float edge_threshold,
    int* global_hist)
{
    dim3 block(16, 16);
    dim3 grid((roi_w + 15) / 16, (roi_h + 15) / 16);
    unsigned int shared_mem = CUDA_HISTOGRAM_BINS * sizeof(int);

    void* args[] = {
        &gray, &stride, &imgW, &imgH,
        &roi_x, &roi_y, &roi_w, &roi_h,
        &edge_threshold, &global_hist
    };
    return launch(func_histogram_, grid, block, args, shared_mem);
}
