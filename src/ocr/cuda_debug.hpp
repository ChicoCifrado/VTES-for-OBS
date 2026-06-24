#ifndef CUDA_DEBUG_HPP
#define CUDA_DEBUG_HPP

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ─── CUDA error checking macro ────────────────────────────────────────
// Usage:
//   CUDA_SAFE(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
//   CUDA_SAFE(cudaGetLastError());                // after kernel launch
//   CUDA_SAFE(cudaDeviceSynchronize());           // wait + check async
//
// On error: logs to stderr (OBS will capture it) and returns false.
//
// ─── Environment variable for extra CUDA driver logging ──────────────
//   Set CUDA_LOG_FILE=stdout or CUDA_LOG_FILE=cuda.log before running
//   OBS to get detailed error messages from the CUDA driver itself
//   (driver r570+). Example:
//     set CUDA_LOG_FILE=stdout
//     obs64.exe
//   This logs errors like invalid block dimensions, illegal memory
//   accesses, etc. even if the application has no error checking.

#ifdef _MSC_VER
#define CUDA_FUNCTION __FUNCTION__
#else
#define CUDA_FUNCTION __func__
#endif

#define CUDA_SAFE(call)                                                 \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr,                                             \
                "[CUDA ERROR] %s:%d (%s) — %s (code %d)\n",             \
                __FILE__, __LINE__, CUDA_FUNCTION,                      \
                cudaGetErrorString(err), (int)err);                     \
            fflush(stderr);                                             \
            return false;                                               \
        }                                                               \
    } while (0)

// Kernel launch + synchronize in one shot
#define CUDA_LAUNCH(kernel, grid, block, ...)                           \
    do {                                                                \
        kernel<<<grid, block>>>(__VA_ARGS__);                           \
        CUDA_SAFE(cudaGetLastError());                                  \
        CUDA_SAFE(cudaDeviceSynchronize());                             \
    } while (0)

#endif
