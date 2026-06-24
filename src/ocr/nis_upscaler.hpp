#ifndef NIS_UPSCALER_HPP
#define NIS_UPSCALER_HPP

#include <opencv2/core.hpp>
#include "cuda/cuda_kernel_launcher.hpp"
#include <memory>

class NISUpscaler {
public:
    NISUpscaler();
    ~NISUpscaler();

    bool init(int device_id = 0);
    bool is_available() const { return available_; }
    int scale() const { return 4; }

    bool upscale(const cv::Mat& bgr_input, cv::Mat& bgr_output);

private:
    bool available_ = false;
    int device_id_ = 0;
    CudaKernelLauncher launcher_;

    void* d_input_ = nullptr;
    void* d_temp_ = nullptr;
    void* d_output_ = nullptr;
    int alloc_h_ = 0;
    int alloc_w_ = 0;

    bool ensure_buffers(int h, int w);
    void free_buffers();
};

#endif
