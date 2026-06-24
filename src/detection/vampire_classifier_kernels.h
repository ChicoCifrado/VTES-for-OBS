#ifndef VAMPIRE_CLASSIFIER_KERNELS_H
#define VAMPIRE_CLASSIFIER_KERNELS_H

#include <cuda_runtime.h>

extern "C" {
cudaError_t nv_edge_orientation_histogram(const uint8_t* gray, int stride,
                                          int imgW, int imgH,
                                          int roi_x, int roi_y,
                                          int roi_w, int roi_h,
                                          float edge_threshold,
                                          int* global_hist);
}

#endif
