#pragma once

#include <cuda_runtime.h>

namespace wgkernel::cuda {

cudaError_t upsample_nearest2d_nchw(
    const float* input,
    float* output,
    int n,
    int c,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    cudaStream_t stream = nullptr);

cudaError_t upsample_bilinear2d_nchw(
    const float* input,
    float* output,
    int n,
    int c,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    bool align_corners,
    cudaStream_t stream = nullptr);

}  // namespace wgkernel::cuda
