#pragma once

#include <cuda_runtime.h>

namespace wgkernel::cuda {

cudaError_t maxpool2d_nchw(
    const float* input,
    float* output,
    int n,
    int c,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    int k_h,
    int k_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    cudaStream_t stream = nullptr);

}  // namespace wgkernel::cuda
