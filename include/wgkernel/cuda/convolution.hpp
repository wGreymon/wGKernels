#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace wgkernel::cuda {

cudaError_t conv2d_nchw(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int n,
    int c_in,
    int h_in,
    int w_in,
    int c_out,
    int k_h,
    int k_w,
    int h_out,
    int w_out,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    cudaStream_t stream = nullptr);

}  // namespace wgkernel::cuda
