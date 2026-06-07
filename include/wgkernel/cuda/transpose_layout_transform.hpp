#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace wgkernel::cuda {

cudaError_t concat_nchw_axis1(
    const float* a,
    const float* b,
    float* output,
    int n,
    int c_a,
    int c_b,
    int h,
    int w,
    cudaStream_t stream = nullptr);

cudaError_t permute_nchw_to_nhwc(
    const float* input,
    float* output,
    int n,
    int c,
    int h,
    int w,
    cudaStream_t stream = nullptr);

cudaError_t permute_nhwc_to_nchw(
    const float* input,
    float* output,
    int n,
    int c,
    int h,
    int w,
    cudaStream_t stream = nullptr);

cudaError_t copy_1d(const float* input, float* output, std::int64_t numel, cudaStream_t stream = nullptr);

}  // namespace wgkernel::cuda
