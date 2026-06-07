#pragma once

#include <cuda_runtime.h>

namespace wgkernel::cuda {

cudaError_t batchnorm2d_inference_nchw(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    int n,
    int c,
    int h,
    int w,
    cudaStream_t stream = nullptr);

}  // namespace wgkernel::cuda
