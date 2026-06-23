#pragma once

#include <cuda_runtime.h>

namespace wgkernel::cuda::conv2d_internal {

// 公共参数校验。参数非法时返回 true。
bool conv2d_args_invalid(
    const float* input,
    const float* weight,
    const float* output,
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
    int dilation_h,
    int dilation_w,
    int groups);

}  // namespace wgkernel::cuda::conv2d_internal
