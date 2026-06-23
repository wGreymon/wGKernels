#include "conv2d_internal.cuh"

namespace wgkernel::cuda::conv2d_internal {

// 公共参数校验。参数非法时返回 true。
bool conv2d_args_invalid(
    const float* input,
    const float* weight,
    const float* output,
    const int n,
    const int c_in,
    const int h_in,
    const int w_in,
    const int c_out,
    const int k_h,
    const int k_w,
    const int h_out,
    const int w_out,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int groups) {
    return input == nullptr || weight == nullptr || output == nullptr || n <= 0 || c_in <= 0 ||
        h_in <= 0 || w_in <= 0 || c_out <= 0 || k_h <= 0 || k_w <= 0 || h_out <= 0 || w_out <= 0 ||
        stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0 || groups <= 0 ||
        c_in % groups != 0 || c_out % groups != 0;
}

}  // namespace wgkernel::cuda::conv2d_internal
