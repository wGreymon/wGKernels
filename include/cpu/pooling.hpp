#pragma once

#include <cstdint>

namespace wgkernel::cpu {

void maxpool2d_nchw(
    const float* input,
    float* output,
    std::int64_t n,
    std::int64_t c,
    std::int64_t h_in,
    std::int64_t w_in,
    std::int64_t h_out,
    std::int64_t w_out,
    std::int64_t k_h,
    std::int64_t k_w,
    std::int64_t stride_h,
    std::int64_t stride_w,
    std::int64_t pad_h,
    std::int64_t pad_w);

}  // namespace wgkernel::cpu
