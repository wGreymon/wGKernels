#pragma once

#include <cstdint>

namespace wgkernel::cpu {

void batchnorm2d_inference_nchw(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    std::int64_t n,
    std::int64_t c,
    std::int64_t h,
    std::int64_t w);

void rmsnorm(
    const float* input,
    const float* weight,
    float* output,
    std::int64_t outer_size,
    std::int64_t hidden_size,
    float eps);

}  // namespace wgkernel::cpu
