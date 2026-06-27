#include "cpu/norm.hpp"

#include <cstdint>
#include <stdexcept>

namespace wgkernel::cpu {
namespace {

void check_batchnorm_args(
    const float* input,
    const float* scale,
    const float* bias,
    const float* output,
    const std::int64_t n,
    const std::int64_t c,
    const std::int64_t h,
    const std::int64_t w) {
    if (input == nullptr || scale == nullptr || bias == nullptr || output == nullptr) {
        throw std::invalid_argument("input, scale, bias and output must not be null");
    }
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0) {
        throw std::invalid_argument("batchnorm dimensions must be positive");
    }
}

}  // namespace

void batchnorm2d_inference_nchw(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    const std::int64_t n,
    const std::int64_t c,
    const std::int64_t h,
    const std::int64_t w) {
    check_batchnorm_args(input, scale, bias, output, n, c, h, w);

    const std::int64_t spatial = h * w;
    const std::int64_t total = n * c * spatial;
    for (std::int64_t linear = 0; linear < total; ++linear) {
        const std::int64_t channel = (linear / spatial) % c;
        output[linear] = input[linear] * scale[channel] + bias[channel];
    }
}

}  // namespace wgkernel::cpu
