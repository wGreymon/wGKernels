#include "cpu/norm.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace wgkernel::cpu {
namespace {

void check_rmsnorm_args(
    const float* input,
    const float* weight,
    const float* output,
    const std::int64_t outer_size,
    const std::int64_t hidden_size,
    const float eps) {
    if (input == nullptr || weight == nullptr || output == nullptr) {
        throw std::invalid_argument("input, weight and output must not be null");
    }
    if (outer_size <= 0 || hidden_size <= 0) {
        throw std::invalid_argument("rmsnorm dimensions must be positive");
    }
    if (eps < 0.0F) {
        throw std::invalid_argument("eps must be non-negative");
    }
}

}  // namespace

void rmsnorm(
    const float* input,
    const float* weight,
    float* output,
    const std::int64_t outer_size,
    const std::int64_t hidden_size,
    const float eps) {
    check_rmsnorm_args(input, weight, output, outer_size, hidden_size, eps);

    for (std::int64_t outer = 0; outer < outer_size; ++outer) {
        const std::int64_t row_offset = outer * hidden_size;
        float square_sum = 0.0F;
        for (std::int64_t dim = 0; dim < hidden_size; ++dim) {
            const float value = input[row_offset + dim];
            square_sum += value * value;
        }

        const float mean_square = square_sum / static_cast<float>(hidden_size);
        const float inv_rms = 1.0F / std::sqrt(mean_square + eps);
        for (std::int64_t dim = 0; dim < hidden_size; ++dim) {
            output[row_offset + dim] = input[row_offset + dim] * inv_rms * weight[dim];
        }
    }
}

}  // namespace wgkernel::cpu
