#include "cpu/reduce.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace wgkernel::cpu {
namespace {

void check_softmax_args(const float* input, const float* output, const std::int64_t numel) {
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("input and output must not be null");
    }
    if (numel <= 0) {
        throw std::invalid_argument("numel must be positive");
    }
}

}  // namespace

void softmax(const float* input, float* output, const std::int64_t numel) {
    check_softmax_args(input, output, numel);

    float max_value = -std::numeric_limits<float>::infinity();
    for (std::int64_t index = 0; index < numel; ++index) {
        max_value = input[index] > max_value ? input[index] : max_value;
    }

    float exp_sum = 0.0F;
    for (std::int64_t index = 0; index < numel; ++index) {
        const float value = std::exp(input[index] - max_value);
        output[index] = value;
        exp_sum += value;
    }

    for (std::int64_t index = 0; index < numel; ++index) {
        output[index] /= exp_sum;
    }
}

}  // namespace wgkernel::cpu
