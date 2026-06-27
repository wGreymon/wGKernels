#include "cpu/activation.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace wgkernel::cpu {
namespace {

void check_unary_args(const float* input, const float* output, const std::int64_t numel) {
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("input and output must not be null");
    }
    if (numel <= 0) {
        throw std::invalid_argument("numel must be positive");
    }
}

}  // namespace

void exp(const float* input, float* output, const std::int64_t numel) {
    check_unary_args(input, output, numel);

    for (std::int64_t index = 0; index < numel; ++index) {
        output[index] = std::exp(input[index]);
    }
}

}  // namespace wgkernel::cpu
