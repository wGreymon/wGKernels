#include "cpu/reduce.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace wgkernel::cpu {
namespace {

void check_reduce_args(const float* input, const std::int64_t numel) {
    if (input == nullptr) {
        throw std::invalid_argument("input must not be null");
    }
    if (numel <= 0) {
        throw std::invalid_argument("numel must be positive");
    }
}

}  // namespace

float reduce_max(const float* input, const std::int64_t numel) {
    check_reduce_args(input, numel);

    float max_value = -std::numeric_limits<float>::infinity();
    for (std::int64_t index = 0; index < numel; ++index) {
        max_value = input[index] > max_value ? input[index] : max_value;
    }
    return max_value;
}

}  // namespace wgkernel::cpu
