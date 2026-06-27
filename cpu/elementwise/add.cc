#include "cpu/elementwise.hpp"

#include <cstdint>
#include <stdexcept>

namespace wgkernel::cpu {
namespace {

void check_binary_args(const float* lhs, const float* rhs, const float* output, const std::int64_t numel) {
    if (lhs == nullptr || rhs == nullptr || output == nullptr) {
        throw std::invalid_argument("lhs, rhs and output must not be null");
    }
    if (numel <= 0) {
        throw std::invalid_argument("numel must be positive");
    }
}

}  // namespace

void add(const float* lhs, const float* rhs, float* output, const std::int64_t numel) {
    check_binary_args(lhs, rhs, output, numel);

    for (std::int64_t index = 0; index < numel; ++index) {
        output[index] = lhs[index] + rhs[index];
    }
}

}  // namespace wgkernel::cpu
