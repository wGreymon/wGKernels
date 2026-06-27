#include "cpu/gemm.hpp"

#include <cstdint>
#include <stdexcept>

namespace wgkernel::cpu {
namespace {

void check_sgemm_args(
    const float* lhs,
    const float* rhs,
    const float* output,
    const std::int64_t m,
    const std::int64_t n,
    const std::int64_t k) {
    if (lhs == nullptr || rhs == nullptr || output == nullptr) {
        throw std::invalid_argument("lhs, rhs and output must not be null");
    }
    if (m <= 0 || n <= 0 || k <= 0) {
        throw std::invalid_argument("m, n and k must be positive");
    }
}

}  // namespace

void sgemm(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::int64_t m,
    const std::int64_t n,
    const std::int64_t k) {
    check_sgemm_args(lhs, rhs, output, m, n, k);

    for (std::int64_t row = 0; row < m; ++row) {
        for (std::int64_t col = 0; col < n; ++col) {
            float accumulator = 0.0F;
            for (std::int64_t inner = 0; inner < k; ++inner) {
                accumulator += lhs[row * k + inner] * rhs[inner * n + col];
            }
            output[row * n + col] = accumulator;
        }
    }
}

}  // namespace wgkernel::cpu
