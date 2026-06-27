#pragma once

#include <cstdint>

namespace wgkernel::cpu {

void sgemm(
    const float* lhs,
    const float* rhs,
    float* output,
    std::int64_t m,
    std::int64_t n,
    std::int64_t k);

}  // namespace wgkernel::cpu
