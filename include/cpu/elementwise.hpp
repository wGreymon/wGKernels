#pragma once

#include <cstdint>

namespace wgkernel::cpu {

void add(const float* lhs, const float* rhs, float* output, std::int64_t numel);

}  // namespace wgkernel::cpu
