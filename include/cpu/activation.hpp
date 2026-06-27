#pragma once

#include <cstdint>

namespace wgkernel::cpu {

void silu(const float* input, float* output, std::int64_t numel);
void sigmoid(const float* input, float* output, std::int64_t numel);
void exp(const float* input, float* output, std::int64_t numel);

}  // namespace wgkernel::cpu
